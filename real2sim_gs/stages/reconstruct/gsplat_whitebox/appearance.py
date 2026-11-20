from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilateralGridModule(nn.Module):
    """Per-view 3D Bilateral Grid for automatic exposure & white-balance compensation.

    各视角网格存放在 `nn.Embedding(sparse=True)` 中：每步只有当前视角行产生稀疏梯度，
    配合 `torch.optim.SparseAdam` 使未访问视角的 Adam 动量不会继续漂移。
    """

    def __init__(
        self,
        num_views: int,
        grid_d: int = 8,
        grid_h: int = 16,
        grid_w: int = 16,
    ) -> None:
        super().__init__()
        self.num_views = max(1, num_views)
        self.grid_shape = (12, grid_d, grid_h, grid_w)
        identity = torch.tensor(
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
            dtype=torch.float32,
        ).view(1, 12, 1, 1, 1)
        init_grids = identity.repeat(self.num_views, 1, grid_d, grid_h, grid_w).contiguous()
        self.register_buffer("identity_grid", identity.clone())
        self.embed = nn.Embedding(self.num_views, init_grids[0].numel(), sparse=True)
        with torch.no_grad():
            self.embed.weight.copy_(init_grids.view(self.num_views, -1))
        self._coord_cache: dict[tuple[int, int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def grids(self) -> torch.Tensor:
        """Dense (V, 12, D, H, W) view of all grids (checkpointing / inspection)."""
        return self.embed.weight.view(self.num_views, *self.grid_shape)

    def grid_for(self, view_idx: int) -> torch.Tensor:
        idx = torch.tensor([view_idx % self.num_views], dtype=torch.long, device=self.embed.weight.device)
        return self.embed(idx).view(1, *self.grid_shape)

    def _xy_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (h, w, device, dtype)
        cached = self._coord_cache.get(key)
        if cached is None:
            xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
            ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
            cached = torch.meshgrid(ys, xs, indexing="ij")
            self._coord_cache[key] = cached
        return cached

    def forward(self, rendered_rgb: torch.Tensor, view_idx: int) -> torch.Tensor:
        """Apply slice-sampled 3x4 affine color transformation conditioned on (x, y, luma)."""
        h, w, _ = rendered_rgb.shape
        grid_y, grid_x = self._xy_grid(h, w, rendered_rgb.device, rendered_rgb.dtype)
        luma = (
            0.299 * rendered_rgb[..., 0]
            + 0.587 * rendered_rgb[..., 1]
            + 0.114 * rendered_rgb[..., 2]
        ).clamp(0.0, 1.0)
        grid_z = luma * 2.0 - 1.0

        sample_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)[None, None, ...]
        sampled = F.grid_sample(
            self.grid_for(view_idx),
            sample_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        affine = sampled[0, :, 0].permute(1, 2, 0).reshape(h, w, 3, 4)
        transformed = torch.matmul(affine[..., :3], rendered_rgb.unsqueeze(-1)).squeeze(-1) + affine[..., 3]
        return transformed.clamp(0.0, 1.0)

    def tv_loss(self, view_idx: int | None = None) -> torch.Tensor:
        """3D total variation regularizer (current view only when `view_idx` is given -> sparse grads)."""
        g = self.grids if view_idx is None else self.grid_for(view_idx)
        diff_d = (g[:, :, 1:] - g[:, :, :-1]).pow(2).mean()
        diff_h = (g[:, :, :, 1:] - g[:, :, :, :-1]).pow(2).mean()
        diff_w = (g[:, :, :, :, 1:] - g[:, :, :, :, :-1]).pow(2).mean()
        return diff_d + diff_h + diff_w

    def identity_loss(self, view_idx: int | None = None) -> torch.Tensor:
        """Anchor affine grids near identity so base SH colors do not drift or invert."""
        g = self.grids if view_idx is None else self.grid_for(view_idx)
        return (g - self.identity_grid).pow(2).mean()


class CameraPoseOptModule(nn.Module):
    """Differentiable per-view SE(3) residual refinement for noisy SfM / feed-forward poses.

    - 残差存于 `nn.Embedding(sparse=True)`，配合 SparseAdam 避免未访问视角动量漂移；
    - 平移分量乘以 `trans_scale`（= scene_extent），使学习率与场景尺度无关；
    - `delta_reg_loss` 提供对当前视角残差的 L2 正则，防止位姿漂移吸收几何误差。
    """

    def __init__(self, num_views: int, trans_scale: float = 1.0) -> None:
        super().__init__()
        self.num_views = max(1, num_views)
        self.trans_scale = float(trans_scale)
        self.embed = nn.Embedding(self.num_views, 6, sparse=True)
        with torch.no_grad():
            self.embed.weight.zero_()

    @property
    def se3_deltas(self) -> torch.Tensor:
        return self.embed.weight

    def _delta(self, view_idx: int) -> torch.Tensor:
        idx = torch.tensor([view_idx % self.num_views], dtype=torch.long, device=self.embed.weight.device)
        return self.embed(idx)[0]

    def delta_reg_loss(self, view_idx: int) -> torch.Tensor:
        return self._delta(view_idx).pow(2).sum()

    def refine_w2c(self, world_to_cam: torch.Tensor, view_idx: int) -> torch.Tensor:
        """Compose a small learnable SE(3) perturbation onto the base 4x4 world_to_cam matrix."""
        delta = self._delta(view_idx)
        omega = delta[:3]
        trans = delta[3:] * self.trans_scale

        theta2 = (omega * omega).sum()
        small = theta2 < 1e-8
        safe_theta2 = torch.where(small, torch.ones_like(theta2), theta2)
        safe_theta = torch.sqrt(safe_theta2)

        c1 = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(safe_theta) / safe_theta)
        c2 = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(safe_theta)) / safe_theta2)

        zero = omega.new_zeros(())
        ox, oy, oz = omega[0], omega[1], omega[2]
        skew = torch.stack(
            [
                torch.stack([zero, -oz, oy]),
                torch.stack([oz, zero, -ox]),
                torch.stack([-oy, ox, zero]),
            ]
        )
        eye3 = torch.eye(3, device=omega.device, dtype=omega.dtype)
        rot = eye3 + c1 * skew + c2 * (skew @ skew)

        top = torch.cat([rot, trans.unsqueeze(-1)], dim=1)
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=omega.device, dtype=omega.dtype)
        delta_mat = torch.cat([top, bottom], dim=0)
        return delta_mat @ world_to_cam
