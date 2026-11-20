from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from real2sim_gs.utils.ply_io import SH_C0, GaussianCloud


def quat_to_rotmat(quats_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert normalized (N, 4) (w, x, y, z) quaternions to (N, 3, 3) rotation matrices."""
    q = quats_wxyz / (quats_wxyz.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


def robust_point_extent(points: np.ndarray) -> float:
    """Robust scene radius from a point cloud (90th percentile distance to the median)."""
    if len(points) == 0:
        return 1.0
    center = np.median(points, axis=0)
    radius = float(np.percentile(np.linalg.norm(points - center, axis=1), 90))
    return radius if radius > 1e-6 else 1.0


_SH_DEGREE_TO_REST_COLS = {0: 0, 1: 9, 2: 24, 3: 45}
_REST_COLS_TO_SH_DEGREE = {0: 0, 9: 1, 24: 2, 45: 3}


class GaussianModel(nn.Module):
    """White-box trainable 3D Gaussian primitive container with Mip-Splatting 3D filter & SH0..3."""

    def __init__(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        sh_dc: torch.Tensor,
        sh_rest: torch.Tensor | None = None,
        sh_degree: int = 3,
        max_log_scale: float = 1.5,
    ) -> None:
        super().__init__()
        self.means = nn.Parameter(means.float())
        self.scales = nn.Parameter(scales.float())
        self.quats = nn.Parameter(quats.float())
        self.opacities = nn.Parameter(opacities.float())
        self.sh_dc = nn.Parameter(sh_dc.float())
        if sh_rest is not None:
            if sh_rest.ndim != 2 or int(sh_rest.shape[1]) not in _REST_COLS_TO_SH_DEGREE:
                raise ValueError(
                    f"sh_rest must have 0, 9, 24 or 45 columns, got shape {tuple(sh_rest.shape)}."
                )
            self.max_sh_degree = _REST_COLS_TO_SH_DEGREE[int(sh_rest.shape[1])]
            self.sh_rest = nn.Parameter(sh_rest.float())
        else:
            deg = max(0, min(3, int(sh_degree)))
            self.max_sh_degree = deg
            cols = _SH_DEGREE_TO_REST_COLS[deg]
            self.sh_rest = nn.Parameter(torch.zeros((means.shape[0], cols), dtype=torch.float32, device=means.device))
        self.max_log_scale = float(max_log_scale)
        # Mip-Splatting 逐高斯 3D 低通滤波方差 (N,)；None 表示未启用。拓扑变化后需重新计算。
        self.filter_3d_var: torch.Tensor | None = None

    @classmethod
    def from_point_cloud(
        cls,
        points: np.ndarray,
        colors: np.ndarray,
        device: torch.device,
        init_scale: float = -3.2,
        init_opacity_logit: float = -0.5,
        scene_extent: float | None = None,
        sh_degree: int = 3,
    ) -> GaussianModel:
        if len(points) == 0:
            rng = np.random.default_rng(0)
            points = rng.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32)
            colors = np.full((200, 3), 0.5, dtype=np.float32)

        n = len(points)
        means = torch.tensor(points, dtype=torch.float32, device=device)
        extent = float(scene_extent) if scene_extent and scene_extent > 0 else robust_point_extent(points)

        # k-NN adaptive scale initialization; clamp bounds are relative to scene extent (scale-invariant).
        if n >= 4:
            tree = cKDTree(points)
            dists, _ = tree.query(points, k=4)
            knn_dist = np.clip(dists[:, 1:].mean(axis=1), 1e-6 * extent, 0.1 * extent).astype(np.float32)
            scales_np = np.repeat(np.log(knn_dist)[:, None], 3, axis=1)
            scales = torch.tensor(scales_np, dtype=torch.float32, device=device)
        else:
            scales = torch.full((n, 3), init_scale, dtype=torch.float32, device=device)

        quats = torch.zeros((n, 4), dtype=torch.float32, device=device)
        quats[:, 0] = 1.0
        opacities = torch.full((n,), init_opacity_logit, dtype=torch.float32, device=device)
        rgb = torch.tensor(colors, dtype=torch.float32, device=device).clamp(0.0, 1.0)
        sh_dc = (rgb - 0.5) / SH_C0
        return cls(means=means, scales=scales, quats=quats, opacities=opacities, sh_dc=sh_dc, sh_degree=sh_degree)

    @property
    def num_points(self) -> int:
        return int(self.means.shape[0])

    @property
    def activated_scales(self) -> torch.Tensor:
        return torch.exp(self.scales.clamp(-10.0, self.max_log_scale))

    @torch.no_grad()
    def clamp_scales_(self) -> None:
        """Keep raw log-scales within safe scene bounds to prevent tile intersection blowup."""
        self.scales.data.clamp_(-10.0, self.max_log_scale)

    @property
    def activated_opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacities)

    @property
    def normalized_quats(self) -> torch.Tensor:
        return self.quats / (self.quats.norm(dim=-1, keepdim=True) + 1e-8)

    @property
    def activated_rgb(self) -> torch.Tensor:
        return (self.sh_dc * SH_C0 + 0.5).clamp(0.0, 1.0)

    @property
    def sh_degree(self) -> int:
        return self.max_sh_degree

    @property
    def sh_coeffs(self) -> torch.Tensor:
        """Return (N, (deg+1)^2, 3) degree-0..3 spherical harmonic tensor ordered for gsplat rasterization."""
        if self.sh_rest.shape[1] == 0:
            return self.sh_dc[:, None, :]
        k = int(self.sh_rest.shape[1]) // 3
        rest = self.sh_rest.view(-1, 3, k).transpose(1, 2)  # (N, rgb, band) -> (N, band, rgb)
        return torch.cat([self.sh_dc[:, None, :], rest], dim=1)

    def covariance_3d(self, scales_override: torch.Tensor | None = None) -> torch.Tensor:
        """Compute (N, 3, 3) anisotropic world covariance matrices R @ diag(s^2) @ R^T."""
        rot = quat_to_rotmat(self.normalized_quats)
        s = self.activated_scales if scales_override is None else scales_override
        rot_s = rot * s.unsqueeze(1)
        return torch.bmm(rot_s, rot_s.transpose(1, 2))

    @torch.no_grad()
    def compute_3d_filter(
        self,
        views: Iterable[object],
        filter_variance_scale: float = 0.2,
        near_plane: float = 0.01,
        frustum_margin: float = 0.15,
    ) -> torch.Tensor:
        """Mip-Splatting per-Gaussian 3D filter: var = s · (d / f)² at the max sampling rate ν̂ = max f/d.

        `filter_variance_scale` 即 Mip-Splatting 官方实现中的 0.2（方差系数：filter_std = sqrt(0.2)·d/f）。
        仅统计落在相机视锥（含 15% 边缘余量）内的相机；任何相机都看不到的高斯取已计算值中的最大方差。
        """
        n = self.num_points
        device = self.means.device
        best_rate = torch.zeros(n, dtype=torch.float32, device=device)
        means = self.means.detach()
        for view in views:
            w2c = view.world_to_cam.to(device)  # type: ignore[attr-defined]
            p_cam = means @ w2c[:3, :3].T + w2c[:3, 3]
            z = p_cam[:, 2]
            z_safe = z.clamp_min(1e-8)
            u = view.fx * p_cam[:, 0] / z_safe + view.cx  # type: ignore[attr-defined]
            v = view.fy * p_cam[:, 1] / z_safe + view.cy  # type: ignore[attr-defined]
            w, h = float(view.width), float(view.height)  # type: ignore[attr-defined]
            in_frustum = (
                (z > near_plane)
                & (u >= -frustum_margin * w)
                & (u <= (1.0 + frustum_margin) * w)
                & (v >= -frustum_margin * h)
                & (v <= (1.0 + frustum_margin) * h)
            )
            focal = max(float(view.fx), float(view.fy))  # type: ignore[attr-defined]
            rate = torch.where(in_frustum, focal / z_safe, torch.zeros_like(z))
            best_rate = torch.maximum(best_rate, rate)
        seen = best_rate > 0.0
        var = torch.zeros(n, dtype=torch.float32, device=device)
        if bool(seen.any()):
            var[seen] = filter_variance_scale / (best_rate[seen] ** 2)
            var[~seen] = var[seen].max()
        self.filter_3d_var = var
        return var

    def mip_filtered_primitives(self, mip_3d_var: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Mip-Splatting 3D smoothing: Σ' = Σ + var·I, opacity *= sqrt(det Σ / det Σ') over all 3 axes.

        `mip_3d_var` 为可选全局常量方差（向后兼容），与逐高斯 `filter_3d_var` 相加。
        """
        scales = self.activated_scales
        opac = self.activated_opacities
        var: torch.Tensor | float = float(mip_3d_var) if mip_3d_var > 0.0 else 0.0
        if self.filter_3d_var is not None:
            if self.filter_3d_var.shape[0] != self.num_points:
                raise RuntimeError(
                    f"filter_3d_var size {self.filter_3d_var.shape[0]} != num_points {self.num_points}; "
                    "recompute the 3D filter after topology changes."
                )
            var = self.filter_3d_var.unsqueeze(-1) + var
        if isinstance(var, float) and var <= 0.0:
            return scales, opac

        s2 = scales * scales
        s2_mip = s2 + var
        scales_mip = torch.sqrt(s2_mip)
        det_ratio = (s2.prod(dim=-1) / s2_mip.prod(dim=-1).clamp_min(1e-30)).clamp(0.0, 1.0)
        return scales_mip, opac * torch.sqrt(det_ratio)

    def parameter_groups(self, lr_cfg: dict[str, float]) -> list[dict[str, object]]:
        lr_sh_dc = float(lr_cfg.get("lr_sh_dc", 2.5e-3))
        groups: list[dict[str, object]] = [
            {"params": [self.means], "lr": float(lr_cfg.get("lr_means", 1.6e-4)), "name": "means"},
            {"params": [self.scales], "lr": float(lr_cfg.get("lr_scales", 5e-3)), "name": "scales"},
            {"params": [self.quats], "lr": float(lr_cfg.get("lr_quats", 1e-3)), "name": "quats"},
            {"params": [self.opacities], "lr": float(lr_cfg.get("lr_opacities", 5e-2)), "name": "opacities"},
            {"params": [self.sh_dc], "lr": lr_sh_dc, "name": "sh_dc"},
        ]
        if self.sh_rest.shape[1] > 0:
            groups.append(
                {"params": [self.sh_rest], "lr": float(lr_cfg.get("lr_sh_rest", lr_sh_dc / 20.0)), "name": "sh_rest"}
            )
        return groups

    def named_gaussian_params(self) -> dict[str, nn.Parameter]:
        return {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "sh_dc": self.sh_dc,
            "sh_rest": self.sh_rest,
        }

    @torch.no_grad()
    def select_and_append_(self, src_idx: torch.Tensor, appended: dict[str, torch.Tensor] | None = None) -> None:
        """Rebuild every per-Gaussian parameter as `param[src_idx]` (+ optional appended rows)."""
        for name, param in self.named_gaussian_params().items():
            data = param.data[src_idx]
            if appended is not None and name in appended:
                data = torch.cat([data, appended[name].to(data.dtype)], dim=0)
            setattr(self, name, nn.Parameter(data.contiguous()))
        self.filter_3d_var = None

    @torch.no_grad()
    def to_gaussian_cloud(self, mip_3d_var: float = 0.0, bake_mip_filter: bool = True) -> GaussianCloud:
        """Export to a standard 3DGS cloud; the Mip 3D filter is baked into scales/opacities by default,
        so the PLY rendered by any classic 3DGS viewer matches what was trained."""
        scales_log = self.scales.detach().clamp(-10.0, self.max_log_scale)
        opac_logit = self.opacities.detach()
        if bake_mip_filter and (self.filter_3d_var is not None or mip_3d_var > 0.0):
            scales_mip, opac_mip = self.mip_filtered_primitives(mip_3d_var=mip_3d_var)
            scales_log = torch.log(scales_mip.clamp_min(1e-12))
            opac_logit = torch.logit(opac_mip.clamp(1e-6, 1.0 - 1e-6))
        return GaussianCloud(
            means=self.means.detach().cpu().numpy(),
            scales=scales_log.cpu().numpy(),
            quats=self.normalized_quats.detach().cpu().numpy(),
            opacities=opac_logit.cpu().numpy(),
            sh_dc=self.sh_dc.detach().cpu().numpy(),
            sh_rest=self.sh_rest.detach().cpu().numpy() if self.sh_rest.shape[1] > 0 else None,
        )
