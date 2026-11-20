from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from real2sim_gs.core.doctor import is_gsplat_cuda_ready
from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import PipelineCancelledError, SceneWorkspace, atomic_write_json
from real2sim_gs.eval.metrics import ImagePairEvaluator
from real2sim_gs.stages.reconstruct.gsplat_whitebox.appearance import (
    BilateralGridModule,
    CameraPoseOptModule,
)
from real2sim_gs.stages.reconstruct.gsplat_whitebox.dataset import CameraView, PoseImageDataset
from real2sim_gs.stages.reconstruct.gsplat_whitebox.losses import compute_total_loss
from real2sim_gs.stages.reconstruct.gsplat_whitebox.model import GaussianModel, robust_point_extent
from real2sim_gs.stages.reconstruct.gsplat_whitebox.strategy import (
    DensityControlStrategy,
    MCMCStrategy,
)
from real2sim_gs.utils.ply_io import load_point_cloud_ply, save_gaussian_ply


@dataclass
class RenderOutput:
    """Outputs emitted by a single view rasterization pass."""

    rgb: torch.Tensor           # (H, W, 3)
    depth: torch.Tensor         # (H, W)
    visible_mask: torch.Tensor  # (N,) bool
    meta: dict[str, Any] | None = None


class SelectiveAdam:
    """Visibility-masked sparse Adam optimizer preventing moment drift on occluded Gaussians."""

    def __init__(
        self,
        model: GaussianModel,
        lr_cfg: dict[str, Any],
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-15,
    ) -> None:
        self.betas = betas
        self.eps = eps
        lr_sh_dc = float(lr_cfg.get("lr_sh_dc", 2.5e-3))
        self.lrs: dict[str, float] = {
            "means": float(lr_cfg.get("lr_means", 1.6e-4)),
            "scales": float(lr_cfg.get("lr_scales", 5e-3)),
            "quats": float(lr_cfg.get("lr_quats", 1e-3)),
            "opacities": float(lr_cfg.get("lr_opacities", 5e-2)),
            "sh_dc": lr_sh_dc,
            "sh_rest": float(lr_cfg.get("lr_sh_rest", lr_sh_dc / 20.0)),
        }
        self.params: dict[str, nn.Parameter] = {}
        self.state: dict[str, dict[str, torch.Tensor]] = {}
        self.refresh_parameters(model)

    def refresh_parameters(self, model: GaussianModel) -> None:
        self.params = model.named_gaussian_params()
        self.state = {
            name: {
                "exp_avg": torch.zeros_like(param.data),
                "exp_avg_sq": torch.zeros_like(param.data),
                "steps": torch.zeros(param.shape[0], dtype=torch.long, device=param.device),
            }
            for name, param in self.params.items()
        }

    def reset_param_state(self, name: str) -> None:
        st = self.state.get(name)
        if st is not None:
            st["exp_avg"].zero_()
            st["exp_avg_sq"].zero_()
            st["steps"].zero_()

    def state_dict(self) -> dict[str, Any]:
        return {
            "lrs": dict(self.lrs),
            "state": {
                name: {k: v.detach().cpu() for k, v in st.items()}
                for name, st in self.state.items()
            },
        }

    def load_state_dict(self, payload: dict[str, Any], device: torch.device) -> None:
        lrs = payload.get("lrs")
        if isinstance(lrs, dict):
            for k, v in lrs.items():
                if k in self.lrs:
                    self.lrs[k] = float(v)
        saved_state = payload.get("state")
        if not isinstance(saved_state, dict):
            return
        for name, param in self.params.items():
            st_in = saved_state.get(name)
            if not isinstance(st_in, dict):
                continue
            exp_avg = st_in.get("exp_avg")
            exp_avg_sq = st_in.get("exp_avg_sq")
            steps = st_in.get("steps")
            if (
                isinstance(exp_avg, torch.Tensor)
                and isinstance(exp_avg_sq, torch.Tensor)
                and isinstance(steps, torch.Tensor)
                and exp_avg.shape == param.shape
                and exp_avg_sq.shape == param.shape
                and steps.shape == (param.shape[0],)
            ):
                self.state[name] = {
                    "exp_avg": exp_avg.to(device=device, dtype=param.dtype).contiguous(),
                    "exp_avg_sq": exp_avg_sq.to(device=device, dtype=param.dtype).contiguous(),
                    "steps": steps.to(device=device, dtype=torch.long).contiguous(),
                }

    def set_lr(self, name: str, lr: float) -> None:
        if name in self.lrs:
            self.lrs[name] = float(lr)

    def zero_grad(self) -> None:
        for param in self.params.values():
            param.grad = None

    @torch.no_grad()
    def step(self, visible_mask: torch.Tensor | None = None) -> None:
        beta1, beta2 = self.betas
        if visible_mask is not None and visible_mask.shape[0] > 150_000:
            vis_idx = torch.nonzero(visible_mask, as_tuple=True)[0]
            if vis_idx.numel() == 0:
                return
            for name, param in self.params.items():
                if param.grad is None or param.numel() == 0:
                    continue
                st = self.state[name]
                lr = self.lrs[name]
                g = param.grad.detach()[vis_idx]
                st["steps"][vis_idx] += 1
                steps = st["steps"][vis_idx].float()
                if param.ndim == 2:
                    steps = steps.unsqueeze(-1)

                exp_avg = st["exp_avg"][vis_idx] * beta1 + (1.0 - beta1) * g
                exp_avg_sq = st["exp_avg_sq"][vis_idx] * beta2 + (1.0 - beta2) * (g * g)
                st["exp_avg"][vis_idx] = exp_avg
                st["exp_avg_sq"][vis_idx] = exp_avg_sq

                bc1 = (1.0 - (beta1 ** steps)).clamp_min(1e-8)
                bc2 = (1.0 - (beta2 ** steps)).clamp_min(1e-8)
                m_hat = exp_avg / bc1
                v_hat = exp_avg_sq / bc2
                param.data[vis_idx] -= lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            return

        # Pure-GPU masked Adam update avoids host nonzero() synchronization stalls per step.
        mask_1d = visible_mask
        for name, param in self.params.items():
            if param.grad is None or param.numel() == 0:
                continue
            st = self.state[name]
            lr = self.lrs[name]
            g = param.grad.detach()
            if mask_1d is None:
                st["steps"].add_(1)
                steps = st["steps"].float()
                if param.ndim == 2:
                    steps = steps.unsqueeze(-1)
                st["exp_avg"].mul_(beta1).add_(g, alpha=1.0 - beta1)
                st["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                bc1 = (1.0 - (beta1 ** steps)).clamp_min(1e-8)
                bc2 = (1.0 - (beta2 ** steps)).clamp_min(1e-8)
                param.data.sub_(lr * (st["exp_avg"] / bc1) / (torch.sqrt(st["exp_avg_sq"] / bc2) + self.eps))
            else:
                st["steps"].add_(mask_1d.long())
                steps = st["steps"].clamp_min(1).float()
                m = mask_1d.unsqueeze(-1) if param.ndim == 2 else mask_1d
                if param.ndim == 2:
                    steps = steps.unsqueeze(-1)
                exp_avg = torch.where(m, st["exp_avg"] * beta1 + (1.0 - beta1) * g, st["exp_avg"])
                exp_avg_sq = torch.where(m, st["exp_avg_sq"] * beta2 + (1.0 - beta2) * (g * g), st["exp_avg_sq"])
                st["exp_avg"].copy_(exp_avg)
                st["exp_avg_sq"].copy_(exp_avg_sq)
                bc1 = (1.0 - (beta1 ** steps)).clamp_min(1e-8)
                bc2 = (1.0 - (beta2 ** steps)).clamp_min(1e-8)
                delta = lr * (exp_avg / bc1) / (torch.sqrt(exp_avg_sq / bc2) + self.eps)
                param.data.sub_(torch.where(m, delta, torch.zeros_like(delta)))


def _render_splats_pytorch(
    model: GaussianModel,
    view: CameraView,
    w2c: torch.Tensor,
    mip_3d_var: float = 1e-4,
    pixel_chunk_size: int = 8192,
    splat_chunk_size: int = 1024,
    bg_color: torch.Tensor | None = None,
) -> RenderOutput:
    """Pixel- and splat-chunked differentiable EWA + Mip-filtered Gaussian renderer (unbounded splat count)."""
    device = model.means.device
    h, w = view.height, view.width
    empty_mask = torch.zeros(model.num_points, dtype=torch.bool, device=device)

    def _empty_bg() -> torch.Tensor:
        if bg_color is not None:
            return model.means.sum() * 0.0 + bg_color.view(1, 1, 3).expand(h, w, 3)
        return model.means.sum() * 0.0 + torch.zeros((h, w, 3), device=device, dtype=torch.float32)

    r_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]
    p_cam = model.means @ r_w2c.T + t_w2c
    z = p_cam[:, 2]
    in_front = z > 0.05
    if not in_front.any():
        return RenderOutput(
            rgb=_empty_bg(),
            depth=torch.zeros((h, w), device=device, dtype=torch.float32),
            visible_mask=empty_mask,
        )

    idx = torch.nonzero(in_front, as_tuple=True)[0]
    p_vis = p_cam[idx]
    z_vis = p_vis[:, 2]
    u = view.fx * (p_vis[:, 0] / z_vis) + view.cx
    v = view.fy * (p_vis[:, 1] / z_vis) + view.cy

    in_bounds = (u >= -16.0) & (u <= w + 16.0) & (v >= -16.0) & (v <= h + 16.0)
    if not in_bounds.any():
        return RenderOutput(
            rgb=_empty_bg(),
            depth=torch.zeros((h, w), device=device, dtype=torch.float32),
            visible_mask=empty_mask,
        )

    idx = idx[in_bounds]
    u, v, p_vis = u[in_bounds], v[in_bounds], p_vis[in_bounds]

    order = torch.argsort(p_vis[:, 2])
    idx = idx[order]
    u, v, p_vis = u[order], v[order], p_vis[order]
    x_c, y_c, z_vis = p_vis[:, 0], p_vis[:, 1], p_vis[:, 2]

    visible_mask = empty_mask.clone()
    visible_mask[idx] = True

    scales_mip, opac_mip = model.mip_filtered_primitives(mip_3d_var=mip_3d_var)
    cov3d = model.covariance_3d(scales_override=scales_mip)[idx]
    opac = opac_mip[idx]
    rgb = model.activated_rgb[idx]

    r_exp = r_w2c.unsqueeze(0).expand(idx.shape[0], -1, -1)
    cov_cam = torch.bmm(torch.bmm(r_exp, cov3d), r_exp.transpose(1, 2))

    inv_z = 1.0 / z_vis
    inv_z2 = inv_z * inv_z
    zeros = torch.zeros_like(inv_z)
    j0 = torch.stack([view.fx * inv_z, zeros, -view.fx * x_c * inv_z2], dim=-1)
    j1 = torch.stack([zeros, view.fy * inv_z, -view.fy * y_c * inv_z2], dim=-1)
    jac = torch.stack([j0, j1], dim=1)

    cov2d = torch.bmm(torch.bmm(jac, cov_cam), jac.transpose(1, 2))
    cov_xx = cov2d[:, 0, 0] + 0.3
    cov_xy = cov2d[:, 0, 1]
    cov_yy = cov2d[:, 1, 1] + 0.3
    det = (cov_xx * cov_yy - cov_xy * cov_xy).clamp_min(1e-4)

    conic_xx = cov_yy / det
    conic_xy = -cov_xy / det
    conic_yy = cov_xx / det

    ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    flat_x = grid_x.reshape(-1)
    flat_y = grid_y.reshape(-1)
    total_pixels = flat_x.numel()
    num_vis = idx.shape[0]
    rgb_chunks: list[torch.Tensor] = []
    depth_chunks: list[torch.Tensor] = []
    ones_head = torch.ones((1, 1), device=device, dtype=torch.float32)
    grad_off = not torch.is_grad_enabled()

    for p0 in range(0, total_pixels, pixel_chunk_size):
        p1 = min(total_pixels, p0 + pixel_chunk_size)
        px = flat_x[p0:p1]
        py = flat_y[p0:p1]
        n_pix = p1 - p0

        acc_rgb = torch.zeros((n_pix, 3), device=device, dtype=torch.float32)
        acc_depth = torch.zeros((n_pix,), device=device, dtype=torch.float32)
        t_rem = torch.ones((n_pix,), device=device, dtype=torch.float32)

        for s0 in range(0, num_vis, splat_chunk_size):
            s1 = min(num_vis, s0 + splat_chunk_size)
            dx = px[None, :] - u[s0:s1, None]
            dy = py[None, :] - v[s0:s1, None]
            power = -0.5 * (
                conic_xx[s0:s1, None] * dx * dx
                + 2.0 * conic_xy[s0:s1, None] * dx * dy
                + conic_yy[s0:s1, None] * dy * dy
            )
            alpha = (opac[s0:s1, None] * torch.exp(power.clamp(max=0.0))).clamp(max=0.99)
            one_minus_a = (1.0 - alpha).clamp(min=1e-4)
            cum_in = torch.cumprod(
                torch.cat([ones_head.expand(1, n_pix), one_minus_a[:-1]], dim=0),
                dim=0,
            )
            weights = t_rem[None, :] * cum_in * alpha
            acc_rgb = acc_rgb + (weights.unsqueeze(-1) * rgb[s0:s1, None, :]).sum(dim=0)
            acc_depth = acc_depth + (weights * z_vis[s0:s1, None]).sum(dim=0)
            t_rem = t_rem * one_minus_a.prod(dim=0)
            if grad_off and float(t_rem.max().item()) < 1e-4:
                break

        if bg_color is not None:
            acc_rgb = acc_rgb + t_rem.unsqueeze(-1) * bg_color[None, :]
        rgb_chunks.append(acc_rgb)
        depth_chunks.append(acc_depth)

    rendered_rgb = torch.cat(rgb_chunks, dim=0).view(h, w, 3).clamp(0.0, 1.0)
    rendered_depth = torch.cat(depth_chunks, dim=0).view(h, w)
    return RenderOutput(rgb=rendered_rgb, depth=rendered_depth, visible_mask=visible_mask)


def _render_view(
    model: GaussianModel,
    view: CameraView,
    w2c: torch.Tensor,
    use_gsplat: bool,
    mip_3d_var: float = 0.0,
    bg_color: torch.Tensor | None = None,
    need_absgrad: bool = False,
    need_depth: bool = True,
    sh_degree: int = 0,
    rasterize_mode: str = "classic",
) -> RenderOutput:
    if not use_gsplat:
        return _render_splats_pytorch(model, view, w2c=w2c, mip_3d_var=mip_3d_var, bg_color=bg_color)

    from gsplat.rendering import rasterization

    scales_mip, opac_mip = model.mip_filtered_primitives(mip_3d_var=mip_3d_var)
    k = view.K.to(device=model.means.device)[None]
    use_packed = model.num_points > 32_000
    render_mode = "RGB+ED" if need_depth else "RGB"
    eff_sh = max(0, min(int(sh_degree), model.sh_degree)) if model.sh_degree > 0 else None
    out, alphas, meta = rasterization(
        means=model.means,
        quats=model.normalized_quats,
        scales=scales_mip,
        opacities=opac_mip,
        colors=model.sh_coeffs if eff_sh is not None else model.activated_rgb,
        viewmats=w2c[None],
        Ks=k,
        width=view.width,
        height=view.height,
        sh_degree=eff_sh,
        backgrounds=None,
        rasterize_mode="antialiased" if rasterize_mode == "antialiased" else "classic",
        render_mode=render_mode,
        packed=use_packed,
        absgrad=need_absgrad,
    )
    rgb = out[0, ..., :3]
    if bg_color is not None:
        rgb = rgb + (1.0 - alphas[0]) * bg_color.view(1, 1, 3)
    rgb = rgb.clamp(0.0, 1.0)
    depth = out[0, ..., 3] if need_depth else out.new_zeros((view.height, view.width))

    gids = meta.get("gaussian_ids")
    if gids is not None:
        visible_mask = torch.zeros(model.num_points, dtype=torch.bool, device=model.means.device)
        visible_mask[gids.long()] = True
    elif meta.get("radii") is not None:
        radii = meta["radii"]
        if radii.ndim == 3:
            visible_mask = (radii[0] > 0).all(dim=-1)
        elif radii.ndim >= 2:
            visible_mask = radii[0] > 0
        else:
            visible_mask = torch.ones(model.num_points, dtype=torch.bool, device=model.means.device)
    else:
        visible_mask = torch.ones(model.num_points, dtype=torch.bool, device=model.means.device)

    return RenderOutput(rgb=rgb, depth=depth, visible_mask=visible_mask, meta=meta)


def _extract_per_gaussian_absgrad(
    meta: dict[str, Any] | None,
    num_points: int,
    device: torch.device,
    width: int = 2,
    height: int = 2,
) -> torch.Tensor | None:
    """Convert gsplat pixel-space means2d.absgrad into NDC-equivalent magnitudes per Gaussian (N,)."""
    if not meta or "means2d" not in meta:
        return None
    raw_ag = getattr(meta["means2d"], "absgrad", None)
    if raw_ag is None:
        return None
    raw_ag = raw_ag.detach().float()
    ndc_scale = raw_ag.new_tensor([0.5 * float(width), 0.5 * float(height)])

    gids = meta.get("gaussian_ids")
    if gids is not None:
        scaled = raw_ag[..., :2] * ndc_scale
        out = torch.zeros(num_points, dtype=torch.float32, device=device)
        out.index_add_(0, gids.long(), scaled.norm(dim=-1))
        return out
    if raw_ag.ndim == 3:
        return (raw_ag[0, ..., :2] * ndc_scale).norm(dim=-1)
    if raw_ag.ndim == 2 and raw_ag.shape[0] == num_points:
        return (raw_ag[..., :2] * ndc_scale).norm(dim=-1)
    return None


def _exp_decay_lr(lr_init: float, lr_final: float, step: int, max_steps: int) -> float:
    if max_steps <= 1 or lr_final <= 0.0 or lr_init <= 0.0:
        return lr_init
    t = min(max(step / max(1, max_steps), 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)


def _estimate_init_bg_color(dataset: PoseImageDataset, indices: list[int], device: torch.device) -> torch.Tensor:
    """Estimate initial backdrop RGB from border strips of training views."""
    if not indices:
        return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device)
    border_samples: list[torch.Tensor] = []
    for idx in indices[: min(8, len(indices))]:
        img = dataset[idx].image
        b = max(1, min(4, img.shape[0] // 8, img.shape[1] // 8))
        strips = torch.cat(
            [
                img[:b, :, :].reshape(-1, 3),
                img[-b:, :, :].reshape(-1, 3),
                img[:, :b, :].reshape(-1, 3),
                img[:, -b:, :].reshape(-1, 3),
            ],
            dim=0,
        )
        border_samples.append(strips)
    all_border = torch.cat(border_samples, dim=0)
    return all_border.median(dim=0).values.clamp(0.02, 0.98).to(device)


def _cpu_state_dict(obj: Any) -> Any:
    """Recursively detach/move any tensors inside an optimizer state_dict to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _cpu_state_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_state_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_state_dict(v) for v in obj)
    return obj


def _compute_params_signature(
    params: dict[str, Any],
    *,
    num_views: int,
    test_every: int,
    use_gsplat: bool,
    strategy_mode: str,
    rasterize_mode: str,
    background_mode: str,
    enable_mip_filter: bool,
) -> str:
    relevant = {
        "num_views": num_views,
        "test_every": test_every,
        "use_gsplat": use_gsplat,
        "strategy": strategy_mode,
        "rasterize_mode": rasterize_mode,
        "background_mode": background_mode,
        "enable_mip_filter": enable_mip_filter,
        "iterations": int(params.get("iterations", 1000)),
        "max_points": int(params.get("max_points", 250_000)),
        "max_image_dim": int(params.get("max_image_dim", 1600)),
        "selective_adam": bool(params.get("selective_adam", True)),
        "enable_bilateral_grid": bool(params.get("enable_bilateral_grid", False)),
        "enable_pose_opt": bool(params.get("enable_pose_opt", False)),
        "sh_degree": max(0, min(3, int(params.get("sh_degree", 3 if use_gsplat else 0)))),
        "mip_3d_var": float(params.get("mip_3d_var", 0.0)),
        "mip_filter_scale": float(params.get("mip_filter_scale", 0.2)),
        "mcmc_noise_mode": str(params.get("mcmc_noise_mode", "scaled")).lower(),
    }
    raw = json.dumps(relevant, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _save_reconstruct_checkpoint(
    ws: SceneWorkspace,
    *,
    step: int,
    total_steps: int,
    num_views: int,
    test_every: int,
    params_signature: str,
    last_loss: float,
    strategy_mode: str,
    model: GaussianModel,
    gs_opt: Any,
    sparse_aux_opt: torch.optim.Optimizer | None,
    dense_aux_opt: torch.optim.Optimizer | None,
    adc_strategy: DensityControlStrategy,
    mcmc_strategy: MCMCStrategy,
    rng_perm: torch.Generator,
    view_order: list[int],
    bg_logit: nn.Parameter | None,
    bilateral: BilateralGridModule | None,
    pose_opt: CameraPoseOptModule | None,
) -> None:
    """Persist full training state (weights, optimizer moments, strategy stats, RNG) atomically."""
    ws.gaussians_dir.mkdir(parents=True, exist_ok=True)
    device = model.means.device
    payload: dict[str, Any] = {
        "step": int(step),
        "total_steps": int(total_steps),
        "num_views": int(num_views),
        "test_every": int(test_every),
        "params_signature": params_signature,
        "last_loss": float(last_loss),
        "means": model.means.detach().cpu(),
        "scales": model.scales.detach().cpu(),
        "quats": model.quats.detach().cpu(),
        "opacities": model.opacities.detach().cpu(),
        "sh_dc": model.sh_dc.detach().cpu(),
        "sh_rest": model.sh_rest.detach().cpu(),
        "filter_3d_var": model.filter_3d_var.detach().cpu() if model.filter_3d_var is not None else None,
        "gs_opt_state": _cpu_state_dict(gs_opt.state_dict()),
        "sparse_aux_opt_state": _cpu_state_dict(sparse_aux_opt.state_dict()) if sparse_aux_opt is not None else None,
        "dense_aux_opt_state": _cpu_state_dict(dense_aux_opt.state_dict()) if dense_aux_opt is not None else None,
        "adc_state": adc_strategy.state_dict(),
        "mcmc_state": mcmc_strategy.state_dict(),
        "rng_perm_state": rng_perm.get_state().cpu(),
        "torch_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
        "view_order": [int(v) for v in view_order],
    }
    if bg_logit is not None:
        payload["bg_logit"] = bg_logit.detach().cpu()
    if bilateral is not None:
        payload["bilateral_grids"] = bilateral.grids.detach().cpu()
    if pose_opt is not None:
        payload["pose_se3_deltas"] = pose_opt.se3_deltas.detach().cpu()

    tmp_ckpt = ws.reconstruct_checkpoint_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_ckpt)
    tmp_ckpt.replace(ws.reconstruct_checkpoint_path)

    atomic_write_json(
        ws.reconstruct_checkpoint_meta_path,
        {
            "step": int(step),
            "total_steps": int(total_steps),
            "loss": round(float(last_loss), 5),
            "num_gaussians": int(model.num_points),
            "strategy": strategy_mode,
            "params_signature": params_signature,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _sync_gs_optimizer_state(
    gs_opt: Any,
    model: GaussianModel,
    scaled_params: dict[str, Any],
    make_opt_fn: Any,
    *,
    src_idx: torch.Tensor | None = None,
    n_append: int = 0,
    reset_idx: torch.Tensor | None = None,
) -> Any:
    """Preserve surviving Adam moments across MCMC relocation/grow and ADC clone/split/prune."""
    new_param_map = model.named_gaussian_params()
    if isinstance(gs_opt, SelectiveAdam):
        with torch.no_grad():
            for name, new_param in new_param_map.items():
                st = gs_opt.state.get(name)
                if st is None:
                    gs_opt.refresh_parameters(model)
                    return gs_opt
                exp_avg = st["exp_avg"]
                exp_avg_sq = st["exp_avg_sq"]
                steps = st["steps"]
                if src_idx is not None:
                    exp_avg = exp_avg[src_idx]
                    exp_avg_sq = exp_avg_sq[src_idx]
                    steps = steps[src_idx]
                if reset_idx is not None and reset_idx.numel() > 0:
                    valid_r = reset_idx[reset_idx < exp_avg.shape[0]]
                    if valid_r.numel() > 0:
                        exp_avg[valid_r] = 0.0
                        exp_avg_sq[valid_r] = 0.0
                        steps[valid_r] = 0
                if n_append > 0:
                    exp_avg = torch.cat(
                        [exp_avg, torch.zeros((n_append, *new_param.shape[1:]), dtype=exp_avg.dtype, device=new_param.device)],
                        dim=0,
                    )
                    exp_avg_sq = torch.cat(
                        [exp_avg_sq, torch.zeros((n_append, *new_param.shape[1:]), dtype=exp_avg_sq.dtype, device=new_param.device)],
                        dim=0,
                    )
                    steps = torch.cat(
                        [steps, torch.zeros((n_append,), dtype=steps.dtype, device=new_param.device)],
                        dim=0,
                    )
                if exp_avg.shape != new_param.shape:
                    gs_opt.refresh_parameters(model)
                    return gs_opt
                st["exp_avg"] = exp_avg.contiguous()
                st["exp_avg_sq"] = exp_avg_sq.contiguous()
                st["steps"] = steps.contiguous()
                gs_opt.params[name] = new_param
        return gs_opt

    if hasattr(gs_opt, "param_groups"):
        groups = model.parameter_groups(scaled_params)
        if len(gs_opt.param_groups) != len(groups):
            return make_opt_fn()
        with torch.no_grad():
            for group, grp_cfg in zip(gs_opt.param_groups, groups):
                new_param = grp_cfg["params"][0]
                old_param = group["params"][0]
                st = gs_opt.state.pop(old_param, None)
                group["params"][0] = new_param
                if st and "exp_avg" in st and "exp_avg_sq" in st:
                    exp_avg = st["exp_avg"]
                    exp_avg_sq = st["exp_avg_sq"]
                    if src_idx is not None:
                        exp_avg = exp_avg[src_idx]
                        exp_avg_sq = exp_avg_sq[src_idx]
                    if reset_idx is not None and reset_idx.numel() > 0:
                        valid_r = reset_idx[reset_idx < exp_avg.shape[0]]
                        if valid_r.numel() > 0:
                            exp_avg[valid_r] = 0.0
                            exp_avg_sq[valid_r] = 0.0
                    if n_append > 0:
                        exp_avg = torch.cat(
                            [exp_avg, torch.zeros((n_append, *new_param.shape[1:]), dtype=exp_avg.dtype, device=new_param.device)],
                            dim=0,
                        )
                        exp_avg_sq = torch.cat(
                            [exp_avg_sq, torch.zeros((n_append, *new_param.shape[1:]), dtype=exp_avg_sq.dtype, device=new_param.device)],
                            dim=0,
                        )
                    if exp_avg.shape != new_param.shape:
                        return make_opt_fn()
                    st["exp_avg"] = exp_avg.contiguous()
                    st["exp_avg_sq"] = exp_avg_sq.contiguous()
                    gs_opt.state[new_param] = st
        return gs_opt

    return make_opt_fn()


def _sync_mcmc_optimizer_state(
    gs_opt: Any,
    model: GaussianModel,
    scaled_params: dict[str, Any],
    dead_idx: torch.Tensor | None,
    n_grow: int,
    make_opt_fn: Any,
) -> Any:
    """Backward-compatible alias delegating to `_sync_gs_optimizer_state`."""
    return _sync_gs_optimizer_state(
        gs_opt,
        model,
        scaled_params,
        make_opt_fn,
        src_idx=None,
        n_append=n_grow,
        reset_idx=dead_idx,
    )


def _reset_opacity_optimizer_state(gs_opt: Any, model: GaussianModel) -> None:
    """Zero Adam moments for `opacities` right after a periodic opacity reset."""
    if isinstance(gs_opt, SelectiveAdam):
        gs_opt.reset_param_state("opacities")
        return
    if hasattr(gs_opt, "param_groups"):
        for group in gs_opt.param_groups:
            if group.get("name") == "opacities" or (group.get("params") and group["params"][0] is model.opacities):
                st = gs_opt.state.get(group["params"][0])
                if isinstance(st, dict):
                    if isinstance(st.get("exp_avg"), torch.Tensor):
                        st["exp_avg"].zero_()
                    if isinstance(st.get("exp_avg_sq"), torch.Tensor):
                        st["exp_avg_sq"].zero_()


def _write_refined_cameras_json(
    ws: SceneWorkspace,
    dataset: PoseImageDataset,
    pose_opt: CameraPoseOptModule,
    train_indices: list[int],
) -> Path | None:
    """Persist refined world_to_cam poses for training views while keeping held-out test poses untouched."""
    if not ws.cameras_path.is_file():
        return None
    payload = json.loads(ws.cameras_path.read_text(encoding="utf-8"))
    refined_by_name: dict[str, list[list[float]]] = {}
    with torch.no_grad():
        for idx in train_indices:
            v = dataset.views[idx]
            w2c = pose_opt.refine_w2c(v.world_to_cam, idx).detach().cpu().numpy()
            refined_by_name[v.name] = np.round(w2c, 6).tolist()

    for entry in payload.get("frames", []):
        name = entry.get("filename")
        if name in refined_by_name:
            entry["world_to_cam"] = refined_by_name[name]
            entry["pose_refined"] = True
        else:
            entry["pose_refined"] = False

    payload["pose_opt_applied"] = True
    out_path = ws.gaussians_dir / "refined_cameras.json"
    atomic_write_json(out_path, payload)
    return out_path


@register_stage("reconstruct", "gsplat_whitebox")
class GsplatWhiteboxStage(BaseStage):
    """SOTA white-box 3DGS trainer (MCMC/AbsGS, SelectiveAdam, Mip-Splatting, Sparse Bilateral/PoseOpt)."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        requested_backend = str(params.get("backend", "auto")).lower()
        gsplat_pkg, gsplat_cuda_ready, gsplat_msg = is_gsplat_cuda_ready()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_gsplat = (requested_backend == "gsplat") or (
            requested_backend == "auto" and gsplat_cuda_ready and device.type == "cuda"
        )
        if requested_backend == "gsplat" and not gsplat_pkg:
            raise RuntimeError("backend='gsplat' requested, but package 'gsplat' is not installed.")
        if requested_backend == "gsplat" and not gsplat_cuda_ready:
            raise RuntimeError(f"backend='gsplat' requested, but CUDA extension is not ready: {gsplat_msg}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        max_image_dim = int(params.get("max_image_dim", 1600))
        iterations = int(params.get("iterations", 1000))
        sh_degree = max(0, min(3, int(params.get("sh_degree", 3 if use_gsplat else 0))))
        if not use_gsplat:
            sh_degree = 0
        rasterize_mode = str(params.get("rasterize_mode", "classic")).lower()
        if rasterize_mode not in ("classic", "antialiased"):
            rasterize_mode = "classic"
        if rasterize_mode == "antialiased":
            warnings.warn(
                "rasterize_mode='antialiased' 已启用 2D 抗锯齿补偿；通用 3DGS PLY 查看器默认使用 classic 光栅化。",
                RuntimeWarning,
                stacklevel=2,
            )

        dataset = PoseImageDataset(ws, device=device, max_image_dim=max_image_dim)
        if ws.init_points_path.exists():
            pts, cols = load_point_cloud_ply(ws.init_points_path)
        else:
            pts, cols = load_point_cloud_ply_fallback()

        # Train / test split (Inria convention: every `test_every`th view held out when >=8 views exist).
        test_every = int(params.get("test_every", 8))
        if test_every > 0 and len(dataset) >= 8:
            test_indices = [i for i in range(len(dataset)) if i % test_every == 0]
            train_indices = [i for i in range(len(dataset)) if i % test_every != 0]
        else:
            test_indices = []
            train_indices = list(range(len(dataset)))

        if len(dataset.views) >= 2:
            cam_centers = torch.stack(
                [-v.world_to_cam[:3, :3].T @ v.world_to_cam[:3, 3] for v in dataset.views]
            )
            cam_spread = float(
                torch.linalg.norm(cam_centers - cam_centers.mean(dim=0, keepdim=True), dim=-1).max().item()
            )
            scene_extent = max(1e-3, cam_spread * 1.1) if cam_spread > 1e-4 else max(1e-3, robust_point_extent(pts))
        else:
            scene_extent = max(1e-3, robust_point_extent(pts))

        n_sparse = len(pts)
        n_seed = 0
        if iterations >= 500 and n_sparse >= 32:
            n_seed = min(4000, max(800, n_sparse // 2))
            rng_seed = np.random.default_rng(42)
            p_lo = np.percentile(pts, 5, axis=0)
            p_hi = np.percentile(pts, 95, axis=0)
            seed_pts = rng_seed.uniform(p_lo, p_hi, size=(n_seed, 3)).astype(np.float32)
            mean_col = cols.mean(axis=0, keepdims=True).astype(np.float32)
            seed_cols = np.repeat(mean_col, n_seed, axis=0)
            pts = np.concatenate([pts, seed_pts], axis=0)
            cols = np.concatenate([cols.astype(np.float32), seed_cols], axis=0)

        model = GaussianModel.from_point_cloud(
            pts, cols, device=device, scene_extent=scene_extent, sh_degree=sh_degree
        )
        if n_seed > 0:
            with torch.no_grad():
                model.opacities.data[n_sparse:].fill_(-2.2)

        max_world_scale = max(1e-4, 0.05 * scene_extent)
        max_log_scale = math.log(max_world_scale)
        max_soft_scale = 0.03 * scene_extent
        model.max_log_scale = max_log_scale
        model.clamp_scales_()
        total_vram_bytes = (
            torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
        )

        use_selective_adam = bool(params.get("selective_adam", True))
        lr_means_init = float(params.get("lr_means", 1.6e-4)) * scene_extent
        lr_means_final = float(params.get("lr_means_final", 1.6e-6)) * scene_extent
        scaled_params = {**params, "lr_means": lr_means_init}

        use_native_selective = False
        if use_selective_adam and use_gsplat and device.type == "cuda":
            try:
                from gsplat.optimizers import SelectiveAdam as GsplatSelectiveAdam

                use_native_selective = True
            except ImportError:
                GsplatSelectiveAdam = None  # type: ignore[assignment]
        else:
            GsplatSelectiveAdam = None  # type: ignore[assignment]

        def _make_gs_opt() -> Any:
            if use_native_selective and GsplatSelectiveAdam is not None:
                return GsplatSelectiveAdam(model.parameter_groups(scaled_params), eps=1e-15, betas=(0.9, 0.999))
            if use_selective_adam:
                return SelectiveAdam(model, scaled_params)
            return torch.optim.Adam(model.parameter_groups(scaled_params))

        strategy_mode = str(params.get("strategy", "mcmc")).lower()
        enable_bilateral = bool(params.get("enable_bilateral_grid", False))
        enable_pose_opt = bool(params.get("enable_pose_opt", False))
        enable_mip_filter = bool(params.get("enable_mip_filter", False))
        mip_filter_scale = float(params.get("mip_filter_scale", 0.2))
        mip_3d_var = float(params.get("mip_3d_var", 0.0))

        background_mode = str(params.get("background_mode", "random")).lower()
        if background_mode not in ("random", "learnable", "black", "white"):
            background_mode = "random"

        params_signature = _compute_params_signature(
            params,
            num_views=len(dataset),
            test_every=test_every,
            use_gsplat=use_gsplat,
            strategy_mode=strategy_mode,
            rasterize_mode=rasterize_mode,
            background_mode=background_mode,
            enable_mip_filter=enable_mip_filter,
        )

        bilateral = BilateralGridModule(len(dataset)).to(device) if enable_bilateral else None
        pose_opt = CameraPoseOptModule(len(dataset), trans_scale=scene_extent).to(device) if enable_pose_opt else None
        bg_logit = (
            nn.Parameter(torch.logit(_estimate_init_bg_color(dataset, train_indices, device)))
            if background_mode == "learnable"
            else None
        )
        bg_black = torch.zeros(3, dtype=torch.float32, device=device)
        bg_white = torch.ones(3, dtype=torch.float32, device=device)

        max_points = int(params.get("max_points", 800_000))
        adc_strategy = DensityControlStrategy(max_points=max_points, use_absgrad=True)
        mcmc_strategy = MCMCStrategy(max_points=max_points)

        gs_opt = _make_gs_opt()

        def _make_aux_optimizers() -> tuple[torch.optim.Optimizer | None, torch.optim.Optimizer | None]:
            sparse_groups: list[dict[str, Any]] = []
            if bilateral is not None:
                sparse_groups.append({"params": bilateral.parameters(), "lr": float(params.get("lr_bilateral", 5e-4))})
            if pose_opt is not None:
                sparse_groups.append({"params": pose_opt.parameters(), "lr": float(params.get("lr_pose", 1e-4))})
            sparse_opt = torch.optim.SparseAdam(sparse_groups) if sparse_groups else None
            dense_opt = (
                torch.optim.Adam([{"params": [bg_logit], "lr": float(params.get("lr_background", 5e-3))}])
                if bg_logit is not None
                else None
            )
            return sparse_opt, dense_opt

        sparse_aux_opt, dense_aux_opt = _make_aux_optimizers()

        rng_perm = torch.Generator()
        rng_perm.manual_seed(42)
        view_order: list[int] = []

        last_loss = 0.0
        start_step = 1
        resumed_from_step = 0
        if ws.reconstruct_checkpoint_path.is_file():
            discard_reason: str | None = None
            try:
                ckpt = torch.load(ws.reconstruct_checkpoint_path, map_location=device, weights_only=True)
                saved_step = int(ckpt.get("step", 0))
                if saved_step <= 0 or "means" not in ckpt:
                    discard_reason = "断点文件缺少有效训练步数"
                elif int(ckpt.get("num_views", -1)) != len(dataset):
                    discard_reason = f"视角数变化 ({ckpt.get('num_views')} != {len(dataset)})"
                elif str(ckpt.get("params_signature", "")) != params_signature:
                    discard_reason = "训练超参已修改"
                elif int(ckpt.get("test_every", -1)) != test_every:
                    discard_reason = "训练/测试划分配置已修改"
                else:
                    model = GaussianModel(
                        means=ckpt["means"].to(device),
                        scales=ckpt["scales"].to(device),
                        quats=ckpt["quats"].to(device),
                        opacities=ckpt["opacities"].to(device),
                        sh_dc=ckpt["sh_dc"].to(device),
                        sh_rest=ckpt["sh_rest"].to(device) if "sh_rest" in ckpt else None,
                        max_log_scale=max_log_scale,
                    )
                    model.clamp_scales_()
                    if isinstance(ckpt.get("filter_3d_var"), torch.Tensor):
                        model.filter_3d_var = ckpt["filter_3d_var"].to(device)
                    with torch.no_grad():
                        if bg_logit is not None and isinstance(ckpt.get("bg_logit"), torch.Tensor) and ckpt["bg_logit"].shape == bg_logit.shape:
                            bg_logit.copy_(ckpt["bg_logit"].to(device))
                        if (
                            bilateral is not None
                            and isinstance(ckpt.get("bilateral_grids"), torch.Tensor)
                            and ckpt["bilateral_grids"].shape == bilateral.grids.shape
                        ):
                            bilateral.embed.weight.copy_(ckpt["bilateral_grids"].to(device).view(bilateral.num_views, -1))
                        if (
                            pose_opt is not None
                            and isinstance(ckpt.get("pose_se3_deltas"), torch.Tensor)
                            and ckpt["pose_se3_deltas"].shape == pose_opt.se3_deltas.shape
                        ):
                            pose_opt.embed.weight.copy_(ckpt["pose_se3_deltas"].to(device))

                    gs_opt = _make_gs_opt()
                    gs_state = ckpt.get("gs_opt_state")
                    if isinstance(gs_state, dict):
                        if isinstance(gs_opt, SelectiveAdam):
                            gs_opt.load_state_dict(gs_state, device=device)
                        else:
                            gs_opt.load_state_dict(gs_state)

                    sparse_aux_opt, dense_aux_opt = _make_aux_optimizers()
                    if sparse_aux_opt is not None and isinstance(ckpt.get("sparse_aux_opt_state"), dict):
                        sparse_aux_opt.load_state_dict(ckpt["sparse_aux_opt_state"])
                    if dense_aux_opt is not None and isinstance(ckpt.get("dense_aux_opt_state"), dict):
                        dense_aux_opt.load_state_dict(ckpt["dense_aux_opt_state"])

                    if isinstance(ckpt.get("adc_state"), dict):
                        adc_strategy.load_state_dict(ckpt["adc_state"], device=device)
                    if isinstance(ckpt.get("mcmc_state"), dict):
                        mcmc_strategy.load_state_dict(ckpt["mcmc_state"])

                    if isinstance(ckpt.get("rng_perm_state"), torch.Tensor):
                        rng_perm.set_state(ckpt["rng_perm_state"].cpu())
                    if isinstance(ckpt.get("torch_rng_state"), torch.Tensor):
                        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
                    if device.type == "cuda" and isinstance(ckpt.get("cuda_rng_state"), torch.Tensor):
                        torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu(), device)
                    if isinstance(ckpt.get("view_order"), list):
                        view_order = [int(v) for v in ckpt["view_order"]]

                    last_loss = float(ckpt.get("last_loss", 0.0))
                    resumed_from_step = saved_step
                    start_step = saved_step + 1
            except Exception as exc:  # noqa: BLE001
                discard_reason = f"读取旧断点失败 ({exc})"

            if discard_reason is not None and resumed_from_step == 0:
                ws.report_progress(0.02, f"旧断点已失效（{discard_reason}），将从第 1 步重新训练...")
                ws.reconstruct_checkpoint_path.unlink(missing_ok=True)
                ws.reconstruct_checkpoint_meta_path.unlink(missing_ok=True)
                start_step = 1
                resumed_from_step = 0

        if adc_strategy.grad_accum is None or adc_strategy.grad_accum.shape[0] != model.num_points:
            adc_strategy.reset_stats(model)

        train_views_list = [dataset[i] for i in train_indices]
        if enable_mip_filter and model.filter_3d_var is None:
            model.compute_3d_filter(train_views_list, filter_variance_scale=mip_filter_scale)

        sh_degree_interval = max(1, int(params.get("sh_degree_interval", 1000)))
        ssim_weight = float(params.get("ssim_weight", 0.2))
        flatness_w = float(params.get("flatness_weight", 0.0))
        anisotropy_w = float(params.get("anisotropy_weight", 0.005))
        depth_smooth_w = float(params.get("depth_smooth_weight", 0.0))
        mcmc_opac_w = float(params.get("mcmc_opacity_weight", 0.005)) if strategy_mode == "mcmc" else 0.0
        mcmc_scale_w = float(params.get("mcmc_scale_weight", 0.005)) if strategy_mode == "mcmc" else 0.0
        bilateral_tv_w = float(params.get("bilateral_tv_weight", 1e-2))
        bilateral_id_w = float(params.get("bilateral_identity_weight", 0.1))
        pose_reg_w = float(params.get("pose_reg_weight", 1e-4))

        densify_from = int(params.get("densify_from_iter", 500))
        default_densify_until = max(densify_from + 1, int(iterations * 0.80)) if iterations >= 100 else iterations
        densify_until = int(params.get("densify_until_iter", default_densify_until))
        effective_densify_until = (
            min(densify_until, max(densify_from + 1, int(iterations * 0.80)))
            if iterations >= 100
            else densify_until
        )
        mid_cleanup_step = (
            effective_densify_until + max(1, (iterations - effective_densify_until) // 3)
            if iterations >= 500 and effective_densify_until < iterations
            else -1
        )
        densify_interval = max(1, int(params.get("densify_interval", 100)))
        default_grad_thresh = 4e-4 if strategy_mode == "adc_absgs" else 2e-4
        grad_thresh = float(params.get("grad_threshold", default_grad_thresh))
        min_opac = float(params.get("min_opacity", 0.005))
        mcmc_grow_ratio = float(params.get("mcmc_grow_ratio", 0.06))
        mcmc_noise_scale = float(params.get("mcmc_noise_scale", 200.0))
        mcmc_noise_mode = str(params.get("mcmc_noise_mode", "scaled")).lower()
        opacity_reset_interval = int(params.get("opacity_reset_interval", 3000))

        t_train_start = time.perf_counter()
        report_every = max(1, min(250, iterations // 60))
        num_train = max(1, len(train_indices))
        if resumed_from_step > 0:
            ws.report_progress(
                min(0.97, resumed_from_step / max(1, iterations)),
                f"从上次断点第 {resumed_from_step:,}/{iterations:,} 步继续训练 · 高斯数 {model.num_points:,}",
                {
                    "step": resumed_from_step,
                    "total_steps": iterations,
                    "loss": round(last_loss, 5),
                    "num_gaussians": model.num_points,
                },
            )

        completed_step = resumed_from_step
        try:
            for step in range(start_step, iterations + 1):
                if step % 5 == 0 or step == start_step:
                    ws.check_cancelled()

                offset = (step - 1) % num_train
                if offset == 0 or len(view_order) != num_train:
                    view_order = [train_indices[i] for i in torch.randperm(num_train, generator=rng_perm).tolist()]
                view_idx = view_order[offset]
                view = dataset[view_idx]

                cur_lr_means = _exp_decay_lr(lr_means_init, lr_means_final, step, iterations)
                if isinstance(gs_opt, SelectiveAdam):
                    gs_opt.set_lr("means", cur_lr_means)
                    gs_opt.zero_grad()
                else:
                    gs_opt.param_groups[0]["lr"] = cur_lr_means
                    gs_opt.zero_grad(set_to_none=True)

                if sparse_aux_opt is not None:
                    sparse_aux_opt.zero_grad(set_to_none=True)
                if dense_aux_opt is not None:
                    dense_aux_opt.zero_grad(set_to_none=True)

                w2c = pose_opt.refine_w2c(view.world_to_cam, view_idx) if pose_opt is not None else view.world_to_cam
                if background_mode == "random":
                    active_bg = torch.rand(3, dtype=torch.float32, device=device)
                elif background_mode == "white":
                    active_bg = bg_white
                elif background_mode == "black":
                    active_bg = bg_black
                elif bg_logit is not None:
                    active_bg = torch.sigmoid(bg_logit)
                else:
                    active_bg = bg_black
                active_sh = (
                    (sh_degree if iterations < 50 else min(sh_degree, step // sh_degree_interval))
                    if use_gsplat
                    else 0
                )
                need_absgrad = use_gsplat and (densify_from <= step <= effective_densify_until)
                apply_depth_reg = (iterations < 100) or (step % 4 == 0)
                need_depth = (not use_gsplat) or (apply_depth_reg and depth_smooth_w > 0.0)

                try:
                    render_out = _render_view(
                        model,
                        view,
                        w2c=w2c,
                        use_gsplat=use_gsplat,
                        mip_3d_var=mip_3d_var,
                        bg_color=active_bg,
                        need_absgrad=need_absgrad,
                        need_depth=need_depth,
                        sh_degree=active_sh,
                        rasterize_mode=rasterize_mode,
                    )
                except Exception as exc:  # noqa: BLE001
                    if use_gsplat and requested_backend == "auto" and step == start_step and model.num_points <= 20_000:
                        ws.report_progress(
                            0.04,
                            f"gsplat CUDA 初始化回退到 PyTorch EWA 渲染器: {exc}",
                        )
                        use_gsplat = False
                        active_sh = 0
                        render_out = _render_view(
                            model,
                            view,
                            w2c=w2c,
                            use_gsplat=False,
                            mip_3d_var=mip_3d_var,
                            bg_color=active_bg,
                            need_absgrad=need_absgrad,
                            need_depth=True,
                            sh_degree=0,
                        )
                    else:
                        raise

                pred_rgb = bilateral(render_out.rgb, view_idx) if bilateral is not None else render_out.rgb
                btv = bilateral.tv_loss(view_idx) if bilateral is not None else None
                bid = bilateral.identity_loss(view_idx) if bilateral is not None else None
                pose_reg = (
                    pose_reg_w * pose_opt.delta_reg_loss(view_idx)
                    if (pose_opt is not None and pose_reg_w > 0.0)
                    else None
                )
                gt_img = view.image

                should_report = step == start_step or step % report_every == 0 or step == iterations
                loss, breakdown = compute_total_loss(
                    rendered=pred_rgb,
                    target=gt_img,
                    ssim_weight=ssim_weight,
                    rendered_depth=render_out.depth if need_depth else None,
                    scales_activated=model.activated_scales,
                    opacities_activated=model.activated_opacities,
                    flatness_weight=flatness_w,
                    anisotropy_weight=anisotropy_w,
                    depth_smooth_weight=depth_smooth_w if apply_depth_reg else 0.0,
                    mcmc_opacity_weight=mcmc_opac_w,
                    mcmc_scale_weight=mcmc_scale_w,
                    max_soft_scale=max_soft_scale if strategy_mode == "mcmc" else 0.0,
                    scene_extent=scene_extent,
                    bilateral_tv=btv,
                    bilateral_tv_weight=bilateral_tv_w,
                    bilateral_identity=bid,
                    bilateral_identity_weight=bilateral_id_w,
                    extra_reg=pose_reg,
                    valid_mask=view.valid_mask,
                    compute_breakdown=should_report,
                )
                loss.backward()
                if should_report and "total" in breakdown:
                    last_loss = breakdown["total"]

                if densify_from <= step <= effective_densify_until:
                    per_gs_absgrad = _extract_per_gaussian_absgrad(
                        render_out.meta,
                        model.num_points,
                        model.means.device,
                        width=view.width,
                        height=view.height,
                    )
                    adc_strategy.record_gradients(
                        model,
                        absgrad_override=per_gs_absgrad,
                        visible_mask=render_out.visible_mask,
                        allow_3d_fallback=not use_gsplat,
                    )

                if use_native_selective and GsplatSelectiveAdam is not None and isinstance(gs_opt, GsplatSelectiveAdam):
                    gs_opt.step(visibility=render_out.visible_mask)
                elif isinstance(gs_opt, SelectiveAdam):
                    gs_opt.step(visible_mask=render_out.visible_mask)
                else:
                    gs_opt.step()

                if sparse_aux_opt is not None:
                    sparse_aux_opt.step()
                if dense_aux_opt is not None:
                    dense_aux_opt.step()

                if step % 25 == 0:
                    model.clamp_scales_()

                if strategy_mode == "mcmc" and step <= effective_densify_until:
                    if step >= densify_from and step % densify_interval == 0:
                        grad_scores = None
                        if adc_strategy.grad_accum is not None and adc_strategy.grad_count is not None:
                            grad_scores = adc_strategy.grad_accum / adc_strategy.grad_count.clamp_min(1.0)
                        changed_mcmc = mcmc_strategy.relocate_and_grow(
                            model,
                            min_opacity=min_opac,
                            grow_ratio=mcmc_grow_ratio,
                            grad_scores=grad_scores,
                        )
                        adc_strategy.reset_stats(model)
                        if changed_mcmc:
                            gs_opt = _sync_gs_optimizer_state(
                                gs_opt,
                                model,
                                scaled_params,
                                _make_gs_opt,
                                src_idx=None,
                                n_append=mcmc_strategy.last_num_grown,
                                reset_idx=mcmc_strategy.last_reset_indices,
                            )
                            if enable_mip_filter:
                                model.compute_3d_filter(train_views_list, filter_variance_scale=mip_filter_scale)
                            if device.type == "cuda" and (
                                step % 500 == 0
                                or (total_vram_bytes > 0 and torch.cuda.memory_reserved(device) > 0.78 * total_vram_bytes)
                            ):
                                torch.cuda.empty_cache()
                    mcmc_strategy.inject_langevin_noise(
                        model,
                        lr_means=cur_lr_means / max(1e-3, scene_extent),
                        noise_scale=mcmc_noise_scale,
                        noise_mode=mcmc_noise_mode,
                    )
                    model.clamp_scales_()
                elif strategy_mode == "adc_absgs" and densify_from <= step <= effective_densify_until:
                    if step % densify_interval == 0:
                        if adc_strategy.step_densify_and_prune(
                            model,
                            grad_threshold=grad_thresh,
                            min_opacity=min_opac,
                            split_scale_threshold=0.01 * scene_extent,
                        ):
                            gs_opt = _sync_gs_optimizer_state(
                                gs_opt,
                                model,
                                scaled_params,
                                _make_gs_opt,
                                src_idx=adc_strategy.last_survivor_indices,
                                n_append=adc_strategy.last_num_appended,
                            )
                            if enable_mip_filter:
                                model.compute_3d_filter(train_views_list, filter_variance_scale=mip_filter_scale)
                            if device.type == "cuda" and (
                                step % 500 == 0
                                or (total_vram_bytes > 0 and torch.cuda.memory_reserved(device) > 0.78 * total_vram_bytes)
                            ):
                                torch.cuda.empty_cache()
                    if opacity_reset_interval > 0 and step % opacity_reset_interval == 0:
                        adc_strategy.reset_opacities(model)
                        _reset_opacity_optimizer_state(gs_opt, model)

                # 冷却期中段清理真正透明飘浮点与针刺，并留出剩余 ~13% 步数让邻近高斯自愈补洞。
                if step == mid_cleanup_step and step < iterations and model.num_points > 64:
                    with torch.no_grad():
                        prune_opac = max(min_opac, 0.015)
                        s_sorted = torch.sort(model.activated_scales, dim=-1).values
                        s_max = s_sorted[:, 2]
                        s_mid = s_sorted[:, 1].clamp_min(1e-6)
                        needle_mask = ((s_max / s_mid) > 15.0) & (s_max > 0.04 * scene_extent)
                        keep = (model.activated_opacities >= prune_opac) & (~needle_mask)
                        if int(keep.sum().item()) >= 32 and not bool(keep.all().item()):
                            keep_idx = torch.nonzero(keep, as_tuple=True)[0]
                            model.select_and_append_(keep_idx)
                            gs_opt = _sync_gs_optimizer_state(
                                gs_opt,
                                model,
                                scaled_params,
                                _make_gs_opt,
                                src_idx=keep_idx,
                            )
                            adc_strategy.reset_stats(model)
                            if enable_mip_filter:
                                model.compute_3d_filter(train_views_list, filter_variance_scale=mip_filter_scale)

                completed_step = step
                if step % 1000 == 0 and step < iterations:
                    _save_reconstruct_checkpoint(
                        ws=ws,
                        step=completed_step,
                        total_steps=iterations,
                        num_views=len(dataset),
                        test_every=test_every,
                        params_signature=params_signature,
                        last_loss=last_loss,
                        strategy_mode=strategy_mode,
                        model=model,
                        gs_opt=gs_opt,
                        sparse_aux_opt=sparse_aux_opt,
                        dense_aux_opt=dense_aux_opt,
                        adc_strategy=adc_strategy,
                        mcmc_strategy=mcmc_strategy,
                        rng_perm=rng_perm,
                        view_order=view_order,
                        bg_logit=bg_logit,
                        bilateral=bilateral,
                        pose_opt=pose_opt,
                    )

                if should_report:
                    elapsed_s = max(1e-6, time.perf_counter() - t_train_start)
                    steps_done_session = max(1, step - start_step + 1)
                    it_s = steps_done_session / elapsed_s
                    eta_s = (iterations - step) / max(it_s, 1e-6)
                    vram_mb = round(torch.cuda.memory_allocated(device) / (1024 * 1024), 1) if device.type == "cuda" else 0.0
                    ws.report_progress(
                        step / max(1, iterations),
                        f"迭代 {step:,}/{iterations:,} · Loss {last_loss:.4f} · 高斯数 {model.num_points:,}",
                        {
                            "step": step,
                            "total_steps": iterations,
                            "loss": round(last_loss, 5),
                            "num_gaussians": model.num_points,
                            "it_per_sec": round(it_s, 1),
                            "eta_sec": round(eta_s, 1),
                            "vram_mb": vram_mb,
                        },
                    )
        except PipelineCancelledError:
            if completed_step > 0:
                _save_reconstruct_checkpoint(
                    ws=ws,
                    step=completed_step,
                    total_steps=iterations,
                    num_views=len(dataset),
                    test_every=test_every,
                    params_signature=params_signature,
                    last_loss=last_loss,
                    strategy_mode=strategy_mode,
                    model=model,
                    gs_opt=gs_opt,
                    sparse_aux_opt=sparse_aux_opt,
                    dense_aux_opt=dense_aux_opt,
                    adc_strategy=adc_strategy,
                    mcmc_strategy=mcmc_strategy,
                    rng_perm=rng_perm,
                    view_order=view_order,
                    bg_logit=bg_logit,
                    bilateral=bilateral,
                    pose_opt=pose_opt,
                )
                save_gaussian_ply(
                    ws.raw_splats_path,
                    model.to_gaussian_cloud(mip_3d_var=mip_3d_var, bake_mip_filter=True),
                )
            raise

        # Post-loop prune: only remove completely dead dust so converged surfaces stay intact.
        if iterations >= 200 and model.num_points > 64:
            with torch.no_grad():
                dead_thresh = max(min_opac, 0.01)
                keep = model.activated_opacities >= dead_thresh
                if int(keep.sum().item()) >= 32 and not bool(keep.all().item()):
                    model.select_and_append_(torch.nonzero(keep, as_tuple=True)[0])
                    if enable_mip_filter:
                        model.compute_3d_filter(train_views_list, filter_variance_scale=mip_filter_scale)

        if enable_mip_filter and model.filter_3d_var is None:
            model.compute_3d_filter(train_views_list, filter_variance_scale=mip_filter_scale)

        ws.report_progress(0.98, "正在渲染训练/测试视角并计算 PSNR / SSIM 指标...")
        renders_dir = ws.gaussians_dir / "renders"
        if renders_dir.exists():
            for old_img in renders_dir.glob("*.png"):
                old_img.unlink(missing_ok=True)
        renders_dir.mkdir(parents=True, exist_ok=True)

        if background_mode == "learnable" and bg_logit is not None:
            eval_bg = torch.sigmoid(bg_logit).detach()
        elif background_mode == "white":
            eval_bg = bg_white
        else:
            eval_bg = bg_black
        eval_sh = sh_degree if use_gsplat else 0
        max_saved = min(8, len(dataset.views))
        save_indices = (
            set(np.linspace(0, len(dataset.views) - 1, num=max_saved, dtype=int).tolist())
            if max_saved > 0
            else set()
        )
        saved_count = 0
        train_eval = ImagePairEvaluator()
        test_eval = ImagePairEvaluator()
        test_set = set(test_indices)
        with torch.no_grad():
            for idx_v, v in enumerate(dataset.views):
                is_test = idx_v in test_set
                # 测试视角禁止泄漏训练阶段学到的逐视角 pose_opt 和 bilateral 参数。
                w2c = v.world_to_cam if (is_test or pose_opt is None) else pose_opt.refine_w2c(v.world_to_cam, idx_v)
                r_out = _render_view(
                    model,
                    v,
                    w2c=w2c,
                    use_gsplat=use_gsplat,
                    mip_3d_var=mip_3d_var,
                    bg_color=eval_bg,
                    need_depth=False,
                    sh_degree=eval_sh,
                    rasterize_mode=rasterize_mode,
                )
                pred_img = (
                    r_out.rgb
                    if (is_test or bilateral is None)
                    else bilateral(r_out.rgb, idx_v)
                )
                gt_img = v.image
                if is_test:
                    test_eval.add(pred_img, gt_img)
                else:
                    train_eval.add(pred_img, gt_img)
                if idx_v in save_indices:
                    rgb_u8 = (pred_img.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
                    Image.fromarray(rgb_u8).save(renders_dir / f"render_{saved_count:04d}.png")
                    saved_count += 1

        train_summary = train_eval.summary()
        has_test = len(test_eval) > 0
        test_summary = test_eval.summary() if has_test else None
        primary_summary = test_summary if test_summary is not None else train_summary
        eval_metrics: dict[str, Any] = {
            "PSNR": primary_summary["PSNR"],
            "SSIM": primary_summary["SSIM"],
            "eval_views": int(primary_summary["eval_views"]),
            "eval_split": "test" if has_test else "train",
            "train_PSNR": train_summary["PSNR"],
            "train_SSIM": train_summary["SSIM"],
            "num_train_views": int(train_summary["eval_views"]),
            "num_test_views": len(test_eval),
        }
        if test_summary is not None:
            eval_metrics["test_PSNR"] = test_summary["PSNR"]
            eval_metrics["test_SSIM"] = test_summary["SSIM"]

        refined_cam_path = (
            _write_refined_cameras_json(ws, dataset, pose_opt, train_indices)
            if pose_opt is not None
            else None
        )

        cloud = model.to_gaussian_cloud(mip_3d_var=mip_3d_var, bake_mip_filter=True)
        del dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        save_gaussian_ply(ws.raw_splats_path, cloud)
        ws.reconstruct_checkpoint_path.unlink(missing_ok=True)
        ws.reconstruct_checkpoint_meta_path.unlink(missing_ok=True)

        metrics: dict[str, Any] = {
            **eval_metrics,
            "num_gaussians": cloud.num_points,
            "final_loss": round(last_loss, 5),
            "render_backend": f"gsplat_{rasterize_mode}" if use_gsplat else f"pytorch_ewa_mip_{device.type}",
            "rasterize_mode": rasterize_mode,
            "fallback_renderer_used": not use_gsplat,
            "background_mode": background_mode,
            "sh_degree": sh_degree,
            "strategy": strategy_mode,
            "selective_adam": use_selective_adam,
            "bilateral_grid": enable_bilateral,
            "pose_opt": enable_pose_opt,
            "mip_filter": enable_mip_filter,
            "num_renders": saved_count,
            "resumed_from_step": resumed_from_step,
        }
        if not use_gsplat:
            metrics["renderer_warning"] = "未使用 gsplat CUDA 光栅化，已通过全可见高斯分块 PyTorch EWA 回退渲染。"
        if rasterize_mode == "antialiased":
            metrics["viewer_note"] = "模型使用 antialiased 2D 过滤训练，部分仅支持 classic 光栅化的外部查看器外观可能略有差异。"
        if background_mode == "learnable" and bg_logit is not None:
            metrics["background_rgb"] = [round(float(c), 4) for c in eval_bg.cpu().tolist()]
        metrics_path = ws.gaussians_dir / "metrics.json"
        atomic_write_json(metrics_path, metrics)

        artifacts: dict[str, str] = {
            "raw_splats": ws.to_relative(ws.raw_splats_path),
            "metrics": ws.to_relative(metrics_path),
            "renders_dir": ws.to_relative(renders_dir),
        }
        if refined_cam_path is not None:
            artifacts["refined_cameras"] = ws.to_relative(refined_cam_path)

        return StageResult(
            artifacts=artifacts,
            metrics=metrics,
            summary=(
                f"Trained {cloud.num_points} Gaussians ({metrics['render_backend']}, {strategy_mode}) | "
                f"PSNR={metrics['PSNR']}, SSIM={metrics['SSIM']} ({eval_metrics['eval_split']})."
            ),
        )


def load_point_cloud_ply_fallback() -> tuple[Any, Any]:
    rng = np.random.default_rng(42)
    return rng.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32), np.full((200, 3), 0.5, dtype=np.float32)
