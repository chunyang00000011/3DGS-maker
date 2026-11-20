from __future__ import annotations

import torch
import torch.nn.functional as F

from real2sim_gs.eval.metrics import ssim_tensor


def l1_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is not None and bool(valid_mask.any()):
        return F.l1_loss(rendered[valid_mask], target[valid_mask])
    return F.l1_loss(rendered, target)


def ssim_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is not None and not bool(valid_mask.all()):
        rendered = torch.where(valid_mask.unsqueeze(-1), rendered, target.detach())
    return 1.0 - ssim_tensor(rendered, target)


def flatness_reg_loss(scales_activated: torch.Tensor) -> torch.Tensor:
    """Encourage thin surfel-like Gaussians (2DGS / PhysGaussian prior) for crisp surfaces."""
    return scales_activated.min(dim=-1).values.mean()


def anisotropy_reg_loss(scales_activated: torch.Tensor, max_ratio: float = 8.0) -> torch.Tensor:
    """Penalize needle-like extreme in-plane scale ratios (s_max / s_mid) while allowing flat surfels."""
    sorted_s = torch.sort(scales_activated, dim=-1).values
    max_s = sorted_s[:, 2]
    mid_s = sorted_s[:, 1].clamp_min(1e-6)
    return F.relu(max_s / mid_s - max_ratio).mean()


def mcmc_reg_loss(
    opacities_activated: torch.Tensor,
    scales_activated: torch.Tensor,
    opacity_weight: float = 0.005,
    scale_weight: float = 0.005,
    max_soft_scale: float = 0.0,
    scale_norm: float = 1.0,
) -> torch.Tensor:
    """3DGS-MCMC L1 sparsity regularizer encouraging compact, scale-invariant Gaussian budgets."""
    denom = max(float(scale_norm), 1e-3)
    reg = opacity_weight * opacities_activated.mean() + scale_weight * (scales_activated / denom).mean()
    if max_soft_scale > 0.0:
        reg = reg + 0.05 * F.relu((scales_activated - max_soft_scale) / denom).pow(2).mean()
    return reg


def edge_aware_depth_smoothness_loss(depth: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    """Penalize noisy relative-depth jitter on smooth surfaces while preserving true object edges."""
    if depth.shape[0] < 2 or depth.shape[1] < 2:
        return depth.new_zeros(())
    depth_scale = depth.detach().clamp_min(0.0).mean().clamp_min(1e-3)
    depth_norm = depth / depth_scale
    grad_dx = torch.abs(depth_norm[:, 1:] - depth_norm[:, :-1])
    grad_dy = torch.abs(depth_norm[1:, :] - depth_norm[:-1, :])
    grad_ix = torch.abs(target_rgb[:, 1:, :] - target_rgb[:, :-1, :]).mean(dim=-1)
    grad_iy = torch.abs(target_rgb[1:, :, :] - target_rgb[:-1, :, :]).mean(dim=-1)
    smooth_x = (grad_dx * torch.exp(-10.0 * grad_ix)).mean()
    smooth_y = (grad_dy * torch.exp(-10.0 * grad_iy)).mean()
    return smooth_x + smooth_y


def compute_total_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    ssim_weight: float = 0.2,
    rendered_depth: torch.Tensor | None = None,
    scales_activated: torch.Tensor | None = None,
    opacities_activated: torch.Tensor | None = None,
    flatness_weight: float = 0.0,
    anisotropy_weight: float = 0.0,
    max_anisotropy_ratio: float = 8.0,
    depth_smooth_weight: float = 0.0,
    mcmc_opacity_weight: float = 0.0,
    mcmc_scale_weight: float = 0.0,
    max_soft_scale: float = 0.0,
    scene_extent: float = 1.0,
    bilateral_tv: torch.Tensor | None = None,
    bilateral_tv_weight: float = 0.0,
    bilateral_identity: torch.Tensor | None = None,
    bilateral_identity_weight: float = 0.0,
    extra_reg: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    compute_breakdown: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Photometric + SOTA geometric/MCMC/appearance regularization loss suite."""
    l1 = l1_loss(rendered, target, valid_mask=valid_mask)
    ssim_term = ssim_loss(rendered, target, valid_mask=valid_mask)
    total = (1.0 - ssim_weight) * l1 + ssim_weight * ssim_term
    breakdown: dict[str, float] = {}
    if compute_breakdown:
        breakdown["l1"] = float(l1.detach().item())
        breakdown["ssim_loss"] = float(ssim_term.detach().item())

    if scales_activated is not None and flatness_weight > 0.0:
        flat_l = flatness_weight * flatness_reg_loss(scales_activated)
        total = total + flat_l
        if compute_breakdown:
            breakdown["flatness"] = float(flat_l.detach().item())

    if scales_activated is not None and anisotropy_weight > 0.0:
        aniso_l = anisotropy_weight * anisotropy_reg_loss(scales_activated, max_ratio=max_anisotropy_ratio)
        total = total + aniso_l
        if compute_breakdown:
            breakdown["anisotropy"] = float(aniso_l.detach().item())

    if (
        opacities_activated is not None
        and scales_activated is not None
        and (mcmc_opacity_weight > 0.0 or mcmc_scale_weight > 0.0)
    ):
        mcmc_l = mcmc_reg_loss(
            opacities_activated,
            scales_activated,
            opacity_weight=mcmc_opacity_weight,
            scale_weight=mcmc_scale_weight,
            max_soft_scale=max_soft_scale,
            scale_norm=scene_extent,
        )
        total = total + mcmc_l
        if compute_breakdown:
            breakdown["mcmc_reg"] = float(mcmc_l.detach().item())

    if rendered_depth is not None and depth_smooth_weight > 0.0:
        depth_l = depth_smooth_weight * edge_aware_depth_smoothness_loss(rendered_depth, target)
        total = total + depth_l
        if compute_breakdown:
            breakdown["depth_smooth"] = float(depth_l.detach().item())

    if bilateral_tv is not None and bilateral_tv_weight > 0.0:
        btv_l = bilateral_tv_weight * bilateral_tv
        total = total + btv_l
        if compute_breakdown:
            breakdown["bilateral_tv"] = float(btv_l.detach().item())

    if bilateral_identity is not None and bilateral_identity_weight > 0.0:
        bid_l = bilateral_identity_weight * bilateral_identity
        total = total + bid_l
        if compute_breakdown:
            breakdown["bilateral_id"] = float(bid_l.detach().item())

    if extra_reg is not None:
        total = total + extra_reg
        if compute_breakdown:
            breakdown["extra_reg"] = float(extra_reg.detach().item())

    if compute_breakdown:
        breakdown["total"] = float(total.detach().item())
    return total, breakdown
