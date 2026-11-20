from __future__ import annotations

import math
from typing import Any

import torch

from real2sim_gs.core.doctor import is_gsplat_cuda_ready
from real2sim_gs.stages.reconstruct.gsplat_whitebox.model import GaussianModel, quat_to_rotmat

_MAX_RELOC_BINOM = 51
_BINOM_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _get_binom_table(device: torch.device, n_max: int = _MAX_RELOC_BINOM) -> torch.Tensor:
    key = (device, n_max)
    cached = _BINOM_CACHE.get(key)
    if cached is not None:
        return cached
    table = torch.zeros((n_max, n_max), dtype=torch.float32, device="cpu")
    for n in range(n_max):
        for k in range(n + 1):
            table[n, k] = float(math.comb(n, k))
    table_dev = table.to(device)
    _BINOM_CACHE[key] = table_dev
    return table_dev


def compute_mcmc_relocation(
    opacities: torch.Tensor,
    scales_activated: torch.Tensor,
    ratios: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3DGS-MCMC multiplicity-preserving relocation update (Kheradmand et al. 2024, Eq. 9).

    Given pre-relocation opacity ``o`` and activated scales ``s`` for parents sampled ``ratios - 1``
    times (so ``ratios = bincount(parent_idx) + 1 >= 1``), computes ``(o_new, s_new)`` that
    conserves the 1D integrated opacity distribution across the ``ratio`` resulting Gaussians.
    Uses ``gsplat.relocation.compute_relocation`` on CUDA when available, with an exact pure-PyTorch
    fallback.
    """
    if opacities.numel() == 0:
        return opacities, scales_activated
    r_i32 = ratios.clamp(1, _MAX_RELOC_BINOM - 1).to(torch.int32).contiguous()
    opac_c = opacities.clamp(1e-6, 1.0 - 1e-6).float().contiguous()
    scales_c = scales_activated.clamp_min(1e-8).float().contiguous()
    binoms = _get_binom_table(opac_c.device, _MAX_RELOC_BINOM)

    if opac_c.is_cuda and is_gsplat_cuda_ready()[1]:
        try:
            from gsplat.relocation import compute_relocation

            o_new, s_new = compute_relocation(opac_c, scales_c, r_i32, binoms)
            return o_new.clamp(1e-4, 0.995), s_new.clamp_min(1e-8)
        except Exception:  # noqa: BLE001
            pass

    r_f = r_i32.to(torch.float32)
    o_new = 1.0 - torch.pow((1.0 - opac_c).clamp_min(1e-6), 1.0 / r_f)
    k_idx = torch.arange(1, _MAX_RELOC_BINOM, device=opac_c.device, dtype=torch.float32)
    valid = k_idx[None, :] <= r_f[:, None]
    b_vals = binoms[r_i32.long()][:, 1:]
    signs = torch.where((torch.arange(1, _MAX_RELOC_BINOM, device=opac_c.device) % 2) == 1, 1.0, -1.0)
    denom = torch.where(
        valid,
        b_vals * signs[None, :] * torch.pow(o_new[:, None], k_idx[None, :]) / torch.sqrt(k_idx[None, :]),
        torch.zeros_like(b_vals),
    ).sum(dim=-1)
    s_new = (opac_c / denom.clamp_min(1e-12))[:, None] * scales_c
    return o_new.clamp(1e-4, 0.995), s_new.clamp_min(1e-8)


class DensityControlStrategy:
    """Adaptive density control upgraded with AbsGS NDC gradient metric & periodic opacity reset."""

    def __init__(self, max_points: int = 300_000, use_absgrad: bool = True) -> None:
        self.max_points = max_points
        self.use_absgrad = use_absgrad
        self.grad_accum: torch.Tensor | None = None
        self.grad_count: torch.Tensor | None = None
        self.last_survivor_indices: torch.Tensor | None = None
        self.last_num_appended: int = 0

    def reset_stats(self, model: GaussianModel) -> None:
        n = model.num_points
        device = model.means.device
        self.grad_accum = torch.zeros(n, dtype=torch.float32, device=device)
        self.grad_count = torch.zeros(n, dtype=torch.float32, device=device)

    def state_dict(self) -> dict[str, Any]:
        return {
            "grad_accum": self.grad_accum.detach().cpu() if self.grad_accum is not None else None,
            "grad_count": self.grad_count.detach().cpu() if self.grad_count is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any], device: torch.device) -> None:
        ga = state.get("grad_accum")
        gc = state.get("grad_count")
        self.grad_accum = ga.to(device) if isinstance(ga, torch.Tensor) else None
        self.grad_count = gc.to(device) if isinstance(gc, torch.Tensor) else None

    @torch.no_grad()
    def record_gradients(
        self,
        model: GaussianModel,
        absgrad_override: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        allow_3d_fallback: bool = True,
    ) -> None:
        if self.grad_accum is None or self.grad_accum.shape[0] != model.num_points:
            self.reset_stats(model)

        if absgrad_override is not None:
            metric = (
                absgrad_override.detach().norm(dim=-1)
                if absgrad_override.ndim > 1
                else absgrad_override.detach().abs()
            )
        elif allow_3d_fallback and model.means.grad is not None:
            grad = model.means.grad.detach()
            metric = grad.abs().sum(dim=-1) if self.use_absgrad else grad.norm(dim=-1)
        else:
            return

        assert self.grad_accum is not None and self.grad_count is not None
        if visible_mask is not None:
            self.grad_accum[visible_mask] += metric[visible_mask]
            self.grad_count[visible_mask] += 1.0
        else:
            self.grad_accum += metric
            self.grad_count += 1.0

    @torch.no_grad()
    def reset_opacities(self, model: GaussianModel, max_opacity: float = 0.01) -> None:
        """Clamp opacities downward periodically to flush near-camera floaters."""
        clamped_alpha = max(1e-4, min(0.99, max_opacity))
        max_logit = math.log(clamped_alpha / (1.0 - clamped_alpha))
        model.opacities.data.clamp_(max=max_logit)

    @torch.no_grad()
    def step_densify_and_prune(
        self,
        model: GaussianModel,
        grad_threshold: float = 4e-4,
        min_opacity: float = 0.005,
        split_scale_threshold: float = 0.05,
    ) -> bool:
        """Clone/split high-gradient Gaussians (removing split parents, rotating offsets) and prune transparent ones."""
        self.last_survivor_indices = None
        self.last_num_appended = 0
        if self.grad_accum is None or self.grad_count is None or model.num_points == 0:
            self.reset_stats(model)
            return False

        n = model.num_points
        device = model.means.device
        avg_grad = self.grad_accum / self.grad_count.clamp_min(1.0)
        scales_act = model.activated_scales
        max_scale = scales_act.max(dim=-1).values
        opac_act = model.activated_opacities
        high_grad = (avg_grad >= grad_threshold) & (opac_act >= min_opacity)

        clone_mask = high_grad & (max_scale <= split_scale_threshold)
        split_mask = high_grad & (max_scale > split_scale_threshold)

        budget = max(0, self.max_points - n)
        clone_idx = torch.nonzero(clone_mask, as_tuple=True)[0][:budget]
        rem_budget = budget - int(clone_idx.numel())
        # 分裂会删除 1 个父高斯并增加 2 个子高斯，净增 +1，因此最多处理 rem_budget 个父高斯。
        split_idx = torch.nonzero(split_mask, as_tuple=True)[0][:rem_budget]

        split_chosen = torch.zeros(n, dtype=torch.bool, device=device)
        if split_idx.numel() > 0:
            split_chosen[split_idx] = True

        survivor_mask = (~split_chosen) & (opac_act >= min_opacity)
        if not survivor_mask.any() and clone_idx.numel() == 0 and split_idx.numel() == 0:
            survivor_mask[0] = True
        survivor_idx = torch.nonzero(survivor_mask, as_tuple=True)[0]

        app_means: list[torch.Tensor] = []
        app_scales: list[torch.Tensor] = []
        app_quats: list[torch.Tensor] = []
        app_opac: list[torch.Tensor] = []
        app_sh_dc: list[torch.Tensor] = []
        app_sh_rest: list[torch.Tensor] = []

        if clone_idx.numel() > 0:
            app_means.append(model.means.data[clone_idx].clone())
            app_scales.append(model.scales.data[clone_idx].clone())
            app_quats.append(model.quats.data[clone_idx].clone())
            app_opac.append(model.opacities.data[clone_idx].clone())
            app_sh_dc.append(model.sh_dc.data[clone_idx].clone())
            app_sh_rest.append(model.sh_rest.data[clone_idx].clone())

        if split_idx.numel() > 0:
            std = scales_act[split_idx]
            rot = quat_to_rotmat(model.normalized_quats[split_idx])
            local_off = std * torch.randn_like(std)
            offsets = torch.bmm(rot, local_off.unsqueeze(-1)).squeeze(-1)
            shrunk_scales = torch.log((std / 1.6).clamp_min(1e-5))
            for sign in (-1.0, 1.0):
                app_means.append(model.means.data[split_idx] + sign * offsets)
                app_scales.append(shrunk_scales.clone())
                app_quats.append(model.quats.data[split_idx].clone())
                app_opac.append(model.opacities.data[split_idx].clone())
                app_sh_dc.append(model.sh_dc.data[split_idx].clone())
                app_sh_rest.append(model.sh_rest.data[split_idx].clone())

        has_pruned = not bool(survivor_mask.all().item())
        has_appended = len(app_means) > 0
        if not has_pruned and not has_appended:
            self.reset_stats(model)
            return False

        appended = (
            {
                "means": torch.cat(app_means, dim=0),
                "scales": torch.cat(app_scales, dim=0),
                "quats": torch.cat(app_quats, dim=0),
                "opacities": torch.cat(app_opac, dim=0),
                "sh_dc": torch.cat(app_sh_dc, dim=0),
                "sh_rest": torch.cat(app_sh_rest, dim=0),
            }
            if has_appended
            else None
        )
        n_appended = int(appended["means"].shape[0]) if appended is not None else 0
        model.select_and_append_(survivor_idx, appended)
        self.last_survivor_indices = survivor_idx
        self.last_num_appended = n_appended
        self.reset_stats(model)
        return True


class MCMCStrategy:
    """3DGS-MCMC: multiplicity-preserving dead-particle relocation, growth, and SGLD noise."""

    def __init__(self, max_points: int = 300_000) -> None:
        self.max_points = max_points
        self.last_dead_indices: torch.Tensor | None = None
        self.last_reset_indices: torch.Tensor | None = None
        self.last_num_grown: int = 0

    def state_dict(self) -> dict[str, Any]:
        return {"last_num_grown": int(self.last_num_grown)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_num_grown = int(state.get("last_num_grown", 0))

    @torch.no_grad()
    def relocate_and_grow(
        self,
        model: GaussianModel,
        min_opacity: float = 0.005,
        grow_ratio: float = 0.05,
        grad_scores: torch.Tensor | None = None,
    ) -> bool:
        """Teleport dead Gaussians and split high-gradient parents within the global budget."""
        self.last_dead_indices = None
        self.last_reset_indices = None
        self.last_num_grown = 0
        if model.num_points == 0:
            return False

        max_log_scale = float(getattr(model, "max_log_scale", 1.5))
        opac = model.activated_opacities.detach()
        dead_mask = opac < min_opacity
        weights = opac.clone()
        weights[dead_mask] = 0.0
        if (
            grad_scores is not None
            and grad_scores.shape[0] == model.num_points
            and float(grad_scores.max().item()) > 1e-8
        ):
            g = grad_scores.detach().clamp_min(0.0)
            g_norm = (g / (g.mean().clamp_min(1e-8))).clamp(0.0, 25.0)
            weights = weights * (0.15 + 0.85 * g_norm)

        if not (weights > 0.0).any():
            return False

        reset_parts: list[torch.Tensor] = []
        probs_norm = weights / weights.sum()
        if dead_mask.any():
            dead_idx = torch.nonzero(dead_mask, as_tuple=True)[0]
            self.last_dead_indices = dead_idx
            parent_idx = torch.multinomial(probs_norm, num_samples=int(dead_idx.numel()), replacement=True)

            counts = torch.bincount(parent_idx, minlength=model.num_points)
            uniq_p = torch.nonzero(counts > 0, as_tuple=True)[0]
            ratios_u = counts[uniq_p] + 1
            old_scales_p = torch.exp(model.scales.data[parent_idx].clamp(-10.0, max_log_scale))
            old_scales_u = torch.exp(model.scales.data[uniq_p].clamp(-10.0, max_log_scale))
            o_u, s_u = compute_mcmc_relocation(opac[uniq_p], old_scales_u, ratios_u)
            logit_u = torch.logit(o_u)
            log_scale_u = torch.log(s_u).clamp(-10.0, max_log_scale)

            full_logits = model.opacities.data.clone()
            full_scales = model.scales.data.clone()
            full_logits[uniq_p] = logit_u
            full_scales[uniq_p] = log_scale_u

            rot_p = quat_to_rotmat(model.normalized_quats[parent_idx])
            local_jitter = (old_scales_p * 0.05) * torch.randn_like(model.means.data[parent_idx])
            jitter = torch.bmm(rot_p, local_jitter.unsqueeze(-1)).squeeze(-1)

            model.means.data[dead_idx] = model.means.data[parent_idx] + jitter
            model.scales.data[uniq_p] = log_scale_u
            model.opacities.data[uniq_p] = logit_u
            model.scales.data[dead_idx] = full_scales[parent_idx]
            model.opacities.data[dead_idx] = full_logits[parent_idx]
            model.quats.data[dead_idx] = model.quats.data[parent_idx]
            model.sh_dc.data[dead_idx] = model.sh_dc.data[parent_idx]
            model.sh_rest.data[dead_idx] = model.sh_rest.data[parent_idx]
            reset_parts.extend([dead_idx, uniq_p])

        remaining = max(0, self.max_points - model.num_points)
        n_grow = min(remaining, max(0, int(round(model.num_points * grow_ratio))))
        if n_grow > 0:
            opac_updated = torch.sigmoid(model.opacities.data).clamp_min(1e-6)
            grow_weights = opac_updated.clone()
            if (
                grad_scores is not None
                and grad_scores.shape[0] == model.num_points
                and float(grad_scores.max().item()) > 1e-8
            ):
                g = grad_scores.detach().clamp_min(0.0)
                g_norm = (g / (g.mean().clamp_min(1e-8))).clamp(0.0, 25.0)
                grow_weights = grow_weights * (0.15 + 0.85 * g_norm)
                if dead_mask.any():
                    grow_weights[dead_mask] = 0.0
                if not (grow_weights > 0.0).any():
                    grow_weights = opac_updated

            grow_parents = torch.multinomial(grow_weights / grow_weights.sum(), num_samples=n_grow, replacement=True)
            g_counts = torch.bincount(grow_parents, minlength=model.num_points)
            g_uniq = torch.nonzero(g_counts > 0, as_tuple=True)[0]
            g_ratios = g_counts[g_uniq] + 1
            old_g_scales_p = torch.exp(model.scales.data[grow_parents].clamp(-10.0, max_log_scale))
            old_g_scales_u = torch.exp(model.scales.data[g_uniq].clamp(-10.0, max_log_scale))
            go_u, gs_u = compute_mcmc_relocation(opac_updated[g_uniq], old_g_scales_u, g_ratios)
            g_logit_u = torch.logit(go_u)
            g_log_scale_u = torch.log(gs_u).clamp(-10.0, max_log_scale)

            model.scales.data[g_uniq] = g_log_scale_u
            model.opacities.data[g_uniq] = g_logit_u

            rot_g = quat_to_rotmat(model.normalized_quats[grow_parents])
            local_cj = (old_g_scales_p * 0.05) * torch.randn_like(model.means.data[grow_parents])
            child_jitter = torch.bmm(rot_g, local_cj.unsqueeze(-1)).squeeze(-1)

            appended = {
                "means": model.means.data[grow_parents] + child_jitter,
                "scales": model.scales.data[grow_parents].clone(),
                "quats": model.quats.data[grow_parents].clone(),
                "opacities": model.opacities.data[grow_parents].clone(),
                "sh_dc": model.sh_dc.data[grow_parents].clone(),
                "sh_rest": model.sh_rest.data[grow_parents].clone(),
            }
            src_all = torch.arange(model.num_points, device=model.means.device, dtype=torch.long)
            model.select_and_append_(src_all, appended)
            reset_parts.append(g_uniq)
            self.last_num_grown = int(n_grow)

        if reset_parts:
            model.filter_3d_var = None
            self.last_reset_indices = torch.unique(torch.cat(reset_parts))
            return True
        return False

    @torch.no_grad()
    def inject_langevin_noise(
        self,
        model: GaussianModel,
        lr_means: float,
        noise_scale: float = 2e4,
        noise_mode: str = "scaled",
    ) -> None:
        """Inject anisotropic SGLD noise weighted inversely by opacity for MCMC exploration.

        Args:
            noise_mode:
                - ``"scaled"`` (default): ``R @ (s * eps)`` — scale-equivariant std proportional to
                  Gaussian radius ``s`` (matches ``mcmc_noise_scale`` defaults in YAML configs).
                - ``"paper"``: ``R @ (s^2 * eps)`` — covariance-proportional noise ``Σ · ε`` matching
                  the 3DGS-MCMC paper / ``gsplat`` convention (requires larger ``mcmc_noise_scale`` e.g. 5e5).
        """
        if model.num_points == 0 or lr_means <= 0.0 or noise_scale <= 0.0:
            return
        gate = torch.sigmoid(-100.0 * (model.activated_opacities - 0.005)).unsqueeze(-1)
        rot = quat_to_rotmat(model.normalized_quats)
        s = model.activated_scales
        amp = (s * s) if noise_mode == "paper" else s
        local_noise = amp * torch.randn_like(s)
        world_noise = torch.bmm(rot, local_noise.unsqueeze(-1)).squeeze(-1)
        model.means.data.add_(lr_means * noise_scale * gate * world_noise)
