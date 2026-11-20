from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace
from real2sim_gs.utils.ply_io import load_gaussian_ply, save_gaussian_ply


@register_stage("postprocess", "clean_floaters")
class CleanFloatersStage(BaseStage):
    """Remove transparent, needle-spike, and isolated floater Gaussians (opacity + scale-aware KDTree SOR)."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        in_path = ws.active_splat_path()
        cloud = load_gaussian_ply(in_path)
        initial_count = cloud.num_points
        if initial_count == 0:
            save_gaussian_ply(ws.cleaned_splats_path, cloud)
            ws.set_active_splat(ws.cleaned_splats_path)
            return StageResult(
                artifacts={"cleaned_splats": ws.to_relative(ws.cleaned_splats_path)},
                metrics={
                    "splats_before_clean": 0,
                    "splats_after_clean": 0,
                    "retained_gaussians": 0,
                    "removed_floaters": 0,
                },
                summary="Input cloud empty; skipped floater cleaning.",
            )

        min_opacity = float(params.get("min_opacity", 0.02))
        nb_neighbors = int(params.get("nb_neighbors", 16))
        std_ratio = float(params.get("std_ratio", 4.5))
        max_scale_ratio = float(params.get("max_scale_ratio", 12.0))
        sor_mode = str(params.get("sor_mode", "scale_aware")).strip().lower()
        if sor_mode not in ("scale_aware", "depth_normalized", "global"):
            raise ValueError(
                f"Unsupported sor_mode '{sor_mode}'. Expected 'scale_aware', 'depth_normalized', or 'global'."
            )

        opac_mask = cloud.sigmoid_opacities() >= min_opacity
        if not opac_mask.any():
            warnings.warn(
                f"clean_floaters: 所有 {initial_count} 个高斯点不透明度均低于 {min_opacity}，保留原始点云。",
                RuntimeWarning,
                stacklevel=2,
            )
            save_gaussian_ply(ws.cleaned_splats_path, cloud)
            ws.set_active_splat(ws.cleaned_splats_path)
            return StageResult(
                artifacts={"cleaned_splats": ws.to_relative(ws.cleaned_splats_path)},
                metrics={
                    "splats_before_clean": initial_count,
                    "splats_after_clean": initial_count,
                    "retained_gaussians": initial_count,
                    "removed_floaters": 0,
                },
                summary=f"All {initial_count} splats below min_opacity={min_opacity}; kept unchanged.",
            )
        filtered = cloud.subset(opac_mask)

        # Prune genuine needle spikes (s_max / s_mid >> 1) while preserving flat pancake surface surfels.
        if filtered.num_points >= 64:
            scales_exp = np.exp(np.clip(filtered.scales, -10.0, 4.0))
            s_sorted = np.sort(scales_exp, axis=1)
            s_mid = np.maximum(s_sorted[:, 1], 1e-6)
            s_max = s_sorted[:, 2]
            needle_ratio = s_max / s_mid
            p85_s = float(np.percentile(s_max, 85))
            p99_s = float(np.percentile(s_max, 99))
            shape_mask = ((needle_ratio <= max_scale_ratio) | (s_max <= p85_s)) & (s_max <= max(p99_s * 2.5, 1e-4))
            if shape_mask.any():
                filtered = filtered.subset(shape_mask)

        n = filtered.num_points
        if n > 3 and nb_neighbors >= 1:
            k = min(nb_neighbors + 1, n)
            tree = cKDTree(filtered.means)
            dists, _ = tree.query(filtered.means, k=k, workers=-1)
            mean_knn_dist = dists[:, 1:].mean(axis=1)
            if sor_mode == "scale_aware":
                # 以高斯自身最大半径归一化近邻间距，避免稀疏大块墙面/天花板高斯因中心距较大被误杀；
                # 但不再叠加半径二次除法，防止远离主体的孤立小浮空噪点同时被 radius 放大后逃脱过滤。
                s_max_sor = np.exp(np.clip(filtered.scales, -10.0, 4.0)).max(axis=1)
                scale_floor = max(float(np.percentile(s_max_sor, 50)), 1e-4)
                eff_scale = np.maximum(s_max_sor, scale_floor)
                metric = mean_knn_dist / eff_scale
            elif sor_mode == "depth_normalized":
                pivot = np.median(filtered.means, axis=0)
                radii = np.linalg.norm(filtered.means - pivot[None, :], axis=1)
                floor_r = max(float(np.median(radii)) * 0.5, 1e-4)
                metric = mean_knn_dist / np.maximum(radii, floor_r)
            else:
                metric = mean_knn_dist
            thresh = float(metric.mean()) + std_ratio * float(metric.std())
            sor_mask = metric <= thresh
            if sor_mask.any():
                filtered = filtered.subset(sor_mask)

        save_gaussian_ply(ws.cleaned_splats_path, filtered)
        ws.set_active_splat(ws.cleaned_splats_path)
        removed = initial_count - filtered.num_points
        return StageResult(
            artifacts={"cleaned_splats": ws.to_relative(ws.cleaned_splats_path)},
            metrics={
                "splats_before_clean": initial_count,
                "splats_after_clean": filtered.num_points,
                "retained_gaussians": filtered.num_points,
                "removed_floaters": removed,
                "sor_mode": sor_mode,
            },
            summary=f"Removed {removed} floaters ({initial_count} -> {filtered.num_points}, mode={sor_mode}).",
        )
