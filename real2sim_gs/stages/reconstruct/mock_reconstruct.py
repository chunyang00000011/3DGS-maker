from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace
from real2sim_gs.utils.ply_io import SH_C0, GaussianCloud, save_gaussian_ply


@register_stage("reconstruct", "mock_reconstruct")
class MockReconstructStage(BaseStage):
    """Generate a synthetic tilted 3DGS scene + mid-air floaters to test postprocess & export."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        rng = np.random.default_rng(123)

        # Tilted ground plane (~10 deg around X) + offset so RANSAC alignment performs real work.
        gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, 22), np.linspace(-1.0, 1.0, 22))
        ground_pts = np.column_stack(
            [gx.ravel(), gy.ravel(), np.zeros(gx.size, dtype=np.float32)]
        ).astype(np.float32)
        ground_rgb = np.tile(np.array([[0.35, 0.65, 0.40]], dtype=np.float32), (ground_pts.shape[0], 1))

        # Small central object sitting on the ground (0.04 <= z <= 0.38).
        bx = rng.uniform(-0.22, 0.22, size=120)
        by = rng.uniform(-0.22, 0.22, size=120)
        bz = rng.uniform(0.04, 0.38, size=120)
        box_pts = np.column_stack([bx, by, bz]).astype(np.float32)
        box_rgb = np.tile(np.array([[0.82, 0.36, 0.25]], dtype=np.float32), (120, 1))

        scene_pts = np.vstack([ground_pts, box_pts])
        scene_rgb = np.vstack([ground_rgb, box_rgb])

        tilt = Rotation.from_euler("x", 10.0, degrees=True).as_matrix().astype(np.float32)
        scene_pts = scene_pts @ tilt.T + np.array([0.0, 0.0, -0.35], dtype=np.float32)

        n_scene = scene_pts.shape[0]
        scene_scales = np.full((n_scene, 3), -3.2, dtype=np.float32)
        scene_quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n_scene, 1))
        scene_opac = np.full((n_scene,), 2.2, dtype=np.float32)

        # Add low-opacity & distant mid-air floater outliers for clean_floaters to remove.
        n_floaters = 36
        floater_pts = np.vstack(
            [
                rng.uniform(-0.8, 0.8, size=(18, 3)).astype(np.float32),
                rng.uniform(4.0, 7.0, size=(18, 3)).astype(np.float32),
            ]
        )
        floater_rgb = np.full((n_floaters, 3), 0.9, dtype=np.float32)
        floater_scales = np.full((n_floaters, 3), -3.5, dtype=np.float32)
        floater_quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n_floaters, 1))
        floater_opac = np.concatenate(
            [np.full((18,), -5.0, dtype=np.float32), np.full((18,), 1.5, dtype=np.float32)]
        )

        all_pts = np.vstack([scene_pts, floater_pts])
        all_rgb = np.vstack([scene_rgb, floater_rgb])
        cloud = GaussianCloud(
            means=all_pts,
            scales=np.vstack([scene_scales, floater_scales]),
            quats=np.vstack([scene_quats, floater_quats]),
            opacities=np.concatenate([scene_opac, floater_opac]),
            sh_dc=(all_rgb - 0.5) / SH_C0,
        )
        save_gaussian_ply(ws.raw_splats_path, cloud)

        metrics = {
            "PSNR": 28.5,
            "SSIM": 0.91,
            "num_gaussians": cloud.num_points,
            "render_backend": "mock",
        }
        metrics_path = ws.gaussians_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        return StageResult(
            artifacts={
                "raw_splats": ws.to_relative(ws.raw_splats_path),
                "metrics": ws.to_relative(metrics_path),
            },
            metrics=metrics,
            summary=f"Synthesized {cloud.num_points} mock Gaussians (including {n_floaters} floaters).",
        )
