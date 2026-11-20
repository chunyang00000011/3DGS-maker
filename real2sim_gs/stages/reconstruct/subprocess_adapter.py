from __future__ import annotations

from typing import Any

from real2sim_gs.core.external import run_command_template
from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace, atomic_write_json
from real2sim_gs.utils.ply_io import load_gaussian_ply


@register_stage("reconstruct", "external_subprocess")
class SubprocessReconstructAdapter(BaseStage):
    """Bridge external paper repositories (2DGS, PGSR, etc.) running in separate envs/CLI."""

    default_repo_dir = "./third_party/external_paper"

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        ws.gaussians_dir.mkdir(parents=True, exist_ok=True)
        run_command_template(
            ws,
            params,
            placeholders={
                "frames_dir": ws.frames_dir,
                "pose_dir": ws.pose_dir,
                "cameras_json": ws.cameras_path,
                "init_points_ply": ws.init_points_path,
                "output_dir": ws.gaussians_dir,
                "output_ply": ws.raw_splats_path,
                "iterations": int(params.get("iterations", 30000)),
            },
            label=self.__class__.__name__,
            default_repo_dir=self.default_repo_dir,
            log_path=ws.gaussians_dir / "external_reconstruct.log",
        )

        if not ws.raw_splats_path.exists():
            raise FileNotFoundError(
                f"External reconstructor completed but missing output PLY: {ws.raw_splats_path}"
            )

        cloud = load_gaussian_ply(ws.raw_splats_path)
        metrics = {"num_gaussians": cloud.num_points, "render_backend": "external_subprocess"}
        metrics_path = ws.gaussians_dir / "metrics.json"
        atomic_write_json(metrics_path, metrics)

        return StageResult(
            artifacts={
                "raw_splats": ws.to_relative(ws.raw_splats_path),
                "metrics": ws.to_relative(metrics_path),
            },
            metrics=metrics,
            summary=f"External reconstructor produced {cloud.num_points} Gaussians.",
        )
