from __future__ import annotations

import json
from typing import Any

from real2sim_gs.core.external import run_command_template
from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace


@register_stage("pose", "feedforward_slot")
class FeedForwardPoseAdapter(BaseStage):
    """Adapter seam for COLMAP-free pose solvers (e.g. VGGT, DUSt3R, MASt3R in third_party/)."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        backend = str(params.get("backend", "vggt"))
        ws.pose_dir.mkdir(parents=True, exist_ok=True)
        run_command_template(
            ws,
            params,
            placeholders={
                "frames_dir": ws.frames_dir,
                "pose_dir": ws.pose_dir,
                "cameras_json": ws.cameras_path,
                "init_points_ply": ws.init_points_path,
            },
            label=f"feedforward:{backend}",
            default_repo_dir=f"./third_party/{backend}",
            log_path=ws.pose_dir / f"feedforward_{backend}.log",
        )

        if not ws.cameras_path.exists() or not ws.init_points_path.exists():
            raise FileNotFoundError(
                f"External pose solver '{backend}' finished but did not produce "
                f"{ws.cameras_path} and {ws.init_points_path}."
            )

        cams = json.loads(ws.cameras_path.read_text(encoding="utf-8"))
        return StageResult(
            artifacts={
                "cameras": ws.to_relative(ws.cameras_path),
                "init_points": ws.to_relative(ws.init_points_path),
            },
            metrics={"registered_views": len(cams.get("frames", []))},
            summary=f"Feed-forward pose solver '{backend}' produced standard pose files.",
        )
