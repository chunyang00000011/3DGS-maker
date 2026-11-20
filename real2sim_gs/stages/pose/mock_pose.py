from __future__ import annotations

import json
from typing import Any

import numpy as np

from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace
from real2sim_gs.utils.ply_io import save_point_cloud_ply


def _look_at_w2c(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build OpenCV-convention world-to-camera 4x4 matrix (+X right, +Y down, +Z forward)."""
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    r_c2w = np.stack([right, down, forward], axis=1)
    r_w2c = r_c2w.T
    t_w2c = -r_w2c @ cam_pos
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = r_w2c
    w2c[:3, 3] = t_w2c
    return w2c


@register_stage("pose", "mock_pose")
class MockPoseStage(BaseStage):
    """Synthesize circular inward-looking cameras and sparse 3D points for fast testing."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        meta = json.loads(ws.frames_meta_path.read_text(encoding="utf-8"))
        frame_entries = meta["frames"]
        target = np.array([0.0, 0.0, 0.2], dtype=np.float32)

        cam_frames: list[dict[str, Any]] = []
        n = max(1, len(frame_entries))
        for idx, entry in enumerate(frame_entries):
            angle = 2.0 * np.pi * idx / n
            cam_pos = np.array([2.2 * np.cos(angle), 2.2 * np.sin(angle), 1.1], dtype=np.float32)
            w, h = int(entry["width"]), int(entry["height"])
            focal = float(max(w, h)) * 1.1
            cam_frames.append(
                {
                    "id": idx,
                    "filename": entry["filename"],
                    "width": w,
                    "height": h,
                    "fx": focal,
                    "fy": focal,
                    "cx": w / 2.0,
                    "cy": h / 2.0,
                    "world_to_cam": _look_at_w2c(cam_pos, target).tolist(),
                }
            )

        ws.cameras_path.write_text(
            json.dumps({"backend": "mock_pose", "frames": cam_frames}, indent=2),
            encoding="utf-8",
        )

        rng = np.random.default_rng(42)
        floor_xy = rng.uniform(-1.0, 1.0, size=(180, 2)).astype(np.float32)
        floor_xyz = np.column_stack([floor_xy, np.zeros(180, dtype=np.float32)])
        box_xyz = rng.uniform([-0.25, -0.25, 0.02], [0.25, 0.25, 0.45], size=(70, 3)).astype(np.float32)
        pts = np.vstack([floor_xyz, box_xyz])
        cols = np.vstack(
            [
                np.tile(np.array([[90, 170, 110]], dtype=np.uint8), (180, 1)),
                np.tile(np.array([[200, 100, 80]], dtype=np.uint8), (70, 1)),
            ]
        )
        save_point_cloud_ply(ws.init_points_path, pts, cols)

        return StageResult(
            artifacts={
                "cameras": ws.to_relative(ws.cameras_path),
                "init_points": ws.to_relative(ws.init_points_path),
            },
            metrics={"registered_views": len(cam_frames), "init_points": int(pts.shape[0])},
            summary=f"Synthesized {len(cam_frames)} mock camera poses and {pts.shape[0]} sparse points.",
        )
