from __future__ import annotations

import json
from typing import Any

import numpy as np
from PIL import Image

from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace


@register_stage("ingest", "mock_ingest")
class MockIngestStage(BaseStage):
    """Generate synthetic multi-view frames for fast pipeline testing without video/FFmpeg."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        num_views = int(params.get("mock_num_views", 6))
        size = int(params.get("mock_image_size", 64))
        entries: list[dict[str, Any]] = []

        yy, xx = np.mgrid[0:size, 0:size]
        nx = xx / max(1, size - 1)
        ny = yy / max(1, size - 1)

        for idx in range(num_views):
            phase = idx / max(1, num_views)
            r = np.clip((0.2 + 0.5 * nx + 0.2 * np.sin(phase * 6.28)), 0.0, 1.0)
            g = np.clip((0.3 + 0.4 * ny), 0.0, 1.0)
            b = np.clip((0.6 - 0.3 * ny + 0.2 * np.cos(phase * 6.28)), 0.0, 1.0)
            rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

            fname = f"frame_{idx:04d}.png"
            Image.fromarray(rgb, mode="RGB").save(ws.frames_dir / fname)
            entries.append(
                {
                    "id": idx,
                    "filename": fname,
                    "width": size,
                    "height": size,
                    "sharpness": 100.0,
                    "source": "mock",
                }
            )

        meta_path = ws.frames_meta_path
        meta_path.write_text(
            json.dumps(
                {"method": "mock_ingest", "source_kind": "images", "num_frames": len(entries), "frames": entries},
                indent=2,
            ),
            encoding="utf-8",
        )
        return StageResult(
            artifacts={"frames_meta": ws.to_relative(meta_path)},
            metrics={"num_frames": len(entries)},
            summary=f"Synthesized {len(entries)} mock frames ({size}x{size}).",
        )
