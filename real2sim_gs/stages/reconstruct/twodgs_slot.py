from __future__ import annotations

from real2sim_gs.core.registry import register_stage
from real2sim_gs.stages.reconstruct.subprocess_adapter import SubprocessReconstructAdapter


@register_stage("reconstruct", "twodgs_slot")
class TwoDGSSlotStage(SubprocessReconstructAdapter):
    """Pre-wired slot for 2D Gaussian Splatting / PGSR surface-accurate reconstruction repos."""

    default_repo_dir = "./third_party/2dgs"
