"""Interactive post-hoc 3DGS segmentation (SAM 3 + B3-Seg + FlashSplat + GaussianCut + SAGD) & Scene Graph."""

from __future__ import annotations

from real2sim_gs.editor.sam3_backend import (
    DEFAULT_SAM3_MODEL_ID,
    SAM3_MODEL_PATH_ENV,
    Sam3Backend,
    Sam3MaskPrediction,
    default_local_sam3_dir,
    get_sam3_status,
    get_shared_sam3_backend,
    is_valid_local_sam3_dir,
    resolve_sam3_model_id,
)
from real2sim_gs.editor.scene_graph import (
    OBJECT_PALETTE_RGB,
    SceneEditManager,
    SceneObject,
    apply_scene_edits_to_gpu_model,
    bake_edited_scene,
    export_single_object_ply,
    pick_object_at_pixel,
)
from real2sim_gs.editor.segment_3d import (
    SegmentLiftConfig,
    SegmentLiftResult,
    accumulate_view_mask_weights,
    build_orbit_camera_view,
    cloud_to_gpu_model,
    estimate_target_anchor_3d,
    lift_masks_to_3dgs,
    plan_active_views,
    project_world_point,
    refine_gaussiancut,
    render_camera_rgb,
    solve_flashsplat_scores,
)

__all__ = [
    "DEFAULT_SAM3_MODEL_ID",
    "SAM3_MODEL_PATH_ENV",
    "OBJECT_PALETTE_RGB",
    "Sam3Backend",
    "Sam3MaskPrediction",
    "default_local_sam3_dir",
    "get_sam3_status",
    "get_shared_sam3_backend",
    "is_valid_local_sam3_dir",
    "resolve_sam3_model_id",
    "SegmentLiftConfig",
    "SegmentLiftResult",
    "accumulate_view_mask_weights",
    "build_orbit_camera_view",
    "cloud_to_gpu_model",
    "estimate_target_anchor_3d",
    "lift_masks_to_3dgs",
    "plan_active_views",
    "project_world_point",
    "refine_gaussiancut",
    "render_camera_rgb",
    "solve_flashsplat_scores",
    "SceneObject",
    "SceneEditManager",
    "apply_scene_edits_to_gpu_model",
    "pick_object_at_pixel",
    "export_single_object_ply",
    "bake_edited_scene",
]
