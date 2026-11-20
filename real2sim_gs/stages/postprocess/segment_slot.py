from __future__ import annotations

from typing import Any

import numpy as np
import torch

from real2sim_gs.core.doctor import is_gsplat_cuda_ready
from real2sim_gs.core.external import run_command_template
from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace, atomic_write_json
from real2sim_gs.utils.ply_io import load_gaussian_ply


@register_stage("postprocess", "semantic_segment_slot")
class SemanticSegmentSlotStage(BaseStage):
    """Pluggable SAM 3 + B3-Seg/FlashSplat/GaussianCut/SAGD or external segmentation stage."""

    def _run_sam3_batch(self, ws: SceneWorkspace, params: dict[str, Any], prompts: list[str]) -> StageResult:
        from real2sim_gs.editor import (
            SceneEditManager,
            SegmentLiftConfig,
            build_orbit_camera_view,
            cloud_to_gpu_model,
            estimate_target_anchor_3d,
            get_shared_sam3_backend,
            lift_masks_to_3dgs,
            plan_active_views,
            project_world_point,
            render_camera_rgb,
        )

        active_input = ws.active_splat_path()
        cloud = load_gaussian_ply(active_input)
        if cloud.num_points == 0:
            raise ValueError("当前高斯云点数为 0，无法执行 SAM 3 语义分割。")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, gsplat_ready, _ = is_gsplat_cuda_ready()
        use_gsplat = bool(gsplat_ready and device.type == "cuda")

        sam3 = get_shared_sam3_backend()
        if params.get("model_path"):
            sam3.configure(str(params["model_path"]))

        lo, hi = np.percentile(cloud.means, 5, axis=0), np.percentile(cloud.means, 95, axis=0)
        target_default = ((lo + hi) * 0.5).tolist()
        diag = float(np.linalg.norm(hi - lo))
        dist_default = float(max(1.0, diag * 0.72))

        probe_angles = [0.65, 2.22, 3.79, 5.36]
        num_views = max(4, int(params.get("num_views", 12)))
        w = int(params.get("render_width", 512))
        h = int(params.get("render_height", 384))

        mgr = SceneEditManager(ws)
        registered_names: list[str] = []
        model_gpu = cloud_to_gpu_model(cloud, device)

        try:
            for p_idx, raw_prompt in enumerate(prompts):
                ws.check_cancelled()
                prompt_text = str(raw_prompt).strip()
                if not prompt_text:
                    continue

                # Find highest-scoring probe viewpoint for this open-vocabulary concept
                best_pred = None
                best_yaw = probe_angles[0]
                for yaw in probe_angles:
                    pv = build_orbit_camera_view(
                        yaw=yaw,
                        pitch=0.45,
                        distance=dist_default,
                        target=target_default,
                        width=w,
                        height=h,
                        device=device,
                    )
                    rgb_u8 = render_camera_rgb(model_gpu, pv, use_gsplat=use_gsplat)
                    pred = sam3.predict_image_prompt(rgb_u8, text=prompt_text)
                    if np.any(pred.mask) and (best_pred is None or pred.score > best_pred.score):
                        best_pred = pred
                        best_yaw = yaw

                if best_pred is None or not np.any(best_pred.mask):
                    continue

                anchor_params = {
                    "yaw": best_yaw,
                    "pitch": 0.45,
                    "distance": dist_default,
                    "target": target_default,
                    "fov_scale": 1.05,
                }
                anchor_view = build_orbit_camera_view(
                    yaw=best_yaw,
                    pitch=0.45,
                    distance=dist_default,
                    target=target_default,
                    width=w,
                    height=h,
                    device=device,
                )
                center_3d, radius_3d = estimate_target_anchor_3d(
                    model_gpu, anchor_view, best_pred.mask, use_gsplat=use_gsplat
                )
                views = plan_active_views(
                    model_gpu,
                    anchor_params,
                    center_3d,
                    radius_3d,
                    num_views=num_views,
                    width=w,
                    height=h,
                )
                frames_rgb = [render_camera_rgb(model_gpu, v, use_gsplat=use_gsplat) for v in views]
                proj_seeds = [project_world_point(v, center_3d) for v in views]
                masks_2d = sam3.track_multiview(
                    frames_rgb,
                    anchor_idx=0,
                    text=prompt_text,
                    anchor_mask=best_pred.mask,
                    projected_seeds=proj_seeds,
                )

                _, existing_ids = mgr.load_state()
                lift_cfg = SegmentLiftConfig(
                    num_views=num_views,
                    render_width=w,
                    render_height=h,
                    use_sagd=bool(params.get("use_sagd", True)),
                    use_gaussiancut=bool(params.get("use_gaussiancut", True)),
                )
                res = lift_masks_to_3dgs(
                    cloud,
                    views,
                    masks_2d,
                    lift_cfg,
                    existing_object_ids=existing_ids,
                    center_3d=center_3d,
                    device=device,
                    use_gsplat=use_gsplat,
                )
                if np.any(res.fg_mask):
                    try:
                        mgr.register_segmented_object(
                            cloud=res.cloud,
                            fg_mask=res.fg_mask,
                            candidate_indices=res.candidate_indices,
                            candidate_scores=res.candidate_scores,
                            name=prompt_text,
                            prompt_summary=f"text:{prompt_text}",
                            existing_object_ids=res.existing_object_ids,
                            decomposed=res.decomposed_count > 0,
                            threshold=lift_cfg.threshold,
                        )
                    except ValueError:
                        continue
                    registered_names.append(prompt_text)
                    if res.decomposed_count > 0:
                        cloud = res.cloud
                        model_gpu = cloud_to_gpu_model(cloud, device)

                ws.report_progress((p_idx + 1) / max(1, len(prompts)), f"SAM 3 已分割 '{prompt_text}'")
        finally:
            sam3.unload()

        summary_data = mgr.public_summary()
        manifest_out = ws.processed_dir / "segments_manifest.json"
        atomic_write_json(
            manifest_out,
            {
                "status": "sam3_completed",
                "active_splat": ws.to_relative(active_input),
                "prompts": prompts,
                "objects": summary_data["objects"],
            },
        )
        return StageResult(
            artifacts={"segments_manifest": ws.to_relative(manifest_out)},
            metrics={"segment_prompts": len(prompts), "segmented_objects": len(summary_data["objects"])},
            summary=f"SAM 3 批量语义分割完成，新增 {len(registered_names)} 个可编辑 3DGS 物体。",
        )

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        prompts = list(params.get("prompts", []))
        command_template = str(params.get("command_template", "")).strip()
        backend = str(params.get("backend", "slot")).strip().lower()
        manifest_out = ws.processed_dir / "segments_manifest.json"
        active_input = ws.active_splat_path()

        if command_template:
            run_command_template(
                ws,
                params,
                placeholders={
                    "input_ply": active_input,
                    "processed_dir": ws.processed_dir,
                    "segments_manifest": manifest_out,
                    "prompts": ",".join(str(p) for p in prompts),
                },
                label="semantic_segment_slot",
                default_repo_dir="./third_party/sam_gs",
                log_path=ws.processed_dir / "semantic_segment.log",
            )
        elif prompts and backend == "sam3":
            return self._run_sam3_batch(ws, params, prompts)

        if not manifest_out.exists():
            payload = {
                "status": "slot_ready",
                "active_splat": ws.to_relative(active_input),
                "prompts": prompts,
                "objects": [],
                "note": "Use Studio Interactive Editor or set backend='sam3' / command_template for batch 3DGS segmentation.",
            }
            atomic_write_json(manifest_out, payload)

        return StageResult(
            artifacts={"segments_manifest": ws.to_relative(manifest_out)},
            metrics={"segment_prompts": len(prompts)},
            summary=f"Prepared semantic segmentation manifest ({len(prompts)} prompts).",
        )
