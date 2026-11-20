from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace, atomic_write_json
from real2sim_gs.utils.ply_io import GaussianCloud, load_gaussian_ply, save_gaussian_ply, transform_gaussian_cloud

# Proper SO(3) rotation (det = +1) mapping our aligned Z-up frame
# (+Z up, +X toward front viewer, +Y viewer right) into PlayCanvas SuperSplat /
# standard Inria COLMAP PLY coordinates: (-Y_aligned, -Z_aligned, +X_aligned).
# Applying SuperSplat's automatic Z-180 import transform R_z(180) = diag(-1, -1, 1)
# yields (+Y_aligned, +Z_aligned, +X_aligned) in PlayCanvas world space:
# +Y = skyward UP, +Z = facing initial camera, +X = viewer's right.
_R_ZUP_TO_SUPERSPLAT_PLY = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def _active_cameras_path(ws: SceneWorkspace) -> Path | None:
    refined = ws.gaussians_dir / "refined_cameras.json"
    if refined.is_file():
        return refined
    if ws.cameras_path.is_file():
        return ws.cameras_path
    return None


def _load_alignment_similarity(ws: SceneWorkspace) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Load (rot_align, trans_align, scale_factor) from alignment_transform.json if valid."""
    t_path = ws.processed_dir / "alignment_transform.json"
    if not t_path.is_file():
        return None
    try:
        t_data = json.loads(t_path.read_text(encoding="utf-8"))
        mat = np.asarray(t_data.get("matrix_4x4", []), dtype=np.float64)
        scale_factor = float(t_data.get("scale_factor", 1.0)) or 1.0
        if mat.shape != (4, 4):
            return None
        rot_align = mat[:3, :3] / scale_factor
        trans_align = mat[:3, 3]
        return rot_align, trans_align, scale_factor
    except Exception:  # noqa: BLE001
        return None


def _compute_aligned_foreground_pivot(
    ws: SceneWorkspace,
    cloud: GaussianCloud,
    is_aligned_input: bool,
) -> np.ndarray:
    """Estimate foreground subject center from primary camera viewing rays, clamped by high-opacity splats."""
    if cloud.num_points == 0:
        return np.zeros(3, dtype=np.float32)

    opac_mask = cloud.sigmoid_opacities() >= 0.15
    core_pts = cloud.means[opac_mask] if int(opac_mask.sum()) >= 16 else cloud.means
    core_median = np.median(core_pts, axis=0).astype(np.float64)

    cam_path = _active_cameras_path(ws)
    if cam_path is None:
        return core_median.astype(np.float32)

    try:
        frames = json.loads(cam_path.read_text(encoding="utf-8")).get("frames", [])
    except Exception:  # noqa: BLE001
        return core_median.astype(np.float32)
    if not frames:
        return core_median.astype(np.float32)

    rot_align = np.eye(3, dtype=np.float64)
    trans_align = np.zeros(3, dtype=np.float64)
    scale_align = 1.0
    if is_aligned_input:
        sim = _load_alignment_similarity(ws)
        if sim is not None:
            rot_align, trans_align, scale_align = sim

    sample_idx = np.linspace(0, len(core_pts) - 1, num=min(4096, len(core_pts)), dtype=int)
    core_sample = core_pts[sample_idx].astype(np.float64)

    cam_idx = (
        np.linspace(0, len(frames) - 1, num=min(16, len(frames)), dtype=int).tolist()
        if len(frames) > 1
        else [0]
    )
    targets: list[np.ndarray] = []
    for idx in cam_idx:
        w2c = np.asarray(frames[int(idx)].get("world_to_cam", []), dtype=np.float64)
        if w2c.shape != (4, 4):
            continue
        r_raw = w2c[:3, :3]
        t_raw = w2c[:3, 3]
        eye_raw = -r_raw.T @ t_raw
        fwd_raw = r_raw[2, :]

        eye = scale_align * (rot_align @ eye_raw) + trans_align
        fwd = rot_align @ fwd_raw
        fwd_norm = float(np.linalg.norm(fwd))
        if fwd_norm < 1e-6:
            continue
        fwd /= fwd_norm

        rel = core_sample - eye[None, :]
        depths = rel @ fwd
        perp = np.linalg.norm(rel - depths[:, None] * fwd[None, :], axis=1)
        cone = (depths > 0.25) & (perp < 0.70 * depths)
        valid_d = depths[cone] if cone.any() else depths[depths > 0.25]
        if valid_d.size > 0:
            dist = float(np.clip(np.percentile(valid_d, 35), 0.35, 30.0))
            targets.append(eye + dist * fwd)

    if not targets:
        return core_median.astype(np.float32)

    pivot = np.median(np.stack(targets, axis=0), axis=0)
    lo = np.percentile(core_pts, 10, axis=0).astype(np.float64)
    hi = np.percentile(core_pts, 90, axis=0).astype(np.float64)
    pivot = np.clip(pivot, lo, hi)
    return pivot.astype(np.float32)


def _trim_extreme_bound_outliers(
    cloud: GaussianCloud,
    pivot: np.ndarray,
    radius_factor: float = 1.8,
) -> tuple[GaussianCloud, int]:
    """Prune rare extreme far-field splats that hijack viewer AABB orbit centers."""
    if cloud.num_points < 64 or radius_factor <= 0.0:
        return cloud, 0

    radii = np.linalg.norm(cloud.means - pivot[None, :], axis=1)
    cap = max(
        float(np.percentile(radii, 95)) * radius_factor,
        float(np.percentile(radii, 98.5)) * 1.15,
        1e-3,
    )
    keep = radii <= cap
    trimmed_count = int(cloud.num_points - int(keep.sum()))
    if trimmed_count <= 0 or not keep.any():
        return cloud, 0
    return cloud.subset(keep), trimmed_count


def _export_transformed_cameras_json(
    src_path: Path,
    dst_path: Path,
    rot_total: np.ndarray,
    trans_total: np.ndarray,
    scale_factor: float,
    coord_label: str,
) -> None:
    """Write cameras.json with extrinsics synchronized to the exported PLY coordinate frame."""
    payload = json.loads(src_path.read_text(encoding="utf-8"))
    out_frames: list[dict[str, Any]] = []
    for frame in payload.get("frames", []):
        updated = dict(frame)
        w2c = np.asarray(frame.get("world_to_cam", []), dtype=np.float64)
        if w2c.shape == (4, 4):
            r_cam = w2c[:3, :3]
            t_cam = w2c[:3, 3]
            r_new = r_cam @ rot_total.T
            t_new = scale_factor * t_cam - r_new @ trans_total
            w2c_new = np.eye(4, dtype=np.float64)
            w2c_new[:3, :3] = r_new
            w2c_new[:3, 3] = t_new
            updated["world_to_cam"] = np.round(w2c_new, 6).tolist()
        out_frames.append(updated)

    payload["coordinate_system"] = coord_label
    payload["frames"] = out_frames
    atomic_write_json(dst_path, payload)


@register_stage("export", "ply_3dgs")
class Ply3DGSExporterStage(BaseStage):
    """Export standalone Inria/SuperSplat-ready 3DGS PLY point cloud and camera metadata."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        out_dir = ws.export_dir / "gaussian_splats"
        out_dir.mkdir(parents=True, exist_ok=True)
        scene_name = str(params.get("scene_name", "scene_3dgs"))
        include_raw = bool(params.get("include_raw", True))
        coordinate_frame = str(params.get("coordinate_frame", "supersplat_colmap")).strip().lower()
        trim_outliers = bool(params.get("trim_extreme_outliers", True))
        outlier_radius_factor = float(params.get("outlier_radius_factor", 1.8))

        active_path = ws.active_splat_path()
        cloud = load_gaussian_ply(active_path)
        # 根据 alignment_transform.json 是否存在判断对齐状态，不依赖文件名优先级。
        is_aligned_input = (
            (ws.processed_dir / "alignment_transform.json").is_file()
            and active_path.resolve() != ws.raw_splats_path.resolve()
        )

        dst_ply = out_dir / "scene_3dgs.ply"
        pivot_zup = [0.0, 0.0, 0.0]
        trimmed_count = 0
        coord_label = "z_up_right_handed" if is_aligned_input else "raw_colmap_camera_space"
        sim_for_cameras: tuple[np.ndarray, np.ndarray, float] | None = None

        if coordinate_frame != "z_up" and is_aligned_input and cloud.num_points > 0:
            pivot = _compute_aligned_foreground_pivot(ws, cloud, is_aligned_input=True)
            pivot_zup = [round(float(v), 4) for v in pivot.tolist()]
            if trim_outliers:
                cloud, trimmed_count = _trim_extreme_bound_outliers(
                    cloud, pivot, radius_factor=outlier_radius_factor
                )

            r_ply = _R_ZUP_TO_SUPERSPLAT_PLY.astype(np.float64)
            exported_cloud = transform_gaussian_cloud(
                cloud,
                r_ply,
                translation=-(r_ply @ pivot.astype(np.float64)),
            )
            save_gaussian_ply(dst_ply, exported_cloud)
            cloud = exported_cloud
            coord_label = "inria_colmap_rdf (SuperSplat / SIBR / WebGL Y-up ready)"

            align_sim = _load_alignment_similarity(ws)
            if align_sim is not None:
                rot_align, trans_align, scale_align = align_sim
                rot_total = r_ply @ rot_align
                trans_total = r_ply @ (trans_align - pivot.astype(np.float64))
                sim_for_cameras = (rot_total, trans_total, scale_align)
        else:
            shutil.copy2(active_path, dst_ply)

        artifacts: dict[str, str] = {"scene_3dgs": ws.to_relative(dst_ply)}

        if include_raw and ws.raw_splats_path.exists() and ws.raw_splats_path.resolve() != active_path.resolve():
            raw_dst = out_dir / "raw_camera_space_3dgs.ply"
            shutil.copy2(ws.raw_splats_path, raw_dst)
            artifacts["raw_camera_space_3dgs"] = ws.to_relative(raw_dst)

        src_cam_path = _active_cameras_path(ws)
        if src_cam_path is not None:
            cam_dst = out_dir / "cameras.json"
            if sim_for_cameras is not None:
                rot_total, trans_total, scale_align = sim_for_cameras
                _export_transformed_cameras_json(
                    src_cam_path,
                    cam_dst,
                    rot_total=rot_total,
                    trans_total=trans_total,
                    scale_factor=scale_align,
                    coord_label=coord_label,
                )
            else:
                shutil.copy2(src_cam_path, cam_dst)
            artifacts["cameras"] = ws.to_relative(cam_dst)

        if cloud.num_points > 0:
            bbox_min = cloud.means.min(axis=0).round(4).tolist()
            bbox_max = cloud.means.max(axis=0).round(4).tolist()
        else:
            bbox_min = [0.0, 0.0, 0.0]
            bbox_max = [0.0, 0.0, 0.0]

        recon_metrics_path = ws.gaussians_dir / "metrics.json"
        trained_backend = None
        viewer_note = None
        if recon_metrics_path.is_file():
            try:
                rm = json.loads(recon_metrics_path.read_text(encoding="utf-8"))
                trained_backend = rm.get("render_backend")
                viewer_note = rm.get("viewer_note")
            except (OSError, ValueError):
                pass

        meta: dict[str, Any] = {
            "target": "ply_3dgs",
            "scene_name": scene_name,
            "format": "inria_3dgs_binary_little_endian_ply",
            "coordinate_frame": coordinate_frame if sim_for_cameras is not None or coord_label.startswith("inria") else ("z_up" if is_aligned_input else "raw"),
            "coordinate_system": coord_label,
            "foreground_pivot_zup": pivot_zup,
            "trimmed_extreme_outliers": trimmed_count,
            "num_splats": cloud.num_points,
            "sh_degree": cloud.sh_degree,
            "source_ply": ws.to_relative(active_path),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "has_raw_camera_space_ply": "raw_camera_space_3dgs" in artifacts,
            "has_cameras_json": "cameras" in artifacts,
            "compatible_viewers": [
                "SuperSplat (https://superspl.at/editor)",
                "Antimatter15 / WebGL 3DGS Viewers",
                "Blender 3DGS Add-on",
                "CloudCompare",
                "Inria SIBR_viewers",
            ],
        }
        if trained_backend is not None:
            meta["trained_render_backend"] = trained_backend
        if viewer_note is not None:
            meta["viewer_note"] = viewer_note
        meta_path = out_dir / "splat_meta.json"
        atomic_write_json(meta_path, meta)
        artifacts["splat_meta"] = ws.to_relative(meta_path)

        return StageResult(
            artifacts=artifacts,
            metrics={
                "exported_3dgs_points": cloud.num_points,
                "trimmed_extreme_outliers": trimmed_count,
            },
            summary=f"Exported {coord_label} 3DGS PLY ({cloud.num_points} splats) to {out_dir}.",
        )
