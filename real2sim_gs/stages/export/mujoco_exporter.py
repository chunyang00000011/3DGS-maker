from __future__ import annotations

import importlib.util
import json
import shutil
import warnings
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import trimesh
from scipy.cluster.vq import kmeans2

from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace, atomic_write_json, atomic_write_text
from real2sim_gs.utils.ply_io import GaussianCloud, load_gaussian_ply

_ATTR_ESCAPE_MAP = {'"': "&quot;", "'": "&apos;"}


def _esc_attr(value: Any) -> str:
    return escape(str(value), _ATTR_ESCAPE_MAP)


def _normalize_friction(raw: Any) -> list[float]:
    defaults = [1.0, 0.005, 0.0001]
    if not isinstance(raw, (list, tuple)):
        return defaults
    out: list[float] = []
    for item in raw[:3]:
        try:
            val = float(item)
            out.append(val if np.isfinite(val) and val >= 0.0 else defaults[len(out)])
        except (TypeError, ValueError):
            out.append(defaults[len(out)])
    while len(out) < 3:
        out.append(defaults[len(out)])
    return out


def _decompose_mesh_into_convex_parts(
    mesh: trimesh.Trimesh,
    max_parts: int = 12,
) -> tuple[list[trimesh.Trimesh], str]:
    """Decompose a scene mesh into convex collision hulls (CoACD -> V-HACD -> normal-aware KMeans -> plane_only)."""
    faces = np.asarray(mesh.faces)
    if len(faces) == 0 or max_parts <= 1:
        return [], "plane_only"

    if importlib.util.find_spec("coacd") is not None:
        try:
            import coacd

            cmesh = coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32))
            raw_parts = coacd.run_coacd(cmesh, max_convex_hull=max_parts)
            parts: list[trimesh.Trimesh] = []
            for verts, tris in raw_parts:
                part = trimesh.Trimesh(vertices=verts, faces=tris, process=True)
                if len(part.faces) >= 4:
                    parts.append(part)
            if parts:
                return parts[:max_parts], "coacd"
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"CoACD 分解失败，回退后续分解方式: {exc}", RuntimeWarning, stacklevel=2)

    try:
        if hasattr(trimesh.interfaces, "vhacd") and trimesh.interfaces.vhacd.exists:
            raw_parts = trimesh.decomposition.convex_decomposition(mesh, maxhulls=max_parts)
            part_list = raw_parts if isinstance(raw_parts, list) else [raw_parts]
            parts = [p for p in part_list if isinstance(p, trimesh.Trimesh) and len(p.faces) >= 4]
            if parts:
                return parts[:max_parts], "vhacd"
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"V-HACD 分解失败，回退至法向空间聚类凸包: {exc}", RuntimeWarning, stacklevel=2)

    if len(faces) < 12:
        return [], "plane_only"

    k = min(max_parts, max(1, len(faces) // 8))
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    span = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    centers_norm = (centers - centers.mean(axis=0, keepdims=True)) / max(span, 1e-4)
    # 联合空间中心与面法向聚类，避免同一簇同时跨地面与竖墙而把凹区填实。
    features = np.hstack([centers_norm, 0.35 * normals])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, labels = kmeans2(
            features,
            k,
            minit="points",
            missing="warn",
            rng=np.random.default_rng(42),
        )

    parts = []
    verts = np.asarray(mesh.vertices)
    for cid in range(k):
        mask = labels == cid
        if int(mask.sum()) < 4:
            continue
        vert_ids = np.unique(faces[mask].ravel())
        pts = verts[vert_ids]
        if len(pts) < 4:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                hull = trimesh.convex.convex_hull(pts)
                vol = float(hull.volume)
            if len(hull.faces) >= 4 and vol > 1e-8:
                parts.append(hull)
        except Exception:  # noqa: BLE001
            continue

    if not parts:
        return [], "plane_only"
    return parts, "normal_spatial_kmeans_hull"


def _compute_robot_spawn(
    cloud: GaussianCloud,
    collision_mesh: trimesh.Trimesh | None,
    is_aligned: bool,
) -> tuple[list[float], list[float], list[float]]:
    """Compute (bbox_min, bbox_max, robot_spawn_xyz) by ray-casting downward onto walkable surfaces (avoids ceilings)."""
    if cloud.num_points == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]

    bbox_min = [float(v) for v in cloud.means.min(axis=0)]
    bbox_max = [float(v) for v in cloud.means.max(axis=0)]
    if cloud.num_points >= 32:
        lo = np.percentile(cloud.means, 5, axis=0)
        hi = np.percentile(cloud.means, 95, axis=0)
        cx = float((lo[0] + hi[0]) * 0.5)
        cy = float((lo[1] + hi[1]) * 0.5)
        z_floor_est = 0.0 if is_aligned else float(lo[2])
    else:
        cx = float((bbox_min[0] + bbox_max[0]) * 0.5)
        cy = float((bbox_min[1] + bbox_max[1]) * 0.5)
        z_floor_est = 0.0 if is_aligned else float(bbox_min[2])

    spawn_z = z_floor_est + 0.12
    if collision_mesh is not None and len(collision_mesh.faces) > 0:
        try:
            z_ray_start = float(collision_mesh.bounds[1, 2]) + 1.0
            locs, _, tri_ids = collision_mesh.ray.intersects_location(
                ray_origins=np.array([[cx, cy, z_ray_start]], dtype=np.float64),
                ray_directions=np.array([[0.0, 0.0, -1.0]], dtype=np.float64),
            )
            if len(locs) > 0:
                tri_normals = np.asarray(collision_mesh.face_normals, dtype=np.float64)[tri_ids]
                upward = tri_normals[:, 2] >= 0.5
                walkable = locs[upward] if np.any(upward) else locs
                # 取最低朝上可站立面，避开天花板顶面
                spawn_z = float(walkable[:, 2].min()) + 0.12
        except Exception:  # noqa: BLE001
            pass

    return bbox_min, bbox_max, [round(cx, 4), round(cy, 4), round(spawn_z, 4)]


def _build_mjcf_xml(
    model_name: str,
    friction: list[float],
    part_files: list[str],
    has_full_mesh: bool,
) -> str:
    safe_name = _esc_attr(model_name)
    norm_fric = _normalize_friction(friction)
    fric_str = " ".join(f"{v:.6g}" for v in norm_fric)

    asset_lines: list[str] = []
    geom_lines: list[str] = []
    if has_full_mesh:
        # 开放表面网格加 inertia="shell" 且仅作可视化，防止 MuJoCo 报体积过小或坍缩成单个大凸包。
        asset_lines.append('    <mesh name="env_visual_mesh" file="collision_mesh.obj" inertia="shell"/>')
        geom_lines.append(
            '    <geom name="env_surface_ref" type="mesh" mesh="env_visual_mesh" '
            'contype="0" conaffinity="0" rgba="0.72 0.76 0.80 0.45"/>'
        )

    for idx, rel_file in enumerate(part_files):
        mesh_name = f"col_part_{idx:02d}"
        asset_lines.append(f'    <mesh name="{mesh_name}" file="{_esc_attr(rel_file)}"/>')
        geom_lines.append(
            f'    <geom name="terrain_col_{idx:02d}" type="mesh" mesh="{mesh_name}" '
            f'friction="{fric_str}" rgba="0.62 0.74 0.82 1"/>'
        )

    asset_block = "  <asset>\n" + "\n".join(asset_lines) + "\n  </asset>\n" if asset_lines else ""
    terrain_geoms = "\n".join(geom_lines) + "\n" if geom_lines else ""

    return (
        f'<mujoco model="{safe_name}">\n'
        '  <compiler angle="radian" meshdir="."/>\n'
        '  <option gravity="0 0 -9.81" timestep="0.002"/>\n'
        f"{asset_block}"
        "  <worldbody>\n"
        '    <light name="sun" pos="0 0 4" dir="0 0 -1" directional="true"/>\n'
        '    <geom name="ground_fallback" type="plane" pos="0 0 -0.02" size="10 10 0.1" '
        f'friction="{fric_str}" rgba="0.35 0.38 0.42 1"/>\n'
        f"{terrain_geoms}"
        "  </worldbody>\n"
        "</mujoco>\n"
    )


@register_stage("export", "mujoco")
class MujocoExporterStage(BaseStage):
    """Export aligned 3DGS visuals, decomposed convex collision parts, MJCF XML, and spawn metadata."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        out_dir = ws.export_dir / "mujoco"
        out_dir.mkdir(parents=True, exist_ok=True)

        src_ply = ws.active_splat_path()
        dst_ply = out_dir / "visual_splats.ply"
        shutil.copy2(src_ply, dst_ply)

        dst_mesh = out_dir / "collision_mesh.obj"
        if ws.collision_mesh_path.exists():
            shutil.copy2(ws.collision_mesh_path, dst_mesh)
        else:
            fallback = trimesh.creation.box(extents=[2.0, 2.0, 0.02])
            fallback.export(str(dst_mesh))

        max_parts = int(params.get("convex_parts", 12))
        part_files: list[str] = []
        decomp_method = "plane_only"
        loaded_mesh: trimesh.Trimesh | None = None
        if dst_mesh.exists():
            loaded_mesh = trimesh.load(str(dst_mesh), force="mesh")
            parts, decomp_method = _decompose_mesh_into_convex_parts(loaded_mesh, max_parts=max_parts)
            parts_dir = out_dir / "collision_parts"
            shutil.rmtree(parts_dir, ignore_errors=True)
            if parts:
                parts_dir.mkdir(parents=True, exist_ok=True)
                for idx, part_mesh in enumerate(parts):
                    rel_part = f"collision_parts/part_{idx:02d}.obj"
                    part_mesh.export(str(out_dir / rel_part))
                    part_files.append(rel_part)

        model_name = str(params.get("model_name", "real2sim_scene"))
        friction = _normalize_friction(params.get("friction", [1.0, 0.005, 0.0001]))
        xml_path = out_dir / "scene.xml"
        atomic_write_text(
            xml_path,
            _build_mjcf_xml(
                model_name=model_name,
                friction=friction,
                part_files=part_files,
                has_full_mesh=dst_mesh.exists(),
            ),
        )

        align_path = ws.processed_dir / "alignment_transform.json"
        align_info = json.loads(align_path.read_text(encoding="utf-8")) if align_path.exists() else None
        is_aligned = align_info is not None and src_ply.resolve() != ws.raw_splats_path.resolve()

        cloud = load_gaussian_ply(dst_ply)
        bbox_min, bbox_max, spawn_xyz = _compute_robot_spawn(cloud, loaded_mesh, is_aligned=is_aligned)

        meta = {
            "target": "mujoco",
            "up_axis": "+Z",
            "model_name": model_name,
            "num_splats": cloud.num_points,
            "num_convex_parts": len(part_files),
            "decomposition_method": decomp_method,
            "bbox_min": [round(float(v), 4) for v in bbox_min],
            "bbox_max": [round(float(v), 4) for v in bbox_max],
            "robot_spawn_xyz": spawn_xyz,
            "files": {
                "scene_xml": "scene.xml",
                "visual_splats": "visual_splats.ply",
                "collision_mesh": "collision_mesh.obj",
                "collision_parts": part_files,
            },
            "alignment": align_info,
        }
        meta_path = out_dir / "scene_meta.json"
        atomic_write_json(meta_path, meta)

        return StageResult(
            artifacts={
                "mujoco_xml": ws.to_relative(xml_path),
                "visual_splats": ws.to_relative(dst_ply),
                "collision_mesh": ws.to_relative(dst_mesh),
                "scene_meta": ws.to_relative(meta_path),
            },
            metrics={
                "exported_splats": cloud.num_points,
                "num_convex_parts": len(part_files),
                "mujoco_convex_parts": len(part_files),
                "decomposition_method": decomp_method,
            },
            summary=f"Exported MuJoCo scene bundle ({len(part_files)} convex parts via {decomp_method}) to {out_dir}.",
        )
