from __future__ import annotations

import shutil
from typing import Any

import numpy as np
import trimesh

from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace, atomic_write_json
from real2sim_gs.stages.export.ply_exporter import _R_ZUP_TO_SUPERSPLAT_PLY
from real2sim_gs.utils.ply_io import load_gaussian_ply, save_gaussian_ply, transform_gaussian_cloud

# Proper SO(3) (det = +1): Z-up (x, y, z) -> right-handed Y-up (y, z, x) for Unity's OBJ importer.
_R_ZUP_TO_RH_YUP = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
_M_ZUP_TO_UNITY_LH = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@register_stage("export", "unity")
class UnityExporterStage(BaseStage):
    """Export Unity-ready 3DGS PLY and collision OBJ (`inria_rh` default prevents importer double-flip)."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        out_dir = ws.export_dir / "unity"
        out_dir.mkdir(parents=True, exist_ok=True)
        scene_name = str(params.get("scene_name", "real2sim_unity_scene"))
        coordinate_mode = str(params.get("coordinate_mode", "inria_rh")).strip().lower()
        if coordinate_mode not in ("inria_rh", "unity_lh"):
            raise ValueError(
                f"Unsupported Unity coordinate_mode '{coordinate_mode}'. Expected 'inria_rh' or 'unity_lh'."
            )

        active_path = ws.active_splat_path()
        cloud = load_gaussian_ply(active_path)
        is_aligned = (
            (ws.processed_dir / "alignment_transform.json").is_file()
            and active_path.resolve() != ws.raw_splats_path.resolve()
        )

        dst_ply = out_dir / "visual_splats.ply"
        dst_mesh = out_dir / "collision_mesh.obj"

        if coordinate_mode == "inria_rh":
            if is_aligned and cloud.num_points > 0:
                rh_cloud = transform_gaussian_cloud(cloud, _R_ZUP_TO_SUPERSPLAT_PLY.astype(np.float64))
                save_gaussian_ply(dst_ply, rh_cloud)
            else:
                shutil.copy2(active_path, dst_ply)

            if ws.collision_mesh_path.exists():
                mesh = trimesh.load(str(ws.collision_mesh_path), force="mesh")
                verts = np.asarray(mesh.vertices, dtype=np.float64)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                if is_aligned and len(verts) > 0:
                    verts = verts @ _R_ZUP_TO_RH_YUP.T
                trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces, process=False).export(str(dst_mesh))
            else:
                trimesh.creation.box(extents=[2.0, 0.02, 2.0]).export(str(dst_mesh))

            coord_sys = "right_handed_inria_ply_and_rh_yup_obj"
            importer_note = (
                "默认保守导出（右手系）：visual_splats.ply 遵循 Inria COLMAP 约定供 UnityGaussianSplatting 自动转左手系，"
                "collision_mesh.obj 为右手 Y-up 供 Unity 原生 OBJ 导入器一次翻转；避免二次镜像。"
            )
        else:
            lh_cloud = transform_gaussian_cloud(cloud, _M_ZUP_TO_UNITY_LH)
            save_gaussian_ply(dst_ply, lh_cloud)

            if ws.collision_mesh_path.exists():
                mesh = trimesh.load(str(ws.collision_mesh_path), force="mesh")
                verts = np.asarray(mesh.vertices, dtype=np.float32)[:, [0, 2, 1]]
                faces = np.asarray(mesh.faces, dtype=np.int64)[:, [0, 2, 1]]
                trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(str(dst_mesh))
            else:
                trimesh.creation.box(extents=[2.0, 0.02, 2.0]).export(str(dst_mesh))

            coord_sys = "left_handed_y_up"
            importer_note = "已预转换为 Unity 左手 Y-up 坐标系，导入时请在插件/导入器中关闭重复轴翻转。"

        manifest = {
            "target": "unity",
            "scene_name": scene_name,
            "coordinate_mode": coordinate_mode,
            "coordinate_system": coord_sys,
            "sh_degree": cloud.sh_degree,
            "num_splats": cloud.num_points,
            "importer_note": importer_note,
            "files": {
                "visual_splats": "visual_splats.ply",
                "collision_mesh": "collision_mesh.obj",
            },
        }
        meta_path = out_dir / "unity_import.json"
        atomic_write_json(meta_path, manifest)

        return StageResult(
            artifacts={
                "unity_splats": ws.to_relative(dst_ply),
                "unity_mesh": ws.to_relative(dst_mesh),
                "unity_import": ws.to_relative(meta_path),
            },
            metrics={
                "unity_exported_splats": cloud.num_points,
                "unity_coordinate_mode": coordinate_mode,
            },
            summary=f"Exported Unity ({coordinate_mode}) asset bundle to {out_dir}.",
        )
