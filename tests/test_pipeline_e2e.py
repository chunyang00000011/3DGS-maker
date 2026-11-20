from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from real2sim_gs import run_pipeline
from real2sim_gs.core.config import IOConfig, PipelineConfig, PipelineStagesConfig, StageSpec, load_config
from real2sim_gs.core.pipeline import PipelineRunner
from real2sim_gs.stages.postprocess.align_sim import _fit_ground_plane_ransac
from real2sim_gs.utils.ply_io import load_gaussian_ply


def test_mock_pipeline_end_to_end_and_checkpoint_resume(tmp_path: Path) -> None:
    ws_dir = tmp_path / "scene_01"
    res1 = run_pipeline("configs/default_mujoco.yaml", output_dir=ws_dir, mock=True)

    assert res1.executed_stages == ["ingest", "pose", "reconstruct", "postprocess", "export"]
    assert res1.skipped_stages == []
    assert (ws_dir / "01_frames" / "frames_meta.json").exists()
    assert (ws_dir / "02_pose" / "cameras.json").exists()
    assert (ws_dir / "03_gaussians" / "raw_splats.ply").exists()
    assert (ws_dir / "04_processed" / "cleaned_splats.ply").exists()
    assert (ws_dir / "04_processed" / "aligned_splats.ply").exists()
    assert (ws_dir / "04_processed" / "collision_mesh.obj").exists()
    assert (ws_dir / "05_export" / "gaussian_splats" / "scene_3dgs.ply").exists()
    assert res1.gaussian_ply_path is not None and res1.gaussian_ply_path.exists()
    assert res1.mujoco_xml_path is not None and res1.mujoco_xml_path.exists()

    # Validate MJCF XML parses cleanly and exports both reference mesh & convex collision parts.
    root = ET.parse(res1.mujoco_xml_path).getroot()
    assert root.tag == "mujoco"
    mesh_files = {node.attrib["file"] for node in root.findall("./asset/mesh")}
    assert "collision_mesh.obj" in mesh_files

    scene_meta = json.loads((ws_dir / "05_export" / "mujoco" / "scene_meta.json").read_text(encoding="utf-8"))
    assert scene_meta["num_convex_parts"] >= 1
    for rel_part in scene_meta["files"]["collision_parts"]:
        assert (ws_dir / "05_export" / "mujoco" / rel_part).exists()

    assert res1.metrics["mesh_method"] == "marching_cubes"

    # Verify floater cleaning reduced point count and RANSAC aligned ground to +Z.
    raw_cloud = load_gaussian_ply(ws_dir / "03_gaussians" / "raw_splats.ply")
    cleaned_cloud = load_gaussian_ply(ws_dir / "04_processed" / "cleaned_splats.ply")
    assert cleaned_cloud.num_points < raw_cloud.num_points

    align_meta = json.loads((ws_dir / "04_processed" / "alignment_transform.json").read_text(encoding="utf-8"))
    assert abs(align_meta["aligned_normal"][2]) > 0.98

    # Re-running without force should skip every completed stage via manifest.json.
    res2 = run_pipeline("configs/default_mujoco.yaml", output_dir=ws_dir, mock=True)
    assert res2.executed_stages == []
    assert res2.skipped_stages == ["ingest", "pose", "reconstruct", "postprocess", "export"]

    # Partial resume from postprocess skips ingest/pose/reconstruct and re-runs postprocess/export.
    res3 = run_pipeline(
        "configs/default_mujoco.yaml",
        output_dir=ws_dir,
        from_stage="postprocess",
        force=True,
        mock=True,
    )
    assert res3.skipped_stages == ["ingest", "pose", "reconstruct"]
    assert res3.executed_stages == ["postprocess", "export"]

    # Explicit from_stage='export' must not re-run upstream stages even if their params changed.
    mod_cfg = load_config("configs/default_mujoco.yaml", workspace_dir=ws_dir)
    mod_cfg.pipeline.reconstruct.params["iterations"] = int(mod_cfg.pipeline.reconstruct.params.get("iterations", 8000)) + 123
    res4 = PipelineRunner().run(mod_cfg, from_stage="export", force=True, mock=True)
    assert res4.skipped_stages == ["ingest", "pose", "reconstruct", "postprocess"]
    assert res4.executed_stages == ["export"]


def test_unity_export_in_pipeline(tmp_path: Path) -> None:
    cfg = load_config("configs/exp_2dgs_vggt.yaml", workspace_dir=tmp_path / "unity_ws")
    res = run_pipeline(cfg, mock=True)
    assert res.mujoco_xml_path is not None and res.mujoco_xml_path.exists()
    assert res.unity_dir is not None
    assert (res.unity_dir / "visual_splats.ply").exists()
    assert (res.unity_dir / "collision_mesh.obj").exists()
    assert (res.unity_dir / "unity_import.json").exists()


def test_pure_3dgs_preset_pipeline(tmp_path: Path) -> None:
    ws_dir = tmp_path / "pure_3dgs_ws"
    res = run_pipeline("configs/pure_3dgs.yaml", output_dir=ws_dir, mock=True)
    assert res.executed_stages == ["ingest", "pose", "reconstruct", "postprocess", "export"]
    assert res.mujoco_xml_path is None
    assert res.gaussian_ply_path is not None and res.gaussian_ply_path.exists()
    assert (ws_dir / "05_export" / "gaussian_splats" / "splat_meta.json").exists()
    assert (ws_dir / "05_export" / "gaussian_splats" / "cameras.json").exists()


def test_supersplat_ply_export_orientation_and_centering(tmp_path: Path) -> None:
    ws_dir = tmp_path / "supersplat_ws"
    res = run_pipeline("configs/pure_3dgs.yaml", output_dir=ws_dir, mock=True)
    assert res.gaussian_ply_path is not None and res.gaussian_ply_path.is_file()

    # 1. Internal aligned cloud stays in MuJoCo right-handed +Z-up coordinates.
    aligned_cloud = load_gaussian_ply(ws_dir / "04_processed" / "aligned_splats.ply")
    n_zup, _ = _fit_ground_plane_ransac(
        aligned_cloud.means,
        trials=200,
        dist_threshold=0.05,
        up_prior=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert float(n_zup[2]) > 0.98

    # 2. Exported scene_3dgs.ply transformed by SuperSplat's R_z(180)=diag(-1,-1,1)
    #    must stand upright along PlayCanvas +Y axis and keep normalized quaternions.
    exported = load_gaussian_ply(res.gaussian_ply_path)
    assert exported.num_points > 0
    quat_norms = np.linalg.norm(exported.quats, axis=1)
    assert np.allclose(quat_norms, 1.0, atol=1e-4)

    pts_playcanvas = exported.means * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    n_pc, _ = _fit_ground_plane_ransac(
        pts_playcanvas,
        trials=200,
        dist_threshold=0.05,
        up_prior=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    assert float(n_pc[1]) > 0.98

    meta = json.loads((ws_dir / "05_export" / "gaussian_splats" / "splat_meta.json").read_text(encoding="utf-8"))
    assert meta["coordinate_frame"] == "supersplat_colmap"
    assert len(meta["foreground_pivot_zup"]) == 3

    # 3. Exported cameras.json must sit in the exact same coordinate space as scene_3dgs.ply.
    cams_payload = json.loads((ws_dir / "05_export" / "gaussian_splats" / "cameras.json").read_text(encoding="utf-8"))
    assert "SuperSplat" in cams_payload.get("coordinate_system", "")
    frames = cams_payload.get("frames", [])
    assert len(frames) >= 1
    w2c0 = np.asarray(frames[0]["world_to_cam"], dtype=np.float64)
    eye0 = -w2c0[:3, :3].T @ w2c0[:3, 3]
    fwd0 = w2c0[2, :3]
    to_origin = -eye0
    # Camera 0 optical axis looks forward toward foreground origin (0, 0, 0) rather than away.
    assert float(np.dot(fwd0, to_origin)) > 0.0


def test_input_change_invalidates_ingest_cache(tmp_path: Path) -> None:
    from PIL import Image

    dir_a = tmp_path / "video_a"
    dir_b = tmp_path / "video_b"
    dir_a.mkdir()
    dir_b.mkdir()
    for i in range(4):
        Image.new("RGB", (32, 32), color=(40 + i * 10, 80, 200)).save(dir_a / f"img_{i:02d}.png")
        Image.new("RGB", (32, 32), color=(220, 120, 30 + i * 10)).save(dir_b / f"img_{i:02d}.png")

    ws_dir = tmp_path / "shared_ws"
    cfg = PipelineConfig(
        io=IOConfig(input_path=str(dir_a), workspace_dir=str(ws_dir)),
        pipeline=PipelineStagesConfig(
            ingest=StageSpec(name="uniform_ffmpeg", params={"fps": 2.0, "max_frames": 8}),
            pose=StageSpec(name="mock_pose", params={}),
            reconstruct=StageSpec(name="mock_reconstruct", params={}),
            postprocess=[],
            export=[StageSpec(name="ply_3dgs", params={"scene_name": "cache_test"})],
        ),
    )
    runner = PipelineRunner()
    res1 = runner.run(cfg, mock=False)
    assert res1.executed_stages == ["ingest", "pose", "reconstruct", "export"]

    res2 = runner.run(cfg, mock=False)
    assert res2.executed_stages == []
    assert res2.skipped_stages == ["ingest", "pose", "reconstruct", "export"]

    # Switching to dir_b with identical ingest params must invalidate ingest + downstream.
    cfg.io.input_path = str(dir_b)
    res3 = runner.run(cfg, mock=False)
    assert res3.executed_stages == ["ingest", "pose", "reconstruct", "export"]
