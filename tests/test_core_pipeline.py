from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from real2sim_gs.core.config import IOConfig, PipelineConfig, PipelineStagesConfig, StageSpec
from real2sim_gs.core.external import build_command, run_external_command
from real2sim_gs.core.image_io import imread_unicode, imwrite_unicode, natural_sorted
from real2sim_gs.core.pipeline import PipelineRunner
from real2sim_gs.core.workspace import PipelineCancelledError, SceneWorkspace
from real2sim_gs.stages.pose._colmap_worker import resolve_use_sequential
from real2sim_gs.stages.pose.colmap_pose import _parse_colmap_cameras, _parse_colmap_images
from real2sim_gs.utils.ply_io import GaussianCloud, save_gaussian_ply


def _mock_cfg(ws_dir: Path, input_path: str = "./unused.mp4") -> PipelineConfig:
    return PipelineConfig(
        io=IOConfig(input_path=input_path, workspace_dir=str(ws_dir)),
        pipeline=PipelineStagesConfig(
            ingest=StageSpec(name="mock_ingest", params={"mock_num_views": 4, "mock_image_size": 32}),
            pose=StageSpec(name="mock_pose", params={}),
            reconstruct=StageSpec(name="mock_reconstruct", params={}),
            postprocess=[StageSpec(name="clean_floaters", params={"min_opacity": 0.05})],
            export=[StageSpec(name="ply_3dgs", params={"scene_name": "core_test"})],
        ),
    )


def test_chained_invalidation_and_mid_run_cancel(tmp_path: Path) -> None:
    ws_dir = tmp_path / "chained_ws"
    cfg = _mock_cfg(ws_dir)
    runner = PipelineRunner()
    res1 = runner.run(cfg, mock=False)
    assert res1.executed_stages == ["ingest", "pose", "reconstruct", "postprocess", "export"]

    # Change upstream pose param and cancel right after pose finishes (before reconstruct completes).
    cfg.pipeline.pose.params["extra_tag"] = "v2"
    stop = {"hit": False}

    def _on_prog(stage_kind: str, state: str, _info: dict) -> None:
        if stage_kind == "reconstruct" and state == "running":
            stop["hit"] = True

    with pytest.raises(PipelineCancelledError):
        runner.run(cfg, mock=False, progress_callback=_on_prog, cancel_check=lambda: stop["hit"])

    ws = SceneWorkspace(ws_dir)
    assert ws.is_completed("ingest", None) is True
    assert ws.is_completed("pose", None) is True
    assert ws.is_completed("reconstruct", None) is False
    assert ws.is_completed("postprocess", None) is False
    assert ws.is_completed("export", None) is False
    assert not (ws.export_dir / "gaussian_splats" / "scene_3dgs.ply").exists()

    # Subsequent resume skips already-updated ingest+pose and cleanly rebuilds all downstream stages.
    res2 = runner.run(cfg, mock=False)
    assert res2.skipped_stages == ["ingest", "pose"]
    assert res2.executed_stages == ["reconstruct", "postprocess", "export"]


def test_corrupt_manifest_recovery_and_strict_config(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "corrupt_ws")
    ws.ensure_dirs()
    ws.manifest_path.write_text("{broken json", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="manifest.json"):
        data = ws.load_manifest()
    assert data["stages"] == {}
    assert not ws.manifest_path.exists()
    assert ws.manifest_path.with_name("manifest.json.corrupt").is_file()

    # Unknown YAML keys rejected loudly (extra="forbid")
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(
            {
                "io": {"input_path": "a.mp4", "workspace_dir": "ws"},
                "pipeline": {
                    "ingest": {"name": "mock_ingest"},
                    "pose": {"name": "mock_pose"},
                    "reconstruct": {"name": "mock_reconstruct"},
                    "post_process": [],  # typo!
                },
            }
        )

    # Unknown stage implementation fails fast before touching workspace
    bad_ws = tmp_path / "fail_fast_ws"
    bad_cfg = _mock_cfg(bad_ws)
    bad_cfg.pipeline.export = [StageSpec(name="nonexistent_exporter")]
    with pytest.raises(KeyError, match="nonexistent_exporter"):
        PipelineRunner().run(bad_cfg, mock=False)
    assert not bad_ws.exists()


def test_unicode_opencv_io_and_natural_sort(tmp_path: Path) -> None:
    cn_dir = tmp_path / "中文目录_测试"
    img_path = cn_dir / "图像_2.png"
    arr = np.full((24, 32, 3), (40, 120, 210), dtype=np.uint8)
    imwrite_unicode(img_path, arr)
    loaded = imread_unicode(img_path)
    assert loaded is not None
    assert loaded.shape == (24, 32, 3)
    assert np.array_equal(loaded, arr)

    ordered = natural_sorted(["img_10.png", "img_2.png", "img_1.png", "img_20.png"])
    assert ordered == ["img_1.png", "img_2.png", "img_10.png", "img_20.png"]


def test_colmap_camera_layouts_and_empty_points2d_images(tmp_path: Path) -> None:
    cams_txt = tmp_path / "cameras.txt"
    cams_txt.write_text(
        "\n".join(
            [
                "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                "1 SIMPLE_PINHOLE 640 480 500 320 240",
                "2 PINHOLE 640 480 510 520 320 240",
                "3 SIMPLE_RADIAL 640 480 500 320 240 -0.05",
                "4 RADIAL 640 480 500 320 240 -0.05 0.01",
                "5 OPENCV 640 480 500 505 320 240 0.01 -0.02 0.001 -0.001",
                "6 FULL_OPENCV 640 480 500 505 320 240 0.1 0.2 0.01 0.02 0.3 0.4 0.5 0.6",
                "7 OPENCV_FISHEYE 640 480 400 405 320 240 0.05 -0.01 0.002 0.001",
                "8 SIMPLE_RADIAL_FISHEYE 640 480 410 320 240 0.04",
                "9 RADIAL_FISHEYE 640 480 410 320 240 0.04 -0.02",
            ]
        ),
        encoding="utf-8",
    )
    cams = _parse_colmap_cameras(cams_txt)
    assert cams[1]["fx"] == 500.0 and cams[1]["fy"] == 500.0
    assert cams[2]["fy"] == 520.0
    assert cams[3]["distortion"] == [-0.05, 0.0, 0.0, 0.0]
    assert len(cams[6]["distortion"]) == 8
    assert cams[7]["model"] == "OPENCV_FISHEYE" and cams[7]["cx"] == 320.0
    assert cams[8]["distortion"] == [0.04, 0.0, 0.0, 0.0]

    bad_cams = tmp_path / "bad_cameras.txt"
    bad_cams.write_text("1 UNKNOWN_MODEL 640 480 500 320 240\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported COLMAP camera model"):
        _parse_colmap_cameras(bad_cams)

    # Empty POINTS2D line between images must NOT cause odd/even line misalignment!
    imgs_txt = tmp_path / "images.txt"
    imgs_txt.write_text(
        "\n".join(
            [
                "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                "1 1 0 0 0 0 0 1 1 img_10.png",
                "",  # <-- empty POINTS2D line for img_10!
                "2 1 0 0 0 0 0 2 1 img_2.png",
                "10.0 20.0 -1",
            ]
        ),
        encoding="utf-8",
    )
    frames = _parse_colmap_images(imgs_txt, cams)
    assert [f["filename"] for f in frames] == ["img_2.png", "img_10.png"]

    # Matcher auto switching rules
    assert resolve_use_sequential("exhaustive", total_imgs=500, source_kind="video") is False
    assert resolve_use_sequential("auto", total_imgs=300, source_kind="images") is False
    assert resolve_use_sequential("auto", total_imgs=180, source_kind="video") is False
    assert resolve_use_sequential("auto", total_imgs=181, source_kind="video") is True


def test_external_command_spaces_chinese_and_cancel(tmp_path: Path) -> None:
    work_dir = tmp_path / "sub dir 中文"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_file = work_dir / "result file.txt"

    argv = build_command(
        '"{python_exe}" -c "from pathlib import Path; Path(r\'{out_file}\').write_text(\'ok\', encoding=\'utf-8\')"',
        {"python_exe": sys.executable, "out_file": out_file},
    )
    run_external_command(argv, label="ext_space", cwd=work_dir)
    assert out_file.read_text(encoding="utf-8") == "ok"

    # Cancellation kills slow child subprocess promptly
    ws = SceneWorkspace(tmp_path / "cancel_ext_ws")
    ws.ensure_dirs()
    t0 = time.monotonic()
    ws.cancel_check = lambda: (time.monotonic() - t0) > 0.3
    sleep_argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(PipelineCancelledError):
        run_external_command(sleep_argv, label="ext_sleep", cwd=work_dir, ws=ws)
    assert time.monotonic() - t0 < 4.0


def test_active_splat_pointer_and_worker_no_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ws = SceneWorkspace(Path("workspaces") / "ptr_ws")
    ws.ensure_dirs()
    dummy_cloud = GaussianCloud(
        means=np.zeros((4, 3), dtype=np.float32),
        scales=np.full((4, 3), -3.0, dtype=np.float32),
        quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (4, 1)),
        opacities=np.ones((4,), dtype=np.float32),
        sh_dc=np.zeros((4, 3), dtype=np.float32),
    )
    save_gaussian_ply(ws.aligned_splats_path, dummy_cloud)
    save_gaussian_ply(ws.cleaned_splats_path, dummy_cloud)

    # Even though aligned_splats.ply exists, setting cleaned as active must win!
    ws.set_active_splat(ws.cleaned_splats_path)
    assert ws.active_splat_path().resolve() == ws.cleaned_splats_path.resolve()

    ws.reset_stage_dir("postprocess")
    assert not ws.active_splat_pointer_path.exists()

    # Worker module import must NOT import torch or cv2
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import real2sim_gs.stages.pose._colmap_worker, sys; "
            "assert 'torch' not in sys.modules, 'torch leaked'; "
            "assert 'cv2' not in sys.modules, 'cv2 leaked'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_pipeline_preserves_workspace_on_missing_input_and_skips_cleaned_upload(tmp_path: Path) -> None:
    img_dir = tmp_path / "src_frames"
    img_dir.mkdir()
    for i in range(4):
        Image.new("RGB", (64, 48), color=(60 + i * 20, 100, 180)).save(img_dir / f"f_{i:02d}.png")

    ws_dir = tmp_path / "protect_ws"
    cfg = PipelineConfig(
        io=IOConfig(input_path=str(img_dir), workspace_dir=str(ws_dir)),
        pipeline=PipelineStagesConfig(
            ingest=StageSpec(name="uniform_ffmpeg", params={"fps": 2.0, "max_frames": 8}),
            pose=StageSpec(name="mock_pose", params={}),
            reconstruct=StageSpec(name="mock_reconstruct", params={}),
            postprocess=[],
            export=[StageSpec(name="ply_3dgs", params={"scene_name": "p"})],
        ),
    )
    runner = PipelineRunner()
    res1 = runner.run(cfg, mock=False)
    assert res1.executed_stages == ["ingest", "pose", "reconstruct", "export"]

    # Delete source frames (simulating upload cache purge); missing-input re-ingest must fail BEFORE deleting workspace.
    import shutil

    shutil.rmtree(img_dir)
    bad_cfg = cfg.model_copy(deep=True)
    bad_cfg.io.input_path = str(tmp_path / "nonexistent_video.mp4")
    with pytest.raises(FileNotFoundError, match="输入路径不存在"):
        runner.run(bad_cfg, from_stage=None, force=False, mock=False)
    ws = SceneWorkspace(ws_dir)
    assert (ws.frames_dir / "frames_meta.json").is_file()
    assert ws.raw_splats_path.is_file()
    assert ws.is_completed("ingest", None) is True

    # With original input_path (now deleted after cleanup) and updated reconstruct params,
    # from_stage=None should still skip ingest & pose cleanly without failing!
    cfg_retrain = cfg.model_copy(deep=True)
    cfg_retrain.pipeline.reconstruct.params["iterations"] = 99
    res2 = runner.run(cfg_retrain, from_stage=None, force=False, mock=False)
    assert res2.skipped_stages == ["ingest", "pose"]
    assert res2.executed_stages == ["reconstruct", "export"]
