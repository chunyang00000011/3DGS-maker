from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim_gs.core.external import kill_process_tree, popen_process_group_kwargs, run_external_command
from real2sim_gs.core.image_io import natural_sort_key
from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import PipelineCancelledError, SceneWorkspace, atomic_write_json
from real2sim_gs.stages.pose._colmap_worker import clean_pose_workdir, resolve_use_sequential
from real2sim_gs.utils.ply_io import save_point_cloud_ply

# 显式 COLMAP 相机模型 -> (参数个数, 是否等焦距, 畸变系数个数)
_COLMAP_CAMERA_LAYOUTS: dict[str, tuple[int, bool, int]] = {
    "SIMPLE_PINHOLE": (3, True, 0),
    "PINHOLE": (4, False, 0),
    "SIMPLE_RADIAL": (4, True, 1),
    "RADIAL": (5, True, 2),
    "OPENCV": (8, False, 4),
    "FULL_OPENCV": (12, False, 8),
    "OPENCV_FISHEYE": (8, False, 4),
    "SIMPLE_RADIAL_FISHEYE": (4, True, 1),
    "RADIAL_FISHEYE": (5, True, 2),
}


def _parse_colmap_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1].upper()
        width, height = int(parts[2]), int(parts[3])
        p = [float(v) for v in parts[4:]]

        layout = _COLMAP_CAMERA_LAYOUTS.get(model)
        if layout is None:
            raise ValueError(
                f"Unsupported COLMAP camera model '{model}' (supported: {sorted(_COLMAP_CAMERA_LAYOUTS)})"
            )
        min_params, single_focal, n_dist = layout
        if len(p) < min_params:
            raise ValueError(
                f"COLMAP camera {cam_id} ({model}) expected >= {min_params} params, got {len(p)}"
            )

        if single_focal:
            fx = fy = p[0]
            cx, cy = p[1], p[2]
            dist_raw = p[3 : 3 + n_dist]
        else:
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            dist_raw = p[4 : 4 + n_dist]

        if n_dist <= 4:
            dist = (dist_raw + [0.0, 0.0, 0.0, 0.0])[:4]
        else:
            dist = dist_raw

        cameras[cam_id] = {
            "model": model,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "distortion": dist,
        }
    return cameras


def _parse_colmap_images(path: Path, cameras: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    # 保留空的 POINTS2D 行，确保每张图严格占两行不会错位；仅在奇数行时丢弃文件末尾多余空行。
    lines = [ln.strip() for ln in raw_lines if not ln.lstrip().startswith("#")]
    while len(lines) % 2 == 1 and lines and lines[-1] == "":
        lines.pop()
    if len(lines) % 2 != 0:
        raise ValueError(f"Malformed COLMAP images.txt (odd line count {len(lines)}): {path}")

    for idx in range(0, len(lines), 2):
        header = lines[idx]
        if not header:
            continue
        parts = header.split()
        img_id = int(parts[0])
        qw, qx, qy, qz = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = " ".join(parts[9:])

        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = rot.astype(np.float32)
        w2c[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)

        cam = cameras[cam_id]
        frames.append(
            {
                "id": img_id,
                "filename": Path(name).name,
                "camera_model": cam["model"],
                "width": cam["width"],
                "height": cam["height"],
                "fx": cam["fx"],
                "fy": cam["fy"],
                "cx": cam["cx"],
                "cy": cam["cy"],
                "distortion": cam.get("distortion", [0.0, 0.0, 0.0, 0.0]),
                "world_to_cam": w2c.tolist(),
            }
        )
    frames.sort(key=lambda item: natural_sort_key(item["filename"]))
    return frames


def _parse_colmap_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pts: list[list[float]] = []
    cols: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        cols.append([int(parts[4]), int(parts[5]), int(parts[6])])
    if not pts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.uint8)


def _read_source_kind_and_count(ws: SceneWorkspace) -> tuple[str, int]:
    """Read source_kind ('video'|'images') and frame count from 01_frames/frames_meta.json."""
    if ws.frames_meta_path.is_file():
        try:
            meta = json.loads(ws.frames_meta_path.read_text(encoding="utf-8"))
            kind = str(meta.get("source_kind", "video")).lower()
            count = int(meta.get("num_frames") or len(meta.get("frames", [])))
            return (kind if kind in ("video", "images") else "video"), count
        except (OSError, ValueError, TypeError):
            pass
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    count = (
        sum(1 for p in ws.frames_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts)
        if ws.frames_dir.is_dir()
        else 0
    )
    return "video", count


def _run_pycolmap_sfm(
    frames_dir: Path,
    pose_dir: Path,
    camera_model: str,
    matcher: str,
    progress_cb: Callable[[float, str], None] | None = None,
    params: dict[str, Any] | None = None,
    ws: SceneWorkspace | None = None,
    source_kind: str = "video",
) -> Path:
    """Run C++ pycolmap SfM inside an isolated lightweight worker subprocess."""
    p_cfg = params or {}
    default_threads = min(6, max(2, (os.cpu_count() or 6) // 4))
    worker_payload = {
        "frames_dir": str(frames_dir.resolve()),
        "pose_dir": str(pose_dir.resolve()),
        "camera_model": camera_model,
        "matcher": matcher,
        "source_kind": source_kind,
        "max_image_size": int(p_cfg.get("max_image_size", 1600)),
        "max_num_features": int(p_cfg.get("max_num_features", 6144)),
        "num_threads": int(p_cfg.get("num_threads", default_threads)),
    }
    cmd = [sys.executable, "-m", "real2sim_gs.stages.pose._colmap_worker", json.dumps(worker_payload)]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        **popen_process_group_kwargs(),
    )

    state_lock = threading.Lock()
    latest_prog: list[tuple[float, str] | None] = [None]
    tail_lines: deque[str] = deque(maxlen=30)

    def _read_stdout() -> None:
        if proc.stdout is None:
            return
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\r\n")
            if line.startswith("__R2S_PROGRESS__\t"):
                parts = line.split("\t", 2)
                if len(parts) == 3:
                    try:
                        p_val = float(parts[1])
                        with state_lock:
                            latest_prog[0] = (p_val, parts[2])
                    except ValueError:
                        pass
            elif line:
                with state_lock:
                    tail_lines.append(line)

    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()

    try:
        while proc.poll() is None:
            if ws is not None:
                ws.check_cancelled()
            pending: tuple[float, str] | None = None
            with state_lock:
                pending = latest_prog[0]
                latest_prog[0] = None
            if pending is not None and progress_cb is not None:
                progress_cb(pending[0], pending[1])
            time.sleep(0.12)
    except PipelineCancelledError:
        kill_process_tree(proc)
        reader.join(timeout=1.0)
        raise

    reader.join(timeout=2.0)
    with state_lock:
        final_prog = latest_prog[0]
        err_tail = "\n".join(tail_lines)
    if final_prog is not None and progress_cb is not None:
        progress_cb(final_prog[0], final_prog[1])

    if proc.returncode != 0:
        raise RuntimeError(
            f"COLMAP / pycolmap 特征提取或 SfM 子进程异常退出 (exit_code={proc.returncode})。\n"
            f"建议降低抽帧数或调小 max_image_size。\n最近日志:\n{err_tail}"
        )

    txt_dir = pose_dir / "txt"
    if not (txt_dir / "cameras.txt").is_file():
        raise RuntimeError(f"pycolmap worker finished without cameras.txt in {txt_dir}.\n{err_tail}")
    return txt_dir


def _pick_largest_colmap_model(sparse_dir: Path) -> tuple[Path, int]:
    """Select the COLMAP sparse sub-model directory containing the largest images binary."""
    submodels = [p for p in sparse_dir.iterdir() if p.is_dir()]
    if not submodels:
        raise RuntimeError(f"COLMAP reconstruction produced no sparse model in {sparse_dir}")

    def _score(model_dir: Path) -> int:
        for fname in ("images.bin", "images.txt", "points3D.bin", "points3D.txt"):
            fpath = model_dir / fname
            if fpath.is_file():
                return int(fpath.stat().st_size)
        return 0

    best = max(submodels, key=_score)
    return best, len(submodels)


def _run_colmap_cli_sfm(
    colmap_bin: str,
    frames_dir: Path,
    pose_dir: Path,
    camera_model: str,
    matcher: str,
    progress_cb: Callable[[float, str], None] | None = None,
    params: dict[str, Any] | None = None,
    ws: SceneWorkspace | None = None,
    source_kind: str = "video",
) -> Path:
    """Fallback COLMAP CLI execution using drained process output & shared cancellation."""
    p_cfg = params or {}
    default_threads = min(6, max(2, (os.cpu_count() or 6) // 4))
    max_image_size = int(p_cfg.get("max_image_size", 1600))
    max_num_features = int(p_cfg.get("max_num_features", 6144))
    num_threads = max(1, int(p_cfg.get("num_threads", default_threads)))

    db_path, sparse_dir, txt_dir = clean_pose_workdir(pose_dir)
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    total_imgs = sum(1 for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts)
    use_sequential = resolve_use_sequential(matcher, total_imgs=total_imgs, source_kind=source_kind)
    matcher_cmd = "sequential_matcher" if use_sequential else "exhaustive_matcher"
    match_label = "sequential" if use_sequential else "exhaustive"

    extract_cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(frames_dir),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        camera_model,
        "--SiftExtraction.max_num_features",
        str(max_num_features),
        "--SiftExtraction.num_threads",
        str(num_threads),
    ]
    if max_image_size > 0:
        extract_cmd.extend(["--SiftExtraction.max_image_size", str(max_image_size)])

    match_cmd = [
        colmap_bin,
        matcher_cmd,
        "--database_path",
        str(db_path),
        "--SiftMatching.num_threads",
        str(num_threads),
    ]
    if use_sequential:
        match_cmd.extend(["--SequentialMatching.overlap", "25", "--SequentialMatching.quadratic_overlap", "1"])

    mapper_cmd = [
        colmap_bin,
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(frames_dir),
        "--output_path",
        str(sparse_dir),
        "--Mapper.num_threads",
        str(num_threads),
    ]

    steps = [
        (0.15, "1/3 正在提取图像局部特征点 (SIFT)...", extract_cmd, "extract"),
        (0.45, f"2/3 正在进行多视角特征匹配 ({match_label})...", match_cmd, "match"),
        (0.72, "3/3 正在执行增量式 SfM 三角化与全局捆绑调整...", mapper_cmd, "mapper"),
    ]
    for prog, msg, cmd, tag in steps:
        if ws is not None:
            ws.check_cancelled()
        if progress_cb is not None:
            progress_cb(prog, msg)
        run_external_command(
            cmd,
            label=f"colmap:{tag}",
            ws=ws,
            progress=prog,
            log_path=pose_dir / f"colmap_{tag}.log",
        )

    best_model, num_models = _pick_largest_colmap_model(sparse_dir)
    conv_cmd = [
        colmap_bin,
        "model_converter",
        "--input_path",
        str(best_model),
        "--output_path",
        str(txt_dir),
        "--output_type",
        "TXT",
    ]
    run_external_command(
        conv_cmd,
        label="colmap:convert",
        ws=ws,
        progress=0.92,
        log_path=pose_dir / "colmap_convert.log",
    )
    atomic_write_json(
        pose_dir / "sfm_summary.json",
        {
            "num_models": num_models,
            "total_input_frames": total_imgs,
            "matcher_used": match_label,
        },
    )
    return txt_dir


@register_stage("pose", "colmap")
class ColmapPoseStage(BaseStage):
    """Standard COLMAP Structure-from-Motion camera pose & sparse cloud solver."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        camera_model = str(params.get("camera_model", "SIMPLE_RADIAL")).upper()
        if camera_model not in _COLMAP_CAMERA_LAYOUTS:
            raise ValueError(
                f"Unsupported camera_model '{camera_model}'. Expected one of {sorted(_COLMAP_CAMERA_LAYOUTS)}."
            )
        matcher = str(params.get("matcher", "auto")).lower()
        source_kind, total_input_frames = _read_source_kind_and_count(ws)

        if importlib.util.find_spec("pycolmap") is not None:
            backend_used = "pycolmap"
            txt_dir = _run_pycolmap_sfm(
                ws.frames_dir,
                ws.pose_dir,
                camera_model,
                matcher,
                progress_cb=ws.report_progress,
                params=params,
                ws=ws,
                source_kind=source_kind,
            )
        else:
            colmap_bin = shutil.which("colmap")
            if colmap_bin is None:
                raise RuntimeError(
                    "Neither 'pycolmap' nor 'colmap' executable was found. "
                    "Install pycolmap (`pip install pycolmap`), or switch pose.name to 'feedforward_slot' / run with --mock."
                )
            backend_used = "colmap_cli"
            txt_dir = _run_colmap_cli_sfm(
                colmap_bin,
                ws.frames_dir,
                ws.pose_dir,
                camera_model,
                matcher,
                progress_cb=ws.report_progress,
                params=params,
                ws=ws,
                source_kind=source_kind,
            )

        cameras = _parse_colmap_cameras(txt_dir / "cameras.txt")
        frames = _parse_colmap_images(txt_dir / "images.txt", cameras)
        pts, cols = _parse_colmap_points3d(txt_dir / "points3D.txt")

        num_models = 1
        summary_path = ws.pose_dir / "sfm_summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                num_models = int(summary.get("num_models", 1))
                total_input_frames = int(summary.get("total_input_frames", total_input_frames))
            except (OSError, ValueError, TypeError):
                pass
        total_input_frames = max(total_input_frames, len(frames))
        registered_ratio = round(len(frames) / max(1, total_input_frames), 4)
        if registered_ratio < 0.5:
            msg = (
                f"COLMAP 视角注册率偏低 ({len(frames)}/{total_input_frames} = {registered_ratio:.1%}，"
                f"子图数={num_models})，建议放慢拍摄速度或增加重叠率。"
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            ws.report_progress(0.94, msg)

        atomic_write_json(ws.cameras_path, {"backend": backend_used, "frames": frames})
        save_point_cloud_ply(ws.init_points_path, pts, cols)
        ws.report_progress(
            0.96,
            f"位姿解算成功：注册 {len(frames)}/{total_input_frames} 个相机视角，稀疏点 {pts.shape[0]} 个",
        )

        return StageResult(
            artifacts={
                "cameras": ws.to_relative(ws.cameras_path),
                "init_points": ws.to_relative(ws.init_points_path),
            },
            metrics={
                "pose_backend": backend_used,
                "registered_views": len(frames),
                "total_input_frames": total_input_frames,
                "registered_ratio": registered_ratio,
                "num_models": num_models,
                "init_points": int(pts.shape[0]),
            },
            summary=(
                f"COLMAP ({backend_used}) registered {len(frames)}/{total_input_frames} views "
                f"({num_models} sub-model(s)) and {pts.shape[0]} sparse points."
            ),
        )
