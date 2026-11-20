"""Lightweight isolated worker for pycolmap SfM (avoids importing torch / cv2)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from real2sim_gs.core.image_io import natural_sorted
from real2sim_gs.core.workspace import atomic_write_json

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def clean_pose_workdir(pose_dir: Path) -> tuple[Path, Path, Path]:
    """Remove stale SQLite DB/WAL/SHM and partial SfM directories before running COLMAP."""
    db_path = pose_dir / "database.db"
    for suffix in ("", "-wal", "-shm"):
        (pose_dir / f"database.db{suffix}").unlink(missing_ok=True)
    sparse_dir = pose_dir / "sparse"
    txt_dir = pose_dir / "txt"
    shutil.rmtree(sparse_dir, ignore_errors=True)
    shutil.rmtree(txt_dir, ignore_errors=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    return db_path, sparse_dir, txt_dir


def resolve_use_sequential(matcher: str, total_imgs: int, source_kind: str) -> bool:
    """Use sequential matching when explicitly requested or in `auto` mode on >180 video frames."""
    m = matcher.strip().lower()
    if m == "sequential":
        return True
    if m == "exhaustive":
        return False
    if m == "auto":
        return source_kind == "video" and total_imgs > 180
    raise ValueError(f"Unsupported COLMAP matcher '{matcher}'. Expected 'auto', 'exhaustive', or 'sequential'.")


def run_pycolmap_worker(payload: dict[str, Any]) -> None:
    """Isolated worker entrypoint for C++ pycolmap SfM so native OOM/abort never kills Studio."""
    import pycolmap
    from PIL import Image

    frames_dir = Path(payload["frames_dir"])
    pose_dir = Path(payload["pose_dir"])
    camera_model = str(payload["camera_model"])
    matcher = str(payload["matcher"])
    source_kind = str(payload.get("source_kind", "video")).lower()
    max_image_size = int(payload.get("max_image_size", 1600))
    max_num_features = int(payload.get("max_num_features", 6144))
    default_threads = min(6, max(2, (os.cpu_count() or 6) // 4))
    num_threads = max(1, int(payload.get("num_threads", default_threads)))

    def _emit(prog: float, msg: str) -> None:
        print(f"__R2S_PROGRESS__\t{prog:.4f}\t{msg}", flush=True)

    db_path, sparse_dir, txt_dir = clean_pose_workdir(pose_dir)
    image_files = natural_sorted(p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    total_imgs = len(image_files)
    first_max_dim = 0
    if image_files:
        with Image.open(image_files[0]) as im:
            first_max_dim = max(im.size)

    reader_opts = pycolmap.ImageReaderOptions(camera_model=camera_model)
    extraction_opts = pycolmap.FeatureExtractionOptions()
    if max_image_size > 0:
        extraction_opts.max_image_size = max_image_size
    extraction_opts.num_threads = num_threads
    if hasattr(extraction_opts, "sift"):
        extraction_opts.sift.max_num_features = max_num_features
        # Avoid 2x SIFT first-octave upsampling (33 MP pyramid) on HD/4K inputs.
        if first_max_dim >= 1200:
            extraction_opts.sift.first_octave = 0

    _emit(
        0.15,
        f"1/3 正在提取图像局部特征点 (SIFT, max_size={extraction_opts.max_image_size}, {num_threads}线程)...",
    )
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(frames_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=extraction_opts,
    )

    matching_opts = pycolmap.FeatureMatchingOptions()
    matching_opts.num_threads = num_threads
    use_sequential = resolve_use_sequential(matcher, total_imgs=total_imgs, source_kind=source_kind)
    match_label = "sequential" if use_sequential else "exhaustive"
    _emit(0.45, f"2/3 正在进行多视角特征匹配 ({match_label}, {num_threads}线程)...")
    if use_sequential:
        pairing_opts = pycolmap.SequentialPairingOptions()
        pairing_opts.overlap = 25
        pairing_opts.quadratic_overlap = True
        pairing_opts.num_threads = num_threads
        pycolmap.match_sequential(
            database_path=str(db_path),
            matching_options=matching_opts,
            pairing_options=pairing_opts,
        )
    else:
        pycolmap.match_exhaustive(database_path=str(db_path), matching_options=matching_opts)

    inc_opts = pycolmap.IncrementalPipelineOptions()
    inc_opts.num_threads = num_threads
    inc_opts.min_num_matches = 12
    inc_opts.mapper.abs_pose_min_num_inliers = 20
    inc_opts.mapper.abs_pose_min_inlier_ratio = 0.18
    inc_opts.mapper.init_min_tri_angle = 8.0
    inc_opts.mapper.filter_min_tri_angle = 1.0
    reg_counter = {"n": 0}

    def _on_next_img() -> None:
        reg_counter["n"] += 1
        n = reg_counter["n"]
        if n == 1 or n % 5 == 0 or n >= total_imgs:
            frac = min(0.92, 0.72 + 0.20 * (n / max(1, total_imgs)))
            _emit(frac, f"3/3 正在增量三角化与全局捆绑调整 (已处理 {n}/{total_imgs} 视角)...")

    _emit(0.72, "3/3 正在执行增量式 SfM 三角化与全局捆绑调整...")
    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(frames_dir),
        output_path=str(sparse_dir),
        options=inc_opts,
        next_image_callback=_on_next_img,
    )
    if not maps:
        raise RuntimeError(
            "pycolmap incremental_mapping produced no valid sparse reconstruction. "
            "Check that video frames have sufficient parallax and texture."
        )

    best_rec = max(maps.values(), key=lambda rec: rec.num_reg_images())
    if len(maps) > 1:
        _emit(
            0.94,
            f"共生成 {len(maps)} 个子图，已选取最大连通子图 {best_rec.num_reg_images()}/{total_imgs} 视角",
        )
    best_rec.write_text(str(txt_dir))
    atomic_write_json(
        pose_dir / "sfm_summary.json",
        {
            "num_models": len(maps),
            "registered_views": int(best_rec.num_reg_images()),
            "total_input_frames": int(total_imgs),
            "matcher_used": match_label,
        },
    )


if __name__ == "__main__":
    run_pycolmap_worker(json.loads(sys.argv[1]))
