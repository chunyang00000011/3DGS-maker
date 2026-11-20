from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from real2sim_gs.core.image_io import imread_unicode, imwrite_unicode, natural_sorted
from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import PipelineCancelledError, SceneWorkspace, atomic_write_json

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# sharpness_ffmpeg 候选帧过采样倍率（时间滑窗内挑最清晰帧需要冗余候选）。
_SHARPNESS_OVERSAMPLE = 2.4


def _probe_video_duration(input_path: Path) -> float | None:
    """Quickly read video duration (seconds) via ffprobe to cap temporary candidate explosion."""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None or not input_path.is_file():
        return None
    try:
        proc = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return None
        val = float(proc.stdout.strip())
        return val if math.isfinite(val) and val > 0.0 else None
    except Exception:  # noqa: BLE001
        return None


def _ensure_input(input_path: Path) -> str:
    """Validate/generate the ingest input and return its source kind ('video' or 'images')."""
    if not input_path.exists():
        if input_path.name == "sample_video.mp4":
            from real2sim_gs.utils.sample_video import generate_sample_orbiter_video

            generate_sample_orbiter_video(input_path)
        else:
            raise FileNotFoundError(
                f"输入路径不存在 (Input path does not exist): {input_path} "
                f"(解析路径: {input_path.resolve()})。请先在左上角上传视频文件，或检查本地路径拼写。"
            )
    return "images" if input_path.is_dir() else "video"


def _collect_candidate_frames(
    input_path: Path,
    tmp_dir: Path,
    fps: float,
    max_dimension: int = 1600,
    max_frames: int = 0,
    oversample: float = 1.0,
    ws: SceneWorkspace | None = None,
) -> list[Path]:
    if input_path.is_dir():
        frames = natural_sorted(
            p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        if not frames:
            raise ValueError(f"No image files found in directory: {input_path}")
        return frames

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("FFmpeg not found in PATH. Install FFmpeg or provide an image directory.")

    eff_fps = float(fps)
    if max_frames > 0:
        dur = _probe_video_duration(input_path)
        if dur is not None and dur > 1.0:
            eff_fps = min(eff_fps, max(0.2, (float(max_frames) * float(oversample)) / dur))

    pattern = str(tmp_dir / "raw_%06d.jpg")
    vf_filter = f"fps={eff_fps:.4f}"
    if max_dimension > 0:
        vf_filter = (
            f"fps={eff_fps:.4f},"
            f"scale='if(gt(iw\\,ih)\\,min({max_dimension}\\,iw)\\,-2)':"
            f"'if(gt(iw\\,ih)\\,-2\\,min({max_dimension}\\,ih))'"
        )
    cmd = [ffmpeg_bin, "-nostdin", "-y", "-i", str(input_path), "-vf", vf_filter, "-q:v", "2", pattern]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stderr_tail: deque[str] = deque(maxlen=40)

    def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            stderr_tail.append(line.rstrip("\r\n"))

    t_err = threading.Thread(target=_drain_stderr, daemon=True)
    t_err.start()
    try:
        while proc.poll() is None:
            if ws is not None:
                ws.check_cancelled()
            time.sleep(0.15)
    except PipelineCancelledError:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        t_err.join(timeout=1.0)
        raise

    t_err.join(timeout=2.0)
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg frame extraction failed:\n" + "\n".join(stderr_tail))

    frames = natural_sorted(tmp_dir.glob("raw_*.jpg"))
    if not frames:
        raise RuntimeError(f"FFmpeg extracted zero frames from video: {input_path}")
    return frames


def _compute_sharpness(image_path: Path) -> float:
    img = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _select_windowed_sharp_frames(
    candidates: list[Path],
    keep_ratio: float,
    max_frames: int,
    ws: SceneWorkspace | None = None,
) -> list[tuple[Path, float | None]]:
    """Select locally sharpest frame per temporal window to preserve trajectory coverage."""
    scored: list[tuple[Path, float]] = []
    for idx, p in enumerate(candidates):
        if ws is not None and idx % 10 == 0:
            ws.check_cancelled()
        scored.append((p, _compute_sharpness(p)))
    n = len(scored)
    if n <= 1:
        return list(scored)

    target_k = max(1, int(round(n * keep_ratio)))
    if max_frames > 0:
        target_k = min(target_k, max_frames)
    target_k = min(target_k, n)

    windows = np.array_split(np.arange(n), target_k)
    selected: list[tuple[Path, float | None]] = []
    for win in windows:
        if len(win) == 0:
            continue
        best_idx = max((int(i) for i in win), key=lambda idx: scored[idx][1])
        selected.append(scored[best_idx])
    return selected


def _write_selected_frames(
    selected: list[tuple[Path, float | None]],
    out_dir: Path,
    max_frames: int,
    method_name: str,
    source_kind: str,
    max_dimension: int = 1600,
    ws: SceneWorkspace | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    if max_frames > 0 and len(selected) > max_frames:
        indices = np.linspace(0, len(selected) - 1, num=max_frames, dtype=int)
        selected = [selected[int(i)] for i in indices]

    entries: list[dict[str, Any]] = []
    for idx, (src_path, sharpness) in enumerate(selected):
        if ws is not None and idx % 10 == 0:
            ws.check_cancelled()
        dst_name = f"frame_{idx:04d}.png"
        dst_path = out_dir / dst_name
        img = imread_unicode(src_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read frame image: {src_path}")
        h, w = img.shape[:2]
        if max_dimension > 0 and max(h, w) > max_dimension:
            scale = max_dimension / float(max(h, w))
            tw = max(2, int(round(w * scale / 2.0)) * 2)
            th = max(2, int(round(h * scale / 2.0)) * 2)
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            h, w = th, tw
        imwrite_unicode(dst_path, img)
        entries.append(
            {
                "id": idx,
                "filename": dst_name,
                "width": int(w),
                "height": int(h),
                "sharpness": None if sharpness is None else round(sharpness, 3),
                "source": src_path.name,
            }
        )

    meta_path = out_dir / "frames_meta.json"
    atomic_write_json(
        meta_path,
        {
            "method": method_name,
            "source_kind": source_kind,
            "num_frames": len(entries),
            "frames": entries,
        },
    )
    return entries, meta_path


@register_stage("ingest", "uniform_ffmpeg")
class UniformFFmpegIngest(BaseStage):
    """Extract frames uniformly from video or index an existing image folder."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        if ws.input_path is None:
            raise ValueError("SceneWorkspace.input_path is required for ingest stage.")
        fps = float(params.get("fps", 2.0))
        max_frames = int(params.get("max_frames", 150))
        max_dimension = int(params.get("max_dimension", 1600))
        source_kind = _ensure_input(ws.input_path)

        with tempfile.TemporaryDirectory(prefix="real2sim_ingest_") as tmp:
            ws.report_progress(0.15, "正在解码视频并抽取候选图像帧...")
            candidates = _collect_candidate_frames(
                ws.input_path,
                Path(tmp),
                fps=fps,
                max_dimension=max_dimension,
                max_frames=max_frames,
                oversample=1.0,
                ws=ws,
            )
            ws.report_progress(0.6, f"已获取 {len(candidates)} 张候选帧，正在保存均匀关键帧至工作区...")
            entries, meta_path = _write_selected_frames(
                [(p, None) for p in candidates],
                ws.frames_dir,
                max_frames=max_frames,
                method_name="uniform_ffmpeg",
                source_kind=source_kind,
                max_dimension=max_dimension,
                ws=ws,
            )

        return StageResult(
            artifacts={"frames_meta": ws.to_relative(meta_path)},
            metrics={"num_frames": len(entries)},
            summary=f"Extracted {len(entries)} frames uniformly.",
        )


@register_stage("ingest", "sharpness_ffmpeg")
class SharpnessFFmpegIngest(BaseStage):
    """Filter motion-blurred frames using temporal sliding-window Laplacian sharpness selection."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        if ws.input_path is None:
            raise ValueError("SceneWorkspace.input_path is required for ingest stage.")
        fps = float(params.get("fps", 2.0))
        max_frames = int(params.get("max_frames", 150))
        keep_ratio = float(params.get("sharpness_keep_ratio", 0.8))
        keep_ratio = max(0.1, min(1.0, keep_ratio))
        max_dimension = int(params.get("max_dimension", 1600))
        source_kind = _ensure_input(ws.input_path)

        with tempfile.TemporaryDirectory(prefix="real2sim_ingest_") as tmp:
            ws.report_progress(0.15, "正在解码视频并抽取候选图像帧...")
            candidates = _collect_candidate_frames(
                ws.input_path,
                Path(tmp),
                fps=fps,
                max_dimension=max_dimension,
                max_frames=max_frames,
                oversample=_SHARPNESS_OVERSAMPLE,
                ws=ws,
            )
            ws.report_progress(0.55, f"已获取 {len(candidates)} 张候选帧，正在进行 Laplacian 清晰度筛选...")
            selected = _select_windowed_sharp_frames(
                candidates, keep_ratio=keep_ratio, max_frames=max_frames, ws=ws
            )
            ws.report_progress(0.85, "正在保存优选关键帧至工作区...")
            entries, meta_path = _write_selected_frames(
                selected,
                ws.frames_dir,
                max_frames=max_frames,
                method_name="sharpness_ffmpeg",
                source_kind=source_kind,
                max_dimension=max_dimension,
                ws=ws,
            )

        avg_sharp = sum(float(e["sharpness"] or 0.0) for e in entries) / max(1, len(entries))
        return StageResult(
            artifacts={"frames_meta": ws.to_relative(meta_path)},
            metrics={"num_frames": len(entries), "avg_sharpness": round(avg_sharp, 2)},
            summary=f"Selected {len(entries)} locally sharp frames across temporal windows (ratio={keep_ratio}).",
        )
