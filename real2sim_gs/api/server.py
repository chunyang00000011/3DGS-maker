from __future__ import annotations

import base64
import functools
import gc
import json
import math
import os
import platform
import queue
import re
import secrets
import shutil
import stat
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

import cv2
import numpy as np
import torch
import trimesh
import yaml
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.middleware.trustedhost import TrustedHostMiddleware

from real2sim_gs.api.sdk import check_environment, list_available_stages, run_pipeline
from real2sim_gs.core.config import PipelineConfig, load_config
from real2sim_gs.core.doctor import is_gsplat_cuda_ready
from real2sim_gs.core.workspace import STAGE_ORDER, PipelineCancelledError, SceneWorkspace, atomic_write_text
from real2sim_gs.editor import (
    SceneEditManager,
    SegmentLiftConfig,
    apply_scene_edits_to_gpu_model,
    bake_edited_scene,
    build_orbit_camera_view,
    estimate_target_anchor_3d,
    export_single_object_ply,
    get_sam3_status,
    get_shared_sam3_backend,
    lift_masks_to_3dgs,
    pick_object_at_pixel,
    plan_active_views,
    project_world_point,
    render_camera_rgb,
)
from real2sim_gs.stages.postprocess.align_sim import realign_workspace_scene
from real2sim_gs.stages.reconstruct.gsplat_whitebox.dataset import CameraView
from real2sim_gs.stages.reconstruct.gsplat_whitebox.model import GaussianModel
from real2sim_gs.stages.reconstruct.gsplat_whitebox.trainer import _render_view
from real2sim_gs.ui import UI_STATIC_DIR
from real2sim_gs.utils.ply_io import load_gaussian_ply, load_point_cloud_ply


_RENDER_CACHE_LOCK = threading.Lock()
_RENDER_MODEL_CACHE: OrderedDict[tuple[str, int, str], GaussianModel] = OrderedDict()
# 全局 GPU 渲染串行锁：同一时刻仅允许一个 /api/render-view 光栅化请求占用显存。
_GPU_RENDER_LOCK = threading.Lock()
_EDITOR_MANAGERS_LOCK = threading.Lock()
_EDITOR_MANAGERS: dict[str, SceneEditManager] = {}

_DIR_SIZE_LOCK = threading.Lock()
_DIR_SIZE_CACHE: dict[str, tuple[int, float, int]] = {}  # path -> (dir mtime_ns, cached_at, size)
_DIR_SIZE_TTL_SEC = 30.0

STUDIO_TOKEN_HEADER = "X-Studio-Token"
STUDIO_TOKEN_ENV = "REAL2SIM_STUDIO_TOKEN"
MAX_UPLOAD_ENV = "REAL2SIM_MAX_UPLOAD_GB"
_TOKEN_PLACEHOLDER = "__REAL2SIM_STUDIO_TOKEN__"
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]")
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_DEFAULT_MAX_UPLOAD_BYTES = 8 * 1024**3
_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024
_UPLOAD_GRACE_SEC = 30 * 60
_MAX_TERMINAL_JOBS = 50
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
UPLOAD_ALLOWED_SUFFIXES = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
)
_SAMPLE_VIDEO_NAME = "sample_video.mp4"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WorkspaceActionRequest(BaseModel):
    """Payload referencing a workspace root path."""

    workspace_dir: str


class WorkspaceDeleteRequest(BaseModel):
    """Payload to slim intermediate cache or permanently remove a workspace."""

    workspace_dir: str
    mode: str = "all"  # "all" | "intermediate"


class Sam3ConfigRequest(BaseModel):
    """Configure local/HF SAM 3 checkpoint source or release VRAM."""

    model_path: str | None = None
    unload: bool = False


class EditorPreview2DRequest(BaseModel):
    """2D SAM 3 prompt preview request anchored to the active 3D viewport camera."""

    workspace_dir: str
    yaw: float = 0.72
    pitch: float = 0.48
    distance: float = Field(3.2, ge=0.01, le=500.0)
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.25
    width: int = Field(640, ge=160, le=4096)
    height: int = Field(420, ge=120, le=4096)
    fov_scale: float = Field(1.05, ge=0.5, le=3.0)
    theme: str = "light"
    points: list[list[float]] | None = None
    labels: list[int] | None = None
    box: list[float] | None = None
    text: str | None = None


class EditorSegment3DRequest(EditorPreview2DRequest):
    """Multi-view SAM 3 + B3-Seg + FlashSplat + GaussianCut + SAGD 3D object segmentation request."""

    name: str | None = None
    num_views: int = Field(12, ge=4, le=24)
    use_sagd: bool = True
    use_gaussiancut: bool = True
    bg_bias: float = Field(0.05, ge=0.0, le=0.45)
    threshold: float = Field(0.5, ge=0.05, le=0.95)


class EditorRefineRequest(BaseModel):
    """Instant score threshold & bottom Z-plane clipping adjustment."""

    workspace_dir: str
    object_id: int
    threshold: float | None = None
    z_clip: float | None = None
    record_undo: bool = True


class EditorPickRequest(BaseModel):
    """3D depth-aware object picking request at viewport pixel coordinates."""

    workspace_dir: str
    yaw: float = 0.72
    pitch: float = 0.48
    distance: float = Field(3.2, ge=0.01, le=500.0)
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.25
    width: int = Field(640, ge=160, le=4096)
    height: int = Field(420, ge=120, le=4096)
    fov_scale: float = Field(1.05, ge=0.5, le=3.0)
    u: float
    v: float


class EditorUpdateObjectRequest(BaseModel):
    """Update object visibility, deletion flag, name, or 3D rigid transform + scale."""

    workspace_dir: str
    object_id: int
    name: str | None = None
    visible: bool | None = None
    deleted: bool | None = None
    translation: list[float] | None = None
    rotation_euler_deg: list[float] | None = None
    scale: float | None = None
    record_undo: bool = True


class EditorObjectActionRequest(BaseModel):
    """Execute discrete Scene Graph action (undo / redo / duplicate / delete / restore / reset_transform)."""

    workspace_dir: str
    action: str
    object_id: int | None = None


class EditorBakeRequest(BaseModel):
    """Bake non-destructive Scene Graph edits into edited_splats.ply and update exports."""

    workspace_dir: str
    reexport_ply: bool = True


class EditorAlignSceneRequest(BaseModel):
    """Re-run SOTA auto-leveling/Manhattan heading or apply incremental Studio scene calibration."""

    workspace_dir: str
    mode: str = "auto"  # "auto" | "set_front" | "manual"
    roll_deg: float = Field(0.0, ge=-45.0, le=45.0)
    pitch_deg: float = Field(0.0, ge=-45.0, le=45.0)
    yaw_deg: float = Field(0.0, ge=-180.0, le=180.0)
    z_offset: float = Field(0.0, ge=-10.0, le=10.0)
    current_yaw_rad: float | None = None
    reexport_ply: bool = True


class JobCreateRequest(BaseModel):
    """Payload to trigger a Real-to-Sim pipeline job over HTTP."""

    config_path: str = "configs/default_mujoco.yaml"
    inline_config: dict[str, Any] | None = None
    input_path: str | None = None
    output_dir: str | None = None
    from_stage: str | None = None
    force: bool = False
    mock: bool = False
    run_async: bool = False


class JobRecord(BaseModel):
    """Tracked pipeline job state, per-stage progress, and output artifacts."""

    job_id: str
    status: str
    created_at: str
    finished_at: str | None = None
    workspace_dir: str | None = None
    current_stage: str | None = None
    overall_progress: float = 0.0
    stage_progress: float = 0.0
    status_detail: str = ""
    telemetry: dict[str, Any] = Field(default_factory=dict)
    loss_history: list[dict[str, float]] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    stage_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    executed_stages: list[str] = Field(default_factory=list)
    skipped_stages: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    mujoco_xml_path: str | None = None
    unity_dir: str | None = None
    gaussian_ply_path: str | None = None
    splat_ply_path: str | None = None
    error: str | None = None
    error_traceback: str | None = None
    diagnostic_report: str | None = None


class JobStore:
    """Thread-safe in-memory job state container with a single serial GPU worker queue."""

    def __init__(self, max_terminal_jobs: int = _MAX_TERMINAL_JOBS) -> None:
        self._lock = threading.Lock()
        self.max_terminal_jobs = max_terminal_jobs
        self.jobs: dict[str, JobRecord] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_inputs: dict[str, Path] = {}
        self._job_workspaces: dict[str, Path] = {}
        self._deleting: set[Path] = set()
        self._pending_ids: set[str] = set()
        self._last_log_bucket: dict[tuple[str, str], int] = {}
        self._queue: queue.Queue[tuple[str, Callable[[], None], threading.Event]] = queue.Queue()
        self._worker: threading.Thread | None = None

    def has_running_job(self) -> bool:
        with self._lock:
            return any(rec.status == "running" for rec in self.jobs.values())

    def has_active_job_for_workspace(self, resolved_ws: Path) -> bool:
        with self._lock:
            return self._has_active_job_locked(resolved_ws)

    def _has_active_job_locked(self, resolved_ws: Path) -> bool:
        for jid, ws in self._job_workspaces.items():
            rec = self.jobs.get(jid)
            if rec is not None and rec.status in ("queued", "running") and ws == resolved_ws:
                return True
        return False

    def create(
        self,
        workspace_dir: Path | None = None,
        display_workspace: str | None = None,
        input_path: Path | None = None,
    ) -> JobRecord:
        """Register a queued job; resolved workspace/input are bound immediately (delete/cleanup guards)."""
        with self._lock:
            if workspace_dir is not None and workspace_dir in self._deleting:
                raise HTTPException(status_code=409, detail="该工作区正在被清理或删除，请稍后再提交任务")
            job_id = uuid.uuid4().hex[:12]
            stamp = datetime.now().strftime("%H:%M:%S")
            rec = JobRecord(
                job_id=job_id,
                status="queued",
                created_at=datetime.now(timezone.utc).isoformat(),
                workspace_dir=display_workspace,
                status_detail="任务已排队等待调度",
                logs=[f"[{stamp}] [queue] 任务已加入执行队列"],
                stage_states={stage: {"status": "pending"} for stage in STAGE_ORDER},
            )
            self.jobs[job_id] = rec
            self._cancel_events[job_id] = threading.Event()
            if workspace_dir is not None:
                self._job_workspaces[job_id] = workspace_dir
            if input_path is not None:
                self._active_inputs[job_id] = input_path
            return rec

    def begin_delete(self, resolved_ws: Path) -> None:
        """Atomically verify no queued/running job targets the workspace and mark it as deleting."""
        with self._lock:
            if self._has_active_job_locked(resolved_ws):
                raise HTTPException(status_code=409, detail="该工作区正在训练或执行中，请先点击停止训练再清理数据")
            if resolved_ws in self._deleting:
                raise HTTPException(status_code=409, detail="该工作区正在被清理，请勿重复提交")
            self._deleting.add(resolved_ws)

    def end_delete(self, resolved_ws: Path) -> None:
        with self._lock:
            self._deleting.discard(resolved_ws)

    def get_cancel_event(self, job_id: str) -> threading.Event:
        with self._lock:
            if job_id not in self._cancel_events:
                raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
            return self._cancel_events[job_id]

    def request_cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
            rec = self.jobs[job_id]
            ev = self._cancel_events.get(job_id)
            if ev is not None:
                ev.set()
            if rec.status in _TERMINAL_STATUSES:
                return rec.model_copy(deep=True)
            stamp = datetime.now().strftime("%H:%M:%S")
            if rec.status == "queued":
                rec.status = "cancelled"
                rec.status_detail = "任务已在排队阶段取消"
                rec.finished_at = datetime.now(timezone.utc).isoformat()
                rec.logs.append(f"[{stamp}] [cancel] 任务已取消")
            else:
                rec.status_detail = "正在安全停止训练并保存当前断点..."
                rec.logs.append(f"[{stamp}] [cancel] 已发送停止信号，正在安全保存当前进度...")
            return rec.model_copy(deep=True)

    def active_input_paths(self) -> set[Path]:
        with self._lock:
            return {
                path
                for jid, path in self._active_inputs.items()
                if (rec := self.jobs.get(jid)) is not None and rec.status in ("queued", "running")
            }

    def submit(self, job_id: str, fn: Callable[[], None]) -> threading.Event:
        """Enqueue a job for the single serial worker thread; returns an event set on completion."""
        done = threading.Event()
        with self._lock:
            self._pending_ids.add(job_id)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._worker_loop, name="real2sim-job-worker", daemon=True)
                self._worker.start()
        self._queue.put((job_id, fn, done))
        return done

    def _worker_loop(self) -> None:
        while True:
            job_id, fn, done = self._queue.get()
            try:
                fn()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            finally:
                with self._lock:
                    self._pending_ids.discard(job_id)
                done.set()
                self.prune_terminal()
                self._queue.task_done()

    def prune_terminal(self) -> None:
        """Keep only the most recent terminal jobs and drop their auxiliary bookkeeping."""
        with self._lock:
            terminal = sorted(
                (r for r in self.jobs.values() if r.status in _TERMINAL_STATUSES and r.job_id not in self._pending_ids),
                key=lambda r: r.created_at,
                reverse=True,
            )
            for rec in terminal[self.max_terminal_jobs :]:
                jid = rec.job_id
                self.jobs.pop(jid, None)
                self._cancel_events.pop(jid, None)
                self._active_inputs.pop(jid, None)
                self._job_workspaces.pop(jid, None)
                for key in [k for k in self._last_log_bucket if k[0] == jid]:
                    self._last_log_bucket.pop(key, None)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
            return self.jobs[job_id].model_copy(deep=True)

    def list_recent(self) -> list[JobRecord]:
        with self._lock:
            items = [r.model_copy(deep=True) for r in self.jobs.values()]
        items.sort(key=lambda r: r.created_at, reverse=True)
        return items

    def update_stage(self, job_id: str, stage_kind: str, state: str, info: dict[str, Any]) -> None:
        with self._lock:
            rec = self.jobs[job_id]
            rec.current_stage = stage_kind
            rec.stage_states[stage_kind] = {"status": state, **info}

            stage_p = float(info.get("progress", 1.0 if state in ("completed", "skipped") else 0.0))
            rec.stage_progress = round(max(0.0, min(1.0, stage_p)), 4)

            total_p = 0.0
            for s in STAGE_ORDER:
                st = rec.stage_states.get(s, {})
                st_status = st.get("status")
                if st_status in ("completed", "skipped"):
                    total_p += 1.0
                elif s == stage_kind and state == "running":
                    total_p += rec.stage_progress
            rec.overall_progress = round(total_p / len(STAGE_ORDER), 4)

            detail = str(info.get("detail") or info.get("impl") or state)
            rec.status_detail = detail

            telem = info.get("telemetry") or {}
            if telem:
                rec.telemetry = telem
                if "loss" in telem and "step" in telem:
                    rec.loss_history.append({"step": float(telem["step"]), "loss": float(telem["loss"])})
                    if len(rec.loss_history) > 120:
                        rec.loss_history = rec.loss_history[-120:]

            should_log = True
            if state == "running" and stage_kind == "reconstruct" and "step" in telem:
                bucket = int(stage_p * 10)
                key = (job_id, stage_kind)
                prev_bucket = self._last_log_bucket.get(key, -1)
                step_num = int(telem.get("step", 0))
                total_steps = int(telem.get("total_steps", 0))
                if step_num != 1 and step_num != total_steps and bucket == prev_bucket:
                    should_log = False
                else:
                    self._last_log_bucket[key] = bucket

            if should_log:
                stamp = datetime.now().strftime("%H:%M:%S")
                entry = f"[{stamp}] [{stage_kind}] {detail}"
                if not rec.logs or rec.logs[-1] != entry:
                    rec.logs.append(entry)
                    if len(rec.logs) > 80:
                        rec.logs = rec.logs[-80:]


def _resolve_configs_dir() -> Path:
    cwd_cfg = Path("configs")
    if cwd_cfg.is_dir():
        return cwd_cfg
    return Path(__file__).resolve().parents[2] / "configs"


def _compact_mesh_payload(mesh: trimesh.Trimesh, max_faces: int) -> dict[str, Any]:
    """Subsample mesh faces compactly for responsive browser 3D inspection."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    total_faces = int(len(faces))
    if total_faces == 0 or len(verts) == 0:
        return {"vertices": [], "faces": [], "total_faces": 0}

    if total_faces > max_faces:
        idx = np.linspace(0, total_faces - 1, num=max_faces, dtype=int)
        faces = faces[idx]

    used_verts, inverse = np.unique(faces.ravel(), return_inverse=True)
    compact_verts = np.round(verts[used_verts], 4).tolist()
    compact_faces = inverse.reshape(-1, 3).tolist()
    return {
        "vertices": compact_verts,
        "faces": compact_faces,
        "total_faces": total_faces,
    }


def _extract_aligned_camera_presets(
    ws: SceneWorkspace,
    active_ply_rel: str | None,
    pts_sample: np.ndarray,
    max_cams: int = 12,
) -> list[dict[str, Any]]:
    """Transform captured COLMAP viewpoints into active workspace coordinates as interactive orbit presets."""
    refined = ws.gaussians_dir / "refined_cameras.json"
    cam_path = refined if refined.is_file() else ws.cameras_path
    if not cam_path.is_file():
        return []
    try:
        frames = json.loads(cam_path.read_text(encoding="utf-8")).get("frames", [])
    except Exception:  # noqa: BLE001
        return []
    if not frames:
        return []

    rot_align = np.eye(3, dtype=np.float64)
    trans_align = np.zeros(3, dtype=np.float64)
    scale_align = 1.0
    t_path = ws.processed_dir / "alignment_transform.json"
    is_aligned = t_path.is_file() and (not active_ply_rel or "raw_splats" not in active_ply_rel)
    if is_aligned and t_path.is_file():
        try:
            t_data = json.loads(t_path.read_text(encoding="utf-8"))
            m = np.asarray(t_data.get("matrix_4x4", []), dtype=np.float64)
            scale_align = float(t_data.get("scale_factor", 1.0)) or 1.0
            if m.shape == (4, 4):
                rot_align = m[:3, :3] / scale_align
                trans_align = m[:3, 3]
        except Exception:  # noqa: BLE001
            pass

    pick_idx = (
        np.linspace(0, len(frames) - 1, num=min(max_cams, len(frames)), dtype=int).tolist()
        if len(frames) > 1
        else [0]
    )
    presets: list[dict[str, Any]] = []
    for idx in pick_idx:
        c = frames[int(idx)]
        w2c = np.asarray(c.get("world_to_cam", []), dtype=np.float64)
        if w2c.shape != (4, 4):
            continue
        r_raw = w2c[:3, :3]
        t_raw = w2c[:3, 3]
        eye_raw = -r_raw.T @ t_raw
        fwd_raw = r_raw[2, :]

        eye = scale_align * (rot_align @ eye_raw) + trans_align
        fwd = rot_align @ fwd_raw
        fwd_len = float(np.linalg.norm(fwd))
        if fwd_len < 1e-6:
            continue
        fwd /= fwd_len

        dist = 3.2
        if pts_sample.shape[0] > 0:
            rel = pts_sample - eye[None, :]
            depths = rel @ fwd
            perp = np.linalg.norm(rel - depths[:, None] * fwd[None, :], axis=1)
            cone = (depths > 0.25) & (perp < 0.70 * depths)
            valid_d = depths[cone] if cone.any() else depths[depths > 0.25]
            if valid_d.size > 0:
                dist = float(np.clip(np.percentile(valid_d, 35), 0.45, 30.0))

        target = eye + dist * fwd
        pitch = math.asin(max(-0.9999, min(0.9999, -float(fwd[2]))))
        yaw = math.atan2(-float(fwd[1]), -float(fwd[0]))
        min_dim = max(1.0, float(min(int(c.get("width", 1000)), int(c.get("height", 1000)))))
        fov_scale = float(np.clip(float(c.get("fx", min_dim)) / min_dim, 0.85, 1.55))

        presets.append(
            {
                "name": str(c.get("filename", f"cam_{idx}")),
                "yaw": round(yaw, 5),
                "pitch": round(pitch, 5),
                "distance": round(dist, 4),
                "target": [round(float(v), 4) for v in target.tolist()],
                "fov_scale": round(fov_scale, 3),
            }
        )
    return presets


def _safe_read_json(path: Path, warnings: list[str]) -> Any | None:
    """Read optional JSON metadata; record a warning instead of failing the whole preview."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"{path.name} 解析失败: {exc}")
        return None


def _build_scene_preview(
    ws_root: Path,
    max_splats: int = 18_000,
    max_mesh_faces: int = 4_500,
    display_dir: str | None = None,
) -> dict[str, Any]:
    """Assemble lightweight 3D splats, collision mesh, convex parts, XML, and frames for Studio UI."""
    if not ws_root.exists():
        raise HTTPException(status_code=404, detail=f"Workspace not found: {display_dir or ws_root}")

    ws = SceneWorkspace(ws_root)
    warnings: list[str] = []
    # 只读解析：不调用 ws.load_manifest()，避免 GET 预览请求触发损坏清单的备份改名副作用。
    manifest = _safe_read_json(ws.manifest_path, warnings)
    if not isinstance(manifest, dict):
        if ws.manifest_path.is_file() and not warnings:
            warnings.append("manifest.json 格式非法（非 JSON 对象）")
        manifest = {"stages": {}, "corrupted": bool(warnings)}

    splats_payload: dict[str, Any] = {
        "total_points": 0,
        "shown_points": 0,
        "positions": [],
        "colors": [],
        "opacities": [],
        "scales": [],
        "scales_3d": [],
        "quats": [],
        "source_ply": None,
        "is_sparse_init": False,
    }
    try:
        active_ply = ws.active_splat_path()
        cloud = load_gaussian_ply(active_ply)
        n = cloud.num_points
        splats_payload["total_points"] = n
        splats_payload["source_ply"] = ws.to_relative(active_ply)
        if n > 0:
            alphas = cloud.sigmoid_opacities()
            valid = np.nonzero(alphas >= 0.10)[0]
            if len(valid) < min(32, n):
                valid = np.arange(n, dtype=int)
            if len(valid) > max_splats:
                step_pick = np.linspace(0, len(valid) - 1, num=max_splats, dtype=int)
                sel = valid[step_pick]
            else:
                sel = valid
            sub = cloud.subset(sel)
            scales_3d = np.exp(np.clip(sub.scales, -10.0, 2.0))
            scales_world = scales_3d.mean(axis=1)
            splats_payload.update(
                {
                    "shown_points": sub.num_points,
                    "positions": np.round(sub.means, 4).tolist(),
                    "colors": np.round(sub.rgb_colors(), 3).tolist(),
                    "opacities": np.round(sub.sigmoid_opacities(), 3).tolist(),
                    "scales": np.round(scales_world, 4).tolist(),
                    "scales_3d": np.round(scales_3d, 4).tolist(),
                    "quats": np.round(sub.quats, 4).tolist(),
                }
            )
    except FileNotFoundError:
        if ws.init_points_path.exists():
            pts, cols = load_point_cloud_ply(ws.init_points_path)
            n = int(pts.shape[0])
            splats_payload["total_points"] = n
            splats_payload["source_ply"] = ws.to_relative(ws.init_points_path)
            splats_payload["is_sparse_init"] = True
            if n > 0:
                sel = np.linspace(0, n - 1, num=max_splats, dtype=int) if n > max_splats else np.arange(n, dtype=int)
                sub_pts = pts[sel]
                sub_cols = cols[sel]
                cnt = int(sub_pts.shape[0])
                splats_payload.update(
                    {
                        "shown_points": cnt,
                        "positions": np.round(sub_pts, 4).tolist(),
                        "colors": np.round(sub_cols, 3).tolist(),
                        "opacities": [0.85] * cnt,
                        "scales": [0.015] * cnt,
                        "scales_3d": [[0.015, 0.015, 0.015]] * cnt,
                        "quats": [[1.0, 0.0, 0.0, 0.0]] * cnt,
                    }
                )
    except (OSError, ValueError, KeyError) as exc:
        warnings.append(f"高斯 PLY 读取失败（可能正在写入或已损坏）: {exc}")

    mesh_payload: dict[str, Any] = {"vertices": [], "faces": [], "total_faces": 0}
    if ws.collision_mesh_path.exists():
        try:
            mesh = trimesh.load(str(ws.collision_mesh_path), force="mesh")
            mesh_payload = _compact_mesh_payload(mesh, max_faces=max_mesh_faces)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"collision_mesh.obj 读取失败: {exc}")

    convex_parts: list[dict[str, Any]] = []
    parts_dir = ws.export_dir / "mujoco" / "collision_parts"
    if parts_dir.is_dir():
        for part_file in sorted(parts_dir.glob("*.obj")):
            try:
                part_mesh = trimesh.load(str(part_file), force="mesh")
                part_data = _compact_mesh_payload(part_mesh, max_faces=800)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{part_file.name} 读取失败: {exc}")
                continue
            part_data["name"] = part_file.stem
            convex_parts.append(part_data)

    scene_meta = _safe_read_json(ws.export_dir / "mujoco" / "scene_meta.json", warnings)

    scene_xml = ""
    xml_path = ws.export_dir / "mujoco" / "scene.xml"
    if xml_path.exists():
        try:
            scene_xml = xml_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            warnings.append(f"scene.xml 读取失败: {exc}")

    frames: list[dict[str, Any]] = []
    f_meta = _safe_read_json(ws.frames_meta_path, warnings)
    if isinstance(f_meta, dict):
        for entry in f_meta.get("frames", [])[:24]:
            if isinstance(entry, dict) and entry.get("filename"):
                frames.append(
                    {
                        **entry,
                        "rel_path": f"01_frames/{entry['filename']}",
                    }
                )

    renders: list[dict[str, str]] = []
    renders_dir = ws.gaussians_dir / "renders"
    if renders_dir.is_dir():
        for img_path in sorted(renders_dir.glob("*.png"))[:12]:
            renders.append(
                {
                    "filename": img_path.name,
                    "rel_path": f"03_gaussians/renders/{img_path.name}",
                }
            )

    unity_ready = (ws.export_dir / "unity" / "unity_import.json").exists()
    pure_ply_rel = (
        "05_export/gaussian_splats/scene_3dgs.ply"
        if (ws.export_dir / "gaussian_splats" / "scene_3dgs.ply").is_file()
        else (splats_payload["source_ply"] if not splats_payload.get("is_sparse_init") else None)
    )
    raw_ply_rel = (
        "05_export/gaussian_splats/raw_camera_space_3dgs.ply"
        if (ws.export_dir / "gaussian_splats" / "raw_camera_space_3dgs.ply").is_file()
        else ("03_gaussians/raw_splats.ply" if ws.raw_splats_path.is_file() else None)
    )
    cameras_rel = (
        "05_export/gaussian_splats/cameras.json"
        if (ws.export_dir / "gaussian_splats" / "cameras.json").is_file()
        else ("02_pose/cameras.json" if ws.cameras_path.is_file() else None)
    )
    mujoco_ply_rel = (
        "05_export/mujoco/visual_splats.ply"
        if (ws.export_dir / "mujoco" / "visual_splats.ply").is_file()
        else None
    )
    mesh_rel = (
        "05_export/mujoco/collision_mesh.obj"
        if (ws.export_dir / "mujoco" / "collision_mesh.obj").is_file()
        else ("04_processed/collision_mesh.obj" if ws.collision_mesh_path.is_file() else None)
    )
    has_export_files = ws.export_dir.is_dir() and any(p.is_file() for p in ws.export_dir.rglob("*"))
    pts_sample = (
        np.asarray(splats_payload["positions"][:4096], dtype=np.float64)
        if splats_payload["positions"]
        else np.zeros((0, 3), dtype=np.float64)
    )
    camera_presets = _extract_aligned_camera_presets(
        ws, active_ply_rel=splats_payload["source_ply"], pts_sample=pts_sample
    )

    alignment_transform = _safe_read_json(ws.processed_dir / "alignment_transform.json", warnings)
    if not isinstance(alignment_transform, dict):
        alignment_transform = None

    mgr = _get_editor_manager(ws_root)
    editor_summary = mgr.public_summary()

    return {
        "workspace_dir": display_dir or str(ws.root),
        "manifest": manifest,
        "warnings": warnings,
        "splats": splats_payload,
        "cameras": camera_presets,
        "mesh": mesh_payload,
        "convex_parts": convex_parts,
        "scene_meta": scene_meta,
        "alignment_transform": alignment_transform,
        "scene_xml": scene_xml,
        "frames": frames,
        "renders": renders,
        "neural_render_available": bool(
            splats_payload["total_points"] > 0 and not splats_payload["is_sparse_init"]
        ),
        "unity_ready": unity_ready,
        "editor_objects_count": len(editor_summary["objects"]),
        "editor_has_unbaked": editor_summary["baked_dirty"],
        "exports_available": {
            "pure_3dgs_ply": pure_ply_rel,
            "raw_3dgs_ply": raw_ply_rel,
            "cameras_json": cameras_rel,
            "mujoco_splats_ply": mujoco_ply_rel,
            "collision_obj": mesh_rel,
            "mujoco_xml": bool(scene_xml),
            "unity_ready": unity_ready,
            "bundle_zip_ready": has_export_files,
        },
    }


def _get_editor_manager(ws_root: Path) -> SceneEditManager:
    key = str(ws_root.resolve())
    with _EDITOR_MANAGERS_LOCK:
        mgr = _EDITOR_MANAGERS.get(key)
        if mgr is None:
            mgr = SceneEditManager(SceneWorkspace(ws_root))
            _EDITOR_MANAGERS[key] = mgr
        return mgr


def _evict_all_render_models() -> None:
    with _RENDER_CACHE_LOCK:
        had = bool(_RENDER_MODEL_CACHE)
        _RENDER_MODEL_CACHE.clear()
    if had:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _get_cached_render_model(
    ply_path: Path,
    device: torch.device,
    cache_model: bool = True,
    store: JobStore | None = None,
) -> GaussianModel:
    resolved = ply_path.resolve()
    key = (str(resolved), int(resolved.stat().st_mtime_ns), device.type)
    with _RENDER_CACHE_LOCK:
        cached = _RENDER_MODEL_CACHE.get(key)
        if cached is not None:
            _RENDER_MODEL_CACHE.move_to_end(key)
            return cached

    cloud = load_gaussian_ply(resolved)
    has_sh = cloud.sh_rest is not None and cloud.sh_rest.shape[1] > 0
    model = GaussianModel(
        means=torch.tensor(cloud.means, dtype=torch.float32, device=device),
        scales=torch.tensor(cloud.scales, dtype=torch.float32, device=device),
        quats=torch.tensor(cloud.quats, dtype=torch.float32, device=device),
        opacities=torch.tensor(cloud.opacities, dtype=torch.float32, device=device),
        sh_dc=torch.tensor(cloud.sh_dc, dtype=torch.float32, device=device),
        sh_rest=torch.tensor(cloud.sh_rest, dtype=torch.float32, device=device) if has_sh else None,
        sh_degree=cloud.sh_degree if has_sh else 0,
        max_log_scale=10.0,
    )
    if cache_model and (store is None or not store.has_running_job()):
        with _RENDER_CACHE_LOCK:
            if store is None or not store.has_running_job():
                _RENDER_MODEL_CACHE[key] = model
                _RENDER_MODEL_CACHE.move_to_end(key)
                while len(_RENDER_MODEL_CACHE) > 2:
                    _RENDER_MODEL_CACHE.popitem(last=False)
    return model


def _load_workspace_render_model(
    ws_root: Path,
    device: torch.device,
    *,
    apply_edits: bool = True,
    selected_id: int | None = None,
    highlight: bool = False,
    dim_bg: bool = False,
    cache_model: bool = True,
    store: JobStore | None = None,
) -> tuple[GaussianModel, torch.Tensor | None]:
    """Load cached base GaussianModel for `ws_root` and optionally compose non-destructive Scene Graph edits."""
    ws = SceneWorkspace(ws_root)
    mgr = _get_editor_manager(ws_root)
    try:
        if apply_edits and mgr.has_edits():
            ply_path = mgr.resolve_base_ply_path()
        else:
            ply_path = ws.active_splat_path()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No trained 3DGS PLY found in workspace") from exc

    base_model = _get_cached_render_model(ply_path, device=device, cache_model=cache_model, store=store)
    if not apply_edits or not mgr.has_edits():
        return base_model, None
    state, obj_ids = mgr.load_state()
    return apply_scene_edits_to_gpu_model(
        base_model,
        obj_ids,
        state,
        selected_id=selected_id,
        highlight_selected=highlight,
        dim_background=dim_bg,
    )


def _render_novel_view_jpeg(
    ws_root: Path,
    yaw: float,
    pitch: float,
    distance: float,
    tx: float,
    ty: float,
    tz: float,
    width: int,
    height: int,
    theme: str,
    fov_scale: float = 1.05,
    hq: int = 1,
    cache_model: bool = True,
    store: JobStore | None = None,
    selected_id: int | None = None,
    highlight: int = 1,
    dim_bg: int = 0,
    apply_edits: int = 1,
) -> bytes:
    """Rasterize the trained Gaussian splat cloud from an arbitrary 360-degree interactive orbit pose."""
    if not ws_root.exists():
        raise HTTPException(status_code=404, detail=f"Workspace not found: {ws_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, gsplat_ready, _ = is_gsplat_cuda_ready()
    use_gsplat = gsplat_ready and device.type == "cuda"
    bg_vals = [0.955, 0.968, 0.986] if theme == "light" else [0.055, 0.078, 0.125]

    with _GPU_RENDER_LOCK:
        if store is not None and store.has_running_job():
            raise HTTPException(status_code=409, detail="训练执行期间暂停实时神经渲染以保护显存")
        model, _ = _load_workspace_render_model(
            ws_root,
            device,
            apply_edits=bool(apply_edits),
            selected_id=selected_id,
            highlight=bool(highlight),
            dim_bg=bool(dim_bg),
            cache_model=cache_model,
            store=store,
        )
        if model.num_points == 0:
            bg_u8 = np.array([round(c * 255.0) for c in bg_vals], dtype=np.uint8)
            rgb_u8 = np.broadcast_to(bg_u8, (height, width, 3)).copy()
        else:
            view = build_orbit_camera_view(
                yaw=yaw,
                pitch=pitch,
                distance=distance,
                target=(tx, ty, tz),
                width=width,
                height=height,
                fov_scale=fov_scale,
                device=device,
                name="interactive",
            )
            bg_t = torch.tensor(bg_vals, dtype=torch.float32, device=device)
            render_sh = model.sh_degree if use_gsplat else 0

            with torch.no_grad():
                r_out = _render_view(
                    model,
                    view,
                    w2c=view.world_to_cam,
                    use_gsplat=use_gsplat,
                    mip_3d_var=0.0,
                    bg_color=bg_t,
                    need_depth=False,
                    sh_degree=render_sh,
                )
                rgb_u8 = (r_out.rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)

    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    jpeg_q = 95 if hq >= 1 else 88
    ok, enc = cv2.imencode(".jpg", bgr_u8, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed encoding rendered frame")
    return bytes(enc.tobytes())


def _build_diagnostic_report(
    rec: JobRecord,
    req: JobCreateRequest,
    exc: BaseException,
    tb_text: str,
) -> str:
    """Assemble a formatted error report suitable for one-click copy-paste debugging."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only"
    if req.inline_config is not None:
        cfg_snapshot = yaml.safe_dump(req.inline_config, sort_keys=False, allow_unicode=True)[:3000]
    else:
        cfg_snapshot = (
            f"config_path: {req.config_path}\n"
            f"input_path: {req.input_path}\n"
            f"output_dir: {req.output_dir}"
        )
    logs_block = "\n".join(rec.logs[-15:]) if rec.logs else "(无日志)"
    return (
        "=== Real2Sim-GS 错误诊断报告 ===\n"
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"任务 ID: {rec.job_id}\n"
        f"失败阶段: {rec.current_stage or 'init'}\n"
        f"错误摘要: {type(exc).__name__}: {exc}\n"
        f"运行环境: Python {py_ver} | OS: {platform.platform()} | "
        f"PyTorch: {torch.__version__} | CUDA: {gpu_info}\n"
        f"工作区目录: {rec.workspace_dir or '(未确定)'}\n"
        f"请求参数: from_stage={req.from_stage}, force={req.force}, mock={req.mock}\n"
        f"\n配置快照:\n{cfg_snapshot.strip()}\n"
        f"\n最近阶段日志:\n{logs_block}\n"
        f"\nPython 详细堆栈 (Traceback):\n{tb_text.strip()}\n"
    )


def _format_local_stamp(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%m-%d %H:%M")
    except ValueError:
        return iso_str[:16]


def _dir_size_bytes(path: Path, use_cache: bool = False) -> int:
    """Safely sum file sizes under a directory tree without following symlinks.

    ``use_cache=True`` reuses a previous walk while the directory mtime is unchanged and the entry is
    younger than ``_DIR_SIZE_TTL_SEC`` (cheap invalidation for the history list polling).
    """
    try:
        st = path.stat()
    except OSError:
        return 0
    if path.is_file():
        return int(st.st_size)
    key = str(path.resolve())
    now = time.monotonic()
    if use_cache:
        with _DIR_SIZE_LOCK:
            hit = _DIR_SIZE_CACHE.get(key)
        if hit is not None and hit[0] == st.st_mtime_ns and now - hit[1] < _DIR_SIZE_TTL_SEC:
            return hit[2]
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for fname in files:
            try:
                total += int((Path(root) / fname).stat().st_size)
            except OSError:
                continue
    with _DIR_SIZE_LOCK:
        _DIR_SIZE_CACHE[key] = (st.st_mtime_ns, now, total)
        if len(_DIR_SIZE_CACHE) > 512:
            _DIR_SIZE_CACHE.clear()
    return total


def _intermediate_cache_paths(ws_dir: Path) -> list[Path]:
    """Enumerate heavy intermediate directories/files safe to purge after export."""
    return [
        ws_dir / "01_frames",
        ws_dir / "02_pose" / "database.db",
        ws_dir / "02_pose" / "database.db-wal",
        ws_dir / "02_pose" / "database.db-shm",
        ws_dir / "02_pose" / "sparse",
        ws_dir / "02_pose" / "undistort_cache",
        ws_dir / "03_gaussians" / "checkpoint.pt",
        ws_dir / "03_gaussians" / "checkpoint_meta.json",
    ]


def _evict_render_cache_for_workspace(resolved_ws: Path) -> None:
    """Release cached Gaussian GPU tensors and editor managers pointing inside a workspace."""
    with _RENDER_CACHE_LOCK:
        stale_keys = []
        for key in _RENDER_MODEL_CACHE:
            ply_path_str = key[0]
            try:
                if Path(ply_path_str).resolve().is_relative_to(resolved_ws):
                    stale_keys.append(key)
            except (OSError, ValueError):
                continue
        for key in stale_keys:
            _RENDER_MODEL_CACHE.pop(key, None)
    with _EDITOR_MANAGERS_LOCK:
        for mgr_key in list(_EDITOR_MANAGERS):
            try:
                p = Path(mgr_key).resolve()
                if p == resolved_ws or p.is_relative_to(resolved_ws):
                    _EDITOR_MANAGERS.pop(mgr_key, None)
            except (OSError, ValueError):
                continue
    if stale_keys:
        gc.collect()


def _resolve_workspace_path(
    workspace_dir: str,
    workspaces_root: Path,
    *,
    direct_child: bool = False,
) -> Path:
    """Resolve an HTTP-supplied workspace path and enforce it stays inside the configured workspaces root.

    - UNC / network paths (``\\\\host\\share`` or ``//host``) and anything resolving outside the root -> 403.
    - The root itself or the ``_uploads`` system cache -> 400.
    - ``direct_child=True`` additionally requires ``<root>/<scene_name>`` (used by delete/clean operations).
    """
    raw = str(workspace_dir or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="workspace_dir 不能为空")
    if raw.startswith(("\\\\", "//")):
        raise HTTPException(status_code=403, detail="安全限制：不允许访问 UNC/网络共享路径")
    try:
        target = Path(raw).resolve()
        rel = target.relative_to(workspaces_root)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=403,
            detail=f"安全限制：仅允许访问工作区根目录 {workspaces_root} 内的场景档案",
        ) from exc
    if not rel.parts or rel.parts[0] == "_uploads" or (direct_child and len(rel.parts) != 1):
        raise HTTPException(
            status_code=400,
            detail="仅允许操作具体场景档案目录（不能为根目录或 _uploads 系统缓存目录）",
        )
    return target


def _display_path(path: Path) -> str:
    """Prefer cwd-relative strings (e.g. ``workspaces\\scene``) so the Studio UI can match form paths."""
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _resolve_job_input_path(
    raw: str,
    uploads_dir: Path,
    allowed_inputs: Iterable[Path],
    allowed_roots: Iterable[Path] = (),
) -> Path:
    """HTTP jobs may only read uploads cache files/folders or explicitly allowlisted inputs/roots."""
    if raw.startswith(("\\\\", "//")):
        raise HTTPException(status_code=403, detail="安全限制：输入路径不允许为 UNC/网络共享路径")
    try:
        resolved = Path(raw).resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"非法输入路径: {raw}") from exc
    if resolved.is_relative_to(uploads_dir) and resolved != uploads_dir:
        return resolved
    if resolved in set(allowed_inputs):
        return resolved
    for root in allowed_roots:
        if resolved == root or resolved.is_relative_to(root):
            return resolved
    raise HTTPException(
        status_code=403,
        detail="安全限制：HTTP 任务的输入只能是通过 Studio 上传的文件（_uploads 缓存）或内置示例视频 sample_video.mp4",
    )


def _resolve_job_config_path(config_path: str) -> Path:
    """HTTP jobs may only reference YAML presets inside configs/ (matched by file name)."""
    name = Path(str(config_path or "")).name
    if not name.endswith((".yaml", ".yml")):
        raise HTTPException(status_code=400, detail=f"非法配置文件名: {config_path}")
    cfg_path = _resolve_configs_dir() / name
    if not cfg_path.is_file():
        raise HTTPException(status_code=404, detail=f"Config '{name}' not found in configs/")
    return cfg_path


def _rmtree_strict(target: Path) -> None:
    """Delete a directory tree, retrying read-only files; raises HTTP 500 listing the files that resisted."""
    failures: list[str] = []

    def _onerror(func: Callable[..., Any], path: str, exc_info: Any) -> None:
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError as exc:
            failures.append(f"{path}: {exc}")

    shutil.rmtree(target, onerror=_onerror)
    if failures or target.exists():
        head = "; ".join(failures[:3]) or str(target)
        raise HTTPException(
            status_code=500,
            detail=f"部分文件删除失败（可能被其他程序占用，请关闭相关窗口后重试）: {head}",
        )


def _corrupted_workspace_entry(child: Path, reason: str) -> dict[str, Any]:
    """Minimal history entry for workspaces whose manifest/metadata cannot be parsed."""
    size_bytes = _dir_size_bytes(child, use_cache=True)
    try:
        updated_at = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        updated_at = ""
    return {
        "name": child.name,
        "workspace_dir": _display_path(child),
        "input_path": "",
        "updated_at": updated_at,
        "updated_label": _format_local_stamp(updated_at),
        "completed_stages": [],
        "is_mock": False,
        "total_duration_sec": 0.0,
        "summary_metrics": {},
        "thumbnail_rel": None,
        "thumbnail_url": None,
        "has_saved_config": (child / "ui_config_snapshot.yaml").is_file(),
        "has_mujoco": False,
        "has_unity": False,
        "has_3dgs": False,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "reclaimable_bytes": 0,
        "reclaimable_mb": 0.0,
        "resumable_checkpoint": None,
        "is_incomplete": True,
        "is_corrupted": True,
        "corruption_reason": reason,
    }


def _summarize_workspace_entry(child: Path) -> dict[str, Any] | None:
    manifest_path = child / "manifest.json"
    ckpt_meta_path = child / "03_gaussians" / "checkpoint_meta.json"
    has_snap = (child / "ui_config_snapshot.yaml").is_file()
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return _corrupted_workspace_entry(child, f"manifest.json 解析失败: {exc}")
        if not isinstance(manifest, dict):
            return _corrupted_workspace_entry(child, "manifest.json 格式非法（非 JSON 对象）")
    elif ckpt_meta_path.is_file() or has_snap:
        manifest = {"stages": {}, "updated_at": ""}
    else:
        return None

    resumable_ckpt: dict[str, Any] | None = None
    if ckpt_meta_path.is_file():
        try:
            resumable_ckpt = json.loads(ckpt_meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            resumable_ckpt = None

    stages = manifest.get("stages", {})
    if not isinstance(stages, dict):
        return _corrupted_workspace_entry(child, "manifest.json 中 stages 字段格式非法")
    completed_stages = [s for s in STAGE_ORDER if s in stages]

    merged_metrics: dict[str, Any] = {}
    total_dur = 0.0
    is_mock = False
    for s in STAGE_ORDER:
        st = stages.get(s)
        if not isinstance(st, dict):
            continue
        if "mock" in str(st.get("impl_name", "")).lower():
            is_mock = True
        total_dur += float(st.get("duration_sec", 0.0) or 0.0)
        if isinstance(st.get("metrics"), dict):
            merged_metrics.update(st["metrics"])

    summary_keys = (
        "PSNR",
        "SSIM",
        "num_gaussians",
        "retained_gaussians",
        "exported_3dgs_points",
        "registered_views",
        "mujoco_convex_parts",
    )
    summary_metrics = {k: merged_metrics[k] for k in summary_keys if k in merged_metrics}

    thumb_rel: str | None = None
    render_thumb = child / "03_gaussians" / "renders" / "render_0000.png"
    if render_thumb.is_file():
        thumb_rel = "03_gaussians/renders/render_0000.png"
    else:
        frames_dir = child / "01_frames"
        if frames_dir.is_dir():
            for p in sorted(frames_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    thumb_rel = f"01_frames/{p.name}"
                    break

    ws_str = _display_path(child)
    thumb_url = (
        f"/api/workspace-file?workspace_dir={quote(ws_str)}&rel_path={quote(thumb_rel)}"
        if thumb_rel
        else None
    )
    updated_at = str(manifest.get("updated_at", "")) or str((resumable_ckpt or {}).get("updated_at", ""))
    if not updated_at:
        try:
            updated_at = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            updated_at = ""

    size_bytes = _dir_size_bytes(child, use_cache=True)
    reclaimable_bytes = sum(_dir_size_bytes(p, use_cache=True) for p in _intermediate_cache_paths(child))
    return {
        "name": child.name,
        "workspace_dir": ws_str,
        "input_path": str(manifest.get("input_path", "")),
        "updated_at": updated_at,
        "updated_label": _format_local_stamp(updated_at),
        "completed_stages": completed_stages,
        "is_mock": is_mock,
        "total_duration_sec": round(total_dur, 1),
        "summary_metrics": summary_metrics,
        "thumbnail_rel": thumb_rel,
        "thumbnail_url": thumb_url,
        "has_saved_config": has_snap,
        "has_mujoco": (child / "05_export" / "mujoco" / "scene.xml").is_file(),
        "has_unity": (child / "05_export" / "unity" / "unity_import.json").is_file(),
        "has_3dgs": (child / "05_export" / "gaussian_splats" / "scene_3dgs.ply").is_file()
        or (child / "03_gaussians" / "raw_splats.ply").is_file(),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "reclaimable_bytes": reclaimable_bytes,
        "reclaimable_mb": round(reclaimable_bytes / (1024 * 1024), 2),
        "resumable_checkpoint": resumable_ckpt,
        "is_incomplete": bool(resumable_ckpt is not None or len(completed_stages) < len(STAGE_ORDER)),
        "is_corrupted": False,
    }


def _execute_job(
    store: JobStore,
    job_id: str,
    req: JobCreateRequest,
    cfg: PipelineConfig,
    ws_dir: Path,
) -> None:
    cancel_event = store.get_cancel_event(job_id)
    with store._lock:
        rec = store.jobs[job_id]
        if rec.status == "cancelled" or cancel_event.is_set():
            if rec.status != "cancelled":
                rec.status = "cancelled"
                rec.status_detail = "任务已在排队阶段取消"
                rec.finished_at = datetime.now(timezone.utc).isoformat()
            return
        rec.status = "running"
        rec.status_detail = "正在初始化工作区..."

    def _on_progress(stage_kind: str, state: str, info: dict[str, Any]) -> None:
        store.update_stage(job_id, stage_kind, state, info)

    try:
        if cancel_event.is_set():
            raise PipelineCancelledError("训练/构建任务已由用户手动停止")

        ws_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            ws_dir / "ui_config_snapshot.yaml",
            yaml.safe_dump(cfg.model_dump(), sort_keys=False, allow_unicode=True),
        )

        if not req.mock:
            get_shared_sam3_backend().unload()
            with _RENDER_CACHE_LOCK:
                _RENDER_MODEL_CACHE.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _evict_render_cache_for_workspace(ws_dir)
        res = run_pipeline(
            config=cfg,
            from_stage=req.from_stage,
            force=req.force,
            mock=req.mock,
            progress_callback=_on_progress,
            cancel_check=cancel_event.is_set,
        )
        with store._lock:
            rec = store.jobs[job_id]
            rec.status = "completed"
            rec.overall_progress = 1.0
            rec.stage_progress = 1.0
            rec.status_detail = "全部阶段执行完毕"
            rec.workspace_dir = _display_path(res.workspace_dir)
            rec.executed_stages = res.executed_stages
            rec.skipped_stages = res.skipped_stages
            rec.metrics = res.metrics
            rec.mujoco_xml_path = str(res.mujoco_xml_path) if res.mujoco_xml_path else None
            rec.unity_dir = str(res.unity_dir) if res.unity_dir else None
            rec.gaussian_ply_path = str(res.gaussian_ply_path) if res.gaussian_ply_path else None
            rec.splat_ply_path = str(res.splat_ply_path) if res.splat_ply_path else None
            rec.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [done] 全部阶段执行完毕")
    except PipelineCancelledError:
        with store._lock:
            rec = store.jobs[job_id]
            rec.status = "cancelled"
            stop_msg = "已手动停止 · 已完成阶段已缓存，下次启动将自动跳过并续跑"
            if rec.workspace_dir:
                ckpt_meta_path = ws_dir / "03_gaussians" / "checkpoint_meta.json"
                if ckpt_meta_path.is_file():
                    try:
                        meta = json.loads(ckpt_meta_path.read_text(encoding="utf-8"))
                        stop_msg = (
                            f"已停止并保存训练断点（第 {int(meta['step']):,}/{int(meta['total_steps']):,} 步）"
                            "· 下次启动将自动从此处继续"
                        )
                    except Exception:  # noqa: BLE001
                        pass
            rec.status_detail = stop_msg
            rec.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [stopped] {stop_msg}")
            if rec.current_stage and rec.current_stage in rec.stage_states:
                rec.stage_states[rec.current_stage]["status"] = "cancelled"
    except Exception as exc:  # noqa: BLE001
        tb_text = traceback.format_exc()
        with store._lock:
            rec = store.jobs[job_id]
            rec.status = "failed"
            rec.error = str(exc)
            rec.error_traceback = tb_text
            rec.status_detail = f"执行异常: {exc}"
            rec.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [error] {exc}")
            rec.diagnostic_report = _build_diagnostic_report(rec, req, exc, tb_text)
            if rec.current_stage and rec.current_stage in rec.stage_states:
                rec.stage_states[rec.current_stage]["status"] = "failed"
    finally:
        _evict_render_cache_for_workspace(ws_dir)
        with _DIR_SIZE_LOCK:
            _DIR_SIZE_CACHE.clear()
        with store._lock:
            store.jobs[job_id].finished_at = datetime.now(timezone.utc).isoformat()
            store._active_inputs.pop(job_id, None)


def create_app(
    *,
    workspaces_root: str | Path | None = None,
    allowed_input_roots: Iterable[str | Path] | None = None,
    studio_token: str | None = None,
    bind_host: str = "127.0.0.1",
    max_upload_bytes: int | None = None,
    upload_grace_sec: float = _UPLOAD_GRACE_SEC,
) -> FastAPI:
    """Create the FastAPI application serving both REST APIs and Real2Sim-GS Studio UI."""
    ws_root = Path(workspaces_root).resolve() if workspaces_root is not None else (Path.cwd() / "workspaces").resolve()
    uploads_dir = ws_root / "_uploads"
    token = studio_token or os.environ.get(STUDIO_TOKEN_ENV) or secrets.token_urlsafe(24)

    if max_upload_bytes is not None:
        eff_max_upload = int(max_upload_bytes)
    elif os.environ.get(MAX_UPLOAD_ENV):
        eff_max_upload = int(float(os.environ[MAX_UPLOAD_ENV]) * (1024**3))
    else:
        eff_max_upload = _DEFAULT_MAX_UPLOAD_BYTES

    extra_input_roots: list[Path] = []
    if allowed_input_roots:
        for r in allowed_input_roots:
            extra_input_roots.append(Path(r).resolve())
    env_roots = os.environ.get("REAL2SIM_ALLOWED_INPUT_DIRS", "")
    if env_roots:
        for part in env_roots.split(os.pathsep):
            if part.strip():
                extra_input_roots.append(Path(part.strip()).resolve())

    is_remote_bind = bind_host in ("0.0.0.0", "::")
    app = FastAPI(title="Real2Sim-GS Studio & API", version="0.2.0")
    store = JobStore()
    app.state.job_store = store
    app.state.workspaces_root = ws_root
    app.state.studio_token = token

    if is_remote_bind:
        allowed_hosts = ["*"]
    else:
        allowed_hosts = list(dict.fromkeys([*_LOOPBACK_HOSTS, "testserver", bind_host]))
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.middleware("http")
    async def _studio_token_middleware(request: Request, call_next: Callable[..., Any]) -> Response:
        if request.method.upper() in ("GET", "HEAD") and request.url.path == "/health":
            return await call_next(request)
        supplied = request.headers.get(STUDIO_TOKEN_HEADER) or ""
        if request.method.upper() in _MUTATING_METHODS:
            if not secrets.compare_digest(supplied, token):
                return JSONResponse({"detail": "Missing or invalid X-Studio-Token header"}, status_code=403)
        elif is_remote_bind:
            cookie_tok = request.cookies.get("r2s_studio_token") or ""
            query_tok = request.query_params.get("token") or ""
            authed = (
                secrets.compare_digest(supplied, token)
                or secrets.compare_digest(cookie_tok, token)
                or (request.url.path == "/" and secrets.compare_digest(query_tok, token))
            )
            if not authed:
                return JSONResponse({"detail": "Authentication required (?token= or X-Studio-Token)"}, status_code=403)
        return await call_next(request)

    if UI_STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(UI_STATIC_DIR)), name="studio_static")

    @app.get("/", include_in_schema=False)
    def index(request: Request) -> HTMLResponse:
        index_html = UI_STATIC_DIR / "index.html"
        if not index_html.exists():
            raise HTTPException(status_code=404, detail="Studio UI assets not found.")
        html = index_html.read_text(encoding="utf-8").replace(_TOKEN_PLACEHOLDER, token)
        resp = HTMLResponse(content=html, headers={"Cache-Control": "no-store"})
        query_tok = request.query_params.get("token") or ""
        if query_tok and secrets.compare_digest(query_tok, token):
            resp.set_cookie("r2s_studio_token", token, httponly=True, samesite="strict")
        return resp

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "stages": list_available_stages()}

    @app.get("/api/doctor")
    def api_doctor() -> dict[str, Any]:
        return {"items": [asdict(item) for item in check_environment()]}

    @app.get("/api/configs")
    def api_list_configs() -> dict[str, Any]:
        cfg_dir = _resolve_configs_dir()
        files = sorted(p.name for p in cfg_dir.glob("*.yaml")) if cfg_dir.is_dir() else []
        default_name = "default_mujoco.yaml" if "default_mujoco.yaml" in files else (files[0] if files else "")
        default_cfg = (
            yaml.safe_load((cfg_dir / default_name).read_text(encoding="utf-8"))
            if default_name
            else {}
        )
        return {
            "files": files,
            "default_name": default_name,
            "default_config": default_cfg,
        }

    @app.get("/api/configs/{config_name}")
    def api_get_config(config_name: str) -> dict[str, Any]:
        if Path(config_name).name != config_name or not config_name.endswith(".yaml"):
            raise HTTPException(status_code=400, detail="Invalid config filename")
        cfg_path = _resolve_configs_dir() / config_name
        if not cfg_path.is_file():
            raise HTTPException(status_code=404, detail=f"Config '{config_name}' not found")
        return {
            "name": config_name,
            "config": yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {},
        }

    @app.post("/api/upload")
    def api_upload_video(file: UploadFile = File(...)) -> dict[str, str]:
        safe_name = Path(file.filename or "input_video.mp4").name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in UPLOAD_ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的上传文件类型 '{suffix}'（支持: {sorted(UPLOAD_ALLOWED_SUFFIXES)}）",
            )
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = uploads_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        written = 0
        try:
            with open(dest, "wb") as out_f:
                while True:
                    chunk = file.file.read(_UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > eff_max_upload:
                        raise HTTPException(
                            status_code=413,
                            detail=f"上传文件超出大小限制 ({eff_max_upload / (1024**3):.2f} GB)",
                        )
                    out_f.write(chunk)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        with _DIR_SIZE_LOCK:
            _DIR_SIZE_CACHE.clear()
        return {"filename": safe_name, "input_path": _display_path(dest)}

    @app.get("/api/workspaces")
    def api_list_workspaces() -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        if ws_root.is_dir():
            for child in ws_root.iterdir():
                if not child.is_dir() or child.name == "_uploads":
                    continue
                entry = _summarize_workspace_entry(child)
                if entry is not None:
                    items.append(entry)
        items.sort(key=lambda item: item["updated_at"], reverse=True)
        uploads_bytes = _dir_size_bytes(uploads_dir, use_cache=True)
        total_bytes = sum(int(i.get("size_bytes", 0)) for i in items) + uploads_bytes
        return {
            "workspaces": items,
            "uploads_size_mb": round(uploads_bytes / (1024 * 1024), 2),
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        }

    @app.post("/api/delete-workspace")
    def api_delete_workspace(req: WorkspaceDeleteRequest) -> dict[str, Any]:
        mode = str(req.mode or "all").lower()
        if mode not in ("all", "intermediate"):
            raise HTTPException(status_code=400, detail=f"Invalid delete mode '{req.mode}'. Expected 'all' or 'intermediate'.")

        target = _resolve_workspace_path(req.workspace_dir, ws_root, direct_child=True)
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"Workspace not found: {req.workspace_dir}")

        store.begin_delete(target)
        try:
            _evict_render_cache_for_workspace(target)
            if mode == "all":
                freed = _dir_size_bytes(target)
                _rmtree_strict(target)
                result_payload: dict[str, Any] = {
                    "deleted": True,
                    "mode": "all",
                    "workspace_dir": _display_path(target),
                    "freed_mb": round(freed / (1024 * 1024), 2),
                }
            else:
                before = _dir_size_bytes(target)
                for p in _intermediate_cache_paths(target):
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    elif p.is_file():
                        p.unlink(missing_ok=True)
                after = _dir_size_bytes(target)
                result_payload = {
                    "deleted": True,
                    "mode": "intermediate",
                    "workspace_dir": _display_path(target),
                    "freed_mb": round(max(0, before - after) / (1024 * 1024), 2),
                    "remaining_mb": round(after / (1024 * 1024), 2),
                }
            with _DIR_SIZE_LOCK:
                _DIR_SIZE_CACHE.clear()
            return result_payload
        finally:
            store.end_delete(target)

    @app.post("/api/cleanup-uploads")
    def api_cleanup_uploads(
        grace_sec: float | None = Query(None, ge=0.0),
        keep_input_path: str | None = Query(None),
    ) -> dict[str, Any]:
        if not uploads_dir.is_dir():
            return {"cleaned": True, "removed_files": 0, "freed_mb": 0.0}
        in_use = store.active_input_paths()
        if keep_input_path:
            try:
                kept = Path(keep_input_path).resolve()
                if kept.is_relative_to(uploads_dir) and kept != uploads_dir:
                    in_use.add(kept)
            except OSError:
                pass
        if ws_root.is_dir():
            for child in ws_root.iterdir():
                if not child.is_dir() or child.name == "_uploads":
                    continue
                summary = _summarize_workspace_entry(child)
                if summary is None or not summary.get("is_incomplete"):
                    continue
                cand_inputs: list[str] = [str(summary.get("input_path") or "")]
                snap = child / "ui_config_snapshot.yaml"
                if snap.is_file():
                    try:
                        snap_cfg = yaml.safe_load(snap.read_text(encoding="utf-8")) or {}
                        cand_inputs.append(str((snap_cfg.get("io") or {}).get("input_path") or ""))
                    except (OSError, yaml.YAMLError):
                        pass
                for raw_in in cand_inputs:
                    if not raw_in.strip():
                        continue
                    try:
                        p_in = Path(raw_in).resolve()
                        if p_in.is_relative_to(uploads_dir) and p_in != uploads_dir:
                            in_use.add(p_in)
                    except OSError:
                        pass
        eff_grace = float(upload_grace_sec if grace_sec is None else grace_sec)
        now = time.time()
        freed = 0
        removed = 0
        for p in list(uploads_dir.iterdir()):
            try:
                resolved_p = p.resolve()
                if resolved_p in in_use:
                    continue
                if eff_grace > 0.0 and (now - p.stat().st_mtime) < eff_grace:
                    continue
                if p.is_file():
                    sz = _dir_size_bytes(p)
                    p.unlink(missing_ok=True)
                    freed += sz
                    removed += 1
                elif p.is_dir():
                    sz = _dir_size_bytes(p)
                    shutil.rmtree(p, ignore_errors=True)
                    freed += sz
                    removed += 1
            except OSError:
                continue
        with _DIR_SIZE_LOCK:
            _DIR_SIZE_CACHE.clear()
        return {
            "cleaned": True,
            "removed_files": removed,
            "freed_mb": round(freed / (1024 * 1024), 2),
        }

    @app.get("/api/workspace-config")
    def api_workspace_config(workspace_dir: str = Query(...)) -> dict[str, Any]:
        root = _resolve_workspace_path(workspace_dir, ws_root)
        if not root.is_dir():
            raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_dir}")
        snap_path = root / "ui_config_snapshot.yaml"
        if not snap_path.is_file():
            return {"workspace_dir": _display_path(root), "has_config": False, "config": None}
        try:
            cfg_data = yaml.safe_load(snap_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise HTTPException(status_code=500, detail=f"参数快照读取失败: {exc}") from exc
        return {"workspace_dir": _display_path(root), "has_config": True, "config": cfg_data}

    @app.post("/api/open-workspace")
    def api_open_workspace(req: WorkspaceActionRequest) -> dict[str, Any]:
        root = _resolve_workspace_path(req.workspace_dir, ws_root)
        if not root.is_dir():
            raise HTTPException(status_code=404, detail=f"Workspace not found: {req.workspace_dir}")
        resolved = str(root)
        if sys.platform == "win32" and hasattr(os, "startfile"):
            os.startfile(resolved)  # noqa: S606
            return {"opened": True, "path": resolved}
        return {"opened": False, "path": resolved}

    @app.get("/api/scene-preview")
    def api_scene_preview(
        workspace_dir: str = Query(..., description="Workspace root path"),
        max_splats: int = Query(18_000, ge=500, le=50_000),
    ) -> dict[str, Any]:
        root = _resolve_workspace_path(workspace_dir, ws_root)
        return _build_scene_preview(root, max_splats=max_splats, display_dir=_display_path(root))

    @app.get("/api/render-view")
    def api_render_view(
        workspace_dir: str = Query(...),
        yaw: float = Query(0.72),
        pitch: float = Query(0.48),
        distance: float = Query(3.2, ge=0.01, le=500.0),
        tx: float = Query(0.0),
        ty: float = Query(0.0),
        tz: float = Query(0.25),
        width: int = Query(800, ge=160, le=1920),
        height: int = Query(520, ge=120, le=1200),
        theme: str = Query("light"),
        fov_scale: float = Query(1.05, ge=0.5, le=3.0),
        hq: int = Query(1, ge=0, le=1),
        selected_id: int | None = Query(None),
        highlight: int = Query(1, ge=0, le=1),
        dim_bg: int = Query(0, ge=0, le=1),
        apply_edits: int = Query(1, ge=0, le=1),
    ) -> Response:
        root = _resolve_workspace_path(workspace_dir, ws_root)
        if store.has_running_job():
            raise HTTPException(status_code=409, detail="训练执行期间暂停实时神经渲染以保护显存")
        jpg_bytes = _render_novel_view_jpeg(
            root,
            yaw=yaw,
            pitch=pitch,
            distance=distance,
            tx=tx,
            ty=ty,
            tz=tz,
            width=width,
            height=height,
            theme=theme,
            fov_scale=fov_scale,
            hq=hq,
            cache_model=True,
            store=store,
            selected_id=selected_id,
            highlight=highlight,
            dim_bg=dim_bg,
            apply_edits=apply_edits,
        )
        return Response(
            content=jpg_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/editor/status")
    def api_editor_status(model_path: str | None = Query(None)) -> dict[str, Any]:
        if model_path is not None and str(model_path).strip().startswith(("\\\\", "//")):
            raise HTTPException(status_code=403, detail="安全限制：不允许访问 UNC/网络共享路径")
        return get_sam3_status(model_path=model_path, backend=get_shared_sam3_backend())

    @app.post("/api/editor/sam3-config")
    def api_editor_sam3_config(req: Sam3ConfigRequest) -> dict[str, Any]:
        sam3 = get_shared_sam3_backend()
        if req.model_path is not None:
            raw_path = str(req.model_path).strip()
            if raw_path.startswith(("\\\\", "//")):
                raise HTTPException(status_code=403, detail="安全限制：不允许访问 UNC/网络共享路径")
            sam3.configure(raw_path)
        if req.unload:
            sam3.unload()
        return get_sam3_status(backend=sam3)

    @app.get("/api/editor/objects")
    def api_editor_objects(workspace_dir: str = Query(...)) -> dict[str, Any]:
        root = _resolve_workspace_path(workspace_dir, ws_root)
        if not root.is_dir():
            raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_dir}")
        mgr = _get_editor_manager(root)
        return {"workspace_dir": _display_path(root), **mgr.public_summary()}

    @app.post("/api/editor/preview-2d")
    def api_editor_preview_2d(req: EditorPreview2DRequest) -> dict[str, Any]:
        root = _resolve_workspace_path(req.workspace_dir, ws_root)
        if store.has_running_job():
            raise HTTPException(status_code=409, detail="训练执行期间暂停交互分割以保护显存")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, gsplat_ready, _ = is_gsplat_cuda_ready()
        use_gsplat = bool(gsplat_ready and device.type == "cuda")
        bg_vals = (0.955, 0.968, 0.986) if req.theme == "light" else (0.055, 0.078, 0.125)

        scale = min(1.0, 960.0 / max(1, req.width), 720.0 / max(1, req.height))
        prev_w = max(160, round(req.width * scale))
        prev_h = max(120, round(req.height * scale))
        sx = prev_w / float(max(1, req.width))
        sy = prev_h / float(max(1, req.height))
        scaled_pts = [[p[0] * sx, p[1] * sy] for p in req.points] if req.points else None
        scaled_box = (
            [req.box[0] * sx, req.box[1] * sy, req.box[2] * sx, req.box[3] * sy]
            if req.box and len(req.box) == 4
            else None
        )

        sam3 = get_shared_sam3_backend()
        with _GPU_RENDER_LOCK:
            if store.has_running_job():
                raise HTTPException(status_code=409, detail="训练执行期间暂停交互分割以保护显存")
            model, _ = _load_workspace_render_model(root, device, apply_edits=True, cache_model=True, store=store)
            view = build_orbit_camera_view(
                yaw=req.yaw,
                pitch=req.pitch,
                distance=req.distance,
                target=(req.tx, req.ty, req.tz),
                width=prev_w,
                height=prev_h,
                fov_scale=req.fov_scale,
                device=device,
                name="preview_2d",
            )
            rgb_u8 = render_camera_rgb(model, view, use_gsplat=use_gsplat, bg_vals=bg_vals)
            try:
                pred = sam3.predict_image_prompt(
                    rgb_u8,
                    points=scaled_pts,
                    labels=req.labels,
                    box=scaled_box,
                    text=req.text,
                )
            except torch.cuda.OutOfMemoryError:
                _evict_all_render_models()
                pred = sam3.predict_image_prompt(
                    rgb_u8,
                    points=scaled_pts,
                    labels=req.labels,
                    box=scaled_box,
                    text=req.text,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        overlay = np.zeros((prev_h, prev_w, 4), dtype=np.uint8)
        if np.any(pred.mask):
            overlay[pred.mask] = [34, 211, 238, 92]
            contours, _ = cv2.findContours(pred.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (56, 189, 248, 240), thickness=2)
        ok, png_buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
        if not ok:
            raise HTTPException(status_code=500, detail="Failed encoding 2D mask overlay")

        return {
            "mask_png_b64": base64.b64encode(png_buf.tobytes()).decode("ascii"),
            "score": round(float(pred.score), 4),
            "area_ratio": round(float(np.mean(pred.mask)), 4) if pred.mask.size else 0.0,
            "boxes": pred.boxes,
            "backend": pred.backend,
            "elapsed_ms": pred.elapsed_ms,
            "width": prev_w,
            "height": prev_h,
        }

    @app.post("/api/editor/segment-3d")
    def api_editor_segment_3d(req: EditorSegment3DRequest) -> dict[str, Any]:
        root = _resolve_workspace_path(req.workspace_dir, ws_root)
        if store.has_running_job():
            raise HTTPException(status_code=409, detail="训练执行期间暂停 3D 分割以保护显存")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, gsplat_ready, _ = is_gsplat_cuda_ready()
        use_gsplat = bool(gsplat_ready and device.type == "cuda")
        bg_vals = (0.955, 0.968, 0.986) if req.theme == "light" else (0.055, 0.078, 0.125)

        anchor_w = min(512, max(192, int(req.width)))
        anchor_h = max(144, min(640, round(anchor_w * req.height / max(1, req.width))))
        sx = anchor_w / float(max(1, req.width))
        sy = anchor_h / float(max(1, req.height))
        scaled_pts = [[p[0] * sx, p[1] * sy] for p in req.points] if req.points else None
        scaled_box = (
            [req.box[0] * sx, req.box[1] * sy, req.box[2] * sx, req.box[3] * sy]
            if req.box and len(req.box) == 4
            else None
        )

        mgr = _get_editor_manager(root)
        sam3 = get_shared_sam3_backend()

        with _GPU_RENDER_LOCK:
            if store.has_running_job():
                raise HTTPException(status_code=409, detail="训练执行期间暂停 3D 分割以保护显存")
            composed_model, _ = _load_workspace_render_model(
                root, device, apply_edits=True, cache_model=True, store=store
            )
            anchor_params = {
                "yaw": req.yaw,
                "pitch": req.pitch,
                "distance": req.distance,
                "target": (req.tx, req.ty, req.tz),
                "fov_scale": req.fov_scale,
            }
            anchor_view = build_orbit_camera_view(
                yaw=req.yaw,
                pitch=req.pitch,
                distance=req.distance,
                target=(req.tx, req.ty, req.tz),
                width=anchor_w,
                height=anchor_h,
                fov_scale=req.fov_scale,
                device=device,
                name="anchor_view",
            )
            anchor_rgb = render_camera_rgb(composed_model, anchor_view, use_gsplat=use_gsplat, bg_vals=bg_vals)
            try:
                anchor_pred = sam3.predict_image_prompt(
                    anchor_rgb,
                    points=scaled_pts,
                    labels=req.labels,
                    box=scaled_box,
                    text=req.text,
                )
            except torch.cuda.OutOfMemoryError:
                _evict_all_render_models()
                anchor_pred = sam3.predict_image_prompt(
                    anchor_rgb,
                    points=scaled_pts,
                    labels=req.labels,
                    box=scaled_box,
                    text=req.text,
                )
            except (RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            if not np.any(anchor_pred.mask):
                raise HTTPException(status_code=400, detail="SAM 3 未在当前视角检测到匹配目标，请换个视角或点选物体中心")

            center_3d, radius_3d = estimate_target_anchor_3d(
                composed_model, anchor_view, anchor_pred.mask, use_gsplat=use_gsplat
            )
            views = plan_active_views(
                composed_model,
                anchor_params,
                center_3d,
                radius_3d,
                num_views=req.num_views,
                width=anchor_w,
                height=anchor_h,
            )
            frames_rgb = [render_camera_rgb(composed_model, v, use_gsplat=use_gsplat, bg_vals=bg_vals) for v in views]
            proj_seeds = [project_world_point(v, center_3d) for v in views]
            try:
                masks_2d = sam3.track_multiview(
                    frames_rgb,
                    anchor_idx=0,
                    points=scaled_pts,
                    labels=req.labels,
                    box=scaled_box,
                    text=req.text,
                    anchor_mask=anchor_pred.mask,
                    projected_seeds=proj_seeds,
                )
            except torch.cuda.OutOfMemoryError:
                _evict_all_render_models()
                masks_2d = sam3.track_multiview(
                    frames_rgb,
                    anchor_idx=0,
                    points=scaled_pts,
                    labels=req.labels,
                    box=scaled_box,
                    text=req.text,
                    anchor_mask=anchor_pred.mask,
                    projected_seeds=proj_seeds,
                )
            except (RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            base_cloud = mgr.load_base_cloud()
            state, existing_ids = mgr.load_state()
            lift_cloud = base_cloud
            objs = list(state.get("objects", []))
            if existing_ids is not None and objs and len(existing_ids) == base_cloud.num_points:
                active_by_mid: dict[int, list[dict[str, Any]]] = {}
                for o in objs:
                    if not o.get("deleted") and o.get("visible", True):
                        active_by_mid.setdefault(int(o["source_mask_id"]), []).append(o)

                def _has_non_identity_tf(o: dict[str, Any]) -> bool:
                    return (
                        any(abs(float(v)) > 1e-5 for v in o.get("translation", [0.0, 0.0, 0.0]))
                        or any(abs(float(v)) > 1e-4 for v in o.get("rotation_euler_deg", [0.0, 0.0, 0.0]))
                        or abs(float(o.get("scale", 1.0)) - 1.0) > 1e-5
                    )

                claimed_mids = {int(o["source_mask_id"]) for o in objs}
                need_mute = any(mid not in active_by_mid for mid in claimed_mids)
                need_tf = any(
                    _has_non_identity_tf(o) for o in objs if not o.get("deleted") and o.get("visible", True)
                )
                if need_mute or need_tf:
                    lift_cloud = base_cloud.clone()
                    for mid in claimed_mids:
                        sel = existing_ids == mid
                        if not np.any(sel):
                            continue
                        active_list = active_by_mid.get(mid)
                        if not active_list:
                            lift_cloud.opacities[sel] = -20.0
                        else:
                            primary = next((x for x in active_list if not x.get("is_clone")), active_list[0])
                            if _has_non_identity_tf(primary):
                                sub_tf = SceneEditManager._transform_cloud_cpu(base_cloud.subset(sel), primary)
                                lift_cloud.means[sel] = sub_tf.means
                                lift_cloud.scales[sel] = sub_tf.scales
                                lift_cloud.quats[sel] = sub_tf.quats
                                if lift_cloud.sh_rest is not None and sub_tf.sh_rest is not None:
                                    lift_cloud.sh_rest[sel] = sub_tf.sh_rest

            lift_cfg = SegmentLiftConfig(
                num_views=len(views),
                render_width=anchor_w,
                render_height=anchor_h,
                bg_bias=req.bg_bias,
                threshold=req.threshold,
                use_gaussiancut=req.use_gaussiancut,
                use_sagd=req.use_sagd,
            )
            lift_res = lift_masks_to_3dgs(
                lift_cloud,
                views,
                masks_2d,
                lift_cfg,
                existing_object_ids=existing_ids,
                center_3d=center_3d,
                device=device,
                use_gsplat=use_gsplat,
            )
            if lift_cloud is not base_cloud and existing_ids is not None:
                if lift_res.decomposed_count == 0:
                    lift_res.cloud = base_cloud
                elif lift_res.existing_object_ids is not None:
                    base_occ = existing_ids > 0
                    res_occ = lift_res.existing_object_ids > 0
                    if int(np.count_nonzero(base_occ)) == int(np.count_nonzero(res_occ)):
                        lift_res.cloud.means[res_occ] = base_cloud.means[base_occ]
                        lift_res.cloud.scales[res_occ] = base_cloud.scales[base_occ]
                        lift_res.cloud.quats[res_occ] = base_cloud.quats[base_occ]
                        lift_res.cloud.opacities[res_occ] = base_cloud.opacities[base_occ]
                        lift_res.cloud.sh_dc[res_occ] = base_cloud.sh_dc[base_occ]
                        if lift_res.cloud.sh_rest is not None and base_cloud.sh_rest is not None:
                            lift_res.cloud.sh_rest[res_occ] = base_cloud.sh_rest[base_occ]

        prompt_parts = []
        if req.text and req.text.strip():
            prompt_parts.append(f"text:{req.text.strip()}")
        if req.points:
            prompt_parts.append(f"pts:{len(req.points)}")
        if req.box:
            prompt_parts.append("box")
        try:
            reg = mgr.register_segmented_object(
                cloud=lift_res.cloud,
                fg_mask=lift_res.fg_mask,
                candidate_indices=lift_res.candidate_indices,
                candidate_scores=lift_res.candidate_scores,
                name=req.name,
                prompt_summary=" + ".join(prompt_parts) or "sam3",
                existing_object_ids=lift_res.existing_object_ids,
                decomposed=lift_res.decomposed_count > 0,
                threshold=lift_cfg.threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if lift_res.decomposed_count > 0:
            with _RENDER_CACHE_LOCK:
                _RENDER_MODEL_CACHE.clear()
        return {**reg, "stats": lift_res.stats}

    @app.post("/api/editor/refine-object")
    def api_editor_refine_object(req: EditorRefineRequest) -> dict[str, Any]:
        root = _resolve_workspace_path(req.workspace_dir, ws_root)
        if store.has_active_job_for_workspace(root):
            raise HTTPException(status_code=409, detail="该工作区正在执行管线任务，暂不允许编辑场景图")
        mgr = _get_editor_manager(root)
        try:
            return mgr.refine_object_mask(
                req.object_id,
                threshold=req.threshold,
                z_clip=req.z_clip,
                record_undo=req.record_undo,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/editor/pick-object")
    def api_editor_pick_object(req: EditorPickRequest) -> dict[str, Any]:
        root = _resolve_workspace_path(req.workspace_dir, ws_root)
        if store.has_running_job():
            raise HTTPException(status_code=409, detail="训练执行期间暂停交互拾取以保护显存")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, gsplat_ready, _ = is_gsplat_cuda_ready()
        use_gsplat = bool(gsplat_ready and device.type == "cuda")

        pick_w = min(320, max(160, int(req.width)))
        pick_h = max(120, min(280, round(pick_w * req.height / max(1, req.width))))
        u_scaled = float(req.u) * (pick_w / float(max(1, req.width)))
        v_scaled = float(req.v) * (pick_h / float(max(1, req.height)))

        with _GPU_RENDER_LOCK:
            composed_model, composed_ids = _load_workspace_render_model(
                root, device, apply_edits=True, cache_model=True, store=store
            )
            view = build_orbit_camera_view(
                yaw=req.yaw,
                pitch=req.pitch,
                distance=req.distance,
                target=(req.tx, req.ty, req.tz),
                width=pick_w,
                height=pick_h,
                fov_scale=req.fov_scale,
                device=device,
                name="pick_view",
            )
            oid, hit_xyz = pick_object_at_pixel(
                composed_model,
                composed_ids,
                view,
                u_scaled,
                v_scaled,
                use_gsplat=use_gsplat,
            )
        return {"object_id": oid, "hit_xyz": hit_xyz}

    @app.post("/api/editor/update-object")
    def api_editor_update_object(req: EditorUpdateObjectRequest) -> dict[str, Any]:
        root = _resolve_workspace_path(req.workspace_dir, ws_root)
        if store.has_active_job_for_workspace(root):
            raise HTTPException(status_code=409, detail="该工作区正在执行管线任务，暂不允许编辑场景图")
        mgr = _get_editor_manager(root)
        try:
            return mgr.update_object(
                req.object_id,
                name=req.name,
                visible=req.visible,
                deleted=req.deleted,
                translation=req.translation,
                rotation_euler_deg=req.rotation_euler_deg,
                scale=req.scale,
                record_undo=req.record_undo,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/editor/object-action")
    def api_editor_object_action(req: EditorObjectActionRequest) -> dict[str, Any]:
        root = _resolve_workspace_path(req.workspace_dir, ws_root)
        if store.has_active_job_for_workspace(root):
            raise HTTPException(status_code=409, detail="该工作区正在执行管线任务，暂不允许编辑场景图")
        mgr = _get_editor_manager(root)
        try:
            return mgr.perform_action(req.action, object_id=req.object_id)
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/editor/bake")
    def api_editor_bake(req: EditorBakeRequest) -> dict[str, Any]:
        root = _resolve_workspace_path(req.workspace_dir, ws_root)
        mgr = _get_editor_manager(root)
        with _GPU_RENDER_LOCK:
            if store.has_running_job():
                raise HTTPException(status_code=409, detail="该工作区正在执行管线任务，请完成后再烘焙导出")
            try:
                res = bake_edited_scene(SceneWorkspace(root), manager=mgr, reexport_ply=req.reexport_ply)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            _evict_render_cache_for_workspace(root)
        with _DIR_SIZE_LOCK:
            _DIR_SIZE_CACHE.clear()
        return res

    @app.post("/api/editor/align-scene")
    def api_editor_align_scene(req: EditorAlignSceneRequest) -> dict[str, Any]:
        root = _resolve_workspace_path(req.workspace_dir, ws_root)
        if not root.is_dir():
            raise HTTPException(status_code=404, detail=f"Workspace not found: {req.workspace_dir}")
        with _GPU_RENDER_LOCK:
            if store.has_running_job():
                raise HTTPException(status_code=409, detail="训练执行期间暂停调整场景方向与水平度")
            try:
                res = realign_workspace_scene(
                    SceneWorkspace(root),
                    mode=req.mode,
                    roll_deg=req.roll_deg,
                    pitch_deg=req.pitch_deg,
                    yaw_deg=req.yaw_deg,
                    z_offset=req.z_offset,
                    current_yaw_rad=req.current_yaw_rad,
                    reexport_ply=req.reexport_ply,
                )
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            _evict_render_cache_for_workspace(root)
        with _DIR_SIZE_LOCK:
            _DIR_SIZE_CACHE.clear()
        return res

    @app.get("/api/editor/export-object-ply")
    def api_editor_export_object_ply(
        workspace_dir: str = Query(...),
        object_id: int = Query(...),
        coordinate_frame: str = Query("supersplat_colmap"),
    ) -> FileResponse:
        root = _resolve_workspace_path(workspace_dir, ws_root)
        mgr = _get_editor_manager(root)
        try:
            out_path, dl_name = export_single_object_ply(
                SceneWorkspace(root),
                object_id,
                manager=mgr,
                coordinate_frame=coordinate_frame,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", dl_name).strip("_") or f"object_{object_id:02d}.ply"
        disposition = f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(dl_name)}'
        return FileResponse(
            out_path,
            media_type="application/octet-stream",
            headers={"Content-Disposition": disposition},
        )

    @app.get("/api/workspace-file")
    def api_workspace_file(
        workspace_dir: str = Query(...),
        rel_path: str = Query(...),
    ) -> FileResponse:
        root = _resolve_workspace_path(workspace_dir, ws_root)
        target = (root / rel_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Path escapes workspace boundary") from exc
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {rel_path}")
        return FileResponse(target, filename=target.name)

    @app.get("/api/workspace-bundle.zip")
    def api_workspace_bundle(workspace_dir: str = Query(...)) -> FileResponse:
        root = _resolve_workspace_path(workspace_dir, ws_root)
        export_dir = root / "05_export"
        if not export_dir.is_dir():
            raise HTTPException(status_code=404, detail="05_export directory not found")

        tmp_fd, tmp_name = tempfile.mkstemp(prefix="r2s_bundle_", suffix=".zip")
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            with zipfile.ZipFile(tmp_path, mode="w") as zf:
                for file_path in sorted(export_dir.rglob("*")):
                    if file_path.is_file():
                        comp = zipfile.ZIP_STORED if file_path.suffix.lower() == ".ply" else zipfile.ZIP_DEFLATED
                        zf.write(file_path, arcname=str(file_path.relative_to(root)), compress_type=comp)
                manifest_path = root / "manifest.json"
                if manifest_path.is_file():
                    zf.write(manifest_path, arcname="manifest.json", compress_type=zipfile.ZIP_DEFLATED)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        raw_name = f"{root.name or 'real2sim'}_export.zip"
        ascii_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", root.name or "real2sim").strip("_") or "real2sim"
        ascii_name = f"{ascii_stem}_export.zip"
        disposition = f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(raw_name)}'
        return FileResponse(
            tmp_path,
            media_type="application/zip",
            headers={"Content-Disposition": disposition},
            background=BackgroundTask(os.unlink, str(tmp_path)),
        )

    @app.get("/jobs", response_model=list[JobRecord])
    def list_jobs() -> list[JobRecord]:
        return store.list_recent()

    @app.post("/jobs", response_model=JobRecord)
    def create_job(req: JobCreateRequest) -> JobRecord:
        if req.inline_config is not None:
            cfg = PipelineConfig.model_validate(req.inline_config)
            if req.input_path is not None:
                cfg.io.input_path = req.input_path
            if req.output_dir is not None:
                cfg.io.workspace_dir = req.output_dir
        else:
            cfg_path = _resolve_job_config_path(req.config_path)
            cfg = load_config(cfg_path, input_path=req.input_path, workspace_dir=req.output_dir)

        ws_dir = _resolve_workspace_path(cfg.io.workspace_dir, ws_root)
        allowed_exact = {
            (Path.cwd() / _SAMPLE_VIDEO_NAME).resolve(),
            (_PROJECT_ROOT / _SAMPLE_VIDEO_NAME).resolve(),
        }
        resolved_input = _resolve_job_input_path(
            cfg.io.input_path,
            uploads_dir,
            allowed_exact,
            allowed_roots=extra_input_roots,
        )
        display_ws = _display_path(ws_dir)
        cfg.io.workspace_dir = display_ws
        cfg.io.input_path = str(resolved_input)

        rec = store.create(
            workspace_dir=ws_dir,
            display_workspace=display_ws,
            input_path=resolved_input,
        )
        done_ev = store.submit(rec.job_id, lambda: _execute_job(store, rec.job_id, req, cfg, ws_dir))
        if not req.run_async:
            done_ev.wait()
        return store.get(rec.job_id)

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        return store.get(job_id)

    @app.post("/jobs/{job_id}/cancel", response_model=JobRecord)
    def cancel_job(job_id: str) -> JobRecord:
        return store.request_cancel(job_id)

    return app
