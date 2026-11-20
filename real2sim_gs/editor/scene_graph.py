from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from real2sim_gs.core.workspace import SceneWorkspace, atomic_write_json
from real2sim_gs.stages.export.ply_exporter import _R_ZUP_TO_SUPERSPLAT_PLY, Ply3DGSExporterStage
from real2sim_gs.stages.reconstruct.gsplat_whitebox.dataset import CameraView
from real2sim_gs.stages.reconstruct.gsplat_whitebox.model import GaussianModel
from real2sim_gs.stages.reconstruct.gsplat_whitebox.trainer import _render_view
from real2sim_gs.utils.ply_io import (
    SH_C0,
    GaussianCloud,
    _sh_band_transform,
    load_gaussian_ply,
    save_gaussian_ply,
    transform_gaussian_cloud,
)

OBJECT_PALETTE_RGB: list[list[float]] = [
    [0.14, 0.72, 0.96],  # cyan
    [0.10, 0.78, 0.52],  # emerald
    [0.96, 0.62, 0.08],  # amber
    [0.52, 0.55, 0.96],  # indigo
    [0.95, 0.30, 0.45],  # rose
    [0.66, 0.38, 0.95],  # violet
    [0.12, 0.78, 0.72],  # teal
    [0.98, 0.52, 0.22],  # orange
    [0.52, 0.80, 0.16],  # lime
    [0.92, 0.32, 0.78],  # pink
    [0.24, 0.60, 0.98],  # blue
    [0.95, 0.78, 0.15],  # gold
]

_SH_REST_K_TO_DEGREE = {3: 1, 8: 2, 15: 3}


@dataclass
class SceneObject:
    """Non-destructive segmented object node inside a 3DGS Scene Graph."""

    id: int
    source_mask_id: int
    name: str
    prompt_summary: str
    color_rgb: list[float]
    point_count: int
    pivot: list[float]
    bbox_min: list[float]
    bbox_max: list[float]
    visible: bool = True
    deleted: bool = False
    is_clone: bool = False
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_euler_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: float = 1.0
    threshold: float = 0.5
    z_clip: float = 0.0
    z_min_base: float = 0.0
    z_span_base: float = 1.0
    has_score_cache: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def rotation_matrix(self) -> np.ndarray:
        rx, ry, rz = [float(v) for v in self.rotation_euler_deg]
        return Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()

    def transformed_pivot(self) -> list[float]:
        return [
            round(float(self.pivot[i]) + float(self.translation[i]), 5)
            for i in range(3)
        ]

    def transformed_bbox(self) -> tuple[list[float], list[float]]:
        bmin = np.asarray(self.bbox_min, dtype=np.float64)
        bmax = np.asarray(self.bbox_max, dtype=np.float64)
        corners = np.array(
            [
                [x, y, z]
                for x in (bmin[0], bmax[0])
                for y in (bmin[1], bmax[1])
                for z in (bmin[2], bmax[2])
            ],
            dtype=np.float64,
        )
        piv = np.asarray(self.pivot, dtype=np.float64)
        trans = np.asarray(self.translation, dtype=np.float64)
        rot = self.rotation_matrix()
        sc = float(max(0.05, min(20.0, self.scale)))
        transformed = ((corners - piv[None, :]) @ rot.T) * sc + piv[None, :] + trans[None, :]
        return (
            [round(float(v), 5) for v in transformed.min(axis=0)],
            [round(float(v), 5) for v in transformed.max(axis=0)],
        )

    def to_public_dict(self) -> dict[str, Any]:
        t_bmin, t_bmax = self.transformed_bbox()
        data = asdict(self)
        data["transformed_bbox_min"] = t_bmin
        data["transformed_bbox_max"] = t_bmax
        data["transformed_pivot"] = self.transformed_pivot()
        return data


def _euler_deg_to_matrix_and_quat_wxyz(euler_deg: list[float]) -> tuple[np.ndarray, np.ndarray]:
    rot = Rotation.from_euler("xyz", [float(v) for v in euler_deg], degrees=True)
    mat = rot.as_matrix().astype(np.float64)
    xyzw = rot.as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    if wxyz[0] < 0.0:
        wxyz = -wxyz
    return mat, wxyz


def _left_mul_quat_torch(q_rot_wxyz: np.ndarray, quats: torch.Tensor) -> torch.Tensor:
    rw, rx, ry, rz = [float(v) for v in q_rot_wxyz]
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    out = torch.stack(
        [
            rw * w - rx * x - ry * y - rz * z,
            rw * x + rx * w + ry * z - rz * y,
            rw * y - rx * z + ry * w + rz * x,
            rw * z + rx * y - ry * x + rz * w,
        ],
        dim=1,
    )
    return out / (out.norm(dim=-1, keepdim=True) + 1e-8)


def _rotate_sh_rest_torch(sh_rest: torch.Tensor, rot_mat: np.ndarray) -> torch.Tensor:
    """Apply exact O(3) spherical harmonic rotation (degrees 1..3) to GPU sh_rest tensor."""
    if sh_rest.numel() == 0 or sh_rest.shape[1] == 0:
        return sh_rest
    n, cols = int(sh_rest.shape[0]), int(sh_rest.shape[1])
    k = cols // 3
    degree = _SH_REST_K_TO_DEGREE.get(k)
    if degree is None:
        return sh_rest
    coeffs = sh_rest.view(n, 3, k)
    chunks: list[torch.Tensor] = []
    start = 0
    for level in range(1, degree + 1):
        width = 2 * level + 1
        x_np = _sh_band_transform(level, rot_mat).astype(np.float32)
        x_t = torch.from_numpy(x_np).to(device=sh_rest.device, dtype=sh_rest.dtype)
        chunks.append(torch.einsum("ij,ncj->nci", x_t, coeffs[:, :, start : start + width]))
        start += width
    return torch.cat(chunks, dim=2).reshape(n, cols)


def _transform_object_tensors_gpu(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    sh_rest: torch.Tensor,
    obj_dict: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    piv = torch.tensor(obj_dict["pivot"], device=means.device, dtype=torch.float32).view(1, 3)
    trans = torch.tensor(obj_dict["translation"], device=means.device, dtype=torch.float32).view(1, 3)
    sc = float(max(0.05, min(20.0, float(obj_dict.get("scale", 1.0)))))
    euler = [float(v) for v in obj_dict.get("rotation_euler_deg", [0.0, 0.0, 0.0])]
    has_rot = any(abs(v) > 1e-4 for v in euler)

    if has_rot:
        rot_np, q_wxyz = _euler_deg_to_matrix_and_quat_wxyz(euler)
        rot_t = torch.from_numpy(rot_np.astype(np.float32)).to(device=means.device)
        new_means = ((means - piv) @ rot_t.T) * sc + piv + trans
        new_quats = _left_mul_quat_torch(q_wxyz, quats)
        new_sh_rest = _rotate_sh_rest_torch(sh_rest, rot_np) if sh_rest.shape[1] > 0 else sh_rest
    else:
        new_means = (means - piv) * sc + piv + trans
        new_quats = quats
        new_sh_rest = sh_rest

    new_scales = scales + math.log(sc) if abs(sc - 1.0) > 1e-6 else scales
    return new_means, new_scales, new_quats, new_sh_rest


def _apply_tint_sh_dc(sh_dc: torch.Tensor, color_rgb: list[float], mix: float) -> torch.Tensor:
    target = torch.tensor([(float(c) - 0.5) / SH_C0 for c in color_rgb], device=sh_dc.device, dtype=torch.float32)
    return sh_dc * (1.0 - mix) + target.view(1, 3) * mix


def _dim_sh_dc(sh_dc: torch.Tensor, factor: float = 0.45) -> torch.Tensor:
    rgb = (sh_dc * SH_C0 + 0.5) * factor
    return (rgb - 0.5) / SH_C0


@torch.no_grad()
def apply_scene_edits_to_gpu_model(
    base_model: GaussianModel,
    object_ids: np.ndarray | None,
    state: dict[str, Any] | None,
    *,
    selected_id: int | None = None,
    highlight_selected: bool = False,
    dim_background: bool = False,
) -> tuple[GaussianModel, torch.Tensor | None]:
    """Compose an edited GaussianModel on GPU from `base_model` + `scene_objects` state without mutating cache."""
    objects = list((state or {}).get("objects") or [])
    if object_ids is None or len(object_ids) != base_model.num_points or not objects:
        return base_model, None

    device = base_model.means.device
    obj_ids_t = torch.from_numpy(np.asarray(object_ids, dtype=np.int32)).to(device=device)

    claimed_mask_ids = {int(o["source_mask_id"]) for o in objects}
    bg_bool = obj_ids_t <= 0
    # Preserve any orphaned mask IDs not tracked in `objects` as static background
    known_mask = torch.zeros_like(bg_bool)
    for mid in claimed_mask_ids:
        known_mask |= obj_ids_t == mid
    bg_bool = ~known_mask

    parts_means: list[torch.Tensor] = []
    parts_scales: list[torch.Tensor] = []
    parts_quats: list[torch.Tensor] = []
    parts_opac: list[torch.Tensor] = []
    parts_dc: list[torch.Tensor] = []
    parts_rest: list[torch.Tensor] = []
    parts_oids: list[torch.Tensor] = []

    if torch.any(bg_bool):
        bg_dc = base_model.sh_dc[bg_bool]
        if dim_background:
            bg_dc = _dim_sh_dc(bg_dc, 0.42)
        parts_means.append(base_model.means[bg_bool])
        parts_scales.append(base_model.scales[bg_bool])
        parts_quats.append(base_model.quats[bg_bool])
        parts_opac.append(base_model.opacities[bg_bool])
        parts_dc.append(bg_dc)
        parts_rest.append(base_model.sh_rest[bg_bool])
        parts_oids.append(torch.zeros(int(bg_bool.sum().item()), dtype=torch.int32, device=device))

    for obj in objects:
        if obj.get("deleted") or not obj.get("visible", True):
            continue
        oid = int(obj["id"])
        mid = int(obj["source_mask_id"])
        sel = obj_ids_t == mid
        cnt = int(sel.sum().item())
        if cnt == 0:
            continue

        m, s, q, rest = _transform_object_tensors_gpu(
            base_model.means[sel],
            base_model.scales[sel],
            base_model.quats[sel],
            base_model.sh_rest[sel],
            obj,
        )
        dc = base_model.sh_dc[sel]
        if highlight_selected and selected_id is not None and oid == int(selected_id):
            dc = _apply_tint_sh_dc(dc, obj.get("color_rgb", [0.14, 0.72, 0.96]), 0.34)
        elif dim_background and selected_id is not None and oid != int(selected_id):
            dc = _dim_sh_dc(dc, 0.42)

        parts_means.append(m)
        parts_scales.append(s)
        parts_quats.append(q)
        parts_opac.append(base_model.opacities[sel])
        parts_dc.append(dc)
        parts_rest.append(rest)
        parts_oids.append(torch.full((cnt,), oid, dtype=torch.int32, device=device))

    if not parts_means:
        empty_model = GaussianModel(
            means=torch.zeros((0, 3), dtype=torch.float32, device=device),
            scales=torch.zeros((0, 3), dtype=torch.float32, device=device),
            quats=torch.zeros((0, 4), dtype=torch.float32, device=device),
            opacities=torch.zeros((0,), dtype=torch.float32, device=device),
            sh_dc=torch.zeros((0, 3), dtype=torch.float32, device=device),
            sh_degree=base_model.sh_degree,
            max_log_scale=base_model.max_log_scale,
        )
        return empty_model, torch.zeros((0,), dtype=torch.int32, device=device)

    rest_cat = torch.cat(parts_rest, dim=0) if base_model.sh_rest.shape[1] > 0 else None
    composed = GaussianModel(
        means=torch.cat(parts_means, dim=0),
        scales=torch.cat(parts_scales, dim=0),
        quats=torch.cat(parts_quats, dim=0),
        opacities=torch.cat(parts_opac, dim=0),
        sh_dc=torch.cat(parts_dc, dim=0),
        sh_rest=rest_cat,
        sh_degree=base_model.sh_degree,
        max_log_scale=base_model.max_log_scale,
    )
    return composed, torch.cat(parts_oids, dim=0)


@torch.no_grad()
def pick_object_at_pixel(
    composed_model: GaussianModel,
    composed_obj_ids: torch.Tensor | None,
    camera_view: CameraView,
    u_px: float,
    v_px: float,
    *,
    use_gsplat: bool,
) -> tuple[int | None, list[float] | None]:
    """Depth-aware 3D picking: unproject pixel expected depth and match nearest visible object Gaussian."""
    if composed_obj_ids is None or composed_model.num_points == 0:
        return None, None

    obj_mask = composed_obj_ids > 0
    if not torch.any(obj_mask):
        return None, None

    device = composed_model.means.device
    bg_t = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device)
    r_out = _render_view(
        composed_model,
        camera_view,
        w2c=camera_view.world_to_cam,
        use_gsplat=use_gsplat,
        mip_3d_var=0.0,
        bg_color=bg_t,
        need_depth=True,
        sh_degree=0,
    )
    depth_map = r_out.depth.detach().cpu().numpy()
    w, h = camera_view.width, camera_view.height
    xi = int(np.clip(round(float(u_px)), 0, w - 1))
    yi = int(np.clip(round(float(v_px)), 0, h - 1))

    x0, x1 = max(0, xi - 2), min(w, xi + 3)
    y0, y1 = max(0, yi - 2), min(h, yi + 3)
    patch = depth_map[y0:y1, x0:x1]
    valid_d = patch[(patch > 0.05) & np.isfinite(patch)]

    w2c = camera_view.world_to_cam.detach().cpu().numpy()
    r_c2w = w2c[:3, :3].T
    eye = -r_c2w @ w2c[:3, 3]
    ray_cam = np.array(
        [(float(u_px) + 0.5 - camera_view.cx) / max(camera_view.fx, 1e-6),
         (float(v_px) + 0.5 - camera_view.cy) / max(camera_view.fy, 1e-6),
         1.0],
        dtype=np.float64,
    )
    ray_dir = r_c2w @ ray_cam
    ray_dir /= max(float(np.linalg.norm(ray_dir)), 1e-8)

    obj_idx = torch.nonzero(obj_mask, as_tuple=True)[0]
    obj_pts = composed_model.means[obj_idx].detach().cpu().numpy().astype(np.float64)
    obj_ids_np = composed_obj_ids[obj_idx].detach().cpu().numpy()

    rel = obj_pts - eye[None, :]
    along = rel @ ray_dir
    in_front = along > 0.05
    if not np.any(in_front):
        return None, None
    obj_pts = obj_pts[in_front]
    obj_ids_np = obj_ids_np[in_front]
    rel = rel[in_front]
    along = along[in_front]

    perp = np.linalg.norm(rel - along[:, None] * ray_dir[None, :], axis=1)

    if valid_d.size > 0:
        d_star = float(np.median(valid_d))
        # gsplat need_depth=True returns camera-space Z depth along optical axis, so unproject via ray_cam * d_star.
        hit_xyz = (eye + r_c2w @ (ray_cam * d_star)).astype(np.float64)
        dist_3d = np.linalg.norm(obj_pts - hit_xyz[None, :], axis=1)
        best = int(np.argmin(dist_3d))
        tol = max(0.09, 0.06 * d_star)
        if dist_3d[best] <= tol or perp[best] <= max(0.035 * d_star, 0.035):
            return int(obj_ids_np[best]), [round(float(v), 5) for v in hit_xyz]
        return None, [round(float(v), 5) for v in hit_xyz]

    # Fallback ray cylinder intersection if rendered depth was empty
    ray_tol = np.maximum(0.04 * along, 0.04)
    hits = perp <= ray_tol
    if not np.any(hits):
        return None, None
    cand = np.nonzero(hits)[0]
    best = int(cand[np.argmin(along[cand])])
    hit_xyz = obj_pts[best]
    return int(obj_ids_np[best]), [round(float(v), 5) for v in hit_xyz]


class SceneEditManager:
    """Manages non-destructive object masks, transforms, undo/redo history, baking, and PLY exports."""

    def __init__(self, ws: SceneWorkspace) -> None:
        self.ws = ws
        self._lock = threading.RLock()
        self._undo_stack: list[tuple[dict[str, Any], np.ndarray]] = []
        self._redo_stack: list[tuple[dict[str, Any], np.ndarray]] = []
        self._max_history = 40
        self._base_cloud_cache: tuple[str, int, GaussianCloud] | None = None

    @property
    def segments_dir(self) -> Path:
        return self.ws.processed_dir / "segments"

    @property
    def objects_json_path(self) -> Path:
        return self.segments_dir / "scene_objects.json"

    @property
    def object_ids_path(self) -> Path:
        return self.segments_dir / "object_ids.npy"

    @property
    def base_splats_path(self) -> Path:
        return self.segments_dir / "base_splats.ply"

    @property
    def edited_splats_path(self) -> Path:
        return self.ws.processed_dir / "edited_splats.ply"

    def _scores_path(self, mask_id: int) -> Path:
        return self.segments_dir / f"scores_{int(mask_id):03d}.npz"

    def has_edits(self) -> bool:
        return self.objects_json_path.is_file() and self.object_ids_path.is_file()

    def clear_caches(self) -> None:
        with self._lock:
            self._base_cloud_cache = None

    def resolve_base_ply_path(self) -> Path:
        with self._lock:
            if self.objects_json_path.is_file():
                try:
                    data = json.loads(self.objects_json_path.read_text(encoding="utf-8"))
                    rel = str(data.get("base_ply") or "").strip()
                    if rel:
                        cand = Path(rel)
                        if not cand.is_absolute():
                            cand = self.ws.root / cand
                        if cand.is_file():
                            return cand
                except (OSError, ValueError):
                    pass
            if self.base_splats_path.is_file():
                return self.base_splats_path
            return self.ws.active_splat_path()

    def load_base_cloud(self) -> GaussianCloud:
        with self._lock:
            ply_path = self.resolve_base_ply_path().resolve()
            mtime_ns = int(ply_path.stat().st_mtime_ns)
            if (
                self._base_cloud_cache is not None
                and self._base_cloud_cache[0] == str(ply_path)
                and self._base_cloud_cache[1] == mtime_ns
            ):
                return self._base_cloud_cache[2]
            cloud = load_gaussian_ply(ply_path)
            self._base_cloud_cache = (str(ply_path), mtime_ns, cloud)
            return cloud

    @staticmethod
    def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.npy", dir=str(path.parent))
        os.close(fd)
        try:
            with open(tmp_name, "wb") as fh:
                np.save(fh, arr)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_save_ply(path: Path, cloud: GaussianCloud) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.ply", dir=str(path.parent))
        os.close(fd)
        try:
            save_gaussian_ply(Path(tmp_name), cloud)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def load_state(self) -> tuple[dict[str, Any], np.ndarray | None]:
        with self._lock:
            if not self.objects_json_path.is_file():
                base_ply = self.ws.active_splat_path()
                rel_base = self.ws.to_relative(base_ply)
                return (
                    {
                        "version": 1,
                        "source_ply": rel_base,
                        "base_ply": rel_base,
                        "next_id": 1,
                        "next_mask_id": 1,
                        "baked_dirty": False,
                        "objects": [],
                    },
                    None,
                )
            state = json.loads(self.objects_json_path.read_text(encoding="utf-8"))
            obj_ids = np.load(self.object_ids_path) if self.object_ids_path.is_file() else None
            return state, obj_ids

    def _save_state(self, state: dict[str, Any], obj_ids: np.ndarray) -> None:
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_save_npy(self.object_ids_path, obj_ids.astype(np.int32))
        atomic_write_json(self.objects_json_path, state)

    def _push_undo(self, state: dict[str, Any], obj_ids: np.ndarray | None) -> None:
        ids_copy = obj_ids.copy() if obj_ids is not None else np.zeros(0, dtype=np.int32)
        self._undo_stack.append((copy.deepcopy(state), ids_copy))
        if len(self._undo_stack) > self._max_history:
            self._undo_stack = self._undo_stack[-self._max_history :]
        self._redo_stack.clear()

    def public_summary(self) -> dict[str, Any]:
        with self._lock:
            try:
                state, _ = self.load_state()
            except FileNotFoundError:
                return {
                    "objects": [],
                    "can_undo": False,
                    "can_redo": False,
                    "baked_dirty": False,
                    "base_ply": None,
                }
            public_objs = [SceneObject(**o).to_public_dict() for o in state.get("objects", [])]
            return {
                "objects": public_objs,
                "can_undo": len(self._undo_stack) > 0,
                "can_redo": len(self._redo_stack) > 0,
                "baked_dirty": bool(state.get("baked_dirty", False)),
                "base_ply": state.get("base_ply"),
            }

    @staticmethod
    def _compute_stats_from_points(pts: np.ndarray) -> tuple[list[float], list[float], list[float], float, float]:
        if len(pts) == 0:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, 1.0
        bmin = [round(float(v), 5) for v in pts.min(axis=0)]
        bmax = [round(float(v), 5) for v in pts.max(axis=0)]
        piv = [round(float(v), 5) for v in np.median(pts, axis=0)]
        z_min = float(pts[:, 2].min())
        z_span = max(float(pts[:, 2].max()) - z_min, 1e-3)
        return piv, bmin, bmax, z_min, z_span

    def register_segmented_object(
        self,
        *,
        cloud: GaussianCloud,
        fg_mask: np.ndarray,
        candidate_indices: np.ndarray,
        candidate_scores: np.ndarray,
        name: str | None = None,
        prompt_summary: str = "",
        existing_object_ids: np.ndarray | None = None,
        decomposed: bool = False,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Register a newly lifted 3DGS object into the non-destructive Scene Graph."""
        with self._lock:
            state, prev_ids = self.load_state()
            n = cloud.num_points
            if decomposed:
                # SAGD modified topology: persist base_splats.ply and remap existing_object_ids
                self._atomic_save_ply(self.base_splats_path, cloud)
                self._base_cloud_cache = (
                    str(self.base_splats_path.resolve()),
                    int(self.base_splats_path.stat().st_mtime_ns),
                    cloud,
                )
                state["base_ply"] = self.ws.to_relative(self.base_splats_path)
                # Clear undo history when topology changed (existing score caches also shift indices)
                self._undo_stack.clear()
                self._redo_stack.clear()
                for old_obj in state.get("objects", []):
                    old_obj["has_score_cache"] = False
                    self._scores_path(int(old_obj["source_mask_id"])).unlink(missing_ok=True)
                obj_ids = (
                    existing_object_ids.astype(np.int32).copy()
                    if existing_object_ids is not None and len(existing_object_ids) == n
                    else np.zeros(n, dtype=np.int32)
                )
            else:
                if prev_ids is not None and len(prev_ids) == n:
                    obj_ids = prev_ids.astype(np.int32).copy()
                else:
                    obj_ids = np.zeros(n, dtype=np.int32)
                self._push_undo(state, prev_ids if prev_ids is not None else np.zeros(n, dtype=np.int32))

            new_id = int(state.get("next_id", 1))
            new_mask_id = int(state.get("next_mask_id", 1))
            state["next_id"] = new_id + 1
            state["next_mask_id"] = new_mask_id + 1

            claim = np.asarray(fg_mask, dtype=bool) & (obj_ids <= 0)
            if not np.any(claim):
                raise ValueError("未能从当前提示中分割出有效未占用 3D 高斯物体，请调整点击位置或提示词重试。")

            obj_ids[claim] = new_mask_id
            pts = cloud.means[claim]
            piv, bmin, bmax, z_min, z_span = self._compute_stats_from_points(pts)

            # Persist candidate scores & Z heights for instant interactive threshold / Z-clip tuning
            self.segments_dir.mkdir(parents=True, exist_ok=True)
            cand_idx = np.asarray(candidate_indices, dtype=np.int32)
            cand_sc = np.asarray(candidate_scores, dtype=np.float16)
            cand_z = cloud.means[cand_idx, 2].astype(np.float32) if len(cand_idx) > 0 else np.zeros(0, dtype=np.float32)
            np.savez_compressed(
                self._scores_path(new_mask_id),
                indices=cand_idx,
                scores=cand_sc,
                z_values=cand_z,
            )

            clean_name = str(name or "").strip() or (
                f"{prompt_summary[:18]}_{new_id:02d}" if prompt_summary else f"物体_{new_id:02d}"
            )
            color = OBJECT_PALETTE_RGB[(new_id - 1) % len(OBJECT_PALETTE_RGB)]
            obj = SceneObject(
                id=new_id,
                source_mask_id=new_mask_id,
                name=clean_name,
                prompt_summary=prompt_summary,
                color_rgb=color,
                point_count=int(np.count_nonzero(claim)),
                pivot=piv,
                bbox_min=bmin,
                bbox_max=bmax,
                threshold=float(threshold),
                z_clip=0.0,
                z_min_base=z_min,
                z_span_base=z_span,
                has_score_cache=len(cand_idx) > 0,
            )
            state.setdefault("objects", []).append(asdict(obj))
            state["baked_dirty"] = True
            self._save_state(state, obj_ids)
            return {
                "object": obj.to_public_dict(),
                **self.public_summary(),
            }

    def refine_object_mask(
        self,
        object_id: int,
        *,
        threshold: float | None = None,
        z_clip: float | None = None,
        record_undo: bool = True,
    ) -> dict[str, Any]:
        """Instant boundary tightness / bottom Z-plane refinement using cached continuous scores."""
        with self._lock:
            state, obj_ids = self.load_state()
            if obj_ids is None:
                raise FileNotFoundError("尚未建立物体分割缓存。")
            obj = next((o for o in state["objects"] if int(o["id"]) == int(object_id)), None)
            if obj is None:
                raise KeyError(f"未找到物体 ID={object_id}")
            if obj.get("is_clone"):
                raise ValueError("复制物体暂不支持边缘微调，请对原物体微调。")
            if not obj.get("has_score_cache"):
                raise ValueError("该物体的连续置信度缓存已因后续边界分裂失效，请重新分割或微调新物体。")

            mask_id = int(obj["source_mask_id"])
            sp_path = self._scores_path(mask_id)
            if not sp_path.is_file():
                raise ValueError("该物体的连续置信度缓存已因后续边界分裂失效，请重新分割或微调新物体。")

            cloud = self.load_base_cloud()
            if len(obj_ids) != cloud.num_points:
                raise ValueError("物体索引与基础点云长度不匹配。")

            npz = np.load(sp_path)
            cand_idx = npz["indices"].astype(np.int32)
            if len(cand_idx) > 0 and (int(cand_idx.min()) < 0 or int(cand_idx.max()) >= cloud.num_points):
                raise ValueError("物体置信度缓存索引已过期，请重新分割。")
            cand_scores = npz["scores"].astype(np.float32)
            cand_z = npz["z_values"].astype(np.float32)

            eff_thresh = float(obj["threshold"] if threshold is None else np.clip(threshold, 0.05, 0.98))
            eff_z_clip = float(obj["z_clip"] if z_clip is None else z_clip)
            z_cutoff = float(obj.get("z_min_base", 0.0)) + eff_z_clip

            new_ids = obj_ids.copy()
            new_ids[new_ids == mask_id] = 0
            eligible = new_ids[cand_idx] == 0
            keep = eligible & (cand_scores >= eff_thresh) & (cand_z >= z_cutoff)
            sel_idx = cand_idx[keep]
            if len(sel_idx) == 0:
                raise ValueError("当前阈值/Z裁剪过严导致选中点数为 0，已保留原选区。")

            if record_undo:
                self._push_undo(state, obj_ids)

            new_ids[sel_idx] = mask_id
            pts = cloud.means[new_ids == mask_id]
            piv, bmin, bmax, _, _ = self._compute_stats_from_points(pts)

            obj["threshold"] = round(eff_thresh, 4)
            obj["z_clip"] = round(eff_z_clip, 5)
            obj["point_count"] = int(len(sel_idx))
            obj["bbox_min"] = bmin
            obj["bbox_max"] = bmax
            # Keep pivot anchored when user hasn't rotated/translated yet, else update cleanly
            if all(abs(v) < 1e-6 for v in obj.get("translation", [0, 0, 0])) and all(
                abs(v) < 1e-6 for v in obj.get("rotation_euler_deg", [0, 0, 0])
            ):
                obj["pivot"] = piv
            state["baked_dirty"] = True
            self._save_state(state, new_ids)
            return {
                "object": SceneObject(**obj).to_public_dict(),
                **self.public_summary(),
            }

    def update_object(
        self,
        object_id: int,
        *,
        name: str | None = None,
        visible: bool | None = None,
        deleted: bool | None = None,
        translation: list[float] | None = None,
        rotation_euler_deg: list[float] | None = None,
        scale: float | None = None,
        record_undo: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            state, obj_ids = self.load_state()
            if obj_ids is None:
                raise FileNotFoundError("当前工作区尚无分割物体。")
            obj = next((o for o in state["objects"] if int(o["id"]) == int(object_id)), None)
            if obj is None:
                raise KeyError(f"未找到物体 ID={object_id}")

            if record_undo:
                self._push_undo(state, obj_ids)

            if name is not None and str(name).strip():
                obj["name"] = str(name).strip()[:64]
            if visible is not None:
                obj["visible"] = bool(visible)
            if deleted is not None:
                obj["deleted"] = bool(deleted)
            if translation is not None and len(translation) == 3:
                obj["translation"] = [round(float(v), 5) for v in translation]
            if rotation_euler_deg is not None and len(rotation_euler_deg) == 3:
                obj["rotation_euler_deg"] = [round(float(v), 3) for v in rotation_euler_deg]
            if scale is not None:
                obj["scale"] = round(float(max(0.05, min(20.0, float(scale)))), 5)

            state["baked_dirty"] = True
            self._save_state(state, obj_ids)
            return {
                "object": SceneObject(**obj).to_public_dict(),
                **self.public_summary(),
            }

    def duplicate_object(self, object_id: int) -> dict[str, Any]:
        with self._lock:
            state, obj_ids = self.load_state()
            if obj_ids is None:
                raise FileNotFoundError("当前工作区尚无分割物体。")
            src = next((o for o in state["objects"] if int(o["id"]) == int(object_id)), None)
            if src is None:
                raise KeyError(f"未找到物体 ID={object_id}")

            self._push_undo(state, obj_ids)
            new_id = int(state.get("next_id", 1))
            state["next_id"] = new_id + 1

            bmin = np.asarray(src["bbox_min"], dtype=np.float64)
            bmax = np.asarray(src["bbox_max"], dtype=np.float64)
            offset_x = float(max(0.12, (bmax[0] - bmin[0]) * 1.15))

            clone_obj = copy.deepcopy(src)
            clone_obj["id"] = new_id
            clone_obj["name"] = f"{src['name']} 副本"
            clone_obj["is_clone"] = True
            clone_obj["deleted"] = False
            clone_obj["visible"] = True
            clone_obj["has_score_cache"] = False
            clone_obj["color_rgb"] = OBJECT_PALETTE_RGB[(new_id - 1) % len(OBJECT_PALETTE_RGB)]
            clone_obj["translation"] = [
                round(float(src["translation"][0]) + offset_x, 5),
                float(src["translation"][1]),
                float(src["translation"][2]),
            ]
            clone_obj["created_at"] = datetime.now(timezone.utc).isoformat()
            state["objects"].append(clone_obj)
            state["baked_dirty"] = True
            self._save_state(state, obj_ids)
            return {
                "object": SceneObject(**clone_obj).to_public_dict(),
                **self.public_summary(),
            }

    def perform_action(self, action: str, object_id: int | None = None) -> dict[str, Any]:
        act = str(action or "").strip().lower()
        with self._lock:
            state, obj_ids = self.load_state()
            if act == "undo":
                if not self._undo_stack:
                    return {**self.public_summary(), "action": "undo", "changed": False}
                cur_ids = obj_ids.copy() if obj_ids is not None else np.zeros(0, dtype=np.int32)
                self._redo_stack.append((copy.deepcopy(state), cur_ids))
                prev_state, prev_ids = self._undo_stack.pop()
                self._save_state(prev_state, prev_ids)
                return {**self.public_summary(), "action": "undo", "changed": True}

            if act == "redo":
                if not self._redo_stack:
                    return {**self.public_summary(), "action": "redo", "changed": False}
                cur_ids = obj_ids.copy() if obj_ids is not None else np.zeros(0, dtype=np.int32)
                self._undo_stack.append((copy.deepcopy(state), cur_ids))
                next_state, next_ids = self._redo_stack.pop()
                self._save_state(next_state, next_ids)
                return {**self.public_summary(), "action": "redo", "changed": True}

            if object_id is None:
                raise ValueError(f"操作 '{act}' 需要指定 object_id")
            if obj_ids is None:
                raise FileNotFoundError("当前工作区尚无分割物体。")

            if act == "duplicate":
                return {**self.duplicate_object(object_id), "action": "duplicate", "changed": True}
            if act == "delete":
                res = self.update_object(object_id, deleted=True, record_undo=True)
                return {**res, "action": "delete", "changed": True}
            if act == "restore":
                res = self.update_object(object_id, deleted=False, visible=True, record_undo=True)
                return {**res, "action": "restore", "changed": True}
            if act == "reset_transform":
                res = self.update_object(
                    object_id,
                    translation=[0.0, 0.0, 0.0],
                    rotation_euler_deg=[0.0, 0.0, 0.0],
                    scale=1.0,
                    record_undo=True,
                )
                return {**res, "action": "reset_transform", "changed": True}
            if act == "remove_entry":
                target = next((o for o in state["objects"] if int(o["id"]) == int(object_id)), None)
                if target is None:
                    raise KeyError(f"未找到物体 ID={object_id}")
                self._push_undo(state, obj_ids)
                state["objects"] = [o for o in state["objects"] if int(o["id"]) != int(object_id)]
                # Only free base mask ID if no remaining clone still references it
                mid = int(target["source_mask_id"])
                if not any(int(o["source_mask_id"]) == mid for o in state["objects"]):
                    obj_ids = obj_ids.copy()
                    obj_ids[obj_ids == mid] = 0
                state["baked_dirty"] = True
                self._save_state(state, obj_ids)
                return {**self.public_summary(), "action": "remove_entry", "changed": True}

            raise ValueError(f"未知操作类型: {action}")

    @staticmethod
    def _transform_cloud_cpu(sub_cloud: GaussianCloud, obj: dict[str, Any]) -> GaussianCloud:
        piv = np.asarray(obj["pivot"], dtype=np.float64).reshape(3)
        trans = np.asarray(obj.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        sc = float(max(0.05, min(20.0, float(obj.get("scale", 1.0)))))
        rot_np, _ = _euler_deg_to_matrix_and_quat_wxyz(obj.get("rotation_euler_deg", [0.0, 0.0, 0.0]))
        # x' = sc * R @ (x - piv) + piv + trans = sc * (R @ x) + (piv + trans - sc * R @ piv)
        net_trans = piv + trans - sc * (rot_np @ piv)
        return transform_gaussian_cloud(sub_cloud, rot_np, translation=net_trans, scale=sc)

    def _is_aligned_workspace(self) -> bool:
        t_path = self.ws.processed_dir / "alignment_transform.json"
        if not t_path.is_file():
            return False
        base_path = self.resolve_base_ply_path().resolve()
        return base_path != self.ws.raw_splats_path.resolve()

    def export_single_object_ply(
        self,
        object_id: int,
        *,
        coordinate_frame: str = "supersplat_colmap",
    ) -> tuple[Path, str]:
        """Extract one segmented object (with current transform & full SH) into an independent .ply."""
        return export_single_object_ply(self.ws, object_id, manager=self, coordinate_frame=coordinate_frame)

    def bake_edited_scene(self, *, reexport_ply: bool = True) -> dict[str, Any]:
        """Bake all non-deleted objects and transforms into `04_processed/edited_splats.ply`."""
        return bake_edited_scene(self.ws, manager=self, reexport_ply=reexport_ply)


def export_single_object_ply(
    ws: SceneWorkspace,
    object_id: int,
    *,
    manager: SceneEditManager | None = None,
    coordinate_frame: str = "supersplat_colmap",
) -> tuple[Path, str]:
    """Export one segmented object from `ws` as a standalone binary 3DGS .ply file."""
    mgr = manager or SceneEditManager(ws)
    with mgr._lock:
        state, obj_ids = mgr.load_state()
        if obj_ids is None:
            raise FileNotFoundError("当前工作区尚无分割物体。")
        obj = next((o for o in state.get("objects", []) if int(o["id"]) == int(object_id)), None)
        if obj is None:
            raise KeyError(f"未找到物体 ID={object_id}")

        cloud = mgr.load_base_cloud()
        mask = obj_ids == int(obj["source_mask_id"])
        if not np.any(mask):
            raise ValueError(f"物体 '{obj['name']}' 不包含任何高斯点。")

        sub = cloud.subset(mask)
        transformed = SceneEditManager._transform_cloud_cpu(sub, obj)

        is_aligned = mgr._is_aligned_workspace()
        if str(coordinate_frame).strip().lower() != "z_up" and is_aligned and transformed.num_points > 0:
            t_piv = np.asarray(SceneObject(**obj).transformed_pivot(), dtype=np.float64)
            r_ply = _R_ZUP_TO_SUPERSPLAT_PLY.astype(np.float64)
            transformed = transform_gaussian_cloud(
                transformed,
                r_ply,
                translation=-(r_ply @ t_piv),
            )

        out_dir = ws.export_dir / "objects"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(r'[<>:"/\\|?*\s]+', "_", str(obj["name"])).strip("_") or f"object_{object_id:02d}"
        out_path = out_dir / f"{safe_stem}_id{int(object_id):02d}.ply"
        mgr._atomic_save_ply(out_path, transformed)
        return out_path, out_path.name


def bake_edited_scene(
    ws: SceneWorkspace,
    *,
    manager: SceneEditManager | None = None,
    reexport_ply: bool = True,
) -> dict[str, Any]:
    """Bake non-deleted segmented objects and transforms to `04_processed/edited_splats.ply` and update exports."""
    mgr = manager or SceneEditManager(ws)
    with mgr._lock:
        state, obj_ids = mgr.load_state()
        cloud = mgr.load_base_cloud()
        objects = list(state.get("objects") or [])

        if obj_ids is None or len(obj_ids) != cloud.num_points:
            raise ValueError("尚未有可烘焙的分割或编辑状态。")

        claimed_mask_ids = {int(o["source_mask_id"]) for o in objects}
        bg_mask = ~np.isin(obj_ids, list(claimed_mask_ids))
        chunks: list[GaussianCloud] = []
        if np.any(bg_mask):
            chunks.append(cloud.subset(bg_mask))

        kept_objects = 0
        deleted_objects = 0
        for obj in objects:
            if obj.get("deleted"):
                deleted_objects += 1
                continue
            sel = obj_ids == int(obj["source_mask_id"])
            if not np.any(sel):
                continue
            sub = cloud.subset(sel)
            chunks.append(SceneEditManager._transform_cloud_cpu(sub, obj))
            kept_objects += 1

        if chunks:
            has_rest = all(c.sh_rest is not None for c in chunks)
            baked_cloud = GaussianCloud(
                means=np.concatenate([c.means for c in chunks], axis=0),
                scales=np.concatenate([c.scales for c in chunks], axis=0),
                quats=np.concatenate([c.quats for c in chunks], axis=0),
                opacities=np.concatenate([c.opacities for c in chunks], axis=0),
                sh_dc=np.concatenate([c.sh_dc for c in chunks], axis=0),
                sh_rest=np.concatenate([c.sh_rest for c in chunks if c.sh_rest is not None], axis=0)
                if has_rest
                else None,
            )
        else:
            baked_cloud = GaussianCloud(
                means=np.zeros((0, 3), dtype=np.float32),
                scales=np.zeros((0, 3), dtype=np.float32),
                quats=np.zeros((0, 4), dtype=np.float32),
                opacities=np.zeros((0,), dtype=np.float32),
                sh_dc=np.zeros((0, 3), dtype=np.float32),
                sh_rest=np.zeros((0, cloud.sh_rest.shape[1]), dtype=np.float32) if cloud.sh_rest is not None else None,
            )

        edited_path = mgr.edited_splats_path
        mgr._atomic_save_ply(edited_path, baked_cloud)
        ws.set_active_splat(edited_path)

        state["baked_dirty"] = False
        atomic_write_json(mgr.objects_json_path, state)

        manifest_out = ws.processed_dir / "segments_manifest.json"
        atomic_write_json(
            manifest_out,
            {
                "status": "baked",
                "active_splat": ws.to_relative(edited_path),
                "total_points": baked_cloud.num_points,
                "kept_objects": kept_objects,
                "deleted_objects": deleted_objects,
                "objects": [SceneObject(**o).to_public_dict() for o in objects],
            },
        )

        exported_ply_rel: str | None = None
        if reexport_ply:
            exp_res = Ply3DGSExporterStage().run(ws, {"scene_name": ws.root.name or "scene_3dgs"})
            exported_ply_rel = exp_res.artifacts.get("scene_3dgs")

        return {
            "baked": True,
            "edited_splats": ws.to_relative(edited_path),
            "exported_3dgs_ply": exported_ply_rel,
            "total_points": baked_cloud.num_points,
            "kept_objects": kept_objects,
            "deleted_objects": deleted_objects,
            **mgr.public_summary(),
        }
