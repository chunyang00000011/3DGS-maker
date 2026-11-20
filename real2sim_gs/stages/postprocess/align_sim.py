from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace, atomic_write_json
from real2sim_gs.utils.ply_io import (
    GaussianCloud,
    _quats_wxyz_to_matrices,
    load_gaussian_ply,
    save_gaussian_ply,
    transform_gaussian_cloud,
)

_RANSAC_SCORE_SUBSAMPLE = 50_000


def _atomic_save_gaussian_ply(path: Path, cloud: GaussianCloud) -> Path:
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
    return path


def _active_cameras_path(ws: SceneWorkspace) -> Path:
    refined = ws.gaussians_dir / "refined_cameras.json"
    return refined if refined.is_file() else ws.cameras_path


def _extract_camera_axes(ws: SceneWorkspace) -> dict[str, Any]:
    """Extract gravity direction, right/forward vectors, and camera centers from extrinsics.

    When multiple cameras span yaw (>= ~8 deg), camera right axes ``r[0, :]`` lie in the horizontal
    ground plane regardless of downward camera pitch, so their smallest singular vector recovers exact
    gravity without pitch bias.
    """
    empty: dict[str, Any] = {
        "cam_up": None,
        "right_hint": None,
        "fwd_mean": None,
        "fwd_first": None,
        "cam_rights": None,
        "cam_centers": None,
        "has_multi_yaw": False,
    }
    cam_path = _active_cameras_path(ws)
    if not cam_path.is_file():
        return empty
    try:
        payload = json.loads(cam_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty

    right_vecs: list[np.ndarray] = []
    up_vecs: list[np.ndarray] = []
    fwd_vecs: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    for frame in payload.get("frames", []):
        w2c = np.asarray(frame.get("world_to_cam", []), dtype=np.float64)
        if w2c.shape == (4, 4):
            r = w2c[:3, :3]
            right_vecs.append(r[0, :])  # OpenCV +X points right (zero-roll horizontal)
            up_vecs.append(-r[1, :])    # OpenCV -Y points toward camera top
            fwd_vecs.append(r[2, :])    # OpenCV +Z points along camera optical axis
            centers.append(-r.T @ w2c[:3, 3])
    if not up_vecs:
        return empty

    up_mean = np.mean(up_vecs, axis=0)
    up_norm = float(np.linalg.norm(up_mean))
    up_mean_unit = (up_mean / up_norm) if up_norm > 1e-4 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    cam_rights = np.stack(right_vecs, axis=0)
    right_hint: np.ndarray | None = None
    cam_up: np.ndarray | None = None
    has_multi_yaw = False
    if len(right_vecs) >= 2:
        _, svals, vt = np.linalg.svd(cam_rights, full_matrices=False)
        if float(svals[0]) > 1e-6 and float(svals[1]) / float(svals[0]) >= 0.12:
            has_multi_yaw = True
            g_up = vt[-1].copy()
            if float(np.dot(g_up, up_mean_unit)) < 0.0:
                g_up = -g_up
            cam_up = (g_up / (np.linalg.norm(g_up) + 1e-8)).astype(np.float32)
        else:
            cam_up = up_mean_unit.astype(np.float32) if up_norm > 1e-4 else None
            r_mean = np.mean(right_vecs, axis=0)
            r_norm = float(np.linalg.norm(r_mean))
            if r_norm > 1e-4:
                right_hint = (r_mean / r_norm).astype(np.float32)
    else:
        cam_up = up_mean_unit.astype(np.float32) if up_norm > 1e-4 else None
        r_norm = float(np.linalg.norm(right_vecs[0]))
        if r_norm > 1e-4:
            right_hint = (right_vecs[0] / r_norm).astype(np.float32)

    return {
        "cam_up": cam_up,
        "right_hint": right_hint,
        "fwd_mean": np.mean(fwd_vecs, axis=0).astype(np.float32),
        "fwd_first": fwd_vecs[0].astype(np.float32),
        "cam_rights": cam_rights,
        "cam_centers": np.stack(centers, axis=0),
        "has_multi_yaw": has_multi_yaw,
    }


def _extract_camera_gravity_and_heading(
    ws: SceneWorkspace,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Infer world up-against-gravity, horizontal right hint, and front viewing axes from camera extrinsics."""
    info = _extract_camera_axes(ws)
    return info["cam_up"], info["right_hint"], info["fwd_mean"], info["fwd_first"]


def _extract_camera_centers(ws: SceneWorkspace) -> np.ndarray | None:
    """Return (C, 3) raw-space camera centers from refined_cameras.json or cameras.json."""
    return _extract_camera_axes(ws)["cam_centers"]


def _fit_ground_plane_ransac(
    points: np.ndarray,
    trials: int = 200,
    dist_threshold: float = 0.03,
    seed: int = 42,
    up_prior: np.ndarray | None = None,
    min_up_cos: float = 0.68,
    horiz_axis: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find dominant near-horizontal planar surface via gravity-gated & optional area-weighted RANSAC."""
    n = points.shape[0]
    fallback_normal = (
        (up_prior / (np.linalg.norm(up_prior) + 1e-8)).astype(np.float32)
        if up_prior is not None
        else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    if n < 3:
        return fallback_normal, np.ones(n, dtype=bool)

    pts_f64 = np.asarray(points, dtype=np.float64)
    w_f64 = (
        np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-12)
        if weights is not None and len(weights) == n
        else None
    )
    up_unit = fallback_normal.astype(np.float64) if up_prior is not None else None
    horiz_unit = (
        (horiz_axis.astype(np.float64) / (np.linalg.norm(horiz_axis) + 1e-8))
        if horiz_axis is not None
        else None
    )

    rng = np.random.default_rng(seed)
    if n > _RANSAC_SCORE_SUBSAMPLE:
        sub_idx = rng.choice(n, size=_RANSAC_SCORE_SUBSAMPLE, replace=False)
        score_pts = pts_f64[sub_idx]
        w_sub = w_f64[sub_idx] if w_f64 is not None else None
    else:
        score_pts = pts_f64
        w_sub = w_f64
    m = score_pts.shape[0]
    probs_sub = (w_sub / float(w_sub.sum())) if (w_sub is not None and float(w_sub.sum()) > 0.0) else None

    best_normal_f64: np.ndarray | None = None
    best_offset = 0.0
    best_sub_score = -1.0

    for _ in range(max(1, int(trials))):
        idx = rng.choice(m, size=3, replace=False, p=probs_sub) if probs_sub is not None else rng.choice(m, size=3, replace=False)
        p0, p1, p2 = score_pts[idx[0]], score_pts[idx[1]], score_pts[idx[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = float(np.linalg.norm(normal))
        if norm_len < 1e-6:
            continue
        normal = normal / norm_len
        if up_unit is not None:
            if float(np.dot(normal, up_unit)) < 0.0:
                normal = -normal
            if float(np.dot(normal, up_unit)) < min_up_cos:
                continue
        elif normal[2] < 0.0:
            normal = -normal

        if horiz_unit is not None and abs(float(np.dot(normal, horiz_unit))) > 0.35:
            continue

        d = float(np.dot(p0, normal))
        sub_mask = np.abs(score_pts @ normal - d) <= dist_threshold
        score = float(w_sub[sub_mask].sum()) if w_sub is not None else float(np.count_nonzero(sub_mask))
        if score > best_sub_score:
            best_sub_score = score
            best_normal_f64 = normal
            best_offset = d

    if best_normal_f64 is not None:
        best_inliers = np.abs(pts_f64 @ best_normal_f64 - best_offset) <= dist_threshold
        best_count = int(np.count_nonzero(best_inliers))
        best_normal = best_normal_f64.astype(np.float32)
    else:
        best_inliers = np.zeros(n, dtype=bool)
        best_count = 0
        best_normal = fallback_normal.copy()

    if w_f64 is not None:
        total_w = max(float(w_f64.sum()), 1e-12)
        inl_w = float(w_f64[best_inliers].sum()) if best_count > 0 else 0.0
        has_support = (best_count >= 16 and inl_w >= 0.03 * total_w) if (up_unit is not None and n >= 200) else (best_count >= 3)
    else:
        min_required = max(32, int(0.08 * n)) if (up_unit is not None and n >= 200) else 3
        has_support = best_count >= min_required

    if has_support:
        inlier_pts = pts_f64[best_inliers]
        if w_f64 is not None:
            w_inl = w_f64[best_inliers]
            w_norm = w_inl / max(float(w_inl.mean()), 1e-12)
            center = np.average(inlier_pts, axis=0, weights=w_norm)
            centered = (inlier_pts - center[None, :]) * np.sqrt(w_norm)[:, None]
        else:
            centered = inlier_pts - inlier_pts.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        refined = vt[-1]
        if up_unit is not None:
            if float(np.dot(refined, up_unit)) < 0.0:
                refined = -refined
        elif refined[2] < 0.0:
            refined = -refined
        refined = refined / (np.linalg.norm(refined) + 1e-8)
        horiz_ok = horiz_unit is None or abs(float(np.dot(refined, horiz_unit))) <= 0.45
        if horiz_ok and (up_unit is None or float(np.dot(refined, up_unit)) >= min_up_cos * 0.9):
            best_normal = refined.astype(np.float32)
    elif up_unit is not None:
        proj = pts_f64 @ up_unit
        z_floor = float(np.percentile(proj, 10))
        best_normal = up_unit.astype(np.float32)
        best_inliers = np.abs(proj - z_floor) <= dist_threshold * 1.5
        if not best_inliers.any():
            best_inliers = proj <= np.percentile(proj, 15)
    else:
        best_inliers = np.ones(n, dtype=bool)

    return best_normal, best_inliers


def _rotation_matrix_to_z_up(normal: np.ndarray) -> np.ndarray:
    """Compute Rodrigues rotation matrix mapping unit vector `normal` to [0, 0, 1]."""
    src = normal / (np.linalg.norm(normal) + 1e-8)
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = np.cross(src.astype(np.float64), target)
    c = float(np.dot(src.astype(np.float64), target))
    if np.linalg.norm(v) < 1e-7:
        return np.eye(3, dtype=np.float32) if c > 0.0 else np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    rot = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * (1.0 / (1.0 + c))
    return rot.astype(np.float32)


def _orient_normal_upward(
    normal: np.ndarray,
    points: np.ndarray,
    inliers: np.ndarray,
    ws: SceneWorkspace,
) -> np.ndarray:
    """Resolve RANSAC plane sign ambiguity by preferring the half-space containing cameras."""
    if points.shape[0] == 0:
        return normal
    p_ground = points[inliers].mean(axis=0) if inliers.any() else points.mean(axis=0)

    cam_centers = _extract_camera_centers(ws)
    if cam_centers is not None and len(cam_centers) > 0:
        mean_signed = float(((cam_centers - p_ground) @ normal).mean())
        if abs(mean_signed) > 1e-6:
            return (-normal if mean_signed < 0.0 else normal).astype(np.float32)

    outliers = ~inliers
    if int(outliers.sum()) >= 8:
        med_signed = float(np.median((points[outliers] - p_ground) @ normal))
        if abs(med_signed) > 1e-6:
            return (-normal if med_signed < 0.0 else normal).astype(np.float32)

    return normal.astype(np.float32)


def _extract_flat_surfels(
    cloud: GaussianCloud,
    *,
    max_samples: int = 80_000,
    min_opacity: float = 0.25,
    max_flat_ratio: float = 0.42,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Sample opaque flat 3DGS surfels, extracting their shortest-axis unit normals and area weights."""
    if cloud.num_points < 60:
        return None
    op = cloud.sigmoid_opacities()
    scales = np.exp(np.clip(cloud.scales.astype(np.float64), -10.0, 2.0))
    order = np.argsort(scales, axis=1)
    ar = np.arange(scales.shape[0])
    s_min = scales[ar, order[:, 0]]
    s_mid = scales[ar, order[:, 1]]
    s_max = scales[ar, order[:, 2]]

    flat_mask = (op >= min_opacity) & (s_min / np.maximum(s_mid, 1e-8) <= max_flat_ratio)
    flat_idx = np.nonzero(flat_mask)[0]
    if len(flat_idx) < 60:
        return None

    if len(flat_idx) > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(flat_idx, size=max_samples, replace=False))
    else:
        idx = flat_idx

    rots = _quats_wxyz_to_matrices(cloud.quats[idx])
    min_axes = order[idx, 0]
    normals = rots[np.arange(len(idx)), :, min_axes].astype(np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    raw_a = (s_mid[idx] * s_max[idx]).astype(np.float64)
    lo_a, hi_a = np.percentile(raw_a, [10, 98])
    areas = np.clip(raw_a, max(float(lo_a), 1e-10), max(float(hi_a), 1e-8))
    areas /= max(float(areas.mean()), 1e-12)
    return cloud.means[idx].astype(np.float64), normals, areas


def _estimate_gravity_surfel_irls(
    normals: np.ndarray,
    areas: np.ndarray,
    up_prior: np.ndarray,
    cam_rights: np.ndarray | None,
    has_multi_yaw: bool,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Refine world-up gravity via Cauchy-IRLS over horizontal surfels, gated dual walls, and camera horizon."""
    up_p = np.asarray(up_prior, dtype=np.float64)
    up_p = up_p / max(float(np.linalg.norm(up_p)), 1e-12)

    h0 = np.abs(normals @ up_p) >= np.cos(np.deg2rad(20.0))
    horiz_count = int(np.count_nonzero(h0))
    if horiz_count < 40:
        return None

    # Dual-orthogonal wall gate: reject single tilted plane (e.g. angled monitor screen) from dragging gravity.
    v0 = np.abs(normals @ up_p) < np.sin(np.deg2rad(15.0))
    use_walls = False
    wall_sv_ratio = 0.0
    wall_ang_deg = 999.0
    if int(np.count_nonzero(v0)) >= 100:
        nw0 = normals[v0] * np.sqrt(areas[v0])[:, None]
        _, sv_w, vt_w = np.linalg.svd(nw0, full_matrices=False)
        g_w = vt_w[-1].copy()
        if float(np.dot(g_w, up_p)) < 0.0:
            g_w = -g_w
        g_w /= max(float(np.linalg.norm(g_w)), 1e-12)
        wall_sv_ratio = float(sv_w[1] / max(float(sv_w[0]), 1e-8))
        wall_thin_ratio = float(sv_w[2] / max(float(sv_w[1]), 1e-8))
        wall_ang_deg = float(np.rad2deg(np.arccos(np.clip(float(np.dot(g_w, up_p)), -1.0, 1.0))))
        max_wall_dev = 7.0 if has_multi_yaw else 18.0
        use_walls = bool(wall_sv_ratio >= 0.45 and wall_thin_ratio <= 0.30 and wall_ang_deg <= max_wall_dev)

    g = up_p.copy()
    cos_h = np.cos(np.deg2rad(16.0))
    sin_v = np.sin(np.deg2rad(12.0))
    cos_cone = np.cos(np.deg2rad(10.0))
    sin_cone = np.sin(np.deg2rad(10.0))
    eye3 = np.eye(3, dtype=np.float64)

    for _ in range(8):
        mat = np.zeros((3, 3), dtype=np.float64)

        hm = np.abs(normals @ g) > cos_h
        if int(np.count_nonzero(hm)) >= 30:
            nh = normals[hm]
            wh = areas[hm]
            dots = np.clip(np.abs(nh @ g), 0.0, 1.0)
            sin2 = np.maximum(0.0, 1.0 - dots * dots)
            rw = wh / (sin2 + 0.04 * 0.04)
            rw /= max(float(rw.sum()), 1e-12)
            mat += 1.6 * (eye3 - (nh * rw[:, None]).T @ nh)

        if use_walls:
            vm = np.abs(normals @ g) < sin_v
            if int(np.count_nonzero(vm)) >= 80:
                nv = normals[vm]
                wv = areas[vm]
                dotv = nv @ g
                rwv = wv / (dotv * dotv + 0.04 * 0.04)
                rwv /= max(float(rwv.sum()), 1e-12)
                mat += 1.6 * ((nv * rwv[:, None]).T @ nv)

        if has_multi_yaw and cam_rights is not None and len(cam_rights) >= 2:
            rd = cam_rights @ g
            rwr = 1.0 / (rd * rd + 0.05 * 0.05)
            rwr /= max(float(rwr.sum()), 1e-12)
            w_cam = 0.8 if use_walls else 2.2
            mat += w_cam * ((cam_rights * rwr[:, None]).T @ cam_rights)

        if float(np.linalg.norm(mat)) < 1e-10:
            break
        _, eigvecs = np.linalg.eigh(mat)
        g_new = eigvecs[:, 0].copy()
        if float(np.dot(g_new, up_p)) < 0.0:
            g_new = -g_new
        g_new /= max(float(np.linalg.norm(g_new)), 1e-12)

        if has_multi_yaw:
            dot_p = float(np.dot(g_new, up_p))
            if dot_p < cos_cone:
                perp = g_new - dot_p * up_p
                p_len = float(np.linalg.norm(perp))
                if p_len > 1e-8:
                    g_new = cos_cone * up_p + sin_cone * (perp / p_len)
                else:
                    g_new = up_p.copy()
                g_new /= max(float(np.linalg.norm(g_new)), 1e-12)

        g = g_new

    return g.astype(np.float32), {
        "use_walls": use_walls,
        "wall_sv_ratio": round(wall_sv_ratio, 4),
        "wall_ang_deg": round(wall_ang_deg, 2),
        "horiz_surfel_count": horiz_count,
    }


def _find_lowest_floor_height(
    pts_f: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    g: np.ndarray,
    cam_centers: np.ndarray | None,
    core_span: float,
) -> float | None:
    """Locate exact Z=0 height from the lowest substantial horizontal surfel peak below camera height."""
    g64 = np.asarray(g, dtype=np.float64)
    g64 = g64 / max(float(np.linalg.norm(g64)), 1e-12)

    hm = np.abs(normals @ g64) >= np.cos(np.deg2rad(15.0))
    if int(np.count_nonzero(hm)) < 30:
        return None
    hz = pts_f[hm] @ g64
    hw = areas[hm]

    if cam_centers is not None and len(cam_centers) > 0:
        cam_h = float(np.median(cam_centers @ g64))
        below = hz <= cam_h - 0.05
        if int(np.count_nonzero(below)) >= 30:
            hz = hz[below]
            hw = hw[below]

    slab_tol = float(np.clip(core_span * 0.005, 0.015, 0.045))
    z_lo, z_hi = float(np.percentile(hz, 1)), float(np.percentile(hz, 99))
    if z_hi - z_lo < 1e-4:
        return float(np.median(hz))

    n_bins = int(np.clip(round((z_hi - z_lo) / slab_tol), 12, 240))
    hist, edges = np.histogram(hz, bins=n_bins, range=(z_lo, z_hi), weights=hw)
    cnt, _ = np.histogram(hz, bins=n_bins, range=(z_lo, z_hi))
    hist_sm = np.convolve(hist, [0.25, 0.5, 0.25], mode="same")
    cnt_3 = np.convolve(cnt, [1, 1, 1], mode="same")

    cands = np.nonzero((hist_sm >= 0.22 * max(float(hist_sm.max()), 1e-12)) & (cnt_3 >= 25))[0]
    b_floor = int(cands[0]) if len(cands) > 0 else int(np.argmax(hist_sm))
    z_center = 0.5 * float(edges[b_floor] + edges[b_floor + 1])

    slab = np.abs(hz - z_center) <= slab_tol * 1.5
    if int(np.count_nonzero(slab)) >= 10:
        return float(np.average(hz[slab], weights=hw[slab]))
    return z_center


def _snap_heading_manhattan(
    normals: np.ndarray,
    areas: np.ndarray,
    g: np.ndarray,
    cam_fwd_mean: np.ndarray | None,
    cam_fwd_first: np.ndarray | None,
    *,
    min_peak_share: float = 0.26,
) -> tuple[np.ndarray | None, float, float]:
    """Align canonical horizontal heading to dominant 4-fold Manhattan wall orientations near primary view."""
    z_ax = np.asarray(g, dtype=np.float64)
    z_ax = z_ax / max(float(np.linalg.norm(z_ax)), 1e-12)

    fwd_p: np.ndarray | None = None
    for cand, min_norm in ((cam_fwd_mean, 0.25), (cam_fwd_first, 0.12)):
        if cand is None:
            continue
        proj = np.asarray(cand, dtype=np.float64) - float(np.dot(cand, z_ax)) * z_ax
        p_norm = float(np.linalg.norm(proj))
        if p_norm >= min_norm:
            fwd_p = proj / p_norm
            break
    if fwd_p is None:
        return None, 0.0, 0.0

    u0 = fwd_p
    v0 = np.cross(z_ax, u0)
    v0_norm = float(np.linalg.norm(v0))
    if v0_norm < 1e-6:
        return fwd_p, 0.0, 0.0
    v0 /= v0_norm

    vm = np.abs(normals @ z_ax) < np.sin(np.deg2rad(18.0))
    if int(np.count_nonzero(vm)) < 100:
        return fwd_p, 0.0, 0.0

    nv = normals[vm]
    wv = areas[vm]
    nx = nv @ u0
    ny = nv @ v0
    xy_mask = np.hypot(nx, ny) > 0.5
    if int(np.count_nonzero(xy_mask)) < 80:
        return fwd_p, 0.0, 0.0

    nx = nx[xy_mask]
    ny = ny[xy_mask]
    wv = wv[xy_mask]
    theta = np.arctan2(ny, nx)
    ang_deg = np.rad2deg(theta) % 90.0

    hist, _ = np.histogram(ang_deg, bins=90, range=(0.0, 90.0), weights=wv)
    hist_sm = (
        0.1 * np.roll(hist, -2)
        + 0.2 * np.roll(hist, -1)
        + 0.4 * hist
        + 0.2 * np.roll(hist, 1)
        + 0.1 * np.roll(hist, 2)
    )
    p_bin = int(np.argmax(hist_sm))
    p_center = p_bin + 0.5
    diff = np.abs(((ang_deg - p_center + 45.0) % 90.0) - 45.0)
    in_peak = diff <= 8.0
    peak_share = float(wv[in_peak].sum() / max(float(wv.sum()), 1e-12))
    if peak_share < min_peak_share or int(np.count_nonzero(in_peak)) < 20:
        return fwd_p, round(peak_share, 4), 0.0

    th_p = theta[in_peak]
    w_p = wv[in_peak]
    ang_man = 0.25 * float(np.arctan2(np.sum(w_p * np.sin(4.0 * th_p)), np.sum(w_p * np.cos(4.0 * th_p))))

    cands = [
        np.cos(ang_man + k * (np.pi * 0.5)) * u0 + np.sin(ang_man + k * (np.pi * 0.5)) * v0
        for k in range(4)
    ]
    best_cand = max(cands, key=lambda c: float(np.dot(c, fwd_p)))
    best_cand /= max(float(np.linalg.norm(best_cand)), 1e-12)
    snap_deg = float(np.rad2deg(np.arccos(np.clip(float(np.dot(best_cand, fwd_p)), -1.0, 1.0))))
    return best_cand, round(peak_share, 4), round(snap_deg, 2)


def _build_upright_front_rotation(
    up_normal: np.ndarray,
    cam_fwd_mean: np.ndarray | None,
    cam_fwd_first: np.ndarray | None,
    *,
    horiz_override: np.ndarray | None = None,
) -> np.ndarray:
    """Build right-handed SO(3) mapping `up_normal`->+Z and primary camera facing->-X (+X side in front)."""
    z_axis = np.asarray(up_normal, dtype=np.float64)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    horiz: np.ndarray | None = None
    if horiz_override is not None:
        proj = np.asarray(horiz_override, dtype=np.float64) - float(np.dot(horiz_override, z_axis)) * z_axis
        if float(np.linalg.norm(proj)) >= 1e-4:
            horiz = proj / float(np.linalg.norm(proj))

    if horiz is None:
        for cand in (cam_fwd_mean, cam_fwd_first):
            if cand is None:
                continue
            proj = np.asarray(cand, dtype=np.float64) - float(np.dot(cand, z_axis)) * z_axis
            if float(np.linalg.norm(proj)) >= 0.12:
                horiz = proj / float(np.linalg.norm(proj))
                break

    if horiz is None:
        return _rotation_matrix_to_z_up(up_normal)

    x_axis = -horiz
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-6:
        return _rotation_matrix_to_z_up(up_normal)
    y_axis /= y_norm
    return np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)


@register_stage("postprocess", "align_gravity_scale")
class AlignGravityScaleStage(BaseStage):
    """Align reconstructed ground plane to +Z up (z=0), front heading to +X->-X, and metric scale."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        src_override = params.get("source_ply_path")
        src_path = Path(src_override) if src_override else ws.active_splat_path()
        cloud = load_gaussian_ply(src_path)
        scale_factor = float(params.get("scale_factor", 1.0))
        if scale_factor <= 0.0:
            raise ValueError("scale_factor must be strictly positive.")

        trials = max(16, int(params.get("ransac_trials", 200)))
        dist_thresh = float(params.get("distance_threshold", 0.03))
        seed = int(params.get("seed", 42))
        use_surfel_alignment = bool(params.get("use_surfel_alignment", True))
        manhattan_yaw = bool(params.get("manhattan_yaw", True))
        extra_roll_deg = float(params.get("extra_roll_deg", 0.0))
        extra_pitch_deg = float(params.get("extra_pitch_deg", 0.0))
        extra_yaw_deg = float(params.get("extra_yaw_deg", 0.0))
        extra_z_offset = float(params.get("extra_z_offset", 0.0))

        bbox_span = (
            float(np.linalg.norm(np.percentile(cloud.means, 95, axis=0) - np.percentile(cloud.means, 5, axis=0)))
            if cloud.num_points >= 8
            else 1.0
        )
        eff_dist_thresh = dist_thresh * max(1.0, bbox_span * 0.15)

        cam_info = _extract_camera_axes(ws)
        cam_up: np.ndarray | None = cam_info["cam_up"]
        right_hint: np.ndarray | None = cam_info["right_hint"]
        cam_fwd_mean: np.ndarray | None = cam_info["fwd_mean"]
        cam_fwd_first: np.ndarray | None = cam_info["fwd_first"]
        cam_rights: np.ndarray | None = cam_info["cam_rights"]
        cam_centers: np.ndarray | None = cam_info["cam_centers"]
        has_multi_yaw = bool(cam_info["has_multi_yaw"])
        eff_min_up_cos = 0.92 if (cam_up is not None and right_hint is None) else 0.68

        areas: np.ndarray | None = None
        if cloud.num_points >= 16:
            s_sorted = np.sort(np.exp(np.clip(cloud.scales, -10.0, 2.0)), axis=1)
            raw_area = (s_sorted[:, 2] * s_sorted[:, 1]).astype(np.float64)
            lo_a, hi_a = np.percentile(raw_area, [10, 98])
            areas = np.clip(raw_area, max(float(lo_a), 1e-10), max(float(hi_a), 1e-8))

        alignment_method = "ransac_fallback"
        horiz_override: np.ndarray | None = None
        peak_share = 0.0
        snap_deg = 0.0
        irls_meta: dict[str, Any] = {}
        z_floor_surfel: float | None = None
        used_surfel = False

        if use_surfel_alignment and cam_up is not None:
            surfels = _extract_flat_surfels(cloud, seed=seed)
            if surfels is not None:
                pts_f, surfel_normals, surfel_areas = surfels
                irls_res = _estimate_gravity_surfel_irls(
                    surfel_normals,
                    surfel_areas,
                    cam_up,
                    cam_rights,
                    has_multi_yaw,
                )
                if irls_res is not None:
                    g_irls, irls_meta = irls_res
                    z_floor_surfel = _find_lowest_floor_height(
                        pts_f,
                        surfel_normals,
                        surfel_areas,
                        g_irls,
                        cam_centers,
                        bbox_span,
                    )
                    if z_floor_surfel is not None:
                        used_surfel = True
                        normal = g_irls
                        slab_tol = float(np.clip(bbox_span * 0.005, 0.015, 0.045))
                        inliers = np.abs(cloud.means.astype(np.float64) @ normal.astype(np.float64) - z_floor_surfel) <= max(
                            eff_dist_thresh, slab_tol * 2.0
                        )
                        if manhattan_yaw:
                            horiz_override, peak_share, snap_deg = _snap_heading_manhattan(
                                surfel_normals,
                                surfel_areas,
                                normal,
                                cam_fwd_mean,
                                cam_fwd_first,
                            )
                        alignment_method = "surfel_irls_manhattan" if snap_deg > 0.0 else "surfel_irls"

        if not used_surfel:
            used_lower_band = False
            if cam_up is not None and cloud.num_points >= 120:
                up_unit = cam_up.astype(np.float64) / (float(np.linalg.norm(cam_up)) + 1e-8)
                h_proj = cloud.means.astype(np.float64) @ up_unit
                h_cut = float(np.percentile(h_proj, 48))
                if cam_centers is not None and len(cam_centers) >= 2:
                    cam_h_med = float(np.median(cam_centers @ up_unit))
                    if cam_h_med > float(np.percentile(h_proj, 15)):
                        h_cut = min(h_cut, cam_h_med)
                cand_mask = h_proj <= h_cut
                if int(cand_mask.sum()) >= 48:
                    used_lower_band = True
                    normal, cand_inliers = _fit_ground_plane_ransac(
                        cloud.means[cand_mask],
                        trials=trials,
                        dist_threshold=eff_dist_thresh,
                        seed=seed,
                        up_prior=cam_up,
                        min_up_cos=eff_min_up_cos,
                        horiz_axis=right_hint,
                        weights=areas[cand_mask] if areas is not None else None,
                    )
                    d_floor = (
                        float(np.median(cloud.means[cand_mask][cand_inliers] @ normal))
                        if cand_inliers.any()
                        else float(np.percentile(cloud.means @ normal, 10))
                    )
                    inliers = np.abs(cloud.means @ normal - d_floor) <= eff_dist_thresh
                    if int(inliers.sum()) < 8:
                        cand_indices = np.nonzero(cand_mask)[0]
                        inliers = np.zeros(cloud.num_points, dtype=bool)
                        inliers[cand_indices[cand_inliers]] = True

            if not used_lower_band:
                normal, inliers = _fit_ground_plane_ransac(
                    cloud.means,
                    trials=trials,
                    dist_threshold=eff_dist_thresh,
                    seed=seed,
                    up_prior=cam_up,
                    min_up_cos=eff_min_up_cos,
                    horiz_axis=right_hint,
                    weights=areas,
                )

            normal = _orient_normal_upward(normal, cloud.means, inliers, ws)
            if cam_up is not None and float(np.dot(normal, cam_up)) < 0.0:
                normal = (-normal).astype(np.float32)

        rot_mat = _build_upright_front_rotation(
            normal,
            cam_fwd_mean,
            cam_fwd_first,
            horiz_override=horiz_override,
        )

        means_rot = cloud.means @ rot_mat.T
        xy_offset = (
            np.median(means_rot[:, :2], axis=0).astype(np.float32)
            if means_rot.shape[0] > 0
            else np.zeros(2, dtype=np.float32)
        )
        z_ground = (
            float(z_floor_surfel)
            if used_surfel and z_floor_surfel is not None
            else (float(np.median(means_rot[inliers, 2])) if means_rot.shape[0] > 0 else 0.0)
        )
        offset_vec = np.array([xy_offset[0], xy_offset[1], z_ground], dtype=np.float32)

        has_extra = any(abs(v) > 1e-6 for v in (extra_roll_deg, extra_pitch_deg, extra_yaw_deg, extra_z_offset))
        if has_extra:
            r_extra = Rotation.from_euler(
                "xyz", [extra_roll_deg, extra_pitch_deg, extra_yaw_deg], degrees=True
            ).as_matrix().astype(np.float64)
            rot_final = (r_extra @ rot_mat.astype(np.float64)).astype(np.float32)
            trans_final = (
                r_extra @ (-offset_vec.astype(np.float64) * scale_factor)
                + np.array([0.0, 0.0, extra_z_offset], dtype=np.float64)
            ).astype(np.float32)
        else:
            rot_final = rot_mat
            trans_final = (-offset_vec * scale_factor).astype(np.float32)

        aligned_cloud = transform_gaussian_cloud(
            cloud,
            rot_final,
            translation=trans_final,
            scale=scale_factor,
        )
        _atomic_save_gaussian_ply(ws.aligned_splats_path, aligned_cloud)
        ws.set_active_splat(ws.aligned_splats_path)

        aligned_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if used_surfel and abs(extra_roll_deg) < 1e-6 and abs(extra_pitch_deg) < 1e-6:
            aligned_surfels = _extract_flat_surfels(aligned_cloud, max_samples=40_000, seed=seed)
            if aligned_surfels is not None:
                a_pts, a_norms, a_areas = aligned_surfels
                slab_tol_s = float(np.clip(bbox_span * scale_factor * 0.005, 0.015, 0.045))
                floor_s = (np.abs(a_pts[:, 2]) <= slab_tol_s * 1.5) & (
                    np.abs(a_norms[:, 2]) >= np.cos(np.deg2rad(15.0))
                )
                if int(np.count_nonzero(floor_s)) >= 16:
                    fn = a_norms[floor_s].copy()
                    fn[fn[:, 2] < 0.0] *= -1.0
                    mean_fn = np.average(fn, axis=0, weights=a_areas[floor_s])
                    mean_fn /= max(float(np.linalg.norm(mean_fn)), 1e-8)
                    aligned_normal = mean_fn.astype(np.float32)
        else:
            floor_verify_pts = aligned_cloud.means[inliers] if int(inliers.sum()) >= 16 else aligned_cloud.means
            aligned_normal, _ = _fit_ground_plane_ransac(
                floor_verify_pts,
                trials=min(160, trials),
                dist_threshold=eff_dist_thresh * scale_factor,
                seed=seed,
                up_prior=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                weights=areas[inliers] if (areas is not None and int(inliers.sum()) >= 16) else None,
            )

        transform_4x4 = np.eye(4, dtype=np.float32)
        transform_4x4[:3, :3] = rot_final * scale_factor
        transform_4x4[:3, 3] = trans_final

        transform_path = ws.processed_dir / "alignment_transform.json"
        transform_payload = {
            "detected_normal": normal.tolist(),
            "aligned_normal": aligned_normal.tolist(),
            "ground_z_offset": z_ground,
            "xy_offset": xy_offset.tolist(),
            "scale_factor": scale_factor,
            "inlier_ratio": round(float(inliers.mean()) if len(inliers) else 0.0, 4),
            "alignment_method": alignment_method,
            "manhattan_snapped_deg": snap_deg,
            "manhattan_peak_share": peak_share,
            "used_dual_walls": bool(irls_meta.get("use_walls", False)),
            "matrix_4x4": transform_4x4.tolist(),
        }
        atomic_write_json(transform_path, transform_payload)

        return StageResult(
            artifacts={
                "aligned_splats": ws.to_relative(ws.aligned_splats_path),
                "alignment_transform": ws.to_relative(transform_path),
            },
            metrics={
                "aligned_normal_z": round(float(aligned_normal[2]), 4),
                "scale_factor": scale_factor,
                "ground_inlier_ratio": transform_payload["inlier_ratio"],
                "alignment_method": alignment_method,
                "manhattan_snapped_deg": snap_deg,
            },
            summary=(
                f"Aligned ground plane to +Z ({alignment_method}, normal_z={aligned_normal[2]:.4f}, "
                f"manhattan_snap={snap_deg:.2f}deg, scale={scale_factor})."
            ),
        )


def realign_workspace_scene(
    ws: SceneWorkspace,
    *,
    mode: str = "auto",
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    z_offset: float = 0.0,
    current_yaw_rad: float | None = None,
    reexport_ply: bool = True,
) -> dict[str, Any]:
    """Re-run SOTA alignment or apply incremental roll/pitch/yaw/height calibration in Studio."""
    eff_mode = str(mode or "auto").strip().lower()
    if eff_mode not in ("auto", "set_front", "manual"):
        raise ValueError(f"Unsupported alignment mode '{mode}'. Expected 'auto', 'set_front', or 'manual'.")

    seg_dir = ws.processed_dir / "segments"
    obj_json = seg_dir / "scene_objects.json"
    obj_ids_npy = seg_dir / "object_ids.npy"
    if obj_json.is_file() and obj_ids_npy.is_file():
        try:
            seg_state = json.loads(obj_json.read_text(encoding="utf-8"))
            if seg_state.get("objects"):
                raise ValueError(
                    "当前工作区存在已分割的物体场景图，调整全局水平/正方向会导致物体包围盒失步；"
                    "请先完成方向校准再分割物体，或清空分割后重试。"
                )
        except ValueError:
            raise
        except OSError:
            pass

    transform_path = ws.processed_dir / "alignment_transform.json"
    prev_payload: dict[str, Any] = {}
    if transform_path.is_file():
        try:
            loaded = json.loads(transform_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                prev_payload = loaded
        except (OSError, ValueError):
            prev_payload = {}

    if eff_mode == "auto":
        scale_factor = float(prev_payload.get("scale_factor", 1.0) or 1.0)
        try:
            active_resolved = ws.active_splat_path().resolve()
        except FileNotFoundError:
            active_resolved = None
        if ws.cleaned_splats_path.is_file() and (
            not transform_path.is_file() or active_resolved != ws.cleaned_splats_path.resolve()
        ):
            src_ply = ws.cleaned_splats_path
        elif ws.raw_splats_path.is_file():
            src_ply = ws.raw_splats_path
        else:
            src_ply = ws.active_splat_path()

        AlignGravityScaleStage().run(
            ws,
            {
                "scale_factor": scale_factor,
                "source_ply_path": str(src_ply),
                "use_surfel_alignment": True,
                "manhattan_yaw": True,
            },
        )
    else:
        if eff_mode == "set_front":
            if current_yaw_rad is None:
                raise ValueError("current_yaw_rad is required when mode='set_front'.")
            eff_roll = 0.0
            eff_pitch = 0.0
            eff_yaw = -float(np.rad2deg(float(current_yaw_rad)))
            eff_dz = 0.0
        else:
            eff_roll = float(roll_deg)
            eff_pitch = float(pitch_deg)
            eff_yaw = float(yaw_deg)
            eff_dz = float(z_offset)

        cur_path = ws.aligned_splats_path if ws.aligned_splats_path.is_file() else ws.active_splat_path()
        cloud = load_gaussian_ply(cur_path)
        r_delta = Rotation.from_euler("xyz", [eff_roll, eff_pitch, eff_yaw], degrees=True).as_matrix().astype(np.float64)
        t_delta = np.array([0.0, 0.0, eff_dz], dtype=np.float64)
        updated_cloud = transform_gaussian_cloud(cloud, r_delta, translation=t_delta, scale=1.0)
        _atomic_save_gaussian_ply(ws.aligned_splats_path, updated_cloud)
        ws.set_active_splat(ws.aligned_splats_path)

        scale_factor = float(prev_payload.get("scale_factor", 1.0) or 1.0)
        mat_old = np.asarray(prev_payload.get("matrix_4x4", np.eye(4).tolist()), dtype=np.float64)
        if mat_old.shape != (4, 4):
            mat_old = np.eye(4, dtype=np.float64)
        mat_delta = np.eye(4, dtype=np.float64)
        mat_delta[:3, :3] = r_delta
        mat_delta[:3, 3] = t_delta
        mat_new = mat_delta @ mat_old
        rot_new = mat_new[:3, :3] / scale_factor
        det_normal = rot_new.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

        payload = dict(prev_payload)
        payload["detected_normal"] = det_normal.astype(np.float32).tolist()
        payload["aligned_normal"] = [0.0, 0.0, 1.0]
        payload["scale_factor"] = scale_factor
        payload["alignment_method"] = f"manual_{eff_mode}"
        payload["last_delta"] = {
            "roll_deg": round(eff_roll, 3),
            "pitch_deg": round(eff_pitch, 3),
            "yaw_deg": round(eff_yaw, 3),
            "z_offset": round(eff_dz, 4),
        }
        payload["matrix_4x4"] = mat_new.astype(np.float32).tolist()
        atomic_write_json(transform_path, payload)

    exported_ply_rel: str | None = None
    pure_ply = ws.export_dir / "gaussian_splats" / "scene_3dgs.ply"
    if reexport_ply and (pure_ply.is_file() or ws.export_dir.is_dir()):
        from real2sim_gs.stages.export.ply_exporter import Ply3DGSExporterStage

        exp_res = Ply3DGSExporterStage().run(
            ws,
            {"scene_name": ws.root.name or "scene_3dgs", "include_raw": False},
        )
        exported_ply_rel = exp_res.artifacts.get("scene_3dgs")

    mj_vis = ws.export_dir / "mujoco" / "visual_splats.ply"
    if mj_vis.is_file() and ws.aligned_splats_path.is_file():
        shutil.copy2(ws.aligned_splats_path, mj_vis)

    final_transform: dict[str, Any] = {}
    if transform_path.is_file():
        try:
            final_transform = json.loads(transform_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            final_transform = {}

    return {
        "aligned": True,
        "mode": eff_mode,
        "alignment_transform": final_transform,
        "exported_3dgs_ply": exported_ply_rel,
    }
