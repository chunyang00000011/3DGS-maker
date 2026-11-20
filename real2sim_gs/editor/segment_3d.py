from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import breadth_first_order, connected_components, maximum_flow
from scipy.spatial import cKDTree

from real2sim_gs.core.doctor import is_gsplat_cuda_ready
from real2sim_gs.stages.reconstruct.gsplat_whitebox.dataset import CameraView
from real2sim_gs.stages.reconstruct.gsplat_whitebox.model import GaussianModel, quat_to_rotmat
from real2sim_gs.stages.reconstruct.gsplat_whitebox.trainer import _render_view
from real2sim_gs.utils.ply_io import GaussianCloud


@dataclass
class SegmentLiftConfig:
    """Hyperparameters for B3-Seg + FlashSplat + GaussianCut + SAGD 3D lifting."""

    num_views: int = 12
    render_width: int = 512
    render_height: int = 384
    bg_bias: float = 0.05
    threshold: float = 0.5
    use_gaussiancut: bool = True
    smooth_weight: float = 1.2
    use_sagd: bool = True
    max_sagd_splits: int = 6000
    keep_largest_component: bool = True


@dataclass
class SegmentLiftResult:
    """3D Gaussian segmentation result with continuous candidate scores for instant refinement."""

    cloud: GaussianCloud
    fg_mask: np.ndarray  # (N,) bool
    candidate_indices: np.ndarray  # (K,) int32
    candidate_scores: np.ndarray  # (K,) float32
    object_center: np.ndarray  # (3,) float32
    decomposed_count: int = 0
    views_used: int = 0
    existing_object_ids: np.ndarray | None = None
    stats: dict[str, Any] = field(default_factory=dict)


def cloud_to_gpu_model(
    cloud: GaussianCloud,
    device: torch.device,
    max_log_scale: float = 10.0,
) -> GaussianModel:
    """Convert a CPU GaussianCloud into a GaussianModel on `device`."""
    has_sh = cloud.sh_rest is not None and cloud.sh_rest.shape[1] > 0
    return GaussianModel(
        means=torch.tensor(cloud.means, dtype=torch.float32, device=device),
        scales=torch.tensor(cloud.scales, dtype=torch.float32, device=device),
        quats=torch.tensor(cloud.quats, dtype=torch.float32, device=device),
        opacities=torch.tensor(cloud.opacities, dtype=torch.float32, device=device),
        sh_dc=torch.tensor(cloud.sh_dc, dtype=torch.float32, device=device),
        sh_rest=torch.tensor(cloud.sh_rest, dtype=torch.float32, device=device) if has_sh else None,
        sh_degree=cloud.sh_degree if has_sh else 0,
        max_log_scale=max_log_scale,
    )


def build_orbit_camera_view(
    *,
    yaw: float,
    pitch: float,
    distance: float,
    target: tuple[float, float, float] | list[float] | np.ndarray,
    width: int,
    height: int,
    fov_scale: float = 1.05,
    device: torch.device | None = None,
    name: str = "orbit_view",
) -> CameraView:
    """Construct a CameraView using the exact Z-up orbit convention shared with Studio JS & server.py."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp, sp_sin = math.cos(float(pitch)), math.sin(float(pitch))
    cy_ang, sy_ang = math.cos(float(yaw)), math.sin(float(yaw))
    tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
    dist = max(0.05, float(distance))

    eye = np.array(
        [tx + dist * cp * cy_ang, ty + dist * cp * sy_ang, tz + dist * sp_sin],
        dtype=np.float32,
    )
    fwd = np.array([-cp * cy_ang, -cp * sy_ang, -sp_sin], dtype=np.float32)
    up = np.array([-sp_sin * cy_ang, -sp_sin * sy_ang, cp], dtype=np.float32)
    right = np.array([-sy_ang, cy_ang, 0.0], dtype=np.float32)
    r_w2c = np.stack([right, -up, fwd], axis=0).astype(np.float32)
    t_w2c = (-r_w2c @ eye).astype(np.float32)

    w2c_np = np.eye(4, dtype=np.float32)
    w2c_np[:3, :3] = r_w2c
    w2c_np[:3, 3] = t_w2c
    focal = float(min(width, height)) * float(fov_scale)
    return CameraView(
        name=name,
        width=int(width),
        height=int(height),
        fx=focal,
        fy=focal,
        cx=float(width) * 0.5,
        cy=float(height) * 0.5,
        world_to_cam=torch.tensor(w2c_np, dtype=torch.float32, device=dev),
        image=torch.empty((0, 0, 3), dtype=torch.float32, device=dev),
    )


def project_world_point(view: CameraView, point_xyz: np.ndarray) -> tuple[float, float] | None:
    """Project a world 3D point into pixel (u, v) coordinates of `view`, or None if behind/outside."""
    w2c = view.world_to_cam.detach().cpu().numpy()
    p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    p_cam = w2c[:3, :3] @ p + w2c[:3, 3]
    z = float(p_cam[2])
    if z <= 0.05:
        return None
    u = float(view.fx * (p_cam[0] / z) + view.cx)
    v = float(view.fy * (p_cam[1] / z) + view.cy)
    if u < 1.0 or u >= view.width - 1.0 or v < 1.0 or v >= view.height - 1.0:
        return None
    return (u, v)


def _resize_bool_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.shape[:2] == (target_h, target_w):
        return arr > 0
    resized = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized > 0


@torch.no_grad()
def render_camera_rgb(
    model: GaussianModel,
    view: CameraView,
    *,
    use_gsplat: bool,
    bg_vals: tuple[float, float, float] = (0.955, 0.968, 0.986),
) -> np.ndarray:
    """Render uint8 (H, W, 3) RGB frame from `model` at `view`."""
    device = model.means.device
    bg_t = torch.tensor(bg_vals, dtype=torch.float32, device=device)
    render_sh = model.sh_degree if use_gsplat else 0
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
    return (r_out.rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)


@torch.no_grad()
def estimate_target_anchor_3d(
    model: GaussianModel,
    anchor_view: CameraView,
    anchor_mask_2d: np.ndarray,
    *,
    use_gsplat: bool,
) -> tuple[np.ndarray, float]:
    """Unproject expected surface depth inside the 2D anchor mask to locate 3D target center & radius."""
    device = model.means.device
    bg_t = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device)
    r_out = _render_view(
        model,
        anchor_view,
        w2c=anchor_view.world_to_cam,
        use_gsplat=use_gsplat,
        mip_3d_var=0.0,
        bg_color=bg_t,
        need_depth=True,
        sh_degree=0,
    )
    depth = r_out.depth.detach().cpu().numpy()
    mask_bool = _resize_bool_mask(anchor_mask_2d, anchor_view.height, anchor_view.width)
    valid = mask_bool & (depth > 0.05) & np.isfinite(depth)
    w2c = anchor_view.world_to_cam.detach().cpu().numpy()
    r_c2w = w2c[:3, :3].T
    eye = -r_c2w @ w2c[:3, 3]

    if np.count_nonzero(valid) >= 4:
        ys, xs = np.nonzero(valid)
        zs = depth[ys, xs]
        z_lo, z_hi = np.percentile(zs, [15, 85])
        core = (zs >= z_lo * 0.85) & (zs <= z_hi * 1.15)
        if np.any(core):
            xs, ys, zs = xs[core], ys[core], zs[core]
        x_cam = (xs.astype(np.float64) + 0.5 - anchor_view.cx) * zs / max(anchor_view.fx, 1e-6)
        y_cam = (ys.astype(np.float64) + 0.5 - anchor_view.cy) * zs / max(anchor_view.fy, 1e-6)
        pts_cam = np.stack([x_cam, y_cam, zs.astype(np.float64)], axis=1)
        pts_w = pts_cam @ r_c2w.T + eye[None, :]
        center_3d = np.median(pts_w, axis=0).astype(np.float32)
        dists = np.linalg.norm(pts_w - center_3d[None, :], axis=1)
        radius_3d = float(max(0.06, np.percentile(dists, 85) * 1.35)) if len(dists) > 4 else 0.28
        return center_3d, radius_3d

    # Fallback: optical ray midpoint using median scene depth
    fwd = w2c[2, :3]
    pts_all = model.means.detach().cpu().numpy()
    if len(pts_all) > 0:
        rel = pts_all - eye[None, :]
        depths = rel @ fwd
        pos_d = depths[depths > 0.1]
        d_est = float(np.median(pos_d)) if pos_d.size > 0 else 2.0
    else:
        d_est = 2.0
    return (eye + d_est * fwd).astype(np.float32), 0.35


def plan_active_views(
    model: GaussianModel,
    anchor_params: dict[str, Any],
    center_3d: np.ndarray,
    radius_3d: float,
    num_views: int = 12,
    width: int = 512,
    height: int = 384,
) -> list[CameraView]:
    """B3-Seg inspired Analytic Expected Information Gain (EIG) active view planning.

    View 0 is always the user's exact anchor viewpoint. Subsequent views are selected from a
    360-degree multi-ring orbit pool around `center_3d` by greedily maximizing Beta-Bernoulli
    expected information gain on ROI Gaussians, then sorted smoothly by azimuth delta from the
    anchor camera for seamless SAM 3 cross-view tracking.
    """
    device = model.means.device
    anchor_yaw = float(anchor_params["yaw"])
    anchor_pitch = float(anchor_params["pitch"])
    anchor_dist = float(anchor_params["distance"])
    anchor_target = tuple(float(v) for v in anchor_params["target"])
    fov_scale = float(anchor_params.get("fov_scale", 1.05))

    anchor_view = build_orbit_camera_view(
        yaw=anchor_yaw,
        pitch=anchor_pitch,
        distance=anchor_dist,
        target=anchor_target,
        width=width,
        height=height,
        fov_scale=fov_scale,
        device=device,
        name="active_view_00",
    )
    if num_views <= 1:
        return [anchor_view]

    # Sample ROI points around estimated target 3D center for fast EIG evaluation
    pts_all = model.means.detach()
    roi_rad = max(float(radius_3d) * 2.4, 0.35)
    dists_to_c = torch.linalg.norm(pts_all - torch.tensor(center_3d, device=device, dtype=torch.float32), dim=1)
    roi_idx = torch.nonzero(dists_to_c <= roi_rad, as_tuple=True)[0]
    if roi_idx.numel() < 16:
        roi_pts = pts_all[: min(4096, pts_all.shape[0])]
    else:
        if roi_idx.numel() > 4096:
            step = torch.linspace(0, roi_idx.numel() - 1, steps=4096, device=device).long()
            roi_idx = roi_idx[step]
        roi_pts = pts_all[roi_idx]

    roi_np = roi_pts.cpu().numpy() if roi_pts.numel() > 0 else np.zeros((0, 3), dtype=np.float32)

    # Ideal object-centric orbit distance that keeps object framed cleanly without clipping
    obj_dist = float(np.clip(max(radius_3d * 3.1, anchor_dist * 0.75), 0.35, max(anchor_dist * 1.4, 1.5)))
    pool_count = max(24, num_views * 3)
    pitch_offsets = [0.0, 0.26, -0.18]
    candidates: list[dict[str, float]] = []
    for k in range(1, pool_count):
        frac = k / float(pool_count)
        yaw_cand = anchor_yaw + frac * (2.0 * math.pi)
        pitch_cand = float(np.clip(anchor_pitch + pitch_offsets[k % len(pitch_offsets)], -0.15, 1.32))
        candidates.append(
            {
                "yaw": yaw_cand,
                "pitch": pitch_cand,
                "distance": obj_dist,
                "yaw_delta": frac * (2.0 * math.pi),
            }
        )

    def _approx_visibility(view_cfg: dict[str, float], target_xyz: np.ndarray) -> np.ndarray:
        if len(roi_np) == 0:
            return np.zeros((0,), dtype=np.float64)
        cv_tmp = build_orbit_camera_view(
            yaw=view_cfg["yaw"],
            pitch=view_cfg["pitch"],
            distance=view_cfg["distance"],
            target=target_xyz,
            width=width,
            height=height,
            fov_scale=fov_scale,
            device=torch.device("cpu"),
        )
        w2c = cv_tmp.world_to_cam.numpy()
        p_cam = roi_np @ w2c[:3, :3].T + w2c[:3, 3]
        z = p_cam[:, 2]
        z_safe = np.maximum(z, 1e-4)
        u = cv_tmp.fx * (p_cam[:, 0] / z_safe) + cv_tmp.cx
        v = cv_tmp.fy * (p_cam[:, 1] / z_safe) + cv_tmp.cy
        in_view = (z > 0.08) & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        # Prefer views facing front of center with mild distance attenuation (front-side gets higher weight)
        d_min = float(np.percentile(z[in_view], 10)) if np.any(in_view) else 1.0
        vis = np.where(in_view, np.exp(-np.maximum(0.0, z - d_min) / max(radius_3d * 2.0, 0.25)), 0.0)
        return vis.astype(np.float64)

    # Beta-Bernoulli belief state initialized uniformly at Beta(1, 1)
    alpha_b = np.ones(len(roi_np), dtype=np.float64)
    beta_b = np.ones(len(roi_np), dtype=np.float64)
    anchor_vis = _approx_visibility(
        {"yaw": anchor_yaw, "pitch": anchor_pitch, "distance": anchor_dist},
        anchor_target,
    )
    alpha_b += 0.5 * anchor_vis
    beta_b += 0.5 * anchor_vis

    cand_vis = [_approx_visibility(c, center_3d) for c in candidates]
    picked_indices: list[int] = []
    remaining = set(range(len(candidates)))

    for _ in range(max(0, num_views - 1)):
        if not remaining:
            break
        # Analytic Beta-Bernoulli uncertainty term Var(theta_i) = a*b / ((a+b)^2 * (a+b+1))
        total_ab = alpha_b + beta_b
        info_density = (alpha_b * beta_b) / (total_ab * total_ab * (total_ab + 1.0))
        best_idx = -1
        best_eig = -1.0
        for idx in remaining:
            eig = float(np.dot(cand_vis[idx], info_density))
            if eig > best_eig:
                best_eig = eig
                best_idx = idx
        if best_idx < 0:
            break
        picked_indices.append(best_idx)
        remaining.remove(best_idx)
        q = cand_vis[best_idx]
        mu = alpha_b / (alpha_b + beta_b)
        alpha_b += q * mu
        beta_b += q * (1.0 - mu)

    # Sort selected orbit viewpoints continuously around azimuth so SAM 3 video tracking glides smoothly
    picked_cands = sorted((candidates[i] for i in picked_indices), key=lambda c: c["yaw_delta"])
    views = [anchor_view]
    for idx, c in enumerate(picked_cands, start=1):
        views.append(
            build_orbit_camera_view(
                yaw=c["yaw"],
                pitch=c["pitch"],
                distance=c["distance"],
                target=center_3d,
                width=width,
                height=height,
                fov_scale=fov_scale,
                device=device,
                name=f"active_view_{idx:02d}",
            )
        )
    return views


def _accumulate_view_mask_weights_pytorch(
    model: GaussianModel,
    view: CameraView,
    fg_flat: torch.Tensor,
    bg_flat: torch.Tensor,
    pixel_chunk_size: int = 8192,
    splat_chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact transmittance-weighted foreground/background accumulation matching _render_splats_pytorch."""
    device = model.means.device
    n = model.num_points
    w_fg_total = torch.zeros(n, dtype=torch.float32, device=device)
    w_bg_total = torch.zeros(n, dtype=torch.float32, device=device)
    if n == 0:
        return w_fg_total, w_bg_total

    w2c = view.world_to_cam.to(device)
    r_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]
    p_cam = model.means.detach() @ r_w2c.T + t_w2c
    z = p_cam[:, 2]
    in_front = z > 0.05
    if not in_front.any():
        return w_fg_total, w_bg_total

    idx = torch.nonzero(in_front, as_tuple=True)[0]
    p_vis = p_cam[idx]
    z_vis = p_vis[:, 2]
    u = view.fx * (p_vis[:, 0] / z_vis) + view.cx
    v = view.fy * (p_vis[:, 1] / z_vis) + view.cy
    h, w = view.height, view.width

    in_bounds = (u >= -16.0) & (u <= w + 16.0) & (v >= -16.0) & (v <= h + 16.0)
    if not in_bounds.any():
        return w_fg_total, w_bg_total

    idx = idx[in_bounds]
    u, v, p_vis = u[in_bounds], v[in_bounds], p_vis[in_bounds]
    order = torch.argsort(p_vis[:, 2])
    idx = idx[order]
    u, v, p_vis = u[order], v[order], p_vis[order]
    x_c, y_c, z_vis = p_vis[:, 0], p_vis[:, 1], p_vis[:, 2]

    scales_mip, opac_mip = model.mip_filtered_primitives(mip_3d_var=0.0)
    cov3d = model.covariance_3d(scales_override=scales_mip.detach())[idx]
    opac = opac_mip.detach()[idx]

    r_exp = r_w2c.unsqueeze(0).expand(idx.shape[0], -1, -1)
    cov_cam = torch.bmm(torch.bmm(r_exp, cov3d), r_exp.transpose(1, 2))

    inv_z = 1.0 / z_vis
    inv_z2 = inv_z * inv_z
    zeros = torch.zeros_like(inv_z)
    j0 = torch.stack([view.fx * inv_z, zeros, -view.fx * x_c * inv_z2], dim=-1)
    j1 = torch.stack([zeros, view.fy * inv_z, -view.fy * y_c * inv_z2], dim=-1)
    jac = torch.stack([j0, j1], dim=1)

    cov2d = torch.bmm(torch.bmm(jac, cov_cam), jac.transpose(1, 2))
    cov_xx = cov2d[:, 0, 0] + 0.3
    cov_xy = cov2d[:, 0, 1]
    cov_yy = cov2d[:, 1, 1] + 0.3
    det = (cov_xx * cov_yy - cov_xy * cov_xy).clamp_min(1e-4)

    conic_xx = cov_yy / det
    conic_xy = -cov_xy / det
    conic_yy = cov_xx / det

    ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    flat_x = grid_x.reshape(-1)
    flat_y = grid_y.reshape(-1)
    total_pixels = flat_x.numel()
    num_vis = idx.shape[0]
    ones_head = torch.ones((1, 1), device=device, dtype=torch.float32)

    w_fg_vis = torch.zeros(num_vis, dtype=torch.float32, device=device)
    w_bg_vis = torch.zeros(num_vis, dtype=torch.float32, device=device)

    for p0 in range(0, total_pixels, pixel_chunk_size):
        p1 = min(total_pixels, p0 + pixel_chunk_size)
        px = flat_x[p0:p1]
        py = flat_y[p0:p1]
        fg_p = fg_flat[p0:p1]
        bg_p = bg_flat[p0:p1]
        n_pix = p1 - p0
        t_rem = torch.ones((n_pix,), device=device, dtype=torch.float32)

        for s0 in range(0, num_vis, splat_chunk_size):
            s1 = min(num_vis, s0 + splat_chunk_size)
            dx = px[None, :] - u[s0:s1, None]
            dy = py[None, :] - v[s0:s1, None]
            power = -0.5 * (
                conic_xx[s0:s1, None] * dx * dx
                + 2.0 * conic_xy[s0:s1, None] * dx * dy
                + conic_yy[s0:s1, None] * dy * dy
            )
            alpha = (opac[s0:s1, None] * torch.exp(power.clamp(max=0.0))).clamp(max=0.99)
            one_minus_a = (1.0 - alpha).clamp(min=1e-4)
            cum_in = torch.cumprod(
                torch.cat([ones_head.expand(1, n_pix), one_minus_a[:-1]], dim=0),
                dim=0,
            )
            weights = t_rem[None, :] * cum_in * alpha
            w_fg_vis[s0:s1] += (weights * fg_p[None, :]).sum(dim=1)
            w_bg_vis[s0:s1] += (weights * bg_p[None, :]).sum(dim=1)
            t_rem = t_rem * one_minus_a.prod(dim=0)
            if float(t_rem.max().item()) < 1e-4:
                break

    w_fg_total.index_add_(0, idx, w_fg_vis)
    w_bg_total.index_add_(0, idx, w_bg_vis)
    return w_fg_total, w_bg_total


def accumulate_view_mask_weights(
    model: GaussianModel,
    view: CameraView,
    mask_2d: np.ndarray,
    *,
    use_gsplat: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-Gaussian foreground & background transmittance weights (w_fg, w_bg) for one view."""
    device = model.means.device
    n = model.num_points
    if n == 0:
        z = torch.zeros(0, dtype=torch.float32, device=device)
        return z, z.clone()

    mask_bool = _resize_bool_mask(mask_2d, view.height, view.width)
    fg_np = mask_bool.astype(np.float32)
    fg_t = torch.from_numpy(fg_np).to(device=device)
    bg_t = 1.0 - fg_t

    if not use_gsplat:
        with torch.no_grad():
            return _accumulate_view_mask_weights_pytorch(
                model,
                view,
                fg_flat=fg_t.reshape(-1),
                bg_flat=bg_t.reshape(-1),
            )

    from gsplat.rendering import rasterization

    dummy = torch.zeros((n, 3), device=device, dtype=torch.float32, requires_grad=True)
    scales_mip, opac_mip = model.mip_filtered_primitives(mip_3d_var=0.0)
    k = view.K.to(device=device)[None]
    use_packed = n > 32_000

    with torch.enable_grad():
        out, _, _ = rasterization(
            means=model.means.detach(),
            quats=model.normalized_quats.detach(),
            scales=scales_mip.detach(),
            opacities=opac_mip.detach(),
            colors=dummy,
            viewmats=view.world_to_cam.detach()[None],
            Ks=k,
            width=view.width,
            height=view.height,
            sh_degree=None,
            backgrounds=None,
            rasterize_mode="classic",
            render_mode="RGB",
            packed=use_packed,
            absgrad=False,
        )
        loss = (out[0, :, :, 0] * fg_t + out[0, :, :, 1] * bg_t).sum()
        grad = torch.autograd.grad(loss, dummy, retain_graph=False, create_graph=False)[0]
    return grad[:, 0].detach().clamp_min(0.0), grad[:, 1].detach().clamp_min(0.0)


def solve_flashsplat_scores(
    w_fg: np.ndarray,
    w_bg: np.ndarray,
    *,
    bg_bias: float = 0.05,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form FlashSplat LP solution with background-noise bias regularization."""
    w_fg_f = np.maximum(np.asarray(w_fg, dtype=np.float64), 0.0)
    w_bg_f = np.maximum(np.asarray(w_bg, dtype=np.float64), 0.0)
    w_tot = w_fg_f + w_bg_f
    scores = np.where(w_tot > 1e-8, w_fg_f / np.maximum(w_tot, 1e-8), 0.0).astype(np.float32)
    cutoff = float(np.clip(threshold + 0.5 * bg_bias, 0.05, 0.95))
    init_fg = (scores >= cutoff) & (w_tot > 1e-4)
    return scores, w_tot.astype(np.float32), init_fg


def _filter_connected_components(
    means: np.ndarray,
    fg_mask: np.ndarray,
    center_3d: np.ndarray | None,
) -> np.ndarray:
    """Keep the main object component anchored near `center_3d` and prune detached stray blobs."""
    fg_idx = np.nonzero(fg_mask)[0]
    if len(fg_idx) < 8:
        return fg_mask

    pts = means[fg_idx].astype(np.float64)
    k = min(9, len(pts))
    tree = cKDTree(pts)
    dists, nbrs = tree.query(pts, k=k)
    nn_d = dists[:, 1:]
    med_spacing = max(float(np.median(nn_d[:, 0])), 1e-4)
    span = float(np.linalg.norm(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)))
    link_thresh = max(med_spacing * 6.0, span * 0.06, 0.03)

    rows = np.repeat(np.arange(len(pts)), k - 1)
    cols = nbrs[:, 1:].ravel()
    valid = nn_d.ravel() <= link_thresh
    rows, cols = rows[valid], cols[valid]
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(pts), len(pts)))
    adj = (adj + adj.T).tocsr()
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp <= 1:
        return fg_mask

    counts = np.bincount(labels)
    if center_3d is not None:
        c = np.asarray(center_3d, dtype=np.float64).reshape(3)
        best_cid = 0
        best_rank = -1.0
        for cid in range(n_comp):
            c_pts = pts[labels == cid]
            d_c = float(np.linalg.norm(np.median(c_pts, axis=0) - c))
            rank = float(counts[cid]) / (1.0 + d_c / max(span * 0.25, 0.05))
            if rank > best_rank:
                best_rank = rank
                best_cid = cid
    else:
        best_cid = int(np.argmax(counts))

    # Keep primary component and any substantial sibling component close to it
    main_pts = pts[labels == best_cid]
    main_tree = cKDTree(main_pts)
    keep_cids = {best_cid}
    min_sibling_size = max(12, int(0.08 * counts[best_cid]))
    sibling_dist_cap = max(link_thresh * 1.8, span * 0.12)
    for cid in range(n_comp):
        if cid == best_cid or counts[cid] < min_sibling_size:
            continue
        d_to_main, _ = main_tree.query(pts[labels == cid], k=1)
        if float(np.min(d_to_main)) <= sibling_dist_cap:
            keep_cids.add(cid)

    keep_local = np.isin(labels, list(keep_cids))
    out = np.zeros_like(fg_mask, dtype=bool)
    out[fg_idx[keep_local]] = True
    return out


def refine_gaussiancut(
    means: np.ndarray,
    rgb_colors: np.ndarray,
    w_fg: np.ndarray,
    w_bg: np.ndarray,
    init_fg: np.ndarray,
    *,
    center_3d: np.ndarray | None = None,
    bg_bias: float = 0.05,
    smooth_weight: float = 1.2,
    existing_object_ids: np.ndarray | None = None,
    keep_largest_component: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """NeurIPS-style GaussianCut 3D k-NN graph min-cut to fill interior shells & prune stray floaters.

    Returns `(refined_fg_mask, roi_indices)`.
    """
    n = int(means.shape[0])
    if n == 0 or not np.any(init_fg):
        return init_fg.copy(), np.zeros(0, dtype=np.int32)

    seed_pts = means[init_fg]
    if len(seed_pts) >= 16:
        lo = np.percentile(seed_pts, 2, axis=0)
        hi = np.percentile(seed_pts, 98, axis=0)
    else:
        lo = seed_pts.min(axis=0)
        hi = seed_pts.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    pad = max(0.22 * diag, 0.08)
    in_box = np.all((means >= (lo - pad)) & (means <= (hi + pad)), axis=1)
    roi_mask = in_box | init_fg
    roi_idx = np.nonzero(roi_mask)[0].astype(np.int32)

    if len(roi_idx) > 45_000:
        # Tighter ROI for gigantic scenes to guarantee sub-second Dinic min-cut
        tight_pad = max(0.10 * diag, 0.04)
        tight_box = np.all((means >= (lo - tight_pad)) & (means <= (hi + tight_pad)), axis=1)
        roi_mask = ((tight_box & ((w_fg + w_bg) > 1e-5)) | init_fg)
        roi_idx = np.nonzero(roi_mask)[0].astype(np.int32)
        if len(roi_idx) > 45_000:
            step = np.linspace(0, len(roi_idx) - 1, num=45_000, dtype=int)
            roi_idx = roi_idx[step]

    m = len(roi_idx)
    if m < 4:
        return init_fg.copy(), roi_idx

    pts_r = means[roi_idx].astype(np.float64)
    cols_r = rgb_colors[roi_idx].astype(np.float64)
    w_fg_r = np.maximum(w_fg[roi_idx].astype(np.float64), 0.0)
    w_bg_r = np.maximum(w_bg[roi_idx].astype(np.float64), 0.0)
    w_tot_r = w_fg_r + w_bg_r

    obs = w_tot_r > 1e-4
    med_w = max(float(np.median(w_tot_r[obs])) if np.any(obs) else 1.0, 1e-4)

    cap_unit = 1000.0
    cap_src = np.where(obs, (w_fg_r / med_w) * cap_unit, 0.0)
    cap_snk = np.where(obs, ((w_bg_r + bg_bias * w_tot_r) / med_w) * cap_unit, 0.0)

    if existing_object_ids is not None:
        locked = existing_object_ids[roi_idx] > 0
        cap_src[locked] = 0.0
        cap_snk[locked] = cap_unit * 500.0

    # k-NN spatial + color affinity binary smoothness edges
    k = min(9, m)
    tree = cKDTree(pts_r)
    dists, nbrs = tree.query(pts_r, k=k)
    nn_d = dists[:, 1:]
    nn_j = nbrs[:, 1:]
    sigma_x = max(float(np.median(nn_d)), 1e-4)
    sigma_c = 0.25

    src_nodes = np.repeat(np.arange(m, dtype=np.int32), k - 1)
    dst_nodes = nn_j.ravel().astype(np.int32)
    d_sq = (nn_d.ravel() / sigma_x) ** 2
    c_diff = cols_r[src_nodes] - cols_r[dst_nodes]
    c_sq = np.sum(c_diff * c_diff, axis=1) / (sigma_c * sigma_c)
    edge_w = smooth_weight * cap_unit * np.exp(-0.5 * (d_sq + 0.6 * c_sq))

    src_id = m
    snk_id = m + 1
    nodes_idx = np.arange(m, dtype=np.int32)

    cap_src_i = np.clip(np.round(cap_src), 0, 5_000_000).astype(np.int32)
    cap_snk_i = np.clip(np.round(cap_snk), 0, 5_000_000).astype(np.int32)
    edge_w_i = np.clip(np.round(edge_w), 0, 5_000_000).astype(np.int32)

    all_rows = np.concatenate([
        np.full(m, src_id, dtype=np.int32),
        nodes_idx,
        src_nodes,
        dst_nodes,
    ])
    all_cols = np.concatenate([
        nodes_idx,
        np.full(m, snk_id, dtype=np.int32),
        dst_nodes,
        src_nodes,
    ])
    all_vals = np.concatenate([cap_src_i, cap_snk_i, edge_w_i, edge_w_i]).astype(np.int32)

    nz = all_vals > 0
    graph = sp.coo_matrix((all_vals[nz], (all_rows[nz], all_cols[nz])), shape=(m + 2, m + 2), dtype=np.int32).tocsr()

    flow_res = maximum_flow(graph, src_id, snk_id, method="dinic")
    residual = (graph - flow_res.flow).tocsr()
    residual.data = np.maximum(residual.data, 0)
    residual.eliminate_zeros()

    reachable = breadth_first_order(residual, src_id, directed=True, return_predecessors=False)
    fg_local = reachable[reachable < m]

    out_mask = np.zeros(n, dtype=bool)
    out_mask[roi_idx[fg_local]] = True
    if existing_object_ids is not None:
        out_mask &= existing_object_ids <= 0

    if keep_largest_component and np.any(out_mask):
        out_mask = _filter_connected_components(means, out_mask, center_3d=center_3d)
    return out_mask, roi_idx


def _decompose_boundary_gaussians(
    cloud: GaussianCloud,
    fg_mask: np.ndarray,
    scores: np.ndarray,
    w_tot: np.ndarray,
    roi_idx: np.ndarray,
    *,
    existing_object_ids: np.ndarray | None = None,
    max_splits: int = 6000,
) -> tuple[GaussianCloud, np.ndarray | None, int]:
    """SAGD boundary-enhanced Gaussian decomposition: split boundary-crossing Gaussians along major axis."""
    if cloud.num_points == 0 or len(roi_idx) < 4 or max_splits <= 0:
        return cloud, existing_object_ids, 0

    roi_pts = cloud.means[roi_idx].astype(np.float64)
    roi_fg = fg_mask[roi_idx]
    roi_scores = scores[roi_idx]
    roi_wtot = w_tot[roi_idx]

    # Identify true boundary-spanning Gaussians:
    # 1) Mixed FG/BG rays with non-trivial physical extent;
    # 2) Oversized splats touching an opposite-label neighbor across the 3D seam.
    scales_lin = np.exp(np.clip(cloud.scales[roi_idx], -10.0, 2.5))
    max_scale = scales_lin.max(axis=1)
    med_scale = max(float(np.median(max_scale)), 1e-5)

    ambig_local = (roi_scores > 0.22) & (roi_scores < 0.78) & (roi_wtot > 1e-3) & (max_scale > 0.90 * med_scale)

    k = min(7, len(roi_pts))
    tree = cKDTree(roi_pts)
    _, nbrs = tree.query(roi_pts, k=k)
    nbr_fg = roi_fg[nbrs[:, 1:]]
    touches_opp = (roi_fg & (~np.all(nbr_fg, axis=1))) | ((~roi_fg) & np.any(nbr_fg, axis=1))

    oversized_boundary = touches_opp & (max_scale > 1.25 * med_scale)
    cand_local = np.nonzero(ambig_local | oversized_boundary)[0]
    if len(cand_local) == 0:
        return cloud, existing_object_ids, 0

    cand_global = roi_idx[cand_local]
    if existing_object_ids is not None:
        unlocked = existing_object_ids[cand_global] <= 0
        cand_global = cand_global[unlocked]
        cand_local = cand_local[unlocked]

    if len(cand_global) == 0:
        return cloud, existing_object_ids, 0

    # Prioritize highest ambiguity + largest physical extent
    ambiguity = 1.0 - np.abs(scores[cand_global] - 0.5) * 2.0
    cand_scales = np.exp(np.clip(cloud.scales[cand_global], -10.0, 2.5)).max(axis=1)
    priority = ambiguity + 0.35 * (cand_scales / med_scale)
    if len(cand_global) > max_splits:
        top = np.argsort(priority)[::-1][:max_splits]
        cand_global = cand_global[top]

    cand_global = np.sort(cand_global)
    split_count = int(len(cand_global))
    if split_count == 0:
        return cloud, existing_object_ids, 0

    keep_mask = np.ones(cloud.num_points, dtype=bool)
    keep_mask[cand_global] = False

    quats_t = torch.from_numpy(cloud.quats[cand_global].astype(np.float32))
    rots = quat_to_rotmat(quats_t).numpy()  # (S, 3, 3)
    parent_scales_log = cloud.scales[cand_global].astype(np.float32)
    parent_scales_lin = np.exp(np.clip(parent_scales_log, -10.0, 4.0))
    max_axis = np.argmax(parent_scales_lin, axis=1)  # (S,)

    idx_s = np.arange(split_count)
    principal_dir = rots[idx_s, :, max_axis]  # (S, 3)
    offset = 0.45 * parent_scales_lin[idx_s, max_axis][:, None] * principal_dir

    means_p = cloud.means[cand_global]
    means_children = np.concatenate([means_p - offset, means_p + offset], axis=0).astype(np.float32)

    child_scales_log = parent_scales_log.copy()
    child_scales_log += np.log(0.92, dtype=np.float32)
    child_scales_log[idx_s, max_axis] = parent_scales_log[idx_s, max_axis] + np.log(0.55, dtype=np.float32)
    scales_children = np.concatenate([child_scales_log, child_scales_log], axis=0).astype(np.float32)

    quats_children = np.concatenate([cloud.quats[cand_global], cloud.quats[cand_global]], axis=0).astype(np.float32)
    opac_children = np.concatenate([cloud.opacities[cand_global], cloud.opacities[cand_global]], axis=0).astype(
        np.float32
    )
    sh_dc_children = np.concatenate([cloud.sh_dc[cand_global], cloud.sh_dc[cand_global]], axis=0).astype(np.float32)
    if cloud.sh_rest is not None:
        sh_rest_children: np.ndarray | None = np.concatenate(
            [cloud.sh_rest[cand_global], cloud.sh_rest[cand_global]], axis=0
        ).astype(np.float32)
    else:
        sh_rest_children = None

    new_cloud = GaussianCloud(
        means=np.concatenate([cloud.means[keep_mask], means_children], axis=0),
        scales=np.concatenate([cloud.scales[keep_mask], scales_children], axis=0),
        quats=np.concatenate([cloud.quats[keep_mask], quats_children], axis=0),
        opacities=np.concatenate([cloud.opacities[keep_mask], opac_children], axis=0),
        sh_dc=np.concatenate([cloud.sh_dc[keep_mask], sh_dc_children], axis=0),
        sh_rest=(
            np.concatenate([cloud.sh_rest[keep_mask], sh_rest_children], axis=0)
            if cloud.sh_rest is not None and sh_rest_children is not None
            else None
        ),
    )

    new_obj_ids: np.ndarray | None = None
    if existing_object_ids is not None:
        new_obj_ids = np.concatenate(
            [existing_object_ids[keep_mask], np.zeros(split_count * 2, dtype=existing_object_ids.dtype)],
            axis=0,
        )

    return new_cloud, new_obj_ids, split_count


def _accumulate_all_views(
    model: GaussianModel,
    views: list[CameraView],
    masks_2d: list[np.ndarray],
    *,
    use_gsplat: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n = model.num_points
    device = model.means.device
    w_fg_acc = torch.zeros(n, dtype=torch.float32, device=device)
    w_bg_acc = torch.zeros(n, dtype=torch.float32, device=device)

    for view, mask in zip(views, masks_2d):
        w_fg, w_bg = accumulate_view_mask_weights(model, view, mask, use_gsplat=use_gsplat)
        w_fg_acc += w_fg
        w_bg_acc += w_bg

    return w_fg_acc.detach().cpu().numpy(), w_bg_acc.detach().cpu().numpy()


def lift_masks_to_3dgs(
    cloud: GaussianCloud,
    views: list[CameraView],
    masks_2d: list[np.ndarray],
    config: SegmentLiftConfig | None = None,
    *,
    existing_object_ids: np.ndarray | None = None,
    center_3d: np.ndarray | None = None,
    device: torch.device | None = None,
    use_gsplat: bool | None = None,
) -> SegmentLiftResult:
    """Lift multi-view 2D segmentation masks to clean, boundary-sharpened 3D Gaussians.

    Combines:
    1. FlashSplat closed-form occlusion-aware linear programming;
    2. GaussianCut 3D k-NN graph min-cut + main connected-component filtering;
    3. SAGD boundary Gaussian decomposition along principal axes + exact weight re-evaluation.
    """
    cfg = config or SegmentLiftConfig()
    if len(views) != len(masks_2d) or not views:
        raise ValueError("views and masks_2d must be non-empty lists of equal length.")

    dev = device or (views[0].world_to_cam.device if hasattr(views[0], "world_to_cam") else torch.device("cpu"))
    if use_gsplat is None:
        _, gsplat_ready, _ = is_gsplat_cuda_ready()
        eff_gsplat = bool(gsplat_ready and dev.type == "cuda")
    else:
        eff_gsplat = bool(use_gsplat and dev.type == "cuda")

    cur_cloud = cloud
    cur_obj_ids = existing_object_ids.copy() if existing_object_ids is not None else None

    model = cloud_to_gpu_model(cur_cloud, dev)
    w_fg, w_bg = _accumulate_all_views(model, views, masks_2d, use_gsplat=eff_gsplat)
    scores, w_tot, init_fg = solve_flashsplat_scores(w_fg, w_bg, bg_bias=cfg.bg_bias, threshold=cfg.threshold)

    if cur_obj_ids is not None:
        init_fg &= cur_obj_ids <= 0

    if cfg.use_gaussiancut and np.any(init_fg):
        fg_mask, roi_idx = refine_gaussiancut(
            cur_cloud.means,
            cur_cloud.rgb_colors(),
            w_fg,
            w_bg,
            init_fg,
            center_3d=center_3d,
            bg_bias=cfg.bg_bias,
            smooth_weight=cfg.smooth_weight,
            existing_object_ids=cur_obj_ids,
            keep_largest_component=cfg.keep_largest_component,
        )
    else:
        fg_mask = init_fg
        roi_idx = np.nonzero((w_tot > 1e-4) | fg_mask)[0].astype(np.int32)

    decomposed_count = 0
    if cfg.use_sagd and np.any(fg_mask) and len(roi_idx) >= 4:
        decomposed_cloud, decomposed_ids, decomposed_count = _decompose_boundary_gaussians(
            cur_cloud,
            fg_mask,
            scores,
            w_tot,
            roi_idx,
            existing_object_ids=cur_obj_ids,
            max_splits=cfg.max_sagd_splits,
        )
        if decomposed_count > 0:
            cur_cloud = decomposed_cloud
            cur_obj_ids = decomposed_ids
            model_new = cloud_to_gpu_model(cur_cloud, dev)
            w_fg, w_bg = _accumulate_all_views(model_new, views, masks_2d, use_gsplat=eff_gsplat)
            scores, w_tot, init_fg = solve_flashsplat_scores(
                w_fg, w_bg, bg_bias=cfg.bg_bias, threshold=cfg.threshold
            )
            if cur_obj_ids is not None:
                init_fg &= cur_obj_ids <= 0

            if cfg.use_gaussiancut and np.any(init_fg):
                fg_mask, roi_idx = refine_gaussiancut(
                    cur_cloud.means,
                    cur_cloud.rgb_colors(),
                    w_fg,
                    w_bg,
                    init_fg,
                    center_3d=center_3d,
                    bg_bias=cfg.bg_bias,
                    smooth_weight=cfg.smooth_weight,
                    existing_object_ids=cur_obj_ids,
                    keep_largest_component=cfg.keep_largest_component,
                )
            else:
                fg_mask = init_fg
                roi_idx = np.nonzero((w_tot > 1e-4) | fg_mask)[0].astype(np.int32)

    # Record candidate scores across ROI + fg_mask so threshold / Z-clip sliders respond smoothly
    # without hollowing out Graph-Cut-filled occluded interior Gaussians.
    cand_mask = fg_mask | (w_fg > 1e-4)
    if len(roi_idx) > 0:
        cand_mask[roi_idx] = True
    if cur_obj_ids is not None:
        cand_mask &= cur_obj_ids <= 0
    candidate_indices = np.nonzero(cand_mask)[0].astype(np.int32)
    eff_scores = scores[candidate_indices].astype(np.float32)
    cand_fg = fg_mask[candidate_indices]
    cand_unobs = w_tot[candidate_indices] <= 1e-4
    eff_scores = np.where(cand_fg & cand_unobs, np.maximum(eff_scores, 0.90), eff_scores)
    eff_scores = np.where(
        cand_fg & (~cand_unobs),
        np.maximum(eff_scores, np.float32(cfg.threshold)),
        eff_scores,
    )
    candidate_scores = eff_scores.astype(np.float32)

    if np.any(fg_mask):
        obj_center = np.median(cur_cloud.means[fg_mask], axis=0).astype(np.float32)
    elif center_3d is not None:
        obj_center = np.asarray(center_3d, dtype=np.float32).reshape(3)
    else:
        obj_center = np.zeros(3, dtype=np.float32)

    return SegmentLiftResult(
        cloud=cur_cloud,
        fg_mask=fg_mask,
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
        object_center=obj_center,
        decomposed_count=decomposed_count,
        views_used=len(views),
        existing_object_ids=cur_obj_ids,
        stats={
            "num_points_total": cur_cloud.num_points,
            "num_points_selected": int(np.count_nonzero(fg_mask)),
            "decomposed_gaussians": decomposed_count,
            "views_used": len(views),
            "used_gsplat": eff_gsplat,
        },
    )
