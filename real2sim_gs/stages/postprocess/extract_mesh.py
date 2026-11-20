from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from real2sim_gs.core.registry import register_stage
from real2sim_gs.core.stage import BaseStage, StageResult
from real2sim_gs.core.workspace import SceneWorkspace
from real2sim_gs.utils.ply_io import GaussianCloud, load_gaussian_ply

MAX_VOXEL_RESOLUTION = 256


def _build_heightfield_mesh(points: np.ndarray, grid_res: int) -> trimesh.Trimesh:
    """Triangulate an upper-envelope 2.5D heightfield over walkable surfaces (strips indoor ceiling lids)."""
    res = max(4, min(MAX_VOXEL_RESOLUTION * 2, int(grid_res)))
    pts = points
    if len(pts) >= 64:
        z_min = float(pts[:, 2].min())
        z_max = float(pts[:, 2].max())
        z_span = z_max - z_min
        if z_span > 1.8:
            lower_pts = pts[pts[:, 2] <= z_min + 0.65 * z_span]
            if len(lower_pts) >= 64:
                pts = lower_pts

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    xs = np.linspace(x_min, x_max if x_max > x_min else x_min + 1.0, res)
    ys = np.linspace(y_min, y_max if y_max > y_min else y_min + 1.0, res)

    dx = (xs[-1] - xs[0]) / max(1, res - 1)
    dy = (ys[-1] - ys[0]) / max(1, res - 1)

    ix = np.clip(np.round((pts[:, 0] - xs[0]) / max(dx, 1e-6)).astype(int), 0, res - 1)
    iy = np.clip(np.round((pts[:, 1] - ys[0]) / max(dy, 1e-6)).astype(int), 0, res - 1)

    grid_z = np.full((res, res), -np.inf, dtype=np.float32)
    np.maximum.at(grid_z, (iy, ix), pts[:, 2])

    valid_mask = np.isfinite(grid_z)
    if not valid_mask.all() and valid_mask.any():
        vy, vx = np.nonzero(valid_mask)
        my, mx = np.nonzero(~valid_mask)
        tree = cKDTree(np.column_stack([vy, vx]))
        _, nearest_idx = tree.query(np.column_stack([my, mx]), k=1)
        grid_z[my, mx] = grid_z[vy[nearest_idx], vx[nearest_idx]]
    elif not valid_mask.any():
        grid_z.fill(0.0)

    gx, gy = np.meshgrid(xs, ys)
    vertices = np.column_stack([gx.ravel(), gy.ravel(), grid_z.ravel()]).astype(np.float32)

    grid_idx = np.arange(res * res, dtype=np.int64).reshape(res, res)
    v00 = grid_idx[:-1, :-1].ravel()
    v10 = grid_idx[:-1, 1:].ravel()
    v01 = grid_idx[1:, :-1].ravel()
    v11 = grid_idx[1:, 1:].ravel()
    faces = np.vstack([np.column_stack([v00, v10, v11]), np.column_stack([v00, v11, v01])])

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _core_points_and_bounds(
    cloud: GaussianCloud,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Focus voxel/heightfield meshing on opaque foreground core rather than extreme outliers."""
    opac = cloud.sigmoid_opacities()
    opac_mask = opac >= 0.10
    pts_all = cloud.means
    scales_all = np.exp(np.clip(cloud.scales, -10.0, 2.0)).max(axis=1)
    if int(opac_mask.sum()) >= 32:
        core_pts = pts_all[opac_mask]
        core_opac = opac[opac_mask]
        core_scales = scales_all[opac_mask]
    else:
        core_pts = pts_all
        core_opac = opac
        core_scales = scales_all

    if len(core_pts) >= 64:
        bmin = np.percentile(core_pts, 2, axis=0).astype(np.float32)
        bmax = np.percentile(core_pts, 98, axis=0).astype(np.float32)
        span = np.maximum(bmax - bmin, 1e-2)
        pad_min = bmin - 0.05 * span
        pad_max = bmax + 0.05 * span
        in_box = np.all((core_pts >= pad_min[None, :]) & (core_pts <= pad_max[None, :]), axis=1)
        if int(in_box.sum()) >= 32:
            core_pts = core_pts[in_box]
            core_opac = core_opac[in_box]
            core_scales = core_scales[in_box]

    bmin = core_pts.min(axis=0)
    bmax = core_pts.max(axis=0)
    return core_pts, core_opac, core_scales, bmin, bmax


def _build_marching_cubes_mesh(
    cloud: GaussianCloud,
    voxel_resolution: int = 56,
    iso_level: float = 0.15,
    max_voxel_resolution: int = MAX_VOXEL_RESOLUTION,
) -> tuple[trimesh.Trimesh, float, list[int]]:
    """Extract isotropic 3D surface mesh from scale-aware smoothed Gaussian Opacity Field."""
    pts, opac, scales, bmin, bmax = _core_points_and_bounds(cloud)
    cap = max(16, int(max_voxel_resolution))
    res = min(cap, max(12, int(voxel_resolution)))

    span = np.maximum(bmax - bmin, 1e-2)
    pad = 0.08 * span
    origin = bmin - pad
    top = bmax + pad
    total_span = np.maximum(top - origin, 1e-2)

    voxel_size = float(total_span.max()) / max(1, res - 1)
    dims = np.clip(np.ceil(total_span / max(voxel_size, 1e-6)).astype(int) + 1, 4, cap)

    ix = np.clip(np.round((pts[:, 0] - origin[0]) / voxel_size).astype(int), 0, int(dims[0]) - 1)
    iy = np.clip(np.round((pts[:, 1] - origin[1]) / voxel_size).astype(int), 0, int(dims[1]) - 1)
    iz = np.clip(np.round((pts[:, 2] - origin[2]) / voxel_size).astype(int), 0, int(dims[2]) - 1)

    field = np.zeros(tuple(int(d) for d in dims), dtype=np.float32)
    np.add.at(field, (ix, iy, iz), opac)

    # Large surface Gaussians contribute into adjacent voxels so flat floors stay hole-free.
    large = scales > (1.6 * voxel_size)
    if np.any(large):
        lx, ly, lz, lo = ix[large], iy[large], iz[large], opac[large] * 0.35
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx = np.clip(lx + dx, 0, int(dims[0]) - 1)
            ny = np.clip(ly + dy, 0, int(dims[1]) - 1)
            nz = np.clip(lz + dz, 0, int(dims[2]) - 1)
            np.add.at(field, (nx, ny, nz), lo)

    sigma = float(np.clip(np.median(scales) / max(voxel_size, 1e-6), 0.85, 2.2)) if len(scales) else 1.2
    field = gaussian_filter(field, sigma=sigma)

    nonzero = field[field > 0.0]
    if nonzero.size == 0:
        return _build_heightfield_mesh(pts, grid_res=res), float(voxel_size), [int(d) for d in dims]
    p99 = float(np.percentile(nonzero, 99))
    if p99 <= 1e-8:
        return _build_heightfield_mesh(pts, grid_res=res), float(voxel_size), [int(d) for d in dims]
    field = np.clip(field / p99, 0.0, 1.0)

    padded = np.pad(field, 1, mode="constant", constant_values=0.0)
    level = min(0.9, max(0.02, float(iso_level)))
    try:
        verts, faces, _, _ = marching_cubes(padded, level=level, spacing=(voxel_size, voxel_size, voxel_size))
    except ValueError:
        return _build_heightfield_mesh(pts, grid_res=res), float(voxel_size), [int(d) for d in dims]

    verts = verts + (origin - voxel_size)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if len(mesh.faces) == 0:
        return _build_heightfield_mesh(pts, grid_res=res), float(voxel_size), [int(d) for d in dims]

    components = mesh.split(only_watertight=False)
    if components:
        min_faces = max(4, int(0.02 * len(mesh.faces)))
        kept = [c for c in components if len(c.faces) >= min_faces]
        if kept:
            mesh = kept[0] if len(kept) == 1 else trimesh.util.concatenate(kept)

    return mesh, float(voxel_size), [int(d) for d in dims]


@register_stage("postprocess", "extract_collision_mesh")
class ExtractCollisionMeshStage(BaseStage):
    """Extract a 3D physics collision mesh (.obj) from refined Gaussians via isotropic Marching Cubes or heightfield."""

    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        cloud = load_gaussian_ply(ws.active_splat_path())
        method = str(params.get("method", "marching_cubes")).lower()
        grid_res = int(params.get("grid_resolution", 48))
        voxel_res = int(params.get("voxel_resolution", 56))
        max_voxel_res = int(params.get("max_voxel_resolution", MAX_VOXEL_RESOLUTION))
        iso_level = float(params.get("iso_level", 0.15))

        extra_metrics: dict[str, Any] = {}
        if cloud.num_points < 4:
            mesh = trimesh.creation.box(extents=[1.0, 1.0, 0.05])
        elif method == "marching_cubes":
            mesh, voxel_size, grid_dims = _build_marching_cubes_mesh(
                cloud,
                voxel_resolution=voxel_res,
                iso_level=iso_level,
                max_voxel_resolution=max_voxel_res,
            )
            extra_metrics["voxel_size"] = round(voxel_size, 5)
            extra_metrics["grid_dims"] = grid_dims
        elif method == "convex_hull":
            core_pts, _, _, _, _ = _core_points_and_bounds(cloud)
            mesh = trimesh.convex.convex_hull(core_pts)
        elif method == "heightfield":
            core_pts, _, _, _, _ = _core_points_and_bounds(cloud)
            mesh = _build_heightfield_mesh(core_pts, grid_res=grid_res)
        else:
            raise ValueError(f"Unknown collision mesh extraction method: '{method}'")

        ws.collision_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(ws.collision_mesh_path))

        return StageResult(
            artifacts={"collision_mesh": ws.to_relative(ws.collision_mesh_path)},
            metrics={
                "mesh_vertices": int(len(mesh.vertices)),
                "mesh_faces": int(len(mesh.faces)),
                "mesh_method": method,
                **extra_metrics,
            },
            summary=f"Exported collision mesh ({method}) with {len(mesh.faces)} faces.",
        )
