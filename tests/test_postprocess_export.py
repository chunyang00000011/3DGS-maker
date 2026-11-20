from __future__ import annotations

import json
import struct
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from real2sim_gs.core.workspace import SceneWorkspace
from real2sim_gs.stages.export.mujoco_exporter import MujocoExporterStage
from real2sim_gs.stages.export.ply_exporter import Ply3DGSExporterStage, _R_ZUP_TO_SUPERSPLAT_PLY
from real2sim_gs.stages.export.unity_exporter import UnityExporterStage, _M_ZUP_TO_UNITY_LH
from real2sim_gs.stages.postprocess.align_sim import (
    AlignGravityScaleStage,
    _fit_ground_plane_ransac,
    realign_workspace_scene,
)
from real2sim_gs.stages.postprocess.clean_floaters import CleanFloatersStage
from real2sim_gs.stages.postprocess.extract_mesh import _build_heightfield_mesh, _build_marching_cubes_mesh
from real2sim_gs.utils.ply_io import (
    GaussianCloud,
    _matrices_to_quats_wxyz,
    _quats_wxyz_to_matrices,
    eval_sh_rest_color,
    load_gaussian_ply,
    save_gaussian_ply,
    transform_gaussian_cloud,
    transform_sh_rest,
)


def _cov_from_cloud(cloud: GaussianCloud) -> np.ndarray:
    rots = _quats_wxyz_to_matrices(cloud.quats)
    s = np.exp(cloud.scales.astype(np.float64))
    rs = rots * s[:, None, :]
    return np.einsum("nij,nkj->nik", rs, rs)


def test_sh_degree1_and_degree3_o3_invariance() -> None:
    rng = np.random.default_rng(123)
    n = 16
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rot = Rotation.random(random_state=rng).as_matrix()
    unity_refl = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    for cols in (9, 24, 45):
        sh = rng.normal(scale=0.5, size=(n, cols)).astype(np.float32)
        c_orig = eval_sh_rest_color(sh, dirs)
        for mat in (rot, unity_refl):
            sh_rot = transform_sh_rest(sh, mat)
            assert sh_rot is not None
            c_rot = eval_sh_rest_color(sh_rot, dirs @ mat.T)
            assert np.allclose(c_orig, c_rot, atol=1e-5)


def test_covariance_invariance_align_ply_unity() -> None:
    rng = np.random.default_rng(321)
    n = 12
    quats_xyzw = Rotation.random(n, random_state=rng).as_quat()
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    cloud = GaussianCloud(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        scales=rng.uniform(-4.0, -1.0, size=(n, 3)).astype(np.float32),
        quats=quats_wxyz,
        opacities=rng.uniform(-1.0, 2.0, size=(n,)).astype(np.float32),
        sh_dc=rng.normal(size=(n, 3)).astype(np.float32),
        sh_rest=rng.normal(scale=0.3, size=(n, 45)).astype(np.float32),
    )
    cov_before = _cov_from_cloud(cloud)

    rot = Rotation.random(random_state=rng).as_matrix()
    scale = 1.75
    for mat, s in (
        (rot, scale),
        (_R_ZUP_TO_SUPERSPLAT_PLY.astype(np.float64), 1.0),
        (_M_ZUP_TO_UNITY_LH, 1.0),
    ):
        t_cloud = transform_gaussian_cloud(cloud, mat, translation=np.array([0.2, -0.3, 0.5]), scale=s)
        cov_after = _cov_from_cloud(t_cloud)
        expected = (s * s) * np.einsum("ij,njk,lk->nil", mat, cov_before, mat)
        assert np.allclose(cov_after, expected, atol=1e-4)
        assert t_cloud.sh_degree == 3


def _make_pitched_camera_w2c(eye: np.ndarray, look_at: np.ndarray) -> np.ndarray:
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fwd = look_at - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    r_c2w = np.stack([right, down, fwd], axis=1)
    r_w2c = r_c2w.T
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = r_w2c.astype(np.float32)
    w2c[:3, 3] = (-r_w2c @ eye).astype(np.float32)
    return w2c


@pytest.mark.parametrize("yaw_span_deg", [0.0, 25.0])
def test_align_gravity_pitched_forward_cameras_40_deg(tmp_path: Path, yaw_span_deg: float) -> None:
    ws = SceneWorkspace(tmp_path / f"pitch_{int(yaw_span_deg)}")
    ws.ensure_dirs()

    # Cameras pitched down 40 degrees toward a desktop floor at Z=0 + tall back wall at X=0.
    pitch_rad = np.deg2rad(40.0)
    frames = []
    for i in range(6):
        yaw = np.deg2rad(-0.5 * yaw_span_deg + yaw_span_deg * (i / 5.0))
        eye = np.array([2.0 * np.cos(yaw), 2.0 * np.sin(yaw) + (i - 2.5) * 0.15, 2.0 * np.tan(pitch_rad)], dtype=np.float64)
        look = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        frames.append({"id": i, "filename": f"f_{i:02d}.png", "world_to_cam": _make_pitched_camera_w2c(eye, look).tolist()})
    ws.cameras_path.write_text(json.dumps({"backend": "pycolmap", "frames": frames}), encoding="utf-8")

    rng = np.random.default_rng(77)
    floor = np.column_stack([rng.uniform(-1.5, 1.5, 240), rng.uniform(-1.5, 1.5, 240), np.zeros(240)]).astype(np.float32)
    wall = np.column_stack([np.full(320, -1.2), rng.uniform(-1.5, 1.5, 320), rng.uniform(0.1, 2.0, 320)]).astype(np.float32)
    pts = np.vstack([floor, wall])
    n = len(pts)
    save_gaussian_ply(
        ws.raw_splats_path,
        GaussianCloud(
            means=pts,
            scales=np.full((n, 3), -3.5, dtype=np.float32),
            quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
            opacities=np.full((n,), 2.0, dtype=np.float32),
            sh_dc=np.zeros((n, 3), dtype=np.float32),
        ),
    )

    res = AlignGravityScaleStage().run(ws, {"scale_factor": 1.0, "ransac_trials": 200, "distance_threshold": 0.03})
    assert res.metrics["aligned_normal_z"] > 0.995
    aligned = load_gaussian_ply(ws.aligned_splats_path)
    assert abs(float(np.median(aligned.means[:240, 2]))) < 0.05


def test_postprocess_order_swap_align_before_clean(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "swap_ws")
    ws.ensure_dirs()
    w2c = _make_pitched_camera_w2c(np.array([2.0, 0.0, 1.2]), np.array([0.0, 0.0, 0.0]))
    ws.cameras_path.write_text(
        json.dumps({"backend": "mock", "frames": [{"id": 0, "filename": "f0.png", "width": 64, "height": 64, "fx": 60.0, "world_to_cam": w2c.tolist()}]}),
        encoding="utf-8",
    )

    rng = np.random.default_rng(19)
    floor = np.column_stack([rng.uniform(-1.0, 1.0, 120), rng.uniform(-1.0, 1.0, 120), np.zeros(120)]).astype(np.float32)
    outlier = np.array([[0.0, 0.0, 15.0]], dtype=np.float32)
    pts = np.vstack([floor, outlier])
    n = len(pts)
    opac = np.full((n,), 2.0, dtype=np.float32)
    opac[-1] = -8.0  # ghost outlier cleaned only AFTER alignment
    save_gaussian_ply(
        ws.raw_splats_path,
        GaussianCloud(
            means=pts,
            scales=np.full((n, 3), -3.5, dtype=np.float32),
            quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
            opacities=opac,
            sh_dc=np.zeros((n, 3), dtype=np.float32),
        ),
    )

    AlignGravityScaleStage().run(ws, {})
    CleanFloatersStage().run(ws, {"min_opacity": 0.1})
    assert ws.active_splat_path().resolve() == ws.cleaned_splats_path.resolve()

    res = Ply3DGSExporterStage().run(ws, {"scene_name": "swapped"})
    meta = json.loads((ws.export_dir / "gaussian_splats" / "splat_meta.json").read_text(encoding="utf-8"))
    assert meta["coordinate_frame"] == "supersplat_colmap"
    assert meta["source_ply"].endswith("cleaned_splats.ply")
    assert res.metrics["exported_3dgs_points"] <= 120


def test_isotropic_marching_cubes_and_depth_normalized_clean(tmp_path: Path) -> None:
    rng = np.random.default_rng(55)
    # Flat 4.0 x 4.0 x 0.6 box
    pts = rng.uniform([-2.0, -2.0, -0.3], [2.0, 2.0, 0.3], size=(2500, 3)).astype(np.float32)
    cloud = GaussianCloud(
        means=pts,
        scales=np.full((2500, 3), -2.5, dtype=np.float32),
        quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2500, 1)),
        opacities=np.full((2500,), 2.0, dtype=np.float32),
        sh_dc=np.zeros((2500, 3), dtype=np.float32),
    )
    mesh, voxel_size, grid_dims = _build_marching_cubes_mesh(cloud, voxel_resolution=48, iso_level=0.15)
    assert len(mesh.faces) > 0
    # Isotropic grid: Z dimension is much smaller than X/Y for a flat 4.0 x 4.0 x 0.6 slab.
    assert grid_dims[0] >= grid_dims[2] * 3
    assert voxel_size > 0.0
    assert abs(float(mesh.bounds[1, 0] - mesh.bounds[0, 0]) - 4.0) < 0.8

    # All-transparent fallback preserves input unchanged with warning
    ws = SceneWorkspace(tmp_path / "all_trans_ws")
    ws.ensure_dirs()
    trans_cloud = GaussianCloud(
        means=pts[:16],
        scales=np.full((16, 3), -3.0, dtype=np.float32),
        quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (16, 1)),
        opacities=np.full((16,), -8.0, dtype=np.float32),
        sh_dc=np.zeros((16, 3), dtype=np.float32),
    )
    save_gaussian_ply(ws.raw_splats_path, trans_cloud)
    with pytest.warns(RuntimeWarning, match="不透明度"):
        res = CleanFloatersStage().run(ws, {"min_opacity": 0.2})
    assert res.metrics["retained_gaussians"] == 16


def test_ply_io_degree3_big_endian_pre_element_and_truncated(tmp_path: Path) -> None:
    ply_path = tmp_path / "big_endian_deg3.ply"
    n = 3
    num_rest = 45
    props = (
        ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(num_rest)]
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    header = (
        "\n".join(
            [
                "ply",
                "format binary_big_endian 1.0",
                "comment UTF-8 注释测试",
                "element cam 2",
                "property float cx",
                f"element vertex {n}",
                *[f"property float {p}" for p in props],
                "end_header\n",
            ]
        )
    ).encode("utf-8")
    cam_bytes = struct.pack(">2f", 320.0, 240.0)
    vert_vals = []
    for idx in range(n):
        row = [float(idx), float(idx + 1), float(idx + 2), 0.1, 0.2, 0.3]
        row.extend([0.01 * k for k in range(num_rest)])
        row.extend([1.5, -3.0, -3.0, -3.0, 1.0, 0.0, 0.0, 0.0])
        vert_vals.append(struct.pack(f">{len(props)}f", *row))
    ply_path.write_bytes(header + cam_bytes + b"".join(vert_vals))

    loaded = load_gaussian_ply(ply_path)
    assert loaded.num_points == 3
    assert loaded.sh_degree == 3
    assert loaded.sh_rest is not None and loaded.sh_rest.shape == (3, 45)
    assert np.allclose(loaded.means[1], [1.0, 2.0, 3.0], atol=1e-6)

    trunc_path = tmp_path / "truncated.ply"
    trunc_path.write_bytes((header + cam_bytes + b"".join(vert_vals))[:-10])
    with pytest.raises(ValueError, match="Truncated PLY"):
        load_gaussian_ply(trunc_path)


def test_mujoco_xml_escaping_plane_only_and_ceiling_safe_spawn(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "room_ws")
    ws.ensure_dirs()

    # Box room: floor at z=0.0 AND wide ceiling slab at z=2.5.
    floor_box = trimesh.creation.box(extents=[4.0, 4.0, 0.05], transform=trimesh.transformations.translation_matrix([0, 0, 0.0]))
    ceiling_box = trimesh.creation.box(extents=[4.0, 4.0, 0.05], transform=trimesh.transformations.translation_matrix([0, 0, 2.5]))
    room = trimesh.util.concatenate([floor_box, ceiling_box])
    room.export(str(ws.collision_mesh_path))

    pts = np.vstack([floor_box.sample(128), ceiling_box.sample(128)]).astype(np.float32)
    n = len(pts)
    save_gaussian_ply(
        ws.aligned_splats_path,
        GaussianCloud(
            means=pts,
            scales=np.full((n, 3), -3.0, dtype=np.float32),
            quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
            opacities=np.full((n,), 2.0, dtype=np.float32),
            sh_dc=np.zeros((n, 3), dtype=np.float32),
        ),
    )
    (ws.processed_dir / "alignment_transform.json").write_text(
        json.dumps({"matrix_4x4": np.eye(4).tolist(), "scale_factor": 1.0}), encoding="utf-8"
    )
    ws.set_active_splat(ws.aligned_splats_path)

    res = MujocoExporterStage().run(
        ws,
        {"model_name": 'lab "alpha" & <1>', "convex_parts": 1, "friction": [0.8]},
    )
    xml_text = (ws.export_dir / "mujoco" / "scene.xml").read_text(encoding="utf-8")
    root = ET.fromstring(xml_text)
    assert root.attrib["model"] == 'lab "alpha" & <1>'
    # When convex_parts=1 -> plane_only -> visual mesh has contype=0, no collidable full mesh.
    assert root.find("./worldbody/geom[@name='env_terrain']") is None
    ref_geom = root.find("./worldbody/geom[@name='env_surface_ref']")
    assert ref_geom is not None and ref_geom.attrib["contype"] == "0"

    meta = json.loads((ws.export_dir / "mujoco" / "scene_meta.json").read_text(encoding="utf-8"))
    assert meta["decomposition_method"] == "plane_only"
    assert 0.0 < meta["robot_spawn_xyz"][2] < 0.5  # Near floor (z=0.025+0.12), NOT ceiling (2.5)!
    assert res.metrics["decomposition_method"] == "plane_only"

    # Also check Unity default inria_rh mode preserves right-handed metadata
    UnityExporterStage().run(ws, {"scene_name": "room_unity"})
    u_meta = json.loads((ws.export_dir / "unity" / "unity_import.json").read_text(encoding="utf-8"))
    assert u_meta["coordinate_mode"] == "inria_rh"


def test_align_gravity_indoor_room_floor_not_desk_or_ceiling(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "indoor_room_ws")
    ws.ensure_dirs()

    # Multi-yaw indoor tour in OpenCV camera convention (world +Y is up -> w2c uses OpenCV -Y = world +Y).
    frames = []
    for i in range(8):
        yaw = np.deg2rad(i * 45.0)
        # Look horizontally in the XZ plane while circling around room center
        fwd = np.array([np.cos(yaw), -0.15, np.sin(yaw)], dtype=np.float64)
        fwd /= np.linalg.norm(fwd)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        r_c2w = np.stack([right, down, fwd], axis=1)
        r_w2c = r_c2w.T
        eye = np.array([-2.0 * np.cos(yaw), 0.0, -2.0 * np.sin(yaw)], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = r_w2c.astype(np.float32)
        w2c[:3, 3] = (-r_w2c @ eye).astype(np.float32)
        frames.append({"id": i, "filename": f"f_{i:02d}.png", "world_to_cam": w2c.tolist()})
    ws.cameras_path.write_text(json.dumps({"backend": "pycolmap", "frames": frames}), encoding="utf-8")

    rng = np.random.default_rng(101)
    # 40 sparse large floor Gaussians at y = -2.0 (scale [0.3, 0.3, 0.01])
    floor = np.column_stack([rng.uniform(-3.0, 3.0, 40), np.full(40, -2.0), rng.uniform(-3.0, 3.0, 40)]).astype(np.float32)
    floor_s = np.tile(np.log(np.array([[0.3, 0.3, 0.01]], dtype=np.float32)), (40, 1))
    # 400 dense tiny desk clutter Gaussians around y = -0.7 (tilted/dense trap if count-based!)
    desk = np.column_stack([rng.uniform(-1.0, 1.0, 400), np.full(400, -0.7), rng.uniform(-1.0, 1.0, 400)]).astype(np.float32)
    desk_s = np.tile(np.log(np.array([[0.01, 0.01, 0.01]], dtype=np.float32)), (400, 1))
    # 120 ceiling Gaussians at y = +1.8
    ceiling = np.column_stack([rng.uniform(-3.0, 3.0, 120), np.full(120, 1.8), rng.uniform(-3.0, 3.0, 120)]).astype(np.float32)
    ceil_s = np.tile(np.log(np.array([[0.25, 0.25, 0.02]], dtype=np.float32)), (120, 1))

    pts = np.vstack([floor, desk, ceiling])
    scales = np.vstack([floor_s, desk_s, ceil_s])
    n = len(pts)
    save_gaussian_ply(
        ws.raw_splats_path,
        GaussianCloud(
            means=pts,
            scales=scales,
            quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
            opacities=np.full((n,), 2.0, dtype=np.float32),
            sh_dc=np.zeros((n, 3), dtype=np.float32),
        ),
    )

    res = AlignGravityScaleStage().run(ws, {"scale_factor": 1.0, "ransac_trials": 200, "distance_threshold": 0.03})
    assert res.metrics["aligned_normal_z"] > 0.995
    aligned = load_gaussian_ply(ws.aligned_splats_path)
    assert abs(float(np.median(aligned.means[:40, 2]))) < 0.08
    assert float(np.median(aligned.means[40:440, 2])) > 1.0
    assert float(np.median(aligned.means[440:, 2])) > 3.2


def test_heightfield_strips_indoor_ceiling_lid() -> None:
    rng = np.random.default_rng(7)
    floor = np.column_stack([rng.uniform(-2.0, 2.0, 200), rng.uniform(-2.0, 2.0, 200), np.zeros(200)]).astype(np.float32)
    obstacle = np.column_stack([rng.uniform(-0.5, 0.5, 80), rng.uniform(-0.5, 0.5, 80), rng.uniform(0.2, 1.0, 80)]).astype(np.float32)
    ceiling = np.column_stack([rng.uniform(-2.0, 2.0, 200), rng.uniform(-2.0, 2.0, 200), np.full(200, 3.2)]).astype(np.float32)
    pts = np.vstack([floor, obstacle, ceiling])
    mesh = _build_heightfield_mesh(pts, grid_res=32)
    assert len(mesh.faces) > 0
    assert float(mesh.vertices[:, 2].max()) <= 1.5


def _make_surfels(
    pts: np.ndarray,
    normal: np.ndarray,
    *,
    thin: float = 0.004,
    broad: float = 0.14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create flat surfel positions, log-scales, and wxyz quats whose shortest axis equals `normal`."""
    n_vec = np.asarray(normal, dtype=np.float64)
    n_vec /= np.linalg.norm(n_vec) + 1e-12
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(n_vec[0]) < 0.8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n_vec, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n_vec, u)
    rot_single = np.stack([n_vec, u, v], axis=1)  # col 0 = normal (matches shortest scale axis)
    rots = np.broadcast_to(rot_single[None, :, :], (len(pts), 3, 3)).copy()
    quats = _matrices_to_quats_wxyz(rots)
    scales = np.tile(np.log(np.array([[thin, broad, broad]], dtype=np.float32)), (len(pts), 1))
    return pts.astype(np.float32), scales, quats


def test_align_gravity_surfel_irls_and_manhattan_heading(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "surfel_room_ws")
    ws.ensure_dirs()
    rng = np.random.default_rng(2026)

    floor_pts = np.column_stack([rng.uniform(-2.0, 2.0, 240), rng.uniform(-2.0, 2.0, 240), np.zeros(240)])
    ceil_pts = np.column_stack([rng.uniform(-2.0, 2.0, 160), rng.uniform(-2.0, 2.0, 160), np.full(160, 2.6)])
    wall_x_pts = np.column_stack([np.full(220, -2.2), rng.uniform(-2.0, 2.0, 220), rng.uniform(0.1, 2.5, 220)])
    wall_y_pts = np.column_stack([rng.uniform(-2.0, 2.0, 200), np.full(200, 2.0), rng.uniform(0.1, 2.5, 200)])
    clutter_pts = np.column_stack([rng.uniform(-0.8, 0.8, 120), rng.uniform(-0.8, 0.8, 120), rng.uniform(0.7, 1.1, 120)])

    f_p, f_s, f_q = _make_surfels(floor_pts, [0.0, 0.0, 1.0])
    c_p, c_s, c_q = _make_surfels(ceil_pts, [0.0, 0.0, 1.0])
    wx_p, wx_s, wx_q = _make_surfels(wall_x_pts, [1.0, 0.0, 0.0])
    wy_p, wy_s, wy_q = _make_surfels(wall_y_pts, [0.0, 1.0, 0.0])
    # Tilted clutter surfels mimicking messy chairs/cables above floor
    cl_n = np.array([np.sin(np.deg2rad(18)), 0.0, np.cos(np.deg2rad(18))])
    cl_p, cl_s, cl_q = _make_surfels(clutter_pts, cl_n, thin=0.008, broad=0.08)

    pts = np.vstack([f_p, c_p, wx_p, wy_p, cl_p])
    scales = np.vstack([f_s, c_s, wx_s, wy_s, cl_s])
    quats = np.vstack([f_q, c_q, wx_q, wy_q, cl_q])
    n_tot = len(pts)
    canon_cloud = GaussianCloud(
        means=pts,
        scales=scales,
        quats=quats,
        opacities=np.full((n_tot,), 2.0, dtype=np.float32),
        sh_dc=np.zeros((n_tot, 3), dtype=np.float32),
        sh_rest=(rng.normal(scale=0.1, size=(n_tot, 45))).astype(np.float32),
    )

    # Rotate whole scene & cameras into an arbitrary tilted raw SfM frame (6 deg roll, -4 deg pitch, 27 deg yaw).
    r_tilt = Rotation.from_euler("xyz", [6.0, -4.0, 27.0], degrees=True).as_matrix().astype(np.float64)
    raw_cloud = transform_gaussian_cloud(canon_cloud, r_tilt)
    save_gaussian_ply(ws.cleaned_splats_path, raw_cloud)
    ws.set_active_splat(ws.cleaned_splats_path)

    # Cameras walk in a diagonal heading ~28 deg off the corridor walls with downward pitch.
    frames = []
    for i in range(12):
        yaw = np.deg2rad(10.0 + i * 4.0)
        eye_c = np.array([1.8 * np.cos(yaw), 1.8 * np.sin(yaw), 1.35], dtype=np.float64)
        look_c = np.array([-0.5, 0.25, 0.85], dtype=np.float64)
        w2c_c = _make_pitched_camera_w2c(eye_c, look_c).astype(np.float64)
        w2c_raw = np.eye(4, dtype=np.float32)
        w2c_raw[:3, :3] = (w2c_c[:3, :3] @ r_tilt.T).astype(np.float32)
        w2c_raw[:3, 3] = w2c_c[:3, 3].astype(np.float32)
        frames.append({"id": i, "filename": f"f_{i:02d}.png", "world_to_cam": w2c_raw.tolist()})
    ws.cameras_path.write_text(json.dumps({"backend": "pycolmap", "frames": frames}), encoding="utf-8")

    res = AlignGravityScaleStage().run(ws, {"scale_factor": 1.0})
    tr = json.loads((ws.processed_dir / "alignment_transform.json").read_text(encoding="utf-8"))
    assert tr["alignment_method"] == "surfel_irls_manhattan"
    assert tr["used_dual_walls"] is True
    assert res.metrics["aligned_normal_z"] > 0.999

    rot_align = np.asarray(tr["matrix_4x4"], dtype=np.float64)[:3, :3]
    true_up_raw = r_tilt @ np.array([0.0, 0.0, 1.0])
    recovered_up = rot_align @ true_up_raw
    up_err_deg = float(np.rad2deg(np.arccos(np.clip(recovered_up[2], -1.0, 1.0))))
    assert up_err_deg < 0.5

    # Floor placed cleanly at Z=0
    aligned = load_gaussian_ply(ws.aligned_splats_path)
    assert abs(float(np.median(aligned.means[:240, 2]))) < 0.03

    # Wall normals aligned dead-square to Manhattan X or Y axes (< 1.0 deg).
    wall_mats = _quats_wxyz_to_matrices(aligned.quats[400:620])
    wall_norm_mean = wall_mats[:, :, 0].mean(axis=0)
    wall_norm_mean /= np.linalg.norm(wall_norm_mean)
    max_axis_dot = max(abs(float(wall_norm_mean[0])), abs(float(wall_norm_mean[1])))
    manhattan_err_deg = float(np.rad2deg(np.arccos(np.clip(max_axis_dot, -1.0, 1.0))))
    assert manhattan_err_deg < 1.0


def test_align_gravity_tilted_monitor_desk_not_hijacked(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "monitor_desk_ws")
    ws.ensure_dirs()
    rng = np.random.default_rng(99)

    desk_pts = np.column_stack([rng.uniform(-1.0, 1.0, 220), rng.uniform(-1.0, 1.0, 220), np.zeros(220)])
    # Single large monitor tilted 15 deg backward (NO orthogonal side wall -> dual-wall gate must reject it!)
    mon_y = rng.uniform(-0.7, 0.7, 260)
    mon_z = rng.uniform(0.15, 0.85, 260)
    mon_x = -0.6 - mon_z * np.tan(np.deg2rad(15.0))
    mon_pts = np.column_stack([mon_x, mon_y, mon_z])

    d_p, d_s, d_q = _make_surfels(desk_pts, [0.0, 0.0, 1.0], thin=0.003, broad=0.12)
    m_n = np.array([np.cos(np.deg2rad(15.0)), 0.0, np.sin(np.deg2rad(15.0))])
    m_p, m_s, m_q = _make_surfels(mon_pts, m_n, thin=0.003, broad=0.12)

    pts = np.vstack([d_p, m_p])
    scales = np.vstack([d_s, m_s])
    quats = np.vstack([d_q, m_q])
    canon_cloud = GaussianCloud(
        means=pts,
        scales=scales,
        quats=quats,
        opacities=np.full((len(pts),), 2.0, dtype=np.float32),
        sh_dc=np.zeros((len(pts), 3), dtype=np.float32),
    )

    r_tilt = Rotation.from_euler("xyz", [4.0, -3.0, 15.0], degrees=True).as_matrix().astype(np.float64)
    raw_cloud = transform_gaussian_cloud(canon_cloud, r_tilt)
    save_gaussian_ply(ws.cleaned_splats_path, raw_cloud)
    ws.set_active_splat(ws.cleaned_splats_path)

    frames = []
    for i in range(10):
        yaw = np.deg2rad(-35.0 + i * 7.5)
        eye_c = np.array([1.6 * np.cos(yaw), 1.6 * np.sin(yaw), 0.72], dtype=np.float64)
        w2c_c = _make_pitched_camera_w2c(eye_c, np.array([0.0, 0.0, 0.25])).astype(np.float64)
        w2c_raw = np.eye(4, dtype=np.float32)
        w2c_raw[:3, :3] = (w2c_c[:3, :3] @ r_tilt.T).astype(np.float32)
        w2c_raw[:3, 3] = w2c_c[:3, 3].astype(np.float32)
        frames.append({"id": i, "filename": f"f_{i:02d}.png", "world_to_cam": w2c_raw.tolist()})
    ws.cameras_path.write_text(json.dumps({"backend": "pycolmap", "frames": frames}), encoding="utf-8")

    AlignGravityScaleStage().run(ws, {"scale_factor": 1.0})
    tr = json.loads((ws.processed_dir / "alignment_transform.json").read_text(encoding="utf-8"))
    assert tr["used_dual_walls"] is False
    rot_align = np.asarray(tr["matrix_4x4"], dtype=np.float64)[:3, :3]
    recovered_up = rot_align @ (r_tilt @ np.array([0.0, 0.0, 1.0]))
    up_err_deg = float(np.rad2deg(np.arccos(np.clip(recovered_up[2], -1.0, 1.0))))
    assert up_err_deg < 1.5


def test_realign_set_front_and_manual_delta_preserves_sh3_and_cameras(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "realign_ws")
    ws.ensure_dirs()
    rng = np.random.default_rng(71)

    n = 32
    pts = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    sh_rest = rng.normal(scale=0.35, size=(n, 45)).astype(np.float32)
    cloud = GaussianCloud(
        means=pts,
        scales=np.full((n, 3), -3.0, dtype=np.float32),
        quats=quats,
        opacities=np.full((n,), 2.0, dtype=np.float32),
        sh_dc=rng.normal(scale=0.2, size=(n, 3)).astype(np.float32),
        sh_rest=sh_rest,
    )
    save_gaussian_ply(ws.aligned_splats_path, cloud)
    ws.set_active_splat(ws.aligned_splats_path)
    (ws.processed_dir / "alignment_transform.json").write_text(
        json.dumps({"matrix_4x4": np.eye(4).tolist(), "scale_factor": 1.0, "aligned_normal": [0.0, 0.0, 1.0]}),
        encoding="utf-8",
    )
    w2c = _make_pitched_camera_w2c(np.array([2.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
    ws.cameras_path.write_text(
        json.dumps({"backend": "pycolmap", "frames": [{"id": 0, "filename": "f0.png", "world_to_cam": w2c.tolist()}]}),
        encoding="utf-8",
    )
    Ply3DGSExporterStage().run(ws, {"scene_name": "realign_ws"})

    # Test set_front at yaw = 0.65 rad: viewing direction [-cos(0.65), -sin(0.65), 0] must become [-1, 0, 0].
    yaw_rad = 0.65
    d0 = np.tile(np.array([[-np.cos(yaw_rad), -np.sin(yaw_rad), 0.2]], dtype=np.float64), (n, 1))
    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
    c_before = eval_sh_rest_color(sh_rest, d0)

    r1 = realign_workspace_scene(ws, mode="set_front", current_yaw_rad=yaw_rad, reexport_ply=True)
    assert r1["aligned"] is True
    m1 = np.asarray(r1["alignment_transform"]["matrix_4x4"], dtype=np.float64)
    rot1 = m1[:3, :3]
    mapped_front = rot1 @ np.array([-np.cos(yaw_rad), -np.sin(yaw_rad), 0.0], dtype=np.float64)
    assert np.allclose(mapped_front, [-1.0, 0.0, 0.0], atol=1e-5)

    realigned_cloud = load_gaussian_ply(ws.aligned_splats_path)
    c_after = eval_sh_rest_color(realigned_cloud.sh_rest, d0 @ rot1.T)
    assert np.allclose(c_before, c_after, atol=1e-5)

    # Test manual delta composes cleanly onto m1
    r2 = realign_workspace_scene(
        ws, mode="manual", roll_deg=1.5, pitch_deg=-1.0, yaw_deg=5.0, z_offset=-0.05, reexport_ply=True
    )
    m2 = np.asarray(r2["alignment_transform"]["matrix_4x4"], dtype=np.float64)
    m_delta = np.eye(4, dtype=np.float64)
    m_delta[:3, :3] = Rotation.from_euler("xyz", [1.5, -1.0, 5.0], degrees=True).as_matrix()
    m_delta[:3, 3] = [0.0, 0.0, -0.05]
    assert np.allclose(m2, m_delta @ m1, atol=1e-4)
    assert (ws.export_dir / "gaussian_splats" / "scene_3dgs.ply").is_file()
    assert (ws.export_dir / "gaussian_splats" / "cameras.json").is_file()
