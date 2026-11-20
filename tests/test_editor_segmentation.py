from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from real2sim_gs.api.server import STUDIO_TOKEN_HEADER, create_app
from real2sim_gs.core.doctor import is_gsplat_cuda_ready
from real2sim_gs.core.workspace import SceneWorkspace
from real2sim_gs.editor import (
    Sam3MaskPrediction,
    SceneEditManager,
    SegmentLiftConfig,
    apply_scene_edits_to_gpu_model,
    bake_edited_scene,
    build_orbit_camera_view,
    cloud_to_gpu_model,
    export_single_object_ply,
    lift_masks_to_3dgs,
    pick_object_at_pixel,
)
from real2sim_gs.editor.sam3_backend import Sam3Backend
from real2sim_gs.utils.ply_io import (
    GaussianCloud,
    load_gaussian_ply,
    save_gaussian_ply,
    transform_gaussian_cloud,
)


def _make_sh3_cloud(means: np.ndarray, scales: np.ndarray, opacs: np.ndarray, dc: np.ndarray, seed: int = 0) -> GaussianCloud:
    rng = np.random.default_rng(seed)
    n = len(means)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8
    sh_rest = (rng.normal(size=(n, 45)) * 0.15).astype(np.float32)
    return GaussianCloud(
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        quats=quats,
        opacities=opacs.astype(np.float32),
        sh_dc=dc.astype(np.float32),
        sh_rest=sh_rest,
    )


def test_3dgs_lifting_occlusion_gaussiancut_and_sagd() -> None:
    rng = np.random.default_rng(42)
    # 1. Foreground object cluster centered at [0, 0, 0.20]
    grid = np.linspace(-0.12, 0.12, 5)
    gx, gy, gz = np.meshgrid(grid, grid, np.linspace(0.08, 0.30, 5), indexing="ij")
    pts_obj = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    # 2. Occluded background wall at x = -0.85 behind the object
    wy, wz = np.meshgrid(np.linspace(-0.4, 0.4, 8), np.linspace(0.0, 0.5, 6), indexing="ij")
    pts_wall = np.stack([np.full(wy.size, -0.85), wy.ravel(), wz.ravel()], axis=1).astype(np.float32)

    # 3. Table plane below at z = 0.0
    tx, ty = np.meshgrid(np.linspace(-0.55, 0.55, 10), np.linspace(-0.55, 0.55, 10), indexing="ij")
    pts_table = np.stack([tx.ravel(), ty.ravel(), np.zeros(tx.size)], axis=1).astype(np.float32)

    # 4. Oversized boundary-spanning Gaussians right at the table/object seam (z = 0.04)
    pts_seam = np.array(
        [[0.12, 0.0, 0.04], [-0.12, 0.0, 0.04], [0.0, 0.12, 0.04], [0.0, -0.12, 0.04]],
        dtype=np.float32,
    )

    means = np.concatenate([pts_obj, pts_wall, pts_table, pts_seam], axis=0)
    n_obj = len(pts_obj)
    n_wall = len(pts_wall)
    n_table = len(pts_table)
    n_seam = len(pts_seam)

    scales = np.full((len(means), 3), -3.1, dtype=np.float32)
    scales[-n_seam:] = np.log(0.14)  # Oversized across boundary -> SAGD must decompose
    opacs = np.full(len(means), 2.8, dtype=np.float32)
    dc = np.zeros((len(means), 3), dtype=np.float32)
    dc[:n_obj] = 1.2
    dc[n_obj : n_obj + n_wall] = -0.8
    dc[n_obj + n_wall : n_obj + n_wall + n_table] = -0.3

    cloud = _make_sh3_cloud(means, scales, opacs, dc, seed=9)
    # Give seam Gaussians identity rotation so longest axis is clean
    cloud.quats[-n_seam:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, gsplat_ready, _ = is_gsplat_cuda_ready()
    use_gsplat = bool(gsplat_ready and device.type == "cuda")

    views = []
    masks = []
    for yaw in np.linspace(0.0, 2.0 * np.pi, num=6, endpoint=False):
        v = build_orbit_camera_view(
            yaw=float(yaw),
            pitch=0.22,
            distance=1.55,
            target=(0.0, 0.0, 0.18),
            width=80,
            height=64,
            device=device,
        )
        w2c = v.world_to_cam.cpu().numpy()
        p = pts_obj @ w2c[:3, :3].T + w2c[:3, 3]
        u = np.round(v.fx * p[:, 0] / p[:, 2] + v.cx).astype(int)
        y = np.round(v.fy * p[:, 1] / p[:, 2] + v.cy).astype(int)
        m = np.zeros((64, 80), dtype=bool)
        m[
            max(0, int(y.min()) - 2) : min(64, int(y.max()) + 3),
            max(0, int(u.min()) - 2) : min(80, int(u.max()) + 3),
        ] = True
        views.append(v)
        masks.append(m)

    res = lift_masks_to_3dgs(
        cloud,
        views,
        masks,
        SegmentLiftConfig(use_sagd=True, use_gaussiancut=True, max_sagd_splits=200),
        center_3d=np.array([0.0, 0.0, 0.19], dtype=np.float32),
        device=device,
        use_gsplat=use_gsplat,
    )

    sel_pts = res.cloud.means[res.fg_mask]
    assert len(sel_pts) >= int(n_obj * 0.90)
    # Occluded back wall at x = -0.85 must NOT be selected
    wall_false = int(np.count_nonzero(sel_pts[:, 0] < -0.45))
    assert wall_false == 0
    # Selected points stay confined to object volume
    assert float(sel_pts[:, 2].min()) > 0.01
    assert res.decomposed_count >= 1
    assert len(res.candidate_indices) > 0
    assert len(res.candidate_scores) == len(res.candidate_indices)


def test_scene_graph_sh3_rotation_duplicate_refine_undo_export_bake(tmp_path: Path) -> None:
    ws_root = tmp_path / "sg_ws"
    ws = SceneWorkspace(ws_root)
    ws.ensure_dirs()

    rng = np.random.default_rng(123)
    obj_pts = rng.uniform(-0.15, 0.15, size=(50, 3)).astype(np.float32) + np.array([0.0, 0.0, 0.3], dtype=np.float32)
    bg_pts = rng.uniform(-0.8, 0.8, size=(100, 3)).astype(np.float32) + np.array([1.5, 0.0, 0.0], dtype=np.float32)
    means = np.concatenate([obj_pts, bg_pts], axis=0)
    scales = np.full((150, 3), -3.0, dtype=np.float32)
    opacs = np.full(150, 2.5, dtype=np.float32)
    dc = np.zeros((150, 3), dtype=np.float32)
    dc[:50] = 0.8
    cloud = _make_sh3_cloud(means, scales, opacs, dc, seed=21)

    save_gaussian_ply(ws.aligned_splats_path, cloud)
    (ws.processed_dir / "alignment_transform.json").write_text(
        json.dumps({"matrix_4x4": np.eye(4).tolist(), "scale_factor": 1.0}),
        encoding="utf-8",
    )

    mgr = SceneEditManager(ws)
    fg_mask = np.zeros(150, dtype=bool)
    fg_mask[:50] = True
    cand_idx = np.arange(65, dtype=np.int32)
    cand_scores = np.full(65, 0.85, dtype=np.float32)
    cand_scores[45:50] = 0.55
    cand_scores[50:65] = 0.35

    reg = mgr.register_segmented_object(
        cloud=cloud,
        fg_mask=fg_mask,
        candidate_indices=cand_idx,
        candidate_scores=cand_scores,
        name="茶杯",
        prompt_summary="text:cup",
        threshold=0.5,
    )
    assert reg["object"]["id"] == 1
    assert reg["object"]["point_count"] == 50

    # GPU rigid + SH0..3 O(3) rotation equivalence against CPU transform_gaussian_cloud
    euler = [15.0, -25.0, 60.0]
    trans = [0.25, -0.1, 0.05]
    upd = mgr.update_object(1, translation=trans, rotation_euler_deg=euler, scale=1.25, record_undo=True)
    obj_dict = upd["object"]

    device = torch.device("cpu")
    base_model = cloud_to_gpu_model(cloud, device)
    state, obj_ids = mgr.load_state()
    gpu_composed, gpu_oids = apply_scene_edits_to_gpu_model(base_model, obj_ids, state)
    assert gpu_oids is not None

    sel_gpu = gpu_oids.detach().numpy() == 1
    cpu_obj_expected = SceneEditManager._transform_cloud_cpu(cloud.subset(fg_mask), obj_dict)
    np.testing.assert_allclose(gpu_composed.means[sel_gpu].detach().numpy(), cpu_obj_expected.means, atol=1e-4)
    np.testing.assert_allclose(gpu_composed.sh_rest[sel_gpu].detach().numpy(), cpu_obj_expected.sh_rest, atol=1e-4)

    # Pixel picking hits Object 1 when looking right at transformed pivot
    piv_t = obj_dict["transformed_pivot"]
    pick_v = build_orbit_camera_view(
        yaw=0.0,
        pitch=0.1,
        distance=1.2,
        target=piv_t,
        width=96,
        height=72,
        device=device,
    )
    picked_id, hit_xyz = pick_object_at_pixel(
        gpu_composed, gpu_oids, pick_v, 48.0, 36.0, use_gsplat=False
    )
    assert picked_id == 1
    assert hit_xyz is not None

    # Instant threshold refinement & undo/redo
    ref = mgr.refine_object_mask(1, threshold=0.65, z_clip=0.0, record_undo=True)
    assert ref["object"]["point_count"] == 45
    undone = mgr.perform_action("undo")
    assert undone["changed"] is True
    assert undone["objects"][0]["point_count"] == 50

    # Duplicate, delete original, export clone PLY, bake scene
    dup = mgr.perform_action("duplicate", 1)
    clone_id = dup["object"]["id"]
    assert clone_id == 2
    assert dup["object"]["is_clone"] is True

    mgr.perform_action("delete", 1)
    single_path, dl_name = export_single_object_ply(ws, clone_id, manager=mgr)
    assert single_path.is_file()
    assert dl_name.endswith(".ply")
    loaded_single = load_gaussian_ply(single_path)
    assert loaded_single.num_points == 50
    assert loaded_single.sh_degree == 3

    baked = bake_edited_scene(ws, manager=mgr, reexport_ply=True)
    assert baked["baked"] is True
    assert baked["kept_objects"] == 1
    assert baked["deleted_objects"] == 1
    assert baked["total_points"] == 150  # 100 bg + 50 clone (original deleted)
    assert ws.active_splat_path().resolve() == (ws.processed_dir / "edited_splats.ply").resolve()
    assert (ws.export_dir / "gaussian_splats" / "scene_3dgs.ply").is_file()


def test_fastapi_editor_endpoints_and_ui_toolbar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws_root = tmp_path / "workspaces"
    app = create_app(workspaces_root=ws_root, allowed_input_roots=[tmp_path])
    client = TestClient(app, headers={STUDIO_TOKEN_HEADER: app.state.studio_token})
    unauthed = TestClient(app)

    # UI HTML contains dedicated segmentation & editing toolbar controls
    home = client.get("/")
    assert home.status_code == 200
    for dom_id in (
        "btnToggleEditor",
        "editorToolbar",
        "btnEdFocus",
        "chkEdAutoFocus",
        "btnEdExportObjPly",
        "btnEdBake",
        "editorDrawer",
    ):
        assert dom_id in home.text

    # Prepare a completed mock scene
    inline_cfg = yaml.safe_load(Path("configs/default_mujoco.yaml").read_text(encoding="utf-8"))
    ws_dir = ws_root / "editor_ws"
    inline_cfg["io"]["workspace_dir"] = str(ws_dir)
    job_res = client.post("/jobs", json={"inline_config": inline_cfg, "mock": True, "run_async": False})
    assert job_res.status_code == 200

    # Security guard on mutating editor routes
    assert unauthed.post("/api/editor/sam3-config", json={"unload": True}).status_code == 403

    st_res = client.get("/api/editor/status")
    assert st_res.status_code == 200
    assert "transformers_installed" in st_res.json()
    assert "model_id" in st_res.json()

    # Deterministic mock SAM 3 predictions around image center
    def _fake_predict(self, rgb_u8: np.ndarray, *, points=None, labels=None, box=None, text=None) -> Sam3MaskPrediction:
        h, w = rgb_u8.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)] = True
        return Sam3MaskPrediction(mask=mask, score=0.93, boxes=[[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], backend="Sam3Mock", elapsed_ms=1.5)

    def _fake_track(self, frames_rgb, **kwargs):
        return [
            _fake_predict(self, f).mask
            for f in frames_rgb
        ]

    monkeypatch.setattr(Sam3Backend, "predict_image_prompt", _fake_predict)
    monkeypatch.setattr(Sam3Backend, "track_multiview", _fake_track)

    prev2d = client.post(
        "/api/editor/preview-2d",
        json={
            "workspace_dir": str(ws_dir),
            "yaw": 0.72,
            "pitch": 0.48,
            "distance": 2.4,
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.2,
            "width": 320,
            "height": 240,
            "points": [[160.0, 120.0]],
            "labels": [1],
        },
    )
    assert prev2d.status_code == 200
    p2_body = prev2d.json()
    assert len(p2_body["mask_png_b64"]) > 20
    assert p2_body["area_ratio"] > 0.1

    seg3d = client.post(
        "/api/editor/segment-3d",
        json={
            "workspace_dir": str(ws_dir),
            "yaw": 0.72,
            "pitch": 0.48,
            "distance": 2.4,
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.2,
            "width": 320,
            "height": 240,
            "points": [[160.0, 120.0]],
            "labels": [1],
            "text": "object",
            "name": "测试物体",
            "num_views": 4,
            "use_sagd": True,
            "use_gaussiancut": True,
        },
    )
    assert seg3d.status_code == 200
    seg_body = seg3d.json()
    obj = seg_body["object"]
    oid = obj["id"]
    assert obj["name"] == "测试物体"
    assert obj["point_count"] > 0

    list_res = client.get("/api/editor/objects", params={"workspace_dir": str(ws_dir)})
    assert list_res.status_code == 200
    assert len(list_res.json()["objects"]) == 1

    upd_res = client.post(
        "/api/editor/update-object",
        json={
            "workspace_dir": str(ws_dir),
            "object_id": oid,
            "translation": [0.15, -0.05, 0.02],
            "rotation_euler_deg": [0.0, 0.0, 35.0],
            "scale": 1.1,
            "record_undo": True,
        },
    )
    assert upd_res.status_code == 200

    render_res = client.get(
        "/api/render-view",
        params={
            "workspace_dir": str(ws_dir),
            "width": 240,
            "height": 180,
            "selected_id": oid,
            "highlight": 1,
            "dim_bg": 1,
        },
    )
    assert render_res.status_code == 200
    assert render_res.headers["content-type"] == "image/jpeg"

    piv = upd_res.json()["object"]["transformed_pivot"]
    pick_res = client.post(
        "/api/editor/pick-object",
        json={
            "workspace_dir": str(ws_dir),
            "yaw": 0.72,
            "pitch": 0.48,
            "distance": 1.6,
            "tx": piv[0],
            "ty": piv[1],
            "tz": piv[2],
            "width": 240,
            "height": 180,
            "u": 120.0,
            "v": 90.0,
        },
    )
    assert pick_res.status_code == 200
    assert pick_res.json()["object_id"] == oid

    dup_res = client.post(
        "/api/editor/object-action",
        json={"workspace_dir": str(ws_dir), "action": "duplicate", "object_id": oid},
    )
    assert dup_res.status_code == 200
    assert len(dup_res.json()["objects"]) == 2

    undo_res = client.post(
        "/api/editor/object-action",
        json={"workspace_dir": str(ws_dir), "action": "undo"},
    )
    assert undo_res.status_code == 200
    assert len(undo_res.json()["objects"]) == 1

    exp_res = client.get(
        "/api/editor/export-object-ply",
        params={"workspace_dir": str(ws_dir), "object_id": oid},
    )
    assert exp_res.status_code == 200
    assert b"format binary_little_endian" in exp_res.content[:400]
    tmp_obj_ply = tmp_path / "dl_obj.ply"
    tmp_obj_ply.write_bytes(exp_res.content)
    assert load_gaussian_ply(tmp_obj_ply).num_points == upd_res.json()["object"]["point_count"]

    bake_res = client.post(
        "/api/editor/bake",
        json={"workspace_dir": str(ws_dir), "reexport_ply": True},
    )
    assert bake_res.status_code == 200
    assert bake_res.json()["baked"] is True
    assert (ws_dir / "04_processed" / "edited_splats.ply").is_file()
    assert (ws_dir / "05_export" / "gaussian_splats" / "scene_3dgs.ply").is_file()


def test_editor_bugfix_regressions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from real2sim_gs.editor.sam3_backend import (
        default_local_sam3_dir,
        get_shared_sam3_backend,
        is_valid_local_sam3_dir,
        resolve_sam3_model_id,
    )

    ws_root = tmp_path / "workspaces"
    app = create_app(workspaces_root=ws_root, allowed_input_roots=[tmp_path])
    client = TestClient(app, headers={STUDIO_TOKEN_HEADER: app.state.studio_token})

    inline_cfg = yaml.safe_load(Path("configs/default_mujoco.yaml").read_text(encoding="utf-8"))
    ws_dir = ws_root / "reg_ws"
    inline_cfg["io"]["workspace_dir"] = str(ws_dir)
    assert client.post("/jobs", json={"inline_config": inline_cfg, "mock": True, "run_async": False}).status_code == 200

    # 1. UNC guard & model_path="" reset
    try:
        assert client.get("/api/editor/status", params={"model_path": r"\\evil\share\sam3"}).status_code == 403
        assert client.post("/api/editor/sam3-config", json={"model_path": r"\\evil\share\sam3"}).status_code == 403
        custom_dir = tmp_path / "custom_sam3"
        custom_dir.mkdir()
        r_set = client.post("/api/editor/sam3-config", json={"model_path": str(custom_dir)})
        assert r_set.status_code == 200
        assert r_set.json()["model_id"] == str(custom_dir)
        assert r_set.json()["ready"] is False  # Missing config.json & model.safetensors

        r_rst = client.post("/api/editor/sam3-config", json={"model_path": ""})
        assert r_rst.status_code == 200
        assert r_rst.json()["model_id"] == resolve_sam3_model_id()
    finally:
        get_shared_sam3_backend().configure(None)

    # 2. High-res viewport (>= 1920x1080) must pass preview-2d, segment-3d, and pick-object cleanly
    def _fake_predict(self, rgb_u8: np.ndarray, *, points=None, labels=None, box=None, text=None) -> Sam3MaskPrediction:
        h, w = rgb_u8.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)] = True
        return Sam3MaskPrediction(mask=mask, score=0.91, boxes=[[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], backend="Sam3Mock", elapsed_ms=1.0)

    def _fake_track(self, frames_rgb, **kwargs):
        return [_fake_predict(self, f).mask for f in frames_rgb]

    monkeypatch.setattr(Sam3Backend, "predict_image_prompt", _fake_predict)
    monkeypatch.setattr(Sam3Backend, "track_multiview", _fake_track)

    prev2d = client.post(
        "/api/editor/preview-2d",
        json={"workspace_dir": str(ws_dir), "width": 1920, "height": 1080, "points": [[960.0, 540.0]], "labels": [1]},
    )
    assert prev2d.status_code == 200
    assert prev2d.json()["width"] == 960
    assert prev2d.json()["height"] == 540

    seg3d = client.post(
        "/api/editor/segment-3d",
        json={"workspace_dir": str(ws_dir), "width": 1920, "height": 1080, "points": [[960.0, 540.0]], "labels": [1]},
    )
    assert seg3d.status_code == 200
    oid = int(seg3d.json()["object"]["id"])

    pick_hi = client.post(
        "/api/editor/pick-object",
        json={"workspace_dir": str(ws_dir), "width": 1920, "height": 1080, "u": 960.0, "v": 540.0},
    )
    assert pick_hi.status_code == 200

    # 3. Gizmo drag sequence: first event records undo, subsequent events update in place, Undo returns to [0,0,0]
    assert client.post(
        "/api/editor/update-object",
        json={"workspace_dir": str(ws_dir), "object_id": oid, "translation": [0.1, 0.0, 0.0], "record_undo": True},
    ).status_code == 200
    assert client.post(
        "/api/editor/update-object",
        json={"workspace_dir": str(ws_dir), "object_id": oid, "translation": [0.45, -0.2, 0.1], "record_undo": False},
    ).status_code == 200
    u_res = client.post("/api/editor/object-action", json={"workspace_dir": str(ws_dir), "action": "undo"})
    assert u_res.status_code == 200
    assert u_res.json()["objects"][0]["translation"] == [0.0, 0.0, 0.0]

    # 4. Stale SAGD score cache unlinking + no bogus undo entry pushed on refine failure
    ws_sg = SceneWorkspace(tmp_path / "sg_ws")
    ws_sg.ensure_dirs()
    n = 12
    pts = np.stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)], axis=1).astype(np.float32)
    cloud = _make_sh3_cloud(pts, np.full((n, 3), -3.0, dtype=np.float32), np.full(n, 2.0, dtype=np.float32), np.zeros((n, 3), dtype=np.float32))
    save_gaussian_ply(ws_sg.raw_splats_path, cloud)
    ws_sg.set_active_splat(ws_sg.raw_splats_path)

    mgr = SceneEditManager(ws_sg)
    fg1 = np.zeros(n, dtype=bool)
    fg1[:4] = True
    r1 = mgr.register_segmented_object(
        cloud=cloud,
        fg_mask=fg1,
        candidate_indices=np.arange(0, 6, dtype=np.int32),
        candidate_scores=np.array([0.8, 0.8, 0.8, 0.8, 0.2, 0.2], dtype=np.float32),
        name="obj1",
        decomposed=False,
    )
    oid1 = r1["object"]["id"]
    assert mgr._scores_path(1).is_file()

    # Over-strict threshold must fail WITHOUT pushing a duplicate entry onto _undo_stack
    undo_before = len(mgr._undo_stack)
    with pytest.raises(ValueError, match="选中点数为 0"):
        mgr.refine_object_mask(oid1, threshold=0.95, record_undo=True)
    assert len(mgr._undo_stack) == undo_before

    # Registering decomposed=True must clear old score cache file & flag
    fg2 = np.zeros(n + 1, dtype=bool)
    fg2[4:8] = True
    cloud2 = _make_sh3_cloud(
        np.vstack([pts, [[2.0, 0.0, 0.0]]]),
        np.full((n + 1, 3), -3.0, dtype=np.float32),
        np.full(n + 1, 2.0, dtype=np.float32),
        np.zeros((n + 1, 3), dtype=np.float32),
    )
    ids2 = np.zeros(n + 1, dtype=np.int32)
    ids2[:4] = 1
    mgr.register_segmented_object(
        cloud=cloud2,
        fg_mask=fg2,
        candidate_indices=np.arange(4, 10, dtype=np.int32),
        candidate_scores=np.array([0.7, 0.7, 0.7, 0.7, 0.2, 0.2], dtype=np.float32),
        name="obj2",
        existing_object_ids=ids2,
        decomposed=True,
    )
    assert not mgr._scores_path(1).is_file()
    with pytest.raises(ValueError, match="失效"):
        mgr.refine_object_mask(oid1, threshold=0.45)

    # 5. Sam3Backend._store_tracker_video_frame handles 4D (1,1,H,W) bfloat16 tensors with real or stub processor
    pm_4d = torch.ones((1, 1, 16, 16), dtype=torch.bfloat16) * 5.0
    dest: dict[int, np.ndarray] = {}
    if is_valid_local_sam3_dir(default_local_sam3_dir()):
        import transformers
        proc = transformers.Sam3TrackerVideoProcessor.from_pretrained(str(default_local_sam3_dir()))
        Sam3Backend._store_tracker_video_frame(proc, SimpleNamespace(pred_masks=pm_4d), 0, 120, 160, dest)
        assert dest[0].shape == (120, 160)
        assert bool(dest[0].all())
    else:
        class _FakeProc:
            def post_process_masks(self, masks, original_sizes, binarize=True):
                assert isinstance(masks, list) and masks[0].ndim == 4
                return [torch.ones((1, 1, original_sizes[0][0], original_sizes[0][1]), dtype=torch.bool)]
        Sam3Backend._store_tracker_video_frame(_FakeProc(), SimpleNamespace(pred_masks=pm_4d), 0, 120, 160, dest)
        assert dest[0].shape == (120, 160)


def test_api_editor_align_scene_modes_and_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws_root = tmp_path / "workspaces"
    app = create_app(workspaces_root=ws_root, allowed_input_roots=[tmp_path])
    unauthed = TestClient(app)
    client = TestClient(app, headers={STUDIO_TOKEN_HEADER: app.state.studio_token})

    inline_cfg = yaml.safe_load(Path("configs/default_mujoco.yaml").read_text(encoding="utf-8"))
    ws_dir = ws_root / "align_api_ws"
    inline_cfg["io"]["workspace_dir"] = str(ws_dir)
    assert client.post("/jobs", json={"inline_config": inline_cfg, "mock": True, "run_async": False}).status_code == 200

    # Missing X-Studio-Token must return 403
    assert unauthed.post("/api/editor/align-scene", json={"workspace_dir": str(ws_dir), "mode": "auto"}).status_code == 403

    r_auto = client.post("/api/editor/align-scene", json={"workspace_dir": str(ws_dir), "mode": "auto"})
    assert r_auto.status_code == 200
    assert r_auto.json()["aligned"] is True

    r_front = client.post(
        "/api/editor/align-scene",
        json={"workspace_dir": str(ws_dir), "mode": "set_front", "current_yaw_rad": 0.5},
    )
    assert r_front.status_code == 200
    assert r_front.json()["alignment_transform"]["alignment_method"] == "manual_set_front"

    r_man = client.post(
        "/api/editor/align-scene",
        json={"workspace_dir": str(ws_dir), "mode": "manual", "roll_deg": 0.5, "yaw_deg": 12.0, "z_offset": 0.08},
    )
    assert r_man.status_code == 200
    assert r_man.json()["alignment_transform"]["alignment_method"] == "manual_manual"

    prev_res = client.get("/api/scene-preview", params={"workspace_dir": str(ws_dir)})
    assert prev_res.status_code == 200
    assert prev_res.json()["alignment_transform"]["alignment_method"] == "manual_manual"

    # Once segmented objects exist in Scene Graph, global align-scene must cleanly block with 400
    def _fake_predict(self, rgb_u8: np.ndarray, **kwargs) -> Sam3MaskPrediction:
        h, w = rgb_u8.shape[:2]
        m = np.zeros((h, w), dtype=bool)
        m[int(h * 0.3) : int(h * 0.7), int(w * 0.3) : int(w * 0.7)] = True
        return Sam3MaskPrediction(mask=m, score=0.9, boxes=[[w * 0.3, h * 0.3, w * 0.7, h * 0.7]], backend="Mock", elapsed_ms=1.0)

    monkeypatch.setattr(Sam3Backend, "predict_image_prompt", _fake_predict)
    monkeypatch.setattr(Sam3Backend, "track_multiview", lambda self, frames_rgb, **kw: [_fake_predict(self, f).mask for f in frames_rgb])
    seg_res = client.post(
        "/api/editor/segment-3d",
        json={"workspace_dir": str(ws_dir), "width": 320, "height": 240, "points": [[160.0, 120.0]], "labels": [1]},
    )
    assert seg_res.status_code == 200
    blocked = client.post("/api/editor/align-scene", json={"workspace_dir": str(ws_dir), "mode": "auto"})
    assert blocked.status_code == 400
    assert "已分割" in blocked.json()["detail"]

