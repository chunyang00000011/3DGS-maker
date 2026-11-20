from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from real2sim_gs.core.config import IOConfig, PipelineConfig, PipelineStagesConfig, StageSpec
from real2sim_gs.core.doctor import is_gsplat_cuda_ready
from real2sim_gs.core.pipeline import PipelineRunner
from real2sim_gs.core.workspace import PipelineCancelledError, SceneWorkspace
from real2sim_gs.stages.ingest import MockIngestStage
from real2sim_gs.stages.pose import MockPoseStage
from real2sim_gs.stages.postprocess import AlignGravityScaleStage, CleanFloatersStage
from real2sim_gs.stages.reconstruct.gsplat_whitebox import (
    BilateralGridModule,
    CameraPoseOptModule,
    DensityControlStrategy,
    GaussianModel,
    GsplatWhiteboxStage,
    MCMCStrategy,
)
from real2sim_gs.stages.reconstruct.gsplat_whitebox.losses import anisotropy_reg_loss
from real2sim_gs.stages.reconstruct.gsplat_whitebox.trainer import (
    SelectiveAdam,
    _render_splats_pytorch,
    _render_view,
    _sync_mcmc_optimizer_state,
)
from real2sim_gs.stages.reconstruct.gsplat_whitebox.dataset import CameraView, PoseImageDataset
from real2sim_gs.utils.ply_io import SH_C0, GaussianCloud, load_gaussian_ply, save_gaussian_ply, transform_sh_degree1


def test_gaussian_ply_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 32
    cloud = GaussianCloud(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        scales=rng.uniform(-4.0, -1.0, size=(n, 3)).astype(np.float32),
        quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
        opacities=rng.uniform(-2.0, 2.0, size=(n,)).astype(np.float32),
        sh_dc=((rng.uniform(0.0, 1.0, size=(n, 3)) - 0.5) / SH_C0).astype(np.float32),
        sh_rest=rng.normal(scale=0.1, size=(n, 9)).astype(np.float32),
    )
    ply_path = tmp_path / "test_splats.ply"
    save_gaussian_ply(ply_path, cloud)
    loaded = load_gaussian_ply(ply_path)

    assert loaded.num_points == n
    assert np.allclose(loaded.means, cloud.means, atol=1e-6)
    assert np.allclose(loaded.scales, cloud.scales, atol=1e-6)
    assert np.allclose(loaded.quats, cloud.quats, atol=1e-6)
    assert np.allclose(loaded.opacities, cloud.opacities, atol=1e-6)
    assert np.allclose(loaded.sh_dc, cloud.sh_dc, atol=1e-6)
    assert loaded.sh_rest is not None
    assert np.allclose(loaded.sh_rest, cloud.sh_rest, atol=1e-6)

    # Verify SO(3) rotation preserves norm of degree-1 SH vectors and is reversible via R^T.
    theta = 0.73
    rot = np.array(
        [[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rot_sh = transform_sh_degree1(loaded.sh_rest, rot)
    restored_sh = transform_sh_degree1(rot_sh, rot.T)
    assert restored_sh is not None
    assert np.allclose(restored_sh, loaded.sh_rest, atol=1e-5)


def test_whitebox_sota_mcmc_bilateral_pose_opt(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "whitebox_sota_ws")
    ws.ensure_dirs()

    MockIngestStage().run(ws, {"mock_num_views": 4, "mock_image_size": 32})
    MockPoseStage().run(ws, {})

    res = GsplatWhiteboxStage().run(
        ws,
        {
            "iterations": 8,
            "backend": "pytorch",
            "strategy": "mcmc",
            "selective_adam": True,
            "enable_bilateral_grid": True,
            "enable_pose_opt": True,
            "densify_from_iter": 2,
            "densify_until_iter": 8,
            "densify_interval": 3,
            "mcmc_grow_ratio": 0.1,
            "min_opacity": 0.01,
        },
    )

    assert ws.raw_splats_path.exists()
    metrics = json.loads((ws.gaussians_dir / "metrics.json").read_text(encoding="utf-8"))
    assert math.isfinite(metrics["PSNR"])
    assert math.isfinite(metrics["SSIM"])
    assert metrics["strategy"] == "mcmc"
    assert metrics["selective_adam"] is True
    assert metrics["bilateral_grid"] is True
    assert metrics["pose_opt"] is True
    assert res.metrics["num_gaussians"] > 250

    # Verify BilateralGrid & CameraPoseOpt modules backpropagate clean gradients without NaNs.
    bgrid = BilateralGridModule(num_views=4)
    popt = CameraPoseOptModule(num_views=4)
    img = torch.rand(16, 16, 3, requires_grad=True)
    w2c = popt.refine_w2c(torch.eye(4), view_idx=1)
    out = bgrid(img, view_idx=1)
    loss = out.mean() + bgrid.tv_loss(1) + w2c.sum()
    loss.backward()
    b_grad = bgrid.embed.weight.grad
    assert b_grad is not None
    b_vals = b_grad.coalesce().values() if b_grad.is_sparse else b_grad
    assert torch.isfinite(b_vals).all()
    p_grad = popt.embed.weight.grad
    assert p_grad is not None
    p_vals = p_grad.coalesce().values() if p_grad.is_sparse else p_grad
    assert torch.isfinite(p_vals).all()


def test_density_control_absgs_and_opacity_reset() -> None:
    device = torch.device("cpu")
    means = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[-5.0, -5.0, -5.0], [-1.5, -1.5, -1.5], [-4.0, -4.0, -4.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, device=device)
    opacities = torch.tensor([2.0, 2.0, -10.0], device=device)
    sh_dc = torch.zeros((3, 3), device=device)
    model = GaussianModel(means, scales, quats, opacities, sh_dc)

    strategy = DensityControlStrategy(max_points=100, use_absgrad=True)
    strategy.reset_stats(model)
    model.means.grad = torch.tensor([[0.08, -0.05, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)
    strategy.record_gradients(model)

    changed = strategy.step_densify_and_prune(
        model, grad_threshold=0.01, min_opacity=0.01, split_scale_threshold=0.05
    )
    assert changed is True
    # Particle 0 cloned (+1), particle 1 split (-1 parent +2 children), particle 2 pruned (-1) -> 4.
    assert model.num_points == 4

    strategy.reset_opacities(model, max_opacity=0.01)
    assert float(model.activated_opacities.max().item()) <= 0.0101

    mcmc = MCMCStrategy(max_points=50)
    assert mcmc.relocate_and_grow(model, min_opacity=0.001, grow_ratio=0.4) is True
    assert model.num_points == 6


def test_mcmc_scale_clamp_and_optimizer_momentum_sync() -> None:
    device = torch.device("cpu")
    means = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[3.5, 3.5, 3.5], [-2.0, -2.0, -2.0], [-2.0, -2.0, -2.0], [-2.0, -2.0, -2.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, device=device)
    # Particle 2 is dead (sigmoid(-10) < 0.005); others survive.
    opacities = torch.tensor([2.0, 2.0, -10.0, 2.0], device=device)
    sh_dc = torch.zeros((4, 3), device=device)
    model = GaussianModel(means, scales, quats, opacities, sh_dc, max_log_scale=0.5)
    model.clamp_scales_()
    assert float(model.scales.max().item()) <= 0.5001

    scaled_params = {"lr_means": 1e-4, "lr_scales": 5e-3, "lr_quats": 1e-3, "lr_opacities": 5e-2, "lr_sh_dc": 2e-3}
    opt_sel = SelectiveAdam(model, scaled_params)
    opt_sel.state["means"]["exp_avg"].copy_(
        torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    )

    opt_adam = torch.optim.Adam(model.parameter_groups(scaled_params))
    opt_adam.zero_grad()
    (model.means.sum() + model.scales.sum() + model.quats.sum() + model.opacities.sum() + model.sh_dc.sum() + model.sh_rest.sum()).backward()
    opt_adam.step()
    opt_adam.state[model.means]["exp_avg"].copy_(
        torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    )

    mcmc = MCMCStrategy(max_points=10)
    assert mcmc.relocate_and_grow(model, min_opacity=0.005, grow_ratio=0.5) is True
    assert model.num_points == 6
    assert mcmc.last_dead_indices is not None and mcmc.last_dead_indices.tolist() == [2]
    assert mcmc.last_num_grown == 2
    assert float(model.scales.max().item()) <= 0.5001

    opt_sel = _sync_mcmc_optimizer_state(
        opt_sel, model, scaled_params, mcmc.last_dead_indices, mcmc.last_num_grown, lambda: SelectiveAdam(model, scaled_params)
    )
    assert opt_sel.state["means"]["exp_avg"].shape[0] == 6
    assert float(opt_sel.state["means"]["exp_avg"][1, 0].item()) == 2.0  # surviving moment kept
    assert float(opt_sel.state["means"]["exp_avg"][2, 0].item()) == 0.0  # teleported dead zeroed

    opt_adam = _sync_mcmc_optimizer_state(
        opt_adam,
        model,
        scaled_params,
        mcmc.last_dead_indices,
        mcmc.last_num_grown,
        lambda: torch.optim.Adam(model.parameter_groups(scaled_params)),
    )
    adam_m = opt_adam.state[model.means]["exp_avg"]
    assert adam_m.shape[0] == 6
    assert float(adam_m[1, 0].item()) == 2.0
    assert float(adam_m[2, 0].item()) == 0.0
    assert float(adam_m[5, 0].item()) == 0.0


def test_chunked_pytorch_ewa_renderer_consistency() -> None:
    device = torch.device("cpu")
    means = torch.tensor([[0.0, 0.0, 2.0], [0.25, -0.1, 2.5]], device=device)
    scales = torch.tensor([[-3.0, -3.5, -3.2], [-2.8, -3.0, -3.1]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.98, 0.1, 0.1, 0.0]], device=device)
    opacities = torch.tensor([1.5, 1.0], device=device)
    sh_dc = torch.tensor([[1.0, -0.5, 0.2], [-0.5, 1.0, 0.5]], device=device)

    model_full = GaussianModel(means.clone(), scales.clone(), quats.clone(), opacities.clone(), sh_dc.clone())
    model_chunk = GaussianModel(means.clone(), scales.clone(), quats.clone(), opacities.clone(), sh_dc.clone())
    view = CameraView(
        name="v0",
        width=24,
        height=20,
        fx=30.0,
        fy=30.0,
        cx=12.0,
        cy=10.0,
        world_to_cam=torch.eye(4, device=device),
        image=torch.zeros((20, 24, 3), device=device),
    )

    out_full = _render_splats_pytorch(model_full, view, w2c=view.world_to_cam, pixel_chunk_size=10_000)
    out_chunk = _render_splats_pytorch(model_chunk, view, w2c=view.world_to_cam, pixel_chunk_size=137)

    assert torch.allclose(out_full.rgb, out_chunk.rgb, atol=1e-6)
    assert torch.allclose(out_full.depth, out_chunk.depth, atol=1e-6)
    out_chunk.rgb.sum().backward()
    assert model_chunk.means.grad is not None and torch.isfinite(model_chunk.means.grad).all()


def test_gsplat_native_cuda_backend(tmp_path: Path) -> None:
    ws_mcmc = SceneWorkspace(tmp_path / "gsplat_cuda_mcmc")
    ws_mcmc.ensure_dirs()
    MockIngestStage().run(ws_mcmc, {"mock_num_views": 4, "mock_image_size": 64})
    MockPoseStage().run(ws_mcmc, {})

    res_mcmc = GsplatWhiteboxStage().run(
        ws_mcmc,
        {
            "iterations": 6,
            "backend": "gsplat",
            "strategy": "mcmc",
            "selective_adam": True,
            "enable_bilateral_grid": True,
            "enable_pose_opt": True,
            "densify_from_iter": 2,
            "densify_until_iter": 6,
            "densify_interval": 3,
            "mcmc_grow_ratio": 0.1,
            "min_opacity": 0.01,
        },
    )
    assert res_mcmc.metrics["render_backend"] == "gsplat_classic"
    assert res_mcmc.metrics["background_mode"] == "random"
    assert res_mcmc.metrics["sh_degree"] == 3
    assert math.isfinite(res_mcmc.metrics["PSNR"])
    assert math.isfinite(res_mcmc.metrics["SSIM"])
    saved_mcmc = load_gaussian_ply(ws_mcmc.raw_splats_path)
    assert saved_mcmc.sh_degree == 3
    assert saved_mcmc.sh_rest is not None and saved_mcmc.sh_rest.shape[1] == 45

    ws_adc = SceneWorkspace(tmp_path / "gsplat_cuda_adc")
    ws_adc.ensure_dirs()
    MockIngestStage().run(ws_adc, {"mock_num_views": 4, "mock_image_size": 64})
    MockPoseStage().run(ws_adc, {})

    res_adc = GsplatWhiteboxStage().run(
        ws_adc,
        {
            "iterations": 6,
            "backend": "gsplat",
            "strategy": "adc_absgs",
            "selective_adam": True,
            "enable_bilateral_grid": True,
            "enable_pose_opt": False,
            "densify_from_iter": 2,
            "densify_until_iter": 6,
            "densify_interval": 2,
            "grad_threshold": 1e-6,
            "min_opacity": 0.005,
        },
    )
    assert res_adc.metrics["render_backend"] == "gsplat_classic"
    assert res_adc.metrics["strategy"] == "adc_absgs"
    assert math.isfinite(res_adc.metrics["PSNR"])
    assert math.isfinite(res_adc.metrics["SSIM"])


def test_gsplat_packed_background_compositing() -> None:
    _, gsplat_ready, _ = is_gsplat_cuda_ready()
    if not (gsplat_ready and torch.cuda.is_available()):
        pytest.skip("gsplat CUDA backend unavailable")

    device = torch.device("cuda")
    torch.manual_seed(11)
    n = 33_000  # Exceeds 32_000 threshold -> forces packed=True
    means = torch.randn((n, 3), device=device)
    scales = torch.full((n, 3), -3.2, device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1)
    opac = torch.full((n,), 0.4, device=device)
    sh_dc = torch.zeros((n, 3), device=device)
    model = GaussianModel(means, scales, quats, opac, sh_dc)

    w2c = torch.eye(4, device=device)
    w2c[2, 3] = 4.0
    view = CameraView("v", 96, 96, 90.0, 90.0, 48.0, 48.0, w2c, torch.zeros((96, 96, 3), device=device))
    bg = torch.tensor([0.2, 0.6, 0.85], device=device, requires_grad=True)

    r_out = _render_view(
        model, view, w2c=w2c, use_gsplat=True, bg_color=bg, need_absgrad=True, need_depth=False, sh_degree=1
    )
    assert r_out.rgb.shape == (96, 96, 3)
    assert torch.isfinite(r_out.rgb).all()
    r_out.rgb.sum().backward()
    assert bg.grad is not None and torch.isfinite(bg.grad).all() and float(bg.grad.abs().sum().item()) > 0.0


def test_whitebox_cancel_and_checkpoint_resume(tmp_path: Path) -> None:
    ws_dir = tmp_path / "cancel_resume_ws"
    cfg = PipelineConfig(
        io=IOConfig(input_path=str(tmp_path / "unused"), workspace_dir=str(ws_dir)),
        pipeline=PipelineStagesConfig(
            ingest=StageSpec(name="mock_ingest", params={"mock_num_views": 4, "mock_image_size": 32}),
            pose=StageSpec(name="mock_pose", params={}),
            reconstruct=StageSpec(
                name="gsplat_whitebox",
                params={"iterations": 10, "backend": "pytorch", "strategy": "mcmc"},
            ),
            postprocess=[StageSpec(name="clean_floaters", params={"min_opacity": 0.05})],
            export=[StageSpec(name="ply_3dgs", params={"scene_name": "resume_scene"})],
        ),
    )

    stop_flag = {"hit": False}

    def _on_prog(stage_kind: str, state: str, info: dict) -> None:
        if stage_kind == "reconstruct" and state == "running":
            step_val = int((info.get("telemetry") or {}).get("step", 0))
            if step_val >= 5:
                stop_flag["hit"] = True

    runner = PipelineRunner()
    with pytest.raises(PipelineCancelledError):
        runner.run(
            cfg,
            mock=False,
            progress_callback=_on_prog,
            cancel_check=lambda: stop_flag["hit"],
        )

    ws = SceneWorkspace(ws_dir)
    assert ws.reconstruct_checkpoint_path.is_file()
    assert ws.reconstruct_checkpoint_meta_path.is_file()
    assert ws.raw_splats_path.is_file()
    ckpt_meta = json.loads(ws.reconstruct_checkpoint_meta_path.read_text(encoding="utf-8"))
    assert 5 <= int(ckpt_meta["step"]) < 10

    # Resume with force=False: should skip ingest+pose and finish reconstruct from saved step.
    res = runner.run(cfg, from_stage=None, force=False, mock=False)
    assert "ingest" in res.skipped_stages
    assert "pose" in res.skipped_stages
    assert "reconstruct" in res.executed_stages
    assert "export" in res.executed_stages
    assert int(res.metrics["resumed_from_step"]) == int(ckpt_meta["step"])
    assert not ws.reconstruct_checkpoint_path.exists()
    assert not ws.reconstruct_checkpoint_meta_path.exists()
    assert res.gaussian_ply_path is not None and res.gaussian_ply_path.is_file()


def test_pose_image_dataset_4k_downscale_and_uint8_storage(tmp_path: Path) -> None:
    from PIL import Image

    ws = SceneWorkspace(tmp_path / "dataset_4k_ws")
    ws.ensure_dirs()
    img_path = ws.frames_dir / "frame_0000.png"
    arr = np.full((3840, 2160, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    ws.cameras_path.write_text(
        json.dumps(
            {
                "backend": "pycolmap",
                "frames": [
                    {
                        "id": 1,
                        "filename": "frame_0000.png",
                        "camera_model": "SIMPLE_RADIAL",
                        "width": 2160,
                        "height": 3840,
                        "fx": 4608.0,
                        "fy": 4608.0,
                        "cx": 1080.0,
                        "cy": 1920.0,
                        "distortion": [0.0, 0.0, 0.0, 0.0],
                        "world_to_cam": np.eye(4, dtype=np.float32).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ds = PoseImageDataset(ws, device=torch.device("cpu"), max_image_dim=1600)
    assert len(ds) == 1
    v = ds[0]
    assert (v.width, v.height) == (900, 1600)
    assert math.isclose(v.fx, 4608.0 * (900.0 / 2160.0), rel_tol=1e-4)
    assert math.isclose(v.fy, 4608.0 * (1600.0 / 3840.0), rel_tol=1e-4)
    assert math.isclose(v.cx, 450.0, abs_tol=1e-2)
    assert math.isclose(v.cy, 800.0, abs_tol=1e-2)
    assert v._image_storage.dtype == torch.uint8
    assert v._image_storage.shape == (1600, 900, 3)
    img_f = v.image
    assert img_f.dtype == torch.float32
    assert torch.allclose(img_f, torch.full_like(img_f, 128.0 / 255.0), atol=1e-2)

    # Non-zero radial distortion triggers getOptimalNewCameraMatrix(alpha=0) undistortion + updated intrinsics.
    ws_dist = SceneWorkspace(tmp_path / "dataset_dist_ws")
    ws_dist.ensure_dirs()
    Image.fromarray(np.full((480, 640, 3), 160, dtype=np.uint8)).save(ws_dist.frames_dir / "frame_0000.png")
    ws_dist.cameras_path.write_text(
        json.dumps(
            {
                "backend": "pycolmap",
                "frames": [
                    {
                        "id": 1,
                        "filename": "frame_0000.png",
                        "camera_model": "SIMPLE_RADIAL",
                        "width": 640,
                        "height": 480,
                        "fx": 500.0,
                        "fy": 500.0,
                        "cx": 320.0,
                        "cy": 240.0,
                        "distortion": [0.15, 0.0, 0.0, 0.0],
                        "world_to_cam": np.eye(4, dtype=np.float32).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ds_dist = PoseImageDataset(ws_dist, device=torch.device("cpu"), max_image_dim=1600)
    assert ds_dist[0].fx > 500.0


def test_align_gravity_rejects_dominant_vertical_wall(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "wall_scene_ws")
    ws.ensure_dirs()

    # Camera looks horizontally toward +Z_cam = +Z_world, with camera UP = -Y_cam = +Y_world.
    w2c = np.eye(4, dtype=np.float32)
    w2c[1, 1] = -1.0  # R[1, :] = [0, -1, 0] -> camera UP = -R[1, :] = [0, +1, 0]
    w2c[2, 3] = 3.0
    ws.cameras_path.write_text(
        json.dumps({"backend": "pycolmap", "frames": [{"id": 1, "filename": "f0.png", "world_to_cam": w2c.tolist()}]}),
        encoding="utf-8",
    )

    # Construct 300 vertical wall points (at z=2, facing camera) + 120 horizontal floor points (at y=0).
    rng = np.random.default_rng(9)
    wall = np.column_stack(
        [rng.uniform(-2.0, 2.0, 300), rng.uniform(0.1, 2.5, 300), np.full(300, 2.0)]
    ).astype(np.float32)
    floor = np.column_stack(
        [rng.uniform(-2.0, 2.0, 120), np.zeros(120), rng.uniform(0.0, 2.0, 120)]
    ).astype(np.float32)
    pts = np.vstack([wall, floor])
    n = pts.shape[0]
    save_gaussian_ply(
        ws.raw_splats_path,
        GaussianCloud(
            means=pts,
            scales=np.full((n, 3), -3.5, dtype=np.float32),
            quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
            opacities=np.full((n,), 2.0, dtype=np.float32),
            sh_dc=np.zeros((n, 3), dtype=np.float32),
            sh_rest=np.ones((n, 9), dtype=np.float32) * 0.05,
        ),
    )

    res = AlignGravityScaleStage().run(ws, {"scale_factor": 1.0, "ransac_trials": 300, "distance_threshold": 0.03})
    assert res.metrics["aligned_normal_z"] >= 0.95
    aligned = load_gaussian_ply(ws.aligned_splats_path)
    # Horizontal floor points (last 120 points) must land near Z=0, while vertical wall points must rise above Z>0.
    floor_z = aligned.means[300:, 2]
    wall_z = aligned.means[:300, 2]
    assert abs(float(np.median(floor_z))) < 0.08
    assert float(np.median(wall_z)) > 0.5
    assert aligned.sh_rest is not None


def test_clean_floaters_prunes_elongated_needle_spikes(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "needle_clean_ws")
    ws.ensure_dirs()
    rng = np.random.default_rng(13)
    n_good = 100
    n_needles = 6
    pts = rng.uniform(-0.8, 0.8, size=(n_good + n_needles, 3)).astype(np.float32)
    scales_good = np.tile(np.array([[-3.2, -3.2, -3.8]], dtype=np.float32), (n_good, 1))
    # True needle: s_max / s_mid = exp(-0.1) / exp(-4.5) ~= 81.4 > 12.0, s_max > p85
    scales_needles = np.tile(np.array([[-0.1, -4.5, -4.5]], dtype=np.float32), (n_needles, 1))
    scales = np.vstack([scales_good, scales_needles])
    quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n_good + n_needles, 1))
    opac = np.full((n_good + n_needles,), 2.0, dtype=np.float32)  # sigmoid(2.0) ~ 0.88
    sh_dc = np.zeros((n_good + n_needles, 3), dtype=np.float32)

    save_gaussian_ply(
        ws.raw_splats_path,
        GaussianCloud(means=pts, scales=scales, quats=quats, opacities=opac, sh_dc=sh_dc),
    )
    res = CleanFloatersStage().run(ws, {"min_opacity": 0.02, "nb_neighbors": 8, "std_ratio": 5.0})
    cleaned = load_gaussian_ply(ws.cleaned_splats_path)
    # All 6 needle spikes must be pruned cleanly; nearly all surface splats kept.
    assert res.metrics["removed_floaters"] >= n_needles
    assert cleaned.num_points <= n_good
    s_sorted = np.sort(np.exp(cleaned.scales), axis=1)
    assert float((s_sorted[:, 2] / np.maximum(s_sorted[:, 1], 1e-6)).max()) <= 12.0


def test_anisotropy_loss_allows_flat_surfels_and_penalizes_needles() -> None:
    flat_surfels = torch.tensor([[0.05, 0.05, 0.001], [0.08, 0.07, 0.001]], dtype=torch.float32)
    needles = torch.tensor([[0.30, 0.003, 0.003], [0.25, 0.004, 0.002]], dtype=torch.float32)
    assert float(anisotropy_reg_loss(flat_surfels, max_ratio=8.0).item()) == pytest.approx(0.0, abs=1e-6)
    assert float(anisotropy_reg_loss(needles, max_ratio=8.0).item()) > 10.0


def test_clean_floaters_preserves_flat_surfels_and_sparse_walls(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "scale_aware_ws")
    ws.ensure_dirs()
    rng = np.random.default_rng(99)

    # Dense small tabletop surfels (150 flat pancake discs: s_max ~ s_mid = 0.015, s_min = 0.0005 -> s_max/s_min=30, s_max/s_mid=1)
    dense_pts = rng.uniform([-0.4, -0.4, 0.0], [0.4, 0.4, 0.2], size=(150, 3)).astype(np.float32)
    dense_scales = np.tile(np.log(np.array([[0.015, 0.015, 0.0005]], dtype=np.float32)), (150, 1))

    # Sparse large wall/ceiling/floor Gaussians (~0.25m radius, spacing ~0.35m) - must NOT be mistaken for floaters
    wx, wy = np.meshgrid(np.linspace(-2.0, 2.0, 7), np.linspace(-2.0, 2.0, 7))
    wall_pts = np.column_stack([wx.ravel(), wy.ravel(), np.full(wx.size, 2.5)]).astype(np.float32)
    wall_scales = np.tile(np.log(np.array([[0.25, 0.25, 0.01]], dtype=np.float32)), (len(wall_pts), 1))

    # Two isolated tiny mid-air floaters far from any surface and one needle spike
    floaters = np.array([[6.0, 6.0, 6.0], [-6.0, -6.0, 6.0], [0.1, 0.1, 0.1]], dtype=np.float32)
    floater_scales = np.vstack(
        [
            np.log(np.array([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]], dtype=np.float32)),
            np.log(np.array([[0.35, 0.004, 0.004]], dtype=np.float32)),
        ]
    )

    pts = np.vstack([dense_pts, wall_pts, floaters])
    scales = np.vstack([dense_scales, wall_scales, floater_scales])
    n = len(pts)
    quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))
    opac = np.full((n,), 2.0, dtype=np.float32)
    sh_dc = np.zeros((n, 3), dtype=np.float32)

    save_gaussian_ply(ws.raw_splats_path, GaussianCloud(means=pts, scales=scales, quats=quats, opacities=opac, sh_dc=sh_dc))
    res = CleanFloatersStage().run(ws, {})
    assert res.metrics["sor_mode"] == "scale_aware"
    cleaned = load_gaussian_ply(ws.cleaned_splats_path)
    # Sparse large ceiling/wall points must survive cleanly while the distant/needle floaters are removed.
    kept_ceiling = int((cleaned.means[:, 2] > 2.0).sum())
    assert kept_ceiling == len(wall_pts)
    assert cleaned.num_points >= 150 + len(wall_pts) - 2
    assert float(cleaned.means[:, 0].max()) < 5.0
