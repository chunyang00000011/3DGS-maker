from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from real2sim_gs.core.doctor import is_gsplat_cuda_ready
from real2sim_gs.core.pipeline import PipelineRunner
from real2sim_gs.core.config import IOConfig, PipelineConfig, PipelineStagesConfig, StageSpec
from real2sim_gs.core.workspace import PipelineCancelledError, SceneWorkspace
from real2sim_gs.stages.ingest import MockIngestStage
from real2sim_gs.stages.pose import MockPoseStage
from real2sim_gs.stages.reconstruct.gsplat_whitebox.dataset import CameraView
from real2sim_gs.stages.reconstruct.gsplat_whitebox.model import GaussianModel
from real2sim_gs.stages.reconstruct.gsplat_whitebox.strategy import (
    DensityControlStrategy,
    MCMCStrategy,
    compute_mcmc_relocation,
)
from real2sim_gs.stages.reconstruct.gsplat_whitebox.trainer import (
    GsplatWhiteboxStage,
    SelectiveAdam,
    _render_splats_pytorch,
    _reset_opacity_optimizer_state,
    _sync_gs_optimizer_state,
)


def test_mcmc_relocation_multiplicity_and_parent_momentum_reset() -> None:
    opac = torch.tensor([0.3, 0.7, 0.9], dtype=torch.float32)
    scales = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.2, 0.2], [0.05, 0.1, 0.15]], dtype=torch.float32)
    ratios = torch.tensor([1, 3, 5], dtype=torch.int32)
    o_cpu, s_cpu = compute_mcmc_relocation(opac, scales, ratios)
    assert torch.allclose(o_cpu[0], opac[0], atol=1e-5)
    assert torch.allclose(s_cpu[0], scales[0], atol=1e-5)

    if torch.cuda.is_available() and is_gsplat_cuda_ready()[1]:
        o_cuda, s_cuda = compute_mcmc_relocation(opac.cuda(), scales.cuda(), ratios.cuda())
        assert torch.allclose(o_cuda.cpu(), o_cpu, atol=1e-5)
        assert torch.allclose(s_cuda.cpu(), s_cpu, atol=1e-5)

    # Particle 0 is high-opacity parent; particles 1, 2, 3 are dead -> all 3 sample parent 0 (ratio = 4).
    torch.manual_seed(0)
    means = torch.zeros((4, 3), dtype=torch.float32)
    scales_log = torch.full((4, 3), -2.0, dtype=torch.float32)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=torch.float32)
    opac_logits = torch.tensor([1.5, -10.0, -10.0, -10.0], dtype=torch.float32)
    sh_dc = torch.zeros((4, 3), dtype=torch.float32)
    model = GaussianModel(means, scales_log, quats, opac_logits, sh_dc)

    scaled_params = {"lr_means": 1e-4, "lr_scales": 5e-3, "lr_quats": 1e-3, "lr_opacities": 5e-2, "lr_sh_dc": 2e-3}
    opt = SelectiveAdam(model, scaled_params)
    opt.state["means"]["exp_avg"].fill_(3.0)

    mcmc = MCMCStrategy(max_points=4)  # grow = 0, test dead relocation with multiplicity = 3
    changed = mcmc.relocate_and_grow(model, min_opacity=0.005, grow_ratio=0.0)
    assert changed is True

    exp_o, exp_s = compute_mcmc_relocation(
        torch.sigmoid(torch.tensor([1.5])),
        torch.exp(torch.tensor([[-2.0, -2.0, -2.0]])),
        torch.tensor([4], dtype=torch.int32),
    )
    assert torch.allclose(model.activated_opacities, exp_o.expand(4), atol=1e-4)
    assert torch.allclose(model.activated_scales, exp_s.expand(4, 3), atol=1e-4)

    opt = _sync_gs_optimizer_state(
        opt,
        model,
        scaled_params,
        lambda: SelectiveAdam(model, scaled_params),
        reset_idx=mcmc.last_reset_indices,
    )
    # Parent 0 AND dead children 1..3 must all have moments reset after relocation.
    assert float(opt.state["means"]["exp_avg"].abs().max().item()) == 0.0


def test_train_test_holdout_split_and_refined_cameras(tmp_path: Path) -> None:
    ws = SceneWorkspace(tmp_path / "split_ws")
    ws.ensure_dirs()
    MockIngestStage().run(ws, {"mock_num_views": 12, "mock_image_size": 32})
    MockPoseStage().run(ws, {})

    res = GsplatWhiteboxStage().run(
        ws,
        {
            "iterations": 8,
            "backend": "pytorch",
            "strategy": "mcmc",
            "test_every": 4,
            "enable_pose_opt": True,
            "enable_bilateral_grid": True,
            "lr_pose": 0.02,
            "pose_reg_weight": 0.0,
        },
    )
    assert res.metrics["num_train_views"] == 9
    assert res.metrics["num_test_views"] == 3
    assert res.metrics["eval_split"] == "test"
    assert "test_PSNR" in res.metrics and "train_PSNR" in res.metrics
    assert res.metrics["fallback_renderer_used"] is True

    refined_path = ws.gaussians_dir / "refined_cameras.json"
    assert refined_path.is_file()
    raw_cams = json.loads(ws.cameras_path.read_text(encoding="utf-8"))["frames"]
    ref_cams = json.loads(refined_path.read_text(encoding="utf-8"))["frames"]
    # Held-out test indices (0, 4, 8) stay unrefined; at least one visited train view changes pose.
    assert ref_cams[0]["pose_refined"] is False
    assert np.allclose(ref_cams[0]["world_to_cam"], raw_cams[0]["world_to_cam"], atol=1e-6)
    assert ref_cams[4]["pose_refined"] is False
    assert any(r["pose_refined"] for r in ref_cams)


def test_checkpoint_complete_state_resume_and_param_change_reset(tmp_path: Path) -> None:
    ws_dir = tmp_path / "ckpt_full_ws"
    cfg = PipelineConfig(
        io=IOConfig(input_path=str(tmp_path / "u.mp4"), workspace_dir=str(ws_dir)),
        pipeline=PipelineStagesConfig(
            ingest=StageSpec(name="mock_ingest", params={"mock_num_views": 4, "mock_image_size": 32}),
            pose=StageSpec(name="mock_pose", params={}),
            reconstruct=StageSpec(
                name="gsplat_whitebox",
                params={"iterations": 8, "backend": "pytorch", "strategy": "mcmc"},
            ),
            postprocess=[],
            export=[],
        ),
    )
    stop = {"hit": False}

    def _on_prog(stage: str, state: str, info: dict) -> None:
        if stage == "reconstruct" and state == "running":
            if int((info.get("telemetry") or {}).get("step", 0)) >= 4:
                stop["hit"] = True

    runner = PipelineRunner()
    with pytest.raises(PipelineCancelledError):
        runner.run(cfg, mock=False, progress_callback=_on_prog, cancel_check=lambda: stop["hit"])

    ws = SceneWorkspace(ws_dir)
    ckpt = torch.load(ws.reconstruct_checkpoint_path, map_location="cpu", weights_only=True)
    assert "gs_opt_state" in ckpt
    assert "rng_perm_state" in ckpt
    assert "torch_rng_state" in ckpt
    assert ckpt.get("params_signature")

    # Resuming with identical params picks up from saved step
    res_same = runner.run(cfg, mock=False)
    assert 4 <= res_same.metrics["resumed_from_step"] < 8

    # Now cancel again and alter iterations -> old checkpoint must be discarded (resumed_from_step == 0)
    stop["hit"] = False
    cfg.pipeline.reconstruct.params["iterations"] = 10
    with pytest.raises(PipelineCancelledError):
        runner.run(cfg, mock=False, progress_callback=_on_prog, cancel_check=lambda: stop["hit"])

    cfg.pipeline.reconstruct.params["iterations"] = 12
    res_changed = runner.run(cfg, mock=False)
    assert res_changed.metrics["resumed_from_step"] == 0


def test_adc_rotated_split_parent_removed_and_survivor_moments() -> None:
    torch.manual_seed(7)
    # 90-degree rotation around Z maps local X -> world Y!
    s2 = float(np.sqrt(0.5))
    q_rot90 = torch.tensor([[s2, 0.0, 0.0, s2]], dtype=torch.float32)
    n_test = 64
    means = torch.cat([torch.tensor([[0.0, 0.0, 0.0]]), torch.ones((n_test, 3))], dim=0)
    scales = torch.cat([torch.tensor([[-4.0, -4.0, -4.0]]), torch.tensor([[-0.5, -5.0, -5.0]]).repeat(n_test, 1)], dim=0)
    quats = torch.cat([torch.tensor([[1.0, 0.0, 0.0, 0.0]]), q_rot90.repeat(n_test, 1)], dim=0)
    opacities = torch.full((1 + n_test,), 2.0, dtype=torch.float32)
    sh_dc = torch.zeros((1 + n_test, 3), dtype=torch.float32)
    model = GaussianModel(means, scales, quats, opacities, sh_dc)

    scaled_params = {"lr_means": 1e-4, "lr_scales": 5e-3, "lr_quats": 1e-3, "lr_opacities": 5e-2, "lr_sh_dc": 2e-3}
    opt = SelectiveAdam(model, scaled_params)
    opt.state["means"]["exp_avg"][0].fill_(7.5)

    adc = DensityControlStrategy(max_points=500, use_absgrad=True)
    adc.reset_stats(model)
    # High gradient only on particles 1..n_test (split), zero on survivor 0
    ag = torch.zeros((1 + n_test,), dtype=torch.float32)
    ag[1:] = 0.01
    adc.record_gradients(model, absgrad_override=ag)

    changed = adc.step_densify_and_prune(model, grad_threshold=4e-4, min_opacity=0.005, split_scale_threshold=0.05)
    assert changed is True
    # 1 survivor (0) + 2*64 split children (parents removed) = 129
    assert model.num_points == 1 + 2 * n_test
    # Split children offsets must be predominantly along world Y (since rotated by 90 deg around Z).
    children = model.means.data[1:] - 1.0
    assert float(children[:, 1].std().item()) > 5.0 * float(children[:, 0].std().item())

    opt = _sync_gs_optimizer_state(
        opt,
        model,
        scaled_params,
        lambda: SelectiveAdam(model, scaled_params),
        src_idx=adc.last_survivor_indices,
        n_append=adc.last_num_appended,
    )
    # Survivor 0 kept its momentum, new split children started at zero.
    assert float(opt.state["means"]["exp_avg"][0, 0].item()) == 7.5
    assert float(opt.state["means"]["exp_avg"][1:, :].abs().max().item()) == 0.0

    opt.state["opacities"]["exp_avg"].fill_(2.0)
    adc.reset_opacities(model, max_opacity=0.01)
    _reset_opacity_optimizer_state(opt, model)
    assert float(opt.state["opacities"]["exp_avg"].abs().max().item()) == 0.0


def test_pytorch_fallback_renders_beyond_1024_splats() -> None:
    device = torch.device("cpu")
    # First 1500 splats are near camera (z=1.5) in the top-left corner;
    # remaining 1000 splats (indices 1500..2499) sit at z=2.5 right at the image center in bright red!
    n_near = 1500
    n_far = 1000
    means_near = torch.tensor([[-0.9, -0.9, 1.5]], device=device).repeat(n_near, 1)
    means_far = torch.tensor([[0.0, 0.0, 2.5]], device=device).repeat(n_far, 1)
    means = torch.cat([means_near, means_far], dim=0)
    scales = torch.full((n_near + n_far, 3), -2.5, device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n_near + n_far, 1)
    opac = torch.full((n_near + n_far,), 2.0, device=device)
    sh_dc = torch.zeros((n_near + n_far, 3), device=device)
    sh_dc[n_near:, 0] = 2.0  # bright red center splats beyond rank 1024

    model = GaussianModel(means, scales, quats, opac, sh_dc)
    view = CameraView("v", 32, 32, 40.0, 40.0, 16.0, 16.0, torch.eye(4, device=device), torch.zeros((32, 32, 3), device=device))
    bg = torch.zeros(3, device=device)
    out = _render_splats_pytorch(model, view, w2c=view.world_to_cam, bg_color=bg, splat_chunk_size=400)
    assert int(out.visible_mask.sum().item()) == n_near + n_far
    assert float(out.rgb[16, 16, 0].item()) > 0.7


def test_mip_3d_filter_consistency_and_bake() -> None:
    device = torch.device("cpu")
    means = torch.tensor([[0.0, 0.0, 2.0], [0.1, -0.1, 3.0]], device=device)
    scales = torch.tensor([[-4.5, -4.5, -4.5], [-4.0, -4.2, -4.4]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device)
    opac = torch.tensor([1.0, 1.5], device=device)
    sh_dc = torch.zeros((2, 3), device=device)
    model = GaussianModel(means, scales, quats, opac, sh_dc)
    view = CameraView("v", 64, 64, 60.0, 60.0, 32.0, 32.0, torch.eye(4, device=device), torch.zeros((64, 64, 3), device=device))

    f_var = model.compute_3d_filter([view], filter_variance_scale=0.2)
    assert torch.all(f_var > 0.0)
    s_mip, o_mip = model.mip_filtered_primitives()
    cloud = model.to_gaussian_cloud(bake_mip_filter=True)
    assert np.allclose(np.exp(cloud.scales), s_mip.detach().numpy(), atol=1e-4)
    assert np.allclose(cloud.sigmoid_opacities(), o_mip.detach().numpy(), atol=1e-4)
