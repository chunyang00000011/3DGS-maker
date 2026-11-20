from __future__ import annotations

from pathlib import Path
from typing import Callable

from real2sim_gs.core.config import PipelineConfig, load_config
from real2sim_gs.core.doctor import CheckItem, diagnose_environment
from real2sim_gs.core.pipeline import PipelineRunResult, PipelineRunner, ProgressCallback
from real2sim_gs.core.registry import REGISTRY
from real2sim_gs.stages import load_builtin_stages


def run_pipeline(
    config: str | Path | PipelineConfig,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    from_stage: str | None = None,
    force: bool = False,
    mock: bool = False,
    progress_callback: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> PipelineRunResult:
    """Execute or resume the Real-to-Sim 3DGS pipeline from a YAML path or PipelineConfig."""
    if isinstance(config, PipelineConfig):
        cfg = config.model_copy(deep=True)
        if input_path is not None:
            cfg.io.input_path = str(input_path)
        if output_dir is not None:
            cfg.io.workspace_dir = str(output_dir)
    else:
        cfg = load_config(config, input_path=input_path, workspace_dir=output_dir)

    runner = PipelineRunner()
    return runner.run(
        cfg,
        from_stage=from_stage,
        force=force,
        mock=mock,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )


def list_available_stages() -> dict[str, list[str]]:
    """Return all registered stage implementations grouped by pipeline phase."""
    load_builtin_stages()
    return REGISTRY.list_registered()


def check_environment() -> list[CheckItem]:
    """Run local hardware/dependency checks (CUDA, FFmpeg, COLMAP, gsplat, MuJoCo)."""
    return diagnose_environment()
