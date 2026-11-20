"""Real2Sim-GS public package exports (loaded lazily to keep lightweight workers fast)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from real2sim_gs.api.sdk import check_environment, list_available_stages, run_pipeline
    from real2sim_gs.core.config import PipelineConfig
    from real2sim_gs.core.pipeline import PipelineRunResult
    from real2sim_gs.core.workspace import PipelineCancelledError

__all__ = [
    "run_pipeline",
    "list_available_stages",
    "check_environment",
    "PipelineConfig",
    "PipelineRunResult",
    "PipelineCancelledError",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "run_pipeline": ("real2sim_gs.api.sdk", "run_pipeline"),
    "list_available_stages": ("real2sim_gs.api.sdk", "list_available_stages"),
    "check_environment": ("real2sim_gs.api.sdk", "check_environment"),
    "PipelineConfig": ("real2sim_gs.core.config", "PipelineConfig"),
    "PipelineRunResult": ("real2sim_gs.core.pipeline", "PipelineRunResult"),
    "PipelineCancelledError": ("real2sim_gs.core.workspace", "PipelineCancelledError"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'real2sim_gs' has no attribute {name!r}")
    mod_name, attr_name = target
    import importlib

    mod = importlib.import_module(mod_name)
    val = getattr(mod, attr_name)
    globals()[name] = val
    return val
