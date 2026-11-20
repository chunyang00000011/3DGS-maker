from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    """Base model rejecting unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid")


class IOConfig(_StrictModel):
    """Input video/image directory and output workspace root."""

    input_path: str
    workspace_dir: str


class StageSpec(_StrictModel):
    """Configuration block selecting a registered stage implementation and its parameters."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class PipelineStagesConfig(_StrictModel):
    """Ordered stage composition for the Real-to-Sim pipeline."""

    ingest: StageSpec
    pose: StageSpec
    reconstruct: StageSpec
    postprocess: list[StageSpec] = Field(default_factory=list)
    export: list[StageSpec] = Field(default_factory=list)


class PipelineConfig(_StrictModel):
    """Top-level configuration parsed from YAML or constructed programmatically."""

    io: IOConfig
    pipeline: PipelineStagesConfig


def load_config(
    config_path: str | Path,
    input_path: str | Path | None = None,
    workspace_dir: str | Path | None = None,
) -> PipelineConfig:
    """Load a YAML PipelineConfig and apply optional CLI/SDK IO path overrides."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = PipelineConfig.model_validate(raw_data)

    if input_path is not None:
        cfg.io.input_path = str(input_path)
    if workspace_dir is not None:
        cfg.io.workspace_dir = str(workspace_dir)
    return cfg
