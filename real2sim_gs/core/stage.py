from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from real2sim_gs.core.workspace import SceneWorkspace

StageKind = Literal["ingest", "pose", "reconstruct", "postprocess", "export"]


@dataclass
class StageResult:
    """Standard return contract produced by every pluggable stage execution."""

    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


class BaseStage(ABC):
    """Abstract seam that all pipeline stages and external paper adapters implement."""

    stage_kind: StageKind
    name: str

    @abstractmethod
    def run(self, ws: SceneWorkspace, params: dict[str, Any]) -> StageResult:
        """Execute this stage reading from and writing to the standardized SceneWorkspace."""
