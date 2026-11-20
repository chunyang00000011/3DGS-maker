from __future__ import annotations

from typing import Callable

from real2sim_gs.core.stage import BaseStage, StageKind


class StageRegistry:
    """In-memory registry mapping (stage_kind, name) to concrete stage implementations."""

    def __init__(self) -> None:
        self._entries: dict[tuple[StageKind, str], type[BaseStage]] = {}

    def register(self, stage_kind: StageKind, name: str, cls: type[BaseStage]) -> None:
        key = (stage_kind, name)
        self._entries[key] = cls

    def get(self, stage_kind: StageKind, name: str) -> type[BaseStage]:
        key = (stage_kind, name)
        if key not in self._entries:
            available = self.list_registered().get(stage_kind, [])
            raise KeyError(
                f"Unknown implementation '{name}' for stage '{stage_kind}'. "
                f"Available: {available}"
            )
        return self._entries[key]

    def create(self, stage_kind: StageKind, name: str) -> BaseStage:
        cls = self.get(stage_kind, name)
        return cls()

    def list_registered(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {
            "ingest": [],
            "pose": [],
            "reconstruct": [],
            "postprocess": [],
            "export": [],
        }
        for kind, name in sorted(self._entries.keys()):
            grouped.setdefault(kind, []).append(name)
        return grouped


REGISTRY = StageRegistry()


def register_stage(stage_kind: StageKind, name: str) -> Callable[[type[BaseStage]], type[BaseStage]]:
    """Class decorator registering a BaseStage subclass into the global REGISTRY."""

    def decorator(cls: type[BaseStage]) -> type[BaseStage]:
        cls.stage_kind = stage_kind
        cls.name = name
        REGISTRY.register(stage_kind, name, cls)
        return cls

    return decorator
