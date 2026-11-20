from __future__ import annotations

_LOADED = False


def load_builtin_stages() -> None:
    """Import all built-in stage packages once so @register_stage decorators populate REGISTRY."""
    global _LOADED
    if _LOADED:
        return
    import real2sim_gs.stages.export  # noqa: F401
    import real2sim_gs.stages.ingest  # noqa: F401
    import real2sim_gs.stages.pose  # noqa: F401
    import real2sim_gs.stages.postprocess  # noqa: F401
    import real2sim_gs.stages.reconstruct  # noqa: F401
    _LOADED = True
