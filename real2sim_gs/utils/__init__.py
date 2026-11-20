"""Shared 3DGS PLY I/O and sample video utilities (lazy-loaded)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from real2sim_gs.utils.ply_io import (
        GaussianCloud,
        load_gaussian_ply,
        load_point_cloud_ply,
        save_gaussian_ply,
        save_point_cloud_ply,
    )
    from real2sim_gs.utils.sample_video import generate_sample_orbiter_video

__all__ = [
    "GaussianCloud",
    "generate_sample_orbiter_video",
    "load_gaussian_ply",
    "save_gaussian_ply",
    "load_point_cloud_ply",
    "save_point_cloud_ply",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "GaussianCloud": ("real2sim_gs.utils.ply_io", "GaussianCloud"),
    "load_gaussian_ply": ("real2sim_gs.utils.ply_io", "load_gaussian_ply"),
    "save_gaussian_ply": ("real2sim_gs.utils.ply_io", "save_gaussian_ply"),
    "load_point_cloud_ply": ("real2sim_gs.utils.ply_io", "load_point_cloud_ply"),
    "save_point_cloud_ply": ("real2sim_gs.utils.ply_io", "save_point_cloud_ply"),
    "generate_sample_orbiter_video": ("real2sim_gs.utils.sample_video", "generate_sample_orbiter_video"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'real2sim_gs.utils' has no attribute {name!r}")
    import importlib

    mod_name, attr_name = target
    val = getattr(importlib.import_module(mod_name), attr_name)
    globals()[name] = val
    return val
