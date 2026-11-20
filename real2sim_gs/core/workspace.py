from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

STAGE_ORDER: tuple[str, ...] = ("ingest", "pose", "reconstruct", "postprocess", "export")


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically (tmp file in same dir + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


class PipelineCancelledError(RuntimeError):
    """Raised when a running pipeline or training job is cancelled by the user."""


@dataclass
class StageRecord:
    """Persistent execution record stored inside manifest.json for checkpoint resume."""

    stage_kind: str
    impl_name: str
    status: str
    duration_sec: float
    completed_at: str
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    impl_signature: str = ""


class SceneWorkspace:
    """Manages the standardized on-disk data contract for one Real-to-Sim scene."""

    def __init__(self, root: str | Path, input_path: str | Path | None = None) -> None:
        self.root = Path(root)
        self.input_path = Path(input_path) if input_path is not None else None
        self.progress_hook: Callable[[float, str, dict[str, Any]], None] | None = None
        self.cancel_check: Callable[[], bool] | None = None

    def check_cancelled(self) -> None:
        """Raise PipelineCancelledError if the attached cancellation predicate triggers."""
        if self.cancel_check is not None and self.cancel_check():
            raise PipelineCancelledError("训练/构建任务已由用户手动停止")

    def report_progress(
        self,
        progress: float,
        message: str,
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        """Emit lightweight scalar progress and telemetry to the active runner if attached."""
        self.check_cancelled()
        if self.progress_hook is not None:
            clamped = max(0.0, min(1.0, float(progress)))
            self.progress_hook(clamped, message, telemetry or {})

    @property
    def frames_dir(self) -> Path:
        return self.root / "01_frames"

    @property
    def pose_dir(self) -> Path:
        return self.root / "02_pose"

    @property
    def gaussians_dir(self) -> Path:
        return self.root / "03_gaussians"

    @property
    def processed_dir(self) -> Path:
        return self.root / "04_processed"

    @property
    def export_dir(self) -> Path:
        return self.root / "05_export"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def frames_meta_path(self) -> Path:
        return self.frames_dir / "frames_meta.json"

    @property
    def cameras_path(self) -> Path:
        return self.pose_dir / "cameras.json"

    @property
    def init_points_path(self) -> Path:
        return self.pose_dir / "init_points.ply"

    @property
    def raw_splats_path(self) -> Path:
        return self.gaussians_dir / "raw_splats.ply"

    @property
    def reconstruct_checkpoint_path(self) -> Path:
        return self.gaussians_dir / "checkpoint.pt"

    @property
    def reconstruct_checkpoint_meta_path(self) -> Path:
        return self.gaussians_dir / "checkpoint_meta.json"

    @property
    def cleaned_splats_path(self) -> Path:
        return self.processed_dir / "cleaned_splats.ply"

    @property
    def aligned_splats_path(self) -> Path:
        return self.processed_dir / "aligned_splats.ply"

    @property
    def collision_mesh_path(self) -> Path:
        return self.processed_dir / "collision_mesh.obj"

    def stage_dir(self, stage_kind: str) -> Path:
        mapping = {
            "ingest": self.frames_dir,
            "pose": self.pose_dir,
            "reconstruct": self.gaussians_dir,
            "postprocess": self.processed_dir,
            "export": self.export_dir,
        }
        if stage_kind not in mapping:
            raise ValueError(f"Unknown stage_kind: {stage_kind}")
        return mapping[stage_kind]

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for stage in STAGE_ORDER:
            self.stage_dir(stage).mkdir(parents=True, exist_ok=True)

    def reset_stage_dir(self, stage_kind: str) -> None:
        target = self.stage_dir(stage_kind)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    @property
    def active_splat_pointer_path(self) -> Path:
        return self.processed_dir / "active_splat.json"

    def set_active_splat(self, path: str | Path) -> None:
        """Record the given PLY as the active splat for downstream stages (04_processed/active_splat.json)."""
        raw = Path(path)
        if raw.is_absolute() or raw.resolve().is_relative_to(self.root.resolve()):
            target = raw
        else:
            target = self.root / raw
        if not target.exists():
            raise FileNotFoundError(f"set_active_splat: file does not exist: {target}")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.active_splat_pointer_path,
            {"path": self.to_relative(target), "updated_at": datetime.now(timezone.utc).isoformat()},
        )

    def active_splat_path(self) -> Path:
        """Return the active Gaussian PLY: explicit pointer first, else aligned > cleaned > raw."""
        pointer = self.active_splat_pointer_path
        if pointer.exists():
            try:
                rel = json.loads(pointer.read_text(encoding="utf-8")).get("path", "")
            except (json.JSONDecodeError, OSError, AttributeError):
                rel = ""
            if rel:
                candidate = Path(rel)
                if not candidate.is_absolute():
                    candidate = self.root / candidate
                if candidate.exists():
                    return candidate
        for candidate in (self.aligned_splats_path, self.cleaned_splats_path, self.raw_splats_path):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No Gaussian PLY found in workspace {self.root}. Run 'reconstruct' stage first."
        )

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "workspace": str(self.root),
            "input_path": str(self.input_path) if self.input_path else "",
            "stages": {},
        }

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return self._empty_manifest()
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise json.JSONDecodeError("manifest root is not an object", "", 0)
            return data
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            backup = self.manifest_path.with_name(self.manifest_path.name + ".corrupt")
            try:
                os.replace(self.manifest_path, backup)
            except OSError:
                pass
            warnings.warn(
                f"manifest.json 已损坏（{exc}），已备份至 {backup} 并以空清单重新开始。",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._empty_manifest()

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self.manifest_path, manifest)

    def mark_completed(self, record: StageRecord) -> None:
        manifest = self.load_manifest()
        if self.input_path is not None:
            manifest["input_path"] = str(self.input_path)
        manifest.setdefault("stages", {})[record.stage_kind] = asdict(record)
        self.save_manifest(manifest)

    def is_completed(self, stage_kind: str, impl_signature: str | None = None) -> bool:
        manifest = self.load_manifest()
        entry = manifest.get("stages", {}).get(stage_kind)
        if not entry or entry.get("status") != "completed":
            return False
        if impl_signature is not None and entry.get("impl_signature") != impl_signature:
            return False
        artifacts = entry.get("artifacts", {})
        if not artifacts:
            return False
        return all((self.root / rel).exists() for rel in artifacts.values())

    def invalidate_from(self, stage_kind: str) -> None:
        if stage_kind not in STAGE_ORDER:
            raise ValueError(f"Unknown stage '{stage_kind}'. Expected one of {STAGE_ORDER}.")
        start_idx = STAGE_ORDER.index(stage_kind)
        manifest = self.load_manifest()
        stages_map = manifest.get("stages", {})
        for s in STAGE_ORDER[start_idx:]:
            stages_map.pop(s, None)
        self.save_manifest(manifest)

    def to_relative(self, path: Path) -> str:
        resolved = path.resolve()
        root_resolved = self.root.resolve()
        try:
            return str(resolved.relative_to(root_resolved)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")
