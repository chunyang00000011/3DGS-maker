from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from real2sim_gs.core.config import PipelineConfig, StageSpec
from real2sim_gs.core.registry import REGISTRY
from real2sim_gs.core.stage import StageKind
from real2sim_gs.core.workspace import STAGE_ORDER, SceneWorkspace, StageRecord

ProgressCallback = Callable[[str, str, dict[str, Any]], None]


@dataclass
class PipelineRunResult:
    """Summary returned after executing or resuming a Real-to-Sim pipeline run."""

    workspace_dir: Path
    executed_stages: list[str] = field(default_factory=list)
    skipped_stages: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    mujoco_xml_path: Path | None = None
    unity_dir: Path | None = None
    gaussian_ply_path: Path | None = None
    splat_ply_path: Path | None = None


def _input_fingerprint(input_path: str | Path | None) -> dict[str, Any] | None:
    """Capture lightweight file/directory identity so changing input video invalidates ingest cache."""
    if input_path is None or not str(input_path).strip():
        return None
    p = Path(input_path)
    if p.is_file():
        st = p.stat()
        return {
            "path": str(p.resolve()),
            "kind": "file",
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    if p.is_dir():
        img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in img_exts]
        total_size = 0
        latest_mtime = 0
        for f in files:
            try:
                st = f.stat()
                total_size += int(st.st_size)
                latest_mtime = max(latest_mtime, int(st.st_mtime_ns))
            except OSError:
                continue
        return {
            "path": str(p.resolve()),
            "kind": "dir",
            "count": len(files),
            "size": total_size,
            "mtime_ns": latest_mtime,
        }
    return {"path": str(p), "kind": "virtual"}


def _normalize_for_signature(value: Any) -> Any:
    """Canonicalize params so semantically identical configs hash identically (2 vs 2.0, tuple vs list)."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return float(repr(value))
    if isinstance(value, int):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _normalize_for_signature(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_signature(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _signature_hash(signature: str) -> str:
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def _signature_for_specs(
    specs: list[StageSpec],
    mock: bool,
    extra: dict[str, Any] | None = None,
    upstream: str = "",
) -> str:
    """Stage signature chained with the upstream stage signature hash, so any upstream change cascades."""
    payload: dict[str, Any] = {
        "mock": mock,
        "specs": [{"name": s.name, "params": _normalize_for_signature(s.params)} for s in specs],
        "upstream": upstream,
    }
    if extra:
        payload["extra"] = _normalize_for_signature(extra)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _validate_stage_plan(stage_plan: list[tuple[StageKind, list[StageSpec]]]) -> None:
    """Fail fast before touching the workspace if any configured stage name is not registered."""
    errors: list[str] = []
    for stage_kind, specs in stage_plan:
        available = REGISTRY.list_registered().get(stage_kind, [])
        for spec in specs:
            if spec.name not in available:
                errors.append(f"阶段 '{stage_kind}' 未注册实现 '{spec.name}'（可用: {available}）")
    if errors:
        raise KeyError("配置中存在未知阶段实现，已在运行前中止：\n" + "\n".join(errors))


class PipelineRunner:
    """Orchestrates pluggable stages against a SceneWorkspace with manifest checkpointing."""

    def run(
        self,
        config: PipelineConfig,
        from_stage: str | None = None,
        force: bool = False,
        mock: bool = False,
        progress_callback: ProgressCallback | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> PipelineRunResult:
        from real2sim_gs.stages import load_builtin_stages

        load_builtin_stages()

        if from_stage is not None and from_stage not in STAGE_ORDER:
            raise ValueError(f"Invalid from_stage '{from_stage}'. Must be one of {STAGE_ORDER}.")

        stage_plan: list[tuple[StageKind, list[StageSpec]]] = [
            (
                "ingest",
                [StageSpec(name="mock_ingest", params=config.pipeline.ingest.params)]
                if mock
                else [config.pipeline.ingest],
            ),
            (
                "pose",
                [StageSpec(name="mock_pose", params=config.pipeline.pose.params)]
                if mock
                else [config.pipeline.pose],
            ),
            (
                "reconstruct",
                [StageSpec(name="mock_reconstruct", params=config.pipeline.reconstruct.params)]
                if mock
                else [config.pipeline.reconstruct],
            ),
            ("postprocess", config.pipeline.postprocess),
            ("export", config.pipeline.export),
        ]
        _validate_stage_plan(stage_plan)

        ws = SceneWorkspace(config.io.workspace_dir, input_path=config.io.input_path)
        ws.ensure_dirs()
        ws.cancel_check = cancel_check

        ingest_specs = stage_plan[0][1]
        ingest_needs_real_input = (
            not mock
            and bool(ingest_specs)
            and ingest_specs[0].name != "mock_ingest"
        )
        raw_in_str = str(config.io.input_path or "").strip()
        raw_in_path = Path(raw_in_str) if raw_in_str else None

        # 计算 ingest 阶段期望签名；若源临时上传视频已被清理但 01_frames 完好，复用旧指纹以免无辜失效。
        def _resolve_ingest_extra_sig() -> dict[str, Any]:
            fp = _input_fingerprint(config.io.input_path)
            if (
                ingest_needs_real_input
                and raw_in_path is not None
                and not raw_in_path.exists()
                and raw_in_path.name != "sample_video.mp4"
                and not force
                and from_stage != "ingest"
                and ws.is_completed("ingest", None)
            ):
                try:
                    prev_entry = ws.load_manifest().get("stages", {}).get("ingest", {})
                    prev_sig = json.loads(prev_entry.get("impl_signature", "{}"))
                    prev_in = prev_sig.get("extra", {}).get("input", {})
                    if prev_in.get("path") == str(raw_in_path.resolve()):
                        return {"input": prev_in}
                except (ValueError, TypeError, OSError):
                    pass
            return {"input": fp}

        if ingest_needs_real_input:
            ingest_extra_pre = _resolve_ingest_extra_sig()
            ingest_sig_pre = _signature_for_specs(ingest_specs, mock=mock, extra=ingest_extra_pre, upstream="")
            will_skip_ingest = (
                (from_stage is not None and STAGE_ORDER.index(from_stage) > 0 and ws.is_completed("ingest", None))
                or (from_stage is None and not force and ws.is_completed("ingest", ingest_sig_pre))
            )
            if not will_skip_ingest:
                if raw_in_path is None or (not raw_in_path.exists() and raw_in_path.name != "sample_video.mp4"):
                    disp = raw_in_path if raw_in_path is not None else Path("")
                    raise FileNotFoundError(
                        f"输入路径不存在 (Input path does not exist): {disp} "
                        f"(解析路径: {disp.resolve()})。请先在左上角上传视频文件，或检查本地路径拼写。"
                    )

        if from_stage is not None:
            ws.invalidate_from(from_stage)
            if STAGE_ORDER.index(from_stage) <= STAGE_ORDER.index("reconstruct"):
                ws.reconstruct_checkpoint_path.unlink(missing_ok=True)
                ws.reconstruct_checkpoint_meta_path.unlink(missing_ok=True)
        elif force:
            ws.invalidate_from(STAGE_ORDER[0])
            ws.reconstruct_checkpoint_path.unlink(missing_ok=True)
            ws.reconstruct_checkpoint_meta_path.unlink(missing_ok=True)

        start_idx = STAGE_ORDER.index(from_stage) if from_stage else 0
        executed: list[str] = []
        skipped: list[str] = []
        aggregated_metrics: dict[str, Any] = {}
        upstream_hash = ""

        try:
            for idx, (stage_kind, specs) in enumerate(stage_plan):
                ws.check_cancelled()
                if not specs:
                    continue
                extra_sig = (
                    _resolve_ingest_extra_sig()
                    if stage_kind == "ingest" and not mock
                    else None
                )
                sig = _signature_for_specs(specs, mock=mock, extra=extra_sig, upstream=upstream_hash)
                if idx < start_idx:
                    should_run = not ws.is_completed(stage_kind, None)
                else:
                    should_run = not ws.is_completed(stage_kind, sig)

                if not should_run:
                    skipped.append(stage_kind)
                    manifest_entry = ws.load_manifest()["stages"][stage_kind]
                    upstream_hash = _signature_hash(manifest_entry.get("impl_signature", ""))
                    entry_metrics = manifest_entry.get("metrics", {})
                    aggregated_metrics.update(entry_metrics)
                    if progress_callback is not None:
                        progress_callback(
                            stage_kind,
                            "skipped",
                            {
                                "impl": manifest_entry.get("impl_name", ""),
                                "duration_sec": manifest_entry.get("duration_sec", 0.0),
                                "progress": 1.0,
                                "detail": "命中缓存跳过",
                                "metrics": entry_metrics,
                            },
                        )
                    continue

                if stage_kind == "ingest" and ingest_needs_real_input:
                    if raw_in_path is None or (not raw_in_path.exists() and raw_in_path.name != "sample_video.mp4"):
                        disp = raw_in_path if raw_in_path is not None else Path("")
                        raise FileNotFoundError(
                            f"输入路径不存在 (Input path does not exist): {disp} "
                            f"(解析路径: {disp.resolve()})。请先在左上角上传视频文件，或检查本地路径拼写。"
                        )

                # 本阶段需要重跑：立即作废本阶段及全部下游的 manifest 记录与下游产物目录，
                # 保证中途取消/失败后下游旧产物绝不会被误复用。
                ws.invalidate_from(stage_kind)
                for downstream in STAGE_ORDER[STAGE_ORDER.index(stage_kind) + 1 :]:
                    ws.reset_stage_dir(downstream)
                if stage_kind in ("ingest", "pose"):
                    ws.reconstruct_checkpoint_path.unlink(missing_ok=True)
                    ws.reconstruct_checkpoint_meta_path.unlink(missing_ok=True)
                can_resume_reconstruct = (
                    stage_kind == "reconstruct"
                    and not force
                    and from_stage != "reconstruct"
                    and "pose" not in executed
                    and ws.reconstruct_checkpoint_path.is_file()
                )
                if not can_resume_reconstruct:
                    ws.reset_stage_dir(stage_kind)

                impl_label = " -> ".join(s.name for s in specs)
                if progress_callback is not None:
                    progress_callback(
                        stage_kind,
                        "running",
                        {
                            "impl": impl_label,
                            "progress": 0.02,
                            "detail": f"从断点恢复 {impl_label}..." if can_resume_reconstruct else f"启动 {impl_label}...",
                        },
                    )

                t0 = time.perf_counter()
                combined_artifacts: dict[str, str] = {}
                combined_metrics: dict[str, Any] = {}
                n_specs = len(specs)

                try:
                    for step_idx, spec in enumerate(specs):
                        ws.check_cancelled()

                        def _step_hook(
                            sub_p: float,
                            msg: str,
                            telemetry: dict[str, Any],
                            _s_idx: int = step_idx,
                            _n_specs: int = n_specs,
                            _stage: StageKind = stage_kind,
                            _label: str = impl_label,
                        ) -> None:
                            if progress_callback is None:
                                return
                            stage_p = round((_s_idx + sub_p) / max(1, _n_specs), 4)
                            progress_callback(
                                _stage,
                                "running",
                                {
                                    "impl": _label,
                                    "progress": stage_p,
                                    "detail": msg,
                                    "telemetry": telemetry,
                                },
                            )

                        ws.progress_hook = _step_hook
                        ws.report_progress(0.05, f"正在执行 {spec.name} ({step_idx + 1}/{n_specs})...")
                        stage_impl = REGISTRY.create(stage_kind, spec.name)
                        result = stage_impl.run(ws, spec.params)
                        for key, rel_path in result.artifacts.items():
                            artifact_key = key if len(specs) == 1 else f"{spec.name}.{key}"
                            combined_artifacts[artifact_key] = rel_path
                        combined_metrics.update(result.metrics)
                finally:
                    ws.progress_hook = None

                elapsed = round(time.perf_counter() - t0, 4)
                if extra_sig is not None:
                    # 输入可能在 ingest 内部才生成（如 sample_video.mp4），完成后重算指纹，
                    # 保证首次运行与二次运行签名一致、可命中缓存。
                    extra_sig = {"input": _input_fingerprint(config.io.input_path)}
                    sig = _signature_for_specs(specs, mock=mock, extra=extra_sig, upstream=upstream_hash)
                ws.mark_completed(
                    StageRecord(
                        stage_kind=stage_kind,
                        impl_name=impl_label,
                        impl_signature=sig,
                        status="completed",
                        duration_sec=elapsed,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        artifacts=combined_artifacts,
                        metrics=combined_metrics,
                    )
                )
                if progress_callback is not None:
                    progress_callback(
                        stage_kind,
                        "completed",
                        {
                            "impl": impl_label,
                            "duration_sec": elapsed,
                            "progress": 1.0,
                            "detail": "已完成",
                            "metrics": combined_metrics,
                        },
                    )
                executed.append(stage_kind)
                upstream_hash = _signature_hash(sig)
                aggregated_metrics.update(combined_metrics)
        finally:
            ws.cancel_check = None

        mujoco_xml = ws.export_dir / "mujoco" / "scene.xml"
        unity_dir = ws.export_dir / "unity"
        gaussian_ply = ws.export_dir / "gaussian_splats" / "scene_3dgs.ply"
        try:
            active_ply: Path | None = ws.active_splat_path()
        except FileNotFoundError:
            active_ply = None

        return PipelineRunResult(
            workspace_dir=ws.root,
            executed_stages=executed,
            skipped_stages=skipped,
            metrics=aggregated_metrics,
            mujoco_xml_path=mujoco_xml if mujoco_xml.exists() else None,
            unity_dir=unity_dir if unity_dir.exists() else None,
            gaussian_ply_path=gaussian_ply if gaussian_ply.exists() else None,
            splat_ply_path=active_ply,
        )
