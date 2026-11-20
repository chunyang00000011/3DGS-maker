from __future__ import annotations

import contextlib
import gc
import importlib
import importlib.util
import inspect
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

_LOGGER = logging.getLogger(__name__)
DEFAULT_SAM3_MODEL_ID = "facebook/sam3"
SAM3_MODEL_PATH_ENV = "REAL2SIM_SAM3_MODEL_PATH"


@dataclass
class Sam3MaskPrediction:
    """Result returned by SAM 3 single-frame prompt prediction."""

    mask: np.ndarray  # (H, W) bool
    score: float
    boxes: list[list[float]] = field(default_factory=list)
    backend: str = "sam3"
    elapsed_ms: float = 0.0


def default_local_sam3_dir() -> Path:
    """Default repository-local SAM 3 checkpoint directory (`<repo_root>/models/sam3`)."""
    return Path(__file__).resolve().parents[2] / "models" / "sam3"


def is_valid_local_sam3_dir(path: str | Path) -> bool:
    """Return True if *path* is an existing directory containing Hugging Face SAM 3 weights."""
    p = Path(path)
    return p.is_dir() and (p / "config.json").is_file() and (p / "model.safetensors").is_file()


def resolve_sam3_model_id(model_path: str | None = None) -> str:
    """Resolve SAM 3 checkpoint directory or Hugging Face repo ID."""
    explicit = str(model_path or "").strip()
    if explicit:
        return explicit
    env_val = str(os.environ.get(SAM3_MODEL_PATH_ENV, "")).strip()
    if env_val:
        return env_val
    auto_dir = default_local_sam3_dir()
    if is_valid_local_sam3_dir(auto_dir):
        return str(auto_dir)
    return DEFAULT_SAM3_MODEL_ID


def _hf_has_token() -> bool:
    if importlib.util.find_spec("huggingface_hub") is None:
        return False
    try:
        import huggingface_hub

        if hasattr(huggingface_hub, "get_token"):
            return bool(huggingface_hub.get_token())
    except Exception:  # noqa: BLE001
        return False
    return False


def get_sam3_status(model_path: str | None = None, backend: Sam3Backend | None = None) -> dict[str, Any]:
    """Inspect SAM 3 dependencies, weight source, authentication state, and loaded models."""
    resolved = resolve_sam3_model_id(backend.model_id if (backend is not None and not model_path) else model_path)
    has_tf = importlib.util.find_spec("transformers") is not None
    tf_ver = ""
    sam3_classes_ok = False
    if has_tf:
        try:
            tf = importlib.import_module("transformers")
            tf_ver = str(getattr(tf, "__version__", ""))
            sam3_classes_ok = all(
                hasattr(tf, cls_name)
                for cls_name in ("Sam3Model", "Sam3Processor", "Sam3TrackerModel", "Sam3TrackerProcessor")
            )
        except Exception:  # noqa: BLE001
            sam3_classes_ok = False

    is_local_path = os.path.sep in resolved or "/" not in resolved or Path(resolved).exists()
    local_dir_exists = Path(resolved).is_dir() if is_local_path else False
    local_valid = is_valid_local_sam3_dir(resolved) if local_dir_exists else False
    hf_authed = _hf_has_token()
    loaded = backend.loaded_model_names() if backend is not None else []
    dev = str(backend.device) if backend is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    if not has_tf:
        hint = "请先安装 transformers>=4.58 与 accelerate safetensors"
    elif not sam3_classes_ok:
        hint = f"当前 transformers ({tf_ver}) 过旧，请运行 pip install -U transformers accelerate safetensors"
    elif is_local_path and not local_dir_exists:
        hint = f"本地权重目录不存在: {resolved}"
    elif is_local_path and not local_valid:
        hint = f"本地权重目录缺少必要文件（需包含 config.json 与 model.safetensors）: {resolved}"
    elif is_local_path and local_valid:
        hint = f"本地 SAM 3 权重已就绪（{resolved}），首次分割时将按需加载至显存"
    elif not is_local_path and not hf_authed:
        hint = "使用 facebook/sam3 需先登录 Hugging Face (`hf auth login`) 并同意协议，或填写本地权重目录"
    else:
        hint = "SAM 3 推理环境已就绪（按需首次调用时自动加载权重，支持随时释放显存）"

    return {
        "transformers_installed": has_tf,
        "transformers_version": tf_ver,
        "sam3_classes_available": sam3_classes_ok,
        "hf_authenticated": hf_authed,
        "model_id": resolved,
        "local_path_exists": local_dir_exists,
        "loaded_models": loaded,
        "device": dev,
        "ready": bool(has_tf and sam3_classes_ok and (local_valid if is_local_path else (hf_authed or not is_local_path))),
        "hint": hint,
    }


def _translate_sam3_load_error(exc: Exception, model_id: str) -> RuntimeError:
    msg = str(exc)
    low = msg.lower()
    if any(k in low for k in ("401", "403", "gated", "unauthorized", "access", "token")):
        return RuntimeError(
            f"加载 SAM 3 权重 '{model_id}' 被拒绝（Hugging Face gated 模型需授权）。\n"
            "请执行以下步骤：\n"
            "1. 在 https://huggingface.co/facebook/sam3 点击同意访问条款；\n"
            "2. 在终端运行 `hf auth login` 填入读取 Token；\n"
            "3. 或在本地下载权重后，将目录路径填入工具栏「本地权重路径」或设置环境变量 REAL2SIM_SAM3_MODEL_PATH。"
        )
    if any(k in low for k in ("connection", "timeout", "name resolution", "offline", "not found", "entry")):
        return RuntimeError(
            f"无法下载或读取 SAM 3 模型 '{model_id}': {msg}\n"
            "若网络受限，可配置镜像环境变量 HF_ENDPOINT=https://hf-mirror.com，或指定本地 SAM 3 权重目录。"
        )
    return RuntimeError(f"加载 SAM 3 模型 '{model_id}' 失败: {msg}")


def _load_pretrained_model(cls: Any, model_id: str, dtype: torch.dtype, device: torch.device) -> Any:
    kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(cls.from_pretrained)
        if "dtype" in sig.parameters:
            kwargs["dtype"] = dtype
        elif "torch_dtype" in sig.parameters:
            kwargs["torch_dtype"] = dtype
        else:
            kwargs["torch_dtype"] = dtype
    except (TypeError, ValueError):
        kwargs["torch_dtype"] = dtype

    try:
        model = cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs.pop("torch_dtype", None)
        model = cls.from_pretrained(model_id, **kwargs)
    model = model.to(device)
    model.eval()
    return model


def _prepare_inputs(inputs: Any, device: torch.device, model_dtype: torch.dtype) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in dict(inputs).items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                if k == "pixel_values":
                    out[k] = v.to(device=device, dtype=model_dtype, non_blocking=True)
                else:
                    out[k] = v.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def _box_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-8 else 0.0


def _mask_to_xyxy(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


class Sam3Backend:
    """Lazy, thread-safe SAM 3 inference backend (Tracker + Open-Vocab Concept + Video Tracking).

    Designed for 12 GB VRAM + 32 GB RAM:
    - Loads only the requested SAM 3 sub-model on demand in bfloat16 (CUDA) / float32 (CPU).
    - Stores multi-view video frames and inference states in CPU RAM (`video_storage_device='cpu'`).
    - Exposes `unload()` to immediately reclaim VRAM before training or heavy rendering.
    """

    def __init__(self, model_id: str | None = None) -> None:
        self._lock = threading.RLock()
        self._custom_model_id: str | None = str(model_path).strip() if (model_path := model_id) else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self._tracker_model: Any = None
        self._tracker_processor: Any = None
        self._concept_model: Any = None
        self._concept_processor: Any = None
        self._video_tracker_model: Any = None
        self._video_tracker_processor: Any = None
        self._video_concept_model: Any = None
        self._video_concept_processor: Any = None

    @property
    def model_id(self) -> str:
        return resolve_sam3_model_id(self._custom_model_id)

    def configure(self, model_path: str | None = None) -> str:
        with self._lock:
            cleaned = str(model_path or "").strip() or None
            if cleaned != self._custom_model_id:
                self.unload()
                self._custom_model_id = cleaned
            return self.model_id

    def loaded_model_names(self) -> list[str]:
        with self._lock:
            names: list[str] = []
            if self._tracker_model is not None:
                names.append("Sam3TrackerModel")
            if self._concept_model is not None:
                names.append("Sam3Model")
            if self._video_tracker_model is not None:
                names.append("Sam3TrackerVideoModel")
            if self._video_concept_model is not None:
                names.append("Sam3VideoModel")
            return names

    def unload(self) -> dict[str, Any]:
        with self._lock:
            was_loaded = self.loaded_model_names()
            self._tracker_model = None
            self._tracker_processor = None
            self._concept_model = None
            self._concept_processor = None
            self._video_tracker_model = None
            self._video_tracker_processor = None
            self._video_concept_model = None
            self._video_concept_processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"unloaded": was_loaded}

    def _evict_other_models(self, keep: str) -> None:
        """Keep at most one SAM 3 sub-model resident in 12 GB VRAM at a time."""
        evicted = False
        if keep != "tracker" and (self._tracker_model is not None or self._tracker_processor is not None):
            self._tracker_model = None
            self._tracker_processor = None
            evicted = True
        if keep != "concept" and (self._concept_model is not None or self._concept_processor is not None):
            self._concept_model = None
            self._concept_processor = None
            evicted = True
        if keep != "video_tracker" and (self._video_tracker_model is not None or self._video_tracker_processor is not None):
            self._video_tracker_model = None
            self._video_tracker_processor = None
            evicted = True
        if keep != "video_concept" and (self._video_concept_model is not None or self._video_concept_processor is not None):
            self._video_concept_model = None
            self._video_concept_processor = None
            evicted = True
        if evicted:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _import_transformers(self) -> Any:
        try:
            import transformers

            return transformers
        except ImportError as exc:
            raise RuntimeError(
                "尚未安装 transformers 库。请运行 `pip install -U transformers accelerate safetensors` 启用 SAM 3 分割功能。"
            ) from exc

    def _ensure_tracker(self) -> tuple[Any, Any]:
        with self._lock:
            if self._tracker_model is not None and self._tracker_processor is not None:
                return self._tracker_model, self._tracker_processor
            self._evict_other_models("tracker")
            tf = self._import_transformers()
            if not hasattr(tf, "Sam3TrackerModel") or not hasattr(tf, "Sam3TrackerProcessor"):
                raise RuntimeError(
                    "当前 transformers 版本不包含 Sam3TrackerModel，请运行 `pip install -U transformers` 升级。"
                )
            mid = self.model_id
            try:
                proc = tf.Sam3TrackerProcessor.from_pretrained(mid)
                model = _load_pretrained_model(tf.Sam3TrackerModel, mid, self.dtype, self.device)
            except Exception as exc:  # noqa: BLE001
                raise _translate_sam3_load_error(exc, mid) from exc
            self._tracker_processor = proc
            self._tracker_model = model
            return model, proc

    def _ensure_concept(self) -> tuple[Any, Any]:
        with self._lock:
            if self._concept_model is not None and self._concept_processor is not None:
                return self._concept_model, self._concept_processor
            self._evict_other_models("concept")
            tf = self._import_transformers()
            if not hasattr(tf, "Sam3Model") or not hasattr(tf, "Sam3Processor"):
                raise RuntimeError(
                    "当前 transformers 版本不包含 Sam3Model，请运行 `pip install -U transformers` 升级。"
                )
            mid = self.model_id
            try:
                proc = tf.Sam3Processor.from_pretrained(mid)
                model = _load_pretrained_model(tf.Sam3Model, mid, self.dtype, self.device)
            except Exception as exc:  # noqa: BLE001
                raise _translate_sam3_load_error(exc, mid) from exc
            self._concept_processor = proc
            self._concept_model = model
            return model, proc

    def _ensure_video_tracker(self) -> tuple[Any, Any]:
        with self._lock:
            if self._video_tracker_model is not None and self._video_tracker_processor is not None:
                return self._video_tracker_model, self._video_tracker_processor
            self._evict_other_models("video_tracker")
            tf = self._import_transformers()
            if not hasattr(tf, "Sam3TrackerVideoModel") or not hasattr(tf, "Sam3TrackerVideoProcessor"):
                raise RuntimeError(
                    "当前 transformers 版本不包含 Sam3TrackerVideoModel，请运行 `pip install -U transformers` 升级。"
                )
            mid = self.model_id
            try:
                proc = tf.Sam3TrackerVideoProcessor.from_pretrained(mid)
                model = _load_pretrained_model(tf.Sam3TrackerVideoModel, mid, self.dtype, self.device)
            except Exception as exc:  # noqa: BLE001
                raise _translate_sam3_load_error(exc, mid) from exc
            self._video_tracker_processor = proc
            self._video_tracker_model = model
            return model, proc

    def _ensure_video_concept(self) -> tuple[Any, Any]:
        with self._lock:
            if self._video_concept_model is not None and self._video_concept_processor is not None:
                return self._video_concept_model, self._video_concept_processor
            self._evict_other_models("video_concept")
            tf = self._import_transformers()
            if not hasattr(tf, "Sam3VideoModel") or not hasattr(tf, "Sam3VideoProcessor"):
                raise RuntimeError(
                    "当前 transformers 版本不包含 Sam3VideoModel，请运行 `pip install -U transformers` 升级。"
                )
            mid = self.model_id
            try:
                proc = tf.Sam3VideoProcessor.from_pretrained(mid)
                model = _load_pretrained_model(tf.Sam3VideoModel, mid, self.dtype, self.device)
            except Exception as exc:  # noqa: BLE001
                raise _translate_sam3_load_error(exc, mid) from exc
            self._video_concept_processor = proc
            self._video_concept_model = model
            return model, proc

    def _amp_ctx(self) -> Any:
        if self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=self.dtype)
        return contextlib.nullcontext()

    @staticmethod
    def _select_concept_instance(
        masks_t: torch.Tensor,
        scores_t: torch.Tensor,
        boxes_t: torch.Tensor | None,
        points: list[list[float]] | None,
        labels: list[int] | None,
        box: list[float] | None,
        h: int,
        w: int,
    ) -> tuple[np.ndarray, float, list[list[float]]]:
        if masks_t.numel() == 0 or masks_t.shape[0] == 0:
            return np.zeros((h, w), dtype=bool), 0.0, []

        masks_np = masks_t.detach().cpu().numpy().astype(bool)
        scores_np = scores_t.detach().float().cpu().numpy().reshape(-1)
        boxes_np = (
            boxes_t.detach().float().cpu().numpy().reshape(-1, 4)
            if boxes_t is not None and boxes_t.numel() >= 4
            else np.zeros((len(masks_np), 4), dtype=np.float32)
        )

        # If spatial prompts are combined with text prompt, disambiguate matching instance(s).
        pos_pts = [
            (int(round(pt[0])), int(round(pt[1])))
            for pt, lb in zip(points or [], labels or [])
            if int(lb) == 1
        ]
        if pos_pts:
            best_idx = 0
            best_val = -1.0
            for i, m in enumerate(masks_np):
                hits = sum(1 for x, y in pos_pts if 0 <= x < w and 0 <= y < h and m[y, x])
                val = hits * 10.0 + float(scores_np[i])
                if val > best_val:
                    best_val = val
                    best_idx = i
            sel_mask = masks_np[best_idx]
            sel_boxes = [boxes_np[best_idx].tolist()] if len(boxes_np) > best_idx else []
            return sel_mask, float(scores_np[best_idx]), sel_boxes

        if box is not None and len(box) == 4:
            best_idx = 0
            best_val = -1.0
            for i in range(len(masks_np)):
                b = boxes_np[i].tolist() if len(boxes_np) > i else (_mask_to_xyxy(masks_np[i]) or [0, 0, 0, 0])
                val = _box_iou_xyxy(box, b) * 5.0 + float(scores_np[i])
                if val > best_val:
                    best_val = val
                    best_idx = i
            sel_mask = masks_np[best_idx]
            sel_boxes = [boxes_np[best_idx].tolist()] if len(boxes_np) > best_idx else []
            return sel_mask, float(scores_np[best_idx]), sel_boxes

        # Pure open-vocabulary text prompt: keep all confident matching instances (fallback top-1).
        keep = scores_np >= 0.35
        if not np.any(keep):
            best_idx = int(np.argmax(scores_np))
            keep = np.zeros(len(scores_np), dtype=bool)
            keep[best_idx] = True
        merged = np.any(masks_np[keep], axis=0)
        sel_boxes = [boxes_np[i].tolist() for i in np.nonzero(keep)[0] if len(boxes_np) > i]
        return merged, float(scores_np[keep].max()), sel_boxes

    def predict_image_prompt(
        self,
        rgb_u8: np.ndarray,
        *,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        box: list[float] | None = None,
        text: str | None = None,
    ) -> Sam3MaskPrediction:
        """Run SAM 3 single-image prompt segmentation via text, points, and/or bounding box."""
        t0 = time.perf_counter()
        img_arr = np.asarray(rgb_u8, dtype=np.uint8)
        h, w = img_arr.shape[:2]
        pil_img = Image.fromarray(img_arr).convert("RGB")
        clean_text = str(text or "").strip()

        with self._lock:
            if clean_text:
                model, proc = self._ensure_concept()
                raw_inputs = proc(images=pil_img, text=clean_text, return_tensors="pt")
                inputs = _prepare_inputs(raw_inputs, self.device, model.dtype)
                with torch.inference_mode(), self._amp_ctx():
                    outputs = model(**inputs)
                results = proc.post_process_instance_segmentation(
                    outputs, threshold=0.25, mask_threshold=0.5, target_sizes=[(h, w)]
                )
                res0 = results[0] if results else {}
                masks_t = res0.get("masks", torch.zeros((0, h, w), dtype=torch.bool))
                scores_t = res0.get("scores", torch.zeros((0,), dtype=torch.float32))
                boxes_t = res0.get("boxes")
                mask, score, boxes = self._select_concept_instance(
                    masks_t, scores_t, boxes_t, points, labels, box, h, w
                )
                return Sam3MaskPrediction(
                    mask=mask,
                    score=score,
                    boxes=boxes,
                    backend="Sam3Model",
                    elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )

            if not points and box is None:
                raise ValueError("请提供点击点、框选区域或开放词汇文本提示中的至少一种。")

            model, proc = self._ensure_tracker()
            proc_kwargs: dict[str, Any] = {"images": pil_img, "return_tensors": "pt"}
            if points:
                pts_clean = [[[[float(p[0]), float(p[1])] for p in points]]]
                lbls_clean = [[[int(lb) for lb in (labels or [1] * len(points))]]]
                proc_kwargs["input_points"] = pts_clean
                proc_kwargs["input_labels"] = lbls_clean
            if box is not None and len(box) == 4:
                proc_kwargs["input_boxes"] = [[[float(v) for v in box]]]

            raw_inputs = proc(**proc_kwargs)
            inputs = _prepare_inputs(raw_inputs, self.device, model.dtype)
            with torch.inference_mode(), self._amp_ctx():
                outputs = model(multimask_output=True, **inputs)

            orig_sizes = raw_inputs.get("original_sizes")
            if orig_sizes is None:
                orig_sizes = [[h, w]]
            masks_list = proc.post_process_masks(outputs.pred_masks.detach().cpu(), orig_sizes)
            m_t = masks_list[0] if isinstance(masks_list, (list, tuple)) else masks_list
            while m_t.ndim > 3:
                m_t = m_t[0]
            ious = outputs.iou_scores.detach().float().cpu().reshape(-1)
            best_idx = int(torch.argmax(ious).item()) if m_t.shape[0] > 0 and len(ious) == m_t.shape[0] else 0
            mask_np = m_t[best_idx].numpy().astype(bool) if m_t.shape[0] > 0 else np.zeros((h, w), dtype=bool)
            best_score = float(ious[best_idx].item()) if len(ious) > best_idx else 0.0
            bbox = _mask_to_xyxy(mask_np)
            return Sam3MaskPrediction(
                mask=mask_np,
                score=best_score,
                boxes=[bbox] if bbox is not None else [],
                backend="Sam3TrackerModel",
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            )

    @staticmethod
    def _init_video_session(proc: Any, frames_pil: list[Image.Image], device: torch.device, dtype: torch.dtype) -> Any:
        sig = inspect.signature(proc.init_video_session)
        kwargs: dict[str, Any] = {"video": frames_pil, "inference_device": device}
        if "inference_state_device" in sig.parameters:
            kwargs["inference_state_device"] = "cpu"
        if "video_storage_device" in sig.parameters:
            kwargs["video_storage_device"] = "cpu"
        if "processing_device" in sig.parameters:
            kwargs["processing_device"] = "cpu"
        if "dtype" in sig.parameters:
            kwargs["dtype"] = dtype
        return proc.init_video_session(**kwargs)

    def track_multiview(
        self,
        frames_rgb: list[np.ndarray],
        *,
        anchor_idx: int = 0,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        box: list[float] | None = None,
        text: str | None = None,
        anchor_mask: np.ndarray | None = None,
        projected_seeds: list[tuple[float, float] | None] | None = None,
    ) -> list[np.ndarray]:
        """Propagate target object mask consistently across multi-view renders.

        Uses SAM 3 Video Session with CPU RAM storage (`video_storage_device='cpu'`) to stay well
        within 12 GB VRAM, with per-frame projected-anchor tracker fallback if video session fails.
        """
        if not frames_rgb:
            return []

        h, w = frames_rgb[0].shape[:2]
        frames_pil = [Image.fromarray(np.asarray(f, dtype=np.uint8)).convert("RGB") for f in frames_rgb]
        clean_text = str(text or "").strip()

        with self._lock:
            if clean_text:
                try:
                    return self._track_multiview_video_concept(
                        frames_pil,
                        clean_text,
                        anchor_idx=anchor_idx,
                        points=points,
                        labels=labels,
                        box=box,
                        h=h,
                        w=w,
                    )
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "SAM 3 video concept session failed (%s); falling back to per-frame segmentation.", exc
                    )
                    out_masks: list[np.ndarray] = []
                    for idx, frame_rgb in enumerate(frames_rgb):
                        seed_pts = points if idx == anchor_idx else (
                            [[projected_seeds[idx][0], projected_seeds[idx][1]]]
                            if projected_seeds and projected_seeds[idx] is not None
                            else None
                        )
                        seed_lbl = labels if idx == anchor_idx else ([1] if seed_pts else None)
                        pred = self.predict_image_prompt(
                            frame_rgb,
                            text=clean_text,
                            points=seed_pts,
                            labels=seed_lbl,
                            box=box if idx == anchor_idx else None,
                        )
                        out_masks.append(pred.mask)
                    return out_masks

            # Visual prompt tracking (points / box / anchor_mask)
            try:
                return self._track_multiview_video_tracker(
                    frames_pil,
                    anchor_idx=anchor_idx,
                    points=points,
                    labels=labels,
                    box=box,
                    anchor_mask=anchor_mask,
                    h=h,
                    w=w,
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "SAM 3 video tracker session failed (%s); falling back to per-frame tracker.", exc
                )
                out_masks = []
                for idx, frame_rgb in enumerate(frames_rgb):
                    if idx == anchor_idx and anchor_mask is not None:
                        out_masks.append(np.asarray(anchor_mask, dtype=bool))
                        continue
                    if idx == anchor_idx:
                        pred = self.predict_image_prompt(frame_rgb, points=points, labels=labels, box=box)
                        out_masks.append(pred.mask)
                    elif projected_seeds and idx < len(projected_seeds) and projected_seeds[idx] is not None:
                        px, py = projected_seeds[idx]  # type: ignore[misc]
                        pred = self.predict_image_prompt(frame_rgb, points=[[px, py]], labels=[1])
                        out_masks.append(pred.mask)
                    else:
                        out_masks.append(np.zeros((h, w), dtype=bool))
                return out_masks

    def _track_multiview_video_concept(
        self,
        frames_pil: list[Image.Image],
        text: str,
        *,
        anchor_idx: int,
        points: list[list[float]] | None,
        labels: list[int] | None,
        box: list[float] | None,
        h: int,
        w: int,
    ) -> list[np.ndarray]:
        model, proc = self._ensure_video_concept()
        session = self._init_video_session(proc, frames_pil, self.device, self.dtype)
        proc.add_text_prompt(session, text=text)

        per_frame_outputs: dict[int, dict[str, Any]] = {}
        with torch.inference_mode(), self._amp_ctx():
            for raw_out in model.propagate_in_video_iterator(inference_session=session, start_frame_idx=0):
                f_idx = int(getattr(raw_out, "frame_idx", len(per_frame_outputs)))
                post = proc.postprocess_outputs(session, raw_out, original_sizes=[[h, w]])
                per_frame_outputs[f_idx] = post

        # Determine target object ID(s) if anchor spatial prompt disambiguates a specific instance.
        selected_obj_ids: set[int] | None = None
        anchor_out = per_frame_outputs.get(anchor_idx)
        if anchor_out is not None and (points or box is not None):
            masks_t = anchor_out.get("masks")
            obj_ids_t = anchor_out.get("object_ids")
            if masks_t is not None and obj_ids_t is not None and len(obj_ids_t) > 0:
                masks_np = masks_t.detach().cpu().numpy().astype(bool)
                obj_ids = [int(v) for v in obj_ids_t.detach().cpu().tolist()]
                pos_pts = [
                    (int(round(p[0])), int(round(p[1])))
                    for p, lb in zip(points or [], labels or [])
                    if int(lb) == 1
                ]
                best_oid = obj_ids[0]
                best_score = -1.0
                for idx_i, oid in enumerate(obj_ids):
                    m = masks_np[idx_i]
                    sc = 0.0
                    if pos_pts:
                        sc += 10.0 * sum(1 for x, y in pos_pts if 0 <= x < w and 0 <= y < h and m[y, x])
                    if box is not None and len(box) == 4:
                        b = _mask_to_xyxy(m)
                        if b is not None:
                            sc += 5.0 * _box_iou_xyxy(box, b)
                    if sc > best_score:
                        best_score = sc
                        best_oid = oid
                if best_score > 0.0:
                    selected_obj_ids = {best_oid}

        results: list[np.ndarray] = []
        for f_idx in range(len(frames_pil)):
            post = per_frame_outputs.get(f_idx)
            if not post:
                results.append(np.zeros((h, w), dtype=bool))
                continue
            masks_t = post.get("masks")
            obj_ids_t = post.get("object_ids")
            if masks_t is None or masks_t.numel() == 0:
                results.append(np.zeros((h, w), dtype=bool))
                continue
            masks_np = masks_t.detach().cpu().numpy().astype(bool)
            if selected_obj_ids is not None and obj_ids_t is not None:
                oids = [int(v) for v in obj_ids_t.detach().cpu().tolist()]
                picked = [masks_np[i] for i, oid in enumerate(oids) if oid in selected_obj_ids]
                results.append(np.any(picked, axis=0) if picked else np.zeros((h, w), dtype=bool))
            else:
                results.append(np.any(masks_np, axis=0))
        return results

    def _track_multiview_video_tracker(
        self,
        frames_pil: list[Image.Image],
        *,
        anchor_idx: int,
        points: list[list[float]] | None,
        labels: list[int] | None,
        box: list[float] | None,
        anchor_mask: np.ndarray | None,
        h: int,
        w: int,
    ) -> list[np.ndarray]:
        model, proc = self._ensure_video_tracker()
        session = self._init_video_session(proc, frames_pil, self.device, self.dtype)

        add_kwargs: dict[str, Any] = {
            "inference_session": session,
            "frame_idx": anchor_idx,
            "obj_ids": 1,
        }
        if points:
            add_kwargs["input_points"] = [[[[float(p[0]), float(p[1])] for p in points]]]
            add_kwargs["input_labels"] = [[[int(lb) for lb in (labels or [1] * len(points))]]]
        if box is not None and len(box) == 4:
            add_kwargs["input_boxes"] = [[[float(v) for v in box]]]
        if not points and box is None and anchor_mask is not None:
            add_kwargs["input_masks"] = [np.asarray(anchor_mask, dtype=np.uint8)]

        proc.add_inputs_to_inference_session(**add_kwargs)

        masks_by_idx: dict[int, np.ndarray] = {}
        with torch.inference_mode(), self._amp_ctx():
            # Seed anchor frame into session memory before propagating
            first_out = model(inference_session=session, frame_idx=anchor_idx)
            self._store_tracker_video_frame(proc, first_out, anchor_idx, h, w, masks_by_idx)

            for out in model.propagate_in_video_iterator(
                inference_session=session, start_frame_idx=anchor_idx, reverse=False
            ):
                f_idx = int(getattr(out, "frame_idx", anchor_idx))
                self._store_tracker_video_frame(proc, out, f_idx, h, w, masks_by_idx)

            if anchor_idx > 0:
                for out in model.propagate_in_video_iterator(
                    inference_session=session, start_frame_idx=anchor_idx, reverse=True
                ):
                    f_idx = int(getattr(out, "frame_idx", 0))
                    self._store_tracker_video_frame(proc, out, f_idx, h, w, masks_by_idx)

        return [masks_by_idx.get(i, np.zeros((h, w), dtype=bool)) for i in range(len(frames_pil))]

    @staticmethod
    def _store_tracker_video_frame(
        proc: Any,
        out: Any,
        frame_idx: int,
        h: int,
        w: int,
        dest: dict[int, np.ndarray],
    ) -> None:
        pred_masks = getattr(out, "pred_masks", None)
        if pred_masks is None:
            return
        pm = pred_masks.detach().float().cpu()
        if pm.ndim == 5:
            pm4 = pm[0]
        elif pm.ndim == 4:
            pm4 = pm
        elif pm.ndim == 3:
            pm4 = pm.unsqueeze(0)
        elif pm.ndim == 2:
            pm4 = pm.unsqueeze(0).unsqueeze(0)
        else:
            pm4 = pm.reshape(1, 1, int(pm.shape[-2]), int(pm.shape[-1]))
        post = proc.post_process_masks([pm4], original_sizes=[[h, w]], binarize=True)
        m_t = post[0] if isinstance(post, (list, tuple)) else post
        while m_t.ndim > 2:
            m_t = m_t[0]
        dest[frame_idx] = m_t.numpy().astype(bool)


_SHARED_SAM3_LOCK = threading.Lock()
_SHARED_SAM3_BACKEND: Sam3Backend | None = None


def get_shared_sam3_backend() -> Sam3Backend:
    """Return the process-wide lazy SAM 3 backend singleton."""
    global _SHARED_SAM3_BACKEND
    with _SHARED_SAM3_LOCK:
        if _SHARED_SAM3_BACKEND is None:
            _SHARED_SAM3_BACKEND = Sam3Backend()
        return _SHARED_SAM3_BACKEND
