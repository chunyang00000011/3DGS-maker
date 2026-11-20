from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from real2sim_gs.core.workspace import SceneWorkspace

# COLMAP 鱼眼族相机模型（统一走 cv2.fisheye 等距投影去畸变）。
FISHEYE_CAMERA_MODELS = frozenset(
    {"OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE", "FISHEYE"}
)
# 超过该总字节数的 uint8 训练图像改存 CPU 锁页内存，按步 non_blocking 拷贝到 GPU。
DEFAULT_GPU_IMAGE_BUDGET_BYTES = 2 * 1024**3


@dataclass
class CameraView:
    """Single calibrated camera view pairing compact uint8/float32 pixels with OpenCV pinhole intrinsics."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_cam: torch.Tensor   # (4, 4) float32
    _image_storage: torch.Tensor # (H, W, 3) uint8 or float32
    valid_mask: torch.Tensor | None
    camera_model: str

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        world_to_cam: torch.Tensor,
        image: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        camera_model: str = "PINHOLE",
    ) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.world_to_cam = world_to_cam
        self._image_storage = image
        self.valid_mask = valid_mask
        self.camera_model = camera_model
        self._k_cache: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.world_to_cam.device

    @property
    def K(self) -> torch.Tensor:
        """Cached (3, 3) intrinsics tensor on the pose device (avoids per-step host->device copies)."""
        if self._k_cache is None or self._k_cache.device != self.device:
            self._k_cache = torch.tensor(
                [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=self.device,
            )
        return self._k_cache

    @property
    def image(self) -> torch.Tensor:
        """Float32 [0,1] image on the pose device. 每次调用都会转换，热循环中应每步只取一次。"""
        img = self._image_storage
        if img.device != self.device:
            img = img.to(self.device, non_blocking=True)
        if img.dtype == torch.uint8:
            return img.float().div_(255.0)
        return img


def _resolve_camera_model(entry: dict, payload: dict) -> str:
    """Camera model name from per-frame `camera_model`/`model` or a payload-level `camera_model`."""
    raw = entry.get("camera_model") or entry.get("model") or payload.get("camera_model") or ""
    return str(raw).strip().upper()


def _valid_mask_from_remap(map1: np.ndarray, map2: np.ndarray, w: int, h: int) -> np.ndarray | None:
    ones = np.full((h, w), 255, dtype=np.uint8)
    warped = cv2.remap(ones, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = warped >= 250
    return None if bool(mask.all()) else mask


def _undistort(
    raw_u8: np.ndarray,
    camera_model: str,
    dist: list[float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, float, float, float, float, np.ndarray | None]:
    """Undistort to an ideal pinhole image; returns (image, fx', fy', cx', cy', valid_mask or None)."""
    h, w = raw_u8.shape[:2]
    k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if camera_model in FISHEYE_CAMERA_MODELS:
        d = np.zeros(4, dtype=np.float64)
        vals = [float(v) for v in dist[:4]]
        d[: len(vals)] = vals
        d = d.reshape(4, 1)
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            k_mat, d, (w, h), np.eye(3), balance=0.0, new_size=(w, h)
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(k_mat, d, np.eye(3), new_k, (w, h), cv2.CV_32FC1)
    else:
        vals = [float(v) for v in dist]
        if len(vals) not in (4, 5, 8, 12, 14):
            padded = (vals + [0.0, 0.0, 0.0, 0.0])[:4] if len(vals) < 4 else vals[:4]
            vals = padded
        d = np.asarray(vals, dtype=np.float64)
        new_k, _roi = cv2.getOptimalNewCameraMatrix(k_mat, d, (w, h), 0.0, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(k_mat, d, None, new_k, (w, h), cv2.CV_32FC1)
    out = cv2.remap(raw_u8, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = _valid_mask_from_remap(map1, map2, w, h)
    return out, float(new_k[0, 0]), float(new_k[1, 1]), float(new_k[0, 2]), float(new_k[1, 2]), mask


class PoseImageDataset:
    """Loads synchronized 01_frames images and 02_pose/cameras.json with 4K cap & uint8 storage."""

    def __init__(
        self,
        ws: SceneWorkspace,
        device: torch.device,
        max_image_dim: int = 1600,
        gpu_image_budget_bytes: int = DEFAULT_GPU_IMAGE_BUDGET_BYTES,
    ) -> None:
        if not ws.cameras_path.exists():
            raise FileNotFoundError(f"Pose file missing: {ws.cameras_path}")
        payload = json.loads(ws.cameras_path.read_text(encoding="utf-8"))
        frames_raw = payload.get("frames", [])
        total_cam_frames = len(frames_raw)
        self.views: list[CameraView] = []
        cpu_images: list[torch.Tensor] = []
        cpu_masks: list[torch.Tensor | None] = []

        for entry in frames_raw:
            img_path = ws.frames_dir / entry["filename"]
            if not img_path.exists():
                continue
            raw_u8 = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            h_img, w_img = raw_u8.shape[:2]
            cam_w = float(entry.get("width") or w_img)
            cam_h = float(entry.get("height") or h_img)
            if (
                cam_w > 0
                and cam_h > 0
                and w_img > 0
                and h_img > 0
                and (w_img > h_img) != (cam_w > cam_h)
                and abs((w_img / float(h_img)) - (cam_w / float(cam_h))) > 0.25
            ):
                raise ValueError(
                    "检测到抽帧图像与相机位姿分辨率方向不一致（疑似旧缓存混用），"
                    "请勾选强制重跑或从 01 ingest 重新执行。"
                )

            target_w, target_h = w_img, h_img
            if max_image_dim > 0 and max(h_img, w_img) > max_image_dim:
                scale = max_image_dim / float(max(h_img, w_img))
                target_w = max(2, int(round(w_img * scale / 2.0)) * 2)
                target_h = max(2, int(round(h_img * scale / 2.0)) * 2)
                raw_u8 = cv2.resize(raw_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)

            sx = target_w / max(1.0, cam_w)
            sy = target_h / max(1.0, cam_h)
            fx = float(entry["fx"]) * sx
            fy = float(entry["fy"]) * sy
            cx = float(entry["cx"]) * sx
            cy = float(entry["cy"]) * sy
            camera_model = _resolve_camera_model(entry, payload)
            dist = [float(v) for v in (entry.get("distortion") or [])]
            has_dist = bool(dist) and max(abs(v) for v in dist) > 1e-6
            mask_np: np.ndarray | None = None
            if camera_model in FISHEYE_CAMERA_MODELS or has_dist:
                raw_u8, fx, fy, cx, cy, mask_np = _undistort(raw_u8, camera_model, dist, fx, fy, cx, cy)

            cpu_images.append(torch.from_numpy(np.array(raw_u8, copy=True, order="C")))
            cpu_masks.append(torch.from_numpy(mask_np.copy()) if mask_np is not None else None)
            w2c = torch.tensor(entry["world_to_cam"], dtype=torch.float32, device=device)
            self.views.append(
                CameraView(
                    name=entry["filename"],
                    width=int(target_w),
                    height=int(target_h),
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    world_to_cam=w2c,
                    image=cpu_images[-1],
                    camera_model=camera_model or "PINHOLE",
                )
            )

        if total_cam_frames >= 4 and len(self.views) < max(2, int(total_cam_frames * 0.5)):
            raise ValueError(
                f"检测到抽帧图像 ({len(self.views)} 张匹配) 与相机位姿总数 ({total_cam_frames}) 严重不匹配（疑似旧缓存混用），"
                "请勾选强制重跑或从 01 ingest 重新执行。"
            )
        if not self.views:
            raise ValueError(f"No valid paired image/pose views found in {ws.root}")

        # 显存预算：总 uint8 字节超阈值则图像留在 CPU 锁页内存（按步异步拷贝），否则常驻 GPU。
        total_bytes = sum(int(t.numel()) for t in cpu_images)
        self.images_on_host = device.type == "cuda" and total_bytes > gpu_image_budget_bytes
        for view, img, mask in zip(self.views, cpu_images, cpu_masks):
            if device.type != "cuda":
                view._image_storage = img
            elif self.images_on_host:
                view._image_storage = img.pin_memory()
            else:
                view._image_storage = img.to(device)
            view.valid_mask = mask.to(device) if mask is not None else None

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, idx: int) -> CameraView:
        return self.views[idx]
