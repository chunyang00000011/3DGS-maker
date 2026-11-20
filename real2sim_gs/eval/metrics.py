from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_bchw(img: torch.Tensor, clamp: bool = False) -> torch.Tensor:
    """Normalize (H,W,C), (C,H,W), or (B,C,H,W) image tensor into float32 BCHW.

    `clamp=True` 仅用于评测路径；训练 Loss 路径不得 clamp，否则越界像素梯度被截断为 0。
    """
    t = img.detach().float() if not img.requires_grad else img.float()
    if t.ndim == 3:
        if t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4):
            t = t.permute(2, 0, 1)
        elif t.shape[-1] in (1, 3, 4) and t.shape[0] > 4:
            t = t.permute(2, 0, 1)
        t = t.unsqueeze(0)
    elif t.ndim != 4:
        raise ValueError(f"Expected 3D or 4D image tensor, got shape {tuple(t.shape)}")
    return t.clamp(0.0, 1.0) if clamp else t


def _gaussian_1d_kernels(
    window_size: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (window_size, channels, device, dtype)
    cached = _SSIM_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2.0 * 1.5**2))
    gauss = gauss / gauss.sum()
    w_h = gauss.view(1, 1, 1, window_size).expand(channels, 1, 1, window_size).contiguous()
    w_v = gauss.view(1, 1, window_size, 1).expand(channels, 1, window_size, 1).contiguous()
    _SSIM_KERNEL_CACHE[key] = (w_h, w_v)
    return w_h, w_v


_SSIM_KERNEL_CACHE: dict[tuple[int, int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _ssim_bchw(x: torch.Tensor, y: torch.Tensor, window_size: int) -> torch.Tensor:
    channels = x.shape[1]
    win_sz = min(window_size, x.shape[-2], x.shape[-1])
    if win_sz % 2 == 0:
        win_sz = max(1, win_sz - 1)
    w_h, w_v = _gaussian_1d_kernels(win_sz, channels, x.device, x.dtype)
    pad = win_sz // 2

    def _blur(z: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.conv2d(z, w_h, padding=(0, pad), groups=channels), w_v, padding=(pad, 0), groups=channels)

    mu_x = _blur(x)
    mu_y = _blur(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = _blur(x * x) - mu_x2
    sigma_y2 = _blur(y * y) - mu_y2
    sigma_xy = _blur(x * y) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (num / (den + 1e-8)).mean()


def ssim_tensor(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Differentiable SSIM scalar tensor (no clamping -> gradients stay alive for out-of-range pixels)."""
    return _ssim_bchw(_to_bchw(pred), _to_bchw(target), window_size)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) on [0, 1]-clamped image tensors (evaluation only)."""
    with torch.no_grad():
        x = _to_bchw(pred, clamp=True)
        y = _to_bchw(target, clamp=True)
        mse = F.mse_loss(x, y).clamp_min(1e-10)
        return float((10.0 * torch.log10(1.0 / mse)).item())


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Compute Structural Similarity Index (SSIM) on [0, 1]-clamped image tensors (evaluation only)."""
    with torch.no_grad():
        return float(_ssim_bchw(_to_bchw(pred, clamp=True), _to_bchw(target, clamp=True), window_size).item())


class ImagePairEvaluator:
    """Streaming PSNR/SSIM aggregator (avoids holding every rendered view in memory)."""

    def __init__(self) -> None:
        self.psnrs: list[float] = []
        self.ssims: list[float] = []

    def add(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.psnrs.append(compute_psnr(pred, target))
        self.ssims.append(compute_ssim(pred, target))

    def __len__(self) -> int:
        return len(self.psnrs)

    def summary(self) -> dict[str, float]:
        n = len(self.psnrs)
        return {
            "PSNR": round(sum(self.psnrs) / max(1, n), 4),
            "SSIM": round(sum(self.ssims) / max(1, n), 4),
            "eval_views": n,
        }


def evaluate_image_pairs(preds: list[torch.Tensor], targets: list[torch.Tensor]) -> dict[str, float]:
    """Aggregate PSNR and SSIM across rendered/ground-truth view pairs."""
    if not preds or len(preds) != len(targets):
        raise ValueError("preds and targets must be non-empty lists of equal length.")
    evaluator = ImagePairEvaluator()
    for p, t in zip(preds, targets):
        evaluator.add(p, t)
    return evaluator.summary()
