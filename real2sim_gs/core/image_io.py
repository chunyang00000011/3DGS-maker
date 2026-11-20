"""Unicode-safe OpenCV image I/O helpers.

`cv2.imread` / `cv2.imwrite` silently fail on Windows for non-ASCII absolute paths
(e.g. ``d:\\a临时下载\\...``). These helpers route through numpy byte buffers instead.
cv2 is imported lazily so importing this module stays lightweight.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable


def imread_unicode(path: str | Path, flags: int | None = None) -> Any:
    """Read an image from any (incl. non-ASCII) path; returns ndarray or None if undecodable/missing."""
    import cv2
    import numpy as np

    p = Path(path)
    if not p.is_file():
        return None
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR if flags is None else flags)


def imwrite_unicode(path: str | Path, image: Any, params: list[int] | None = None) -> None:
    """Encode by file suffix and write to any (incl. non-ASCII) path; raises OSError on failure."""
    import cv2

    p = Path(path)
    ext = p.suffix or ".png"
    ok, buf = cv2.imencode(ext, image, params or [])
    if not ok:
        raise OSError(f"cv2.imencode 编码失败 ({ext}): {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        buf.tofile(str(p))
    except OSError as exc:
        raise OSError(f"图像写入失败: {p} ({exc})") from exc
    if not p.is_file() or p.stat().st_size == 0:
        raise OSError(f"图像写入失败（文件为空或不存在）: {p}")


_NUM_SPLIT = re.compile(r"(\d+)")


def natural_sort_key(value: str | Path) -> tuple[Any, ...]:
    """Human-friendly sort key: img_2 < img_10 (case-insensitive)."""
    name = value.name if isinstance(value, Path) else str(value)
    parts = _NUM_SPLIT.split(name.lower())
    return tuple((0, int(tok), "") if tok.isdigit() else (1, 0, tok) for tok in parts)


def natural_sorted(items: Iterable[Any]) -> list[Any]:
    return sorted(items, key=natural_sort_key)
