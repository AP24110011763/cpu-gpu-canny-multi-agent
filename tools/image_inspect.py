"""Agent 1's tool: look at the image so the planner can choose sensible parameters.

Plain deterministic function. The LLM may *call* it; it never computes these numbers.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _size_class(megapixels: float) -> tuple[str, int]:
    """Bigger images take longer per run, so fewer repetitions keep the demo quick."""
    if megapixels < 0.5:
        return "small", 20
    if megapixels < 2.0:
        return "medium", 10
    return "large", 5


def inspect_image(image_path: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        return {"ok": False, "error": f"image not found: {image_path}"}
    if path.suffix.lower() not in SUPPORTED_EXT:
        return {"ok": False, "error": f"unsupported image type '{path.suffix}'"}

    # np.fromfile + imdecode (instead of cv2.imread) also works with non-ASCII Windows paths
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        return {"ok": False, "error": f"could not read image: {exc}"}
    if img is None:
        return {"ok": False, "error": "file exists but is not a valid image"}

    height, width = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    megapixels = round(width * height / 1_000_000, 4)
    size_class, suggested = _size_class(megapixels)
    return {
        "ok": True,
        "width": width,
        "height": height,
        "channels": channels,
        "megapixels": megapixels,
        "size_class": size_class,
        "suggested_repetitions": suggested,
    }
