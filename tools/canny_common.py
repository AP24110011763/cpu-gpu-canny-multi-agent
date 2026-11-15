"""Pieces shared by the CPU and GPU Canny tools (Agent 2).

Keeping hysteresis here means BOTH paths run exactly the same code for that stage, so any
speed difference elsewhere is a fair comparison. (On the GPU path it runs on the CPU too,
because it is a connectivity problem that does not parallelise cleanly.)
"""
from __future__ import annotations

from pathlib import Path

import time

import cv2
import numpy as np

# Stage names are part of the contract with Agent 3. Do not rename.
STAGE_KEYS = ["gaussian_blur", "sobel_gradients", "non_max_suppression", "hysteresis"]


def load_gray(image_path: str) -> np.ndarray:
    """Read an image as 8-bit grayscale. Uses imdecode so Windows paths with spaces/unicode work."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"not a valid image: {image_path}")
    return img


def hysteresis(mag_thin: np.ndarray, low: float, high: float) -> np.ndarray:
    """Final Canny stage. Keep strong pixels, plus weak pixels connected (8-neighbour) to a strong one.

    mag_thin: gradient magnitude after non-max suppression (0 where suppressed).
    Returns a uint8 mask, 255 = edge.
    """
    strong = mag_thin >= high
    weak = mag_thin >= low
    n_labels, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros(mag_thin.shape, dtype=np.uint8)
    keep = np.zeros(n_labels, dtype=bool)
    keep[np.unique(labels[strong])] = True   # a component survives if it holds any strong pixel
    keep[0] = False                          # label 0 = background
    return (keep[labels] * 255).astype(np.uint8)


def save_png(path: str, array: np.ndarray) -> str:
    """Write a PNG with imencode + tofile (safe for Windows paths). Creates the folder if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    buf.tofile(str(p))
    return str(p)


# Warm-up policy, shared by the CPU and GPU tools so the comparison stays fair.
# A fixed count of passes is not enough: after a few idle seconds a laptop GPU drops to about 300 MHz and
# needs roughly half a second of real work to reach full clock (measured with scripts/gpu_idle_test.py:
# 3 warm-up passes left the timing 40-90% slower and 20x noisier, 40 passes fixed it). So warm-up keeps going
# until BOTH the planned number of passes AND this much time have passed.
MIN_WARMUP_SECONDS = 1.0
MAX_WARMUP_PASSES = 5000      # safety stop so a broken step can never loop forever


def warm_up(step, min_passes: int, min_seconds: float | None = None) -> int:
    """Call step() until at least min_passes calls AND min_seconds have passed. Returns the passes done."""
    seconds = MIN_WARMUP_SECONDS if min_seconds is None else min_seconds
    start, done = time.perf_counter(), 0
    while done < MAX_WARMUP_PASSES and (done < max(0, min_passes) or time.perf_counter() - start < seconds):
        step()
        done += 1
    return done
