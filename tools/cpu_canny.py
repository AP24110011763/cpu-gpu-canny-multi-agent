"""Agent 2 tool: staged Canny edge detection on the CPU, with honest timing.

Four stages, each timed separately (names from canny_common.STAGE_KEYS):
  1. gaussian_blur         5x5 Gaussian, sigma 1.4
  2. sobel_gradients       Sobel dx/dy, magnitude = |gx| + |gy| (same convention as cv2.Canny)
  3. non_max_suppression   keep a pixel only if it is the local maximum along its gradient direction
  4. hysteresis            shared with the GPU path (see canny_common.py)

Timing method: `warmup_runs` runs are discarded, then `repetitions` runs are timed with
time.perf_counter(); we report the mean and standard deviation. The image is already in memory,
so file reading is NOT part of the timed region. cv2.Canny is timed the same way as a reference
(an optimised library call; it cannot be split into stages).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from schemas import DeviceResult
from tools.canny_common import STAGE_KEYS, hysteresis, load_gray, save_png, warm_up

SIGMA = 1.4
KERNEL = 5


def gaussian_blur(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, (KERNEL, KERNEL), SIGMA)


def sobel_gradients(blurred: np.ndarray):
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, np.abs(gx) + np.abs(gy)


def non_max_suppression(gx: np.ndarray, gy: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Vectorised NMS over four direction sectors (image coordinates: y points down)."""
    angle = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0   # 0..180
    padded = np.pad(mag, 1, mode="constant")
    h, w = mag.shape

    def shifted(dy: int, dx: int) -> np.ndarray:  # neighbour at (y+dy, x+dx)
        return padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]

    sector0 = (angle < 22.5) | (angle >= 157.5)          # gradient along x -> compare left/right
    sector45 = (angle >= 22.5) & (angle < 67.5)          # down-right diagonal
    sector90 = (angle >= 67.5) & (angle < 112.5)         # gradient along y -> compare up/down
    sector135 = (angle >= 112.5) & (angle < 157.5)       # down-left diagonal

    n1 = np.where(sector0, shifted(0, -1), np.where(sector45, shifted(-1, -1),
         np.where(sector90, shifted(-1, 0), shifted(-1, 1))))
    n2 = np.where(sector0, shifted(0, 1), np.where(sector45, shifted(1, 1),
         np.where(sector90, shifted(1, 0), shifted(1, -1))))
    return np.where((mag >= n1) & (mag >= n2), mag, 0.0).astype(np.float32)


def canny_staged(gray: np.ndarray, low: float, high: float):
    """One full pass. Returns (edges, magnitude, per-stage seconds)."""
    t = {}
    t0 = time.perf_counter()
    blurred = gaussian_blur(gray)
    t1 = time.perf_counter(); t["gaussian_blur"] = t1 - t0
    gx, gy, mag = sobel_gradients(blurred)
    t2 = time.perf_counter(); t["sobel_gradients"] = t2 - t1
    thin = non_max_suppression(gx, gy, mag)
    t3 = time.perf_counter(); t["non_max_suppression"] = t3 - t2
    edges = hysteresis(thin, low, high)
    t4 = time.perf_counter(); t["hysteresis"] = t4 - t3
    return edges, mag, t


def run_cpu_canny(
    image_path: str,
    low_threshold: int = 100,
    high_threshold: int = 200,
    repetitions: int = 10,
    warmup_runs: int = 3,
    out_path: str | None = None,
    min_warmup_seconds: float | None = None,
):
    """Time the staged CPU Canny. Returns (DeviceResult, edges, gradient_magnitude).

    Never raises for a bad image: the problem is returned in DeviceResult.error so the agent can react.
    """
    try:
        gray = load_gray(image_path)
    except (FileNotFoundError, ValueError) as exc:
        return (DeviceResult(device="cpu", stage_times_ms={}, total_time_ms=0.0, total_time_std_ms=0.0,
                             repetitions=repetitions, warmup_runs=warmup_runs, error=str(exc)), None, None)

    warmed = warm_up(lambda: canny_staged(gray, low_threshold, high_threshold), warmup_runs, min_warmup_seconds)   # discarded

    stage_runs = {k: [] for k in STAGE_KEYS}
    totals = []
    edges = mag = None
    for _ in range(repetitions):
        edges, mag, t = canny_staged(gray, low_threshold, high_threshold)
        for k in STAGE_KEYS:
            stage_runs[k].append(t[k] * 1000.0)
        totals.append(sum(t.values()) * 1000.0)

    ref = []
    warm_up(lambda: cv2.Canny(gray, low_threshold, high_threshold), warmup_runs, min_warmup_seconds)
    for _ in range(repetitions):
        t0 = time.perf_counter()
        cv2.Canny(gray, low_threshold, high_threshold)
        ref.append((time.perf_counter() - t0) * 1000.0)

    saved = save_png(out_path, edges) if out_path else None
    result = DeviceResult(
        device="cpu",
        stage_times_ms={k: round(float(np.mean(v)), 4) for k, v in stage_runs.items()},
        total_time_ms=round(float(np.mean(totals)), 4),
        total_time_std_ms=round(float(np.std(totals)), 4),
        opencv_reference_time_ms=round(float(np.mean(ref)), 4),
        repetitions=repetitions,
        warmup_runs=warmed,                                    # passes actually done (at least the planned number)
        output_path=saved,
    )
    return result, edges, mag
