"""Agent 2 tool: staged Canny edge detection on the GPU (CuPy), with honest timing.

Same four stages and the same maths as tools/cpu_canny.py, so the comparison is like-for-like:
  1. gaussian_blur         5x5 Gaussian, sigma 1.4 (two 1-D passes)   -> GPU
  2. sobel_gradients       Sobel dx/dy, magnitude = |gx| + |gy|        -> GPU
  3. non_max_suppression   local maximum along the gradient direction  -> GPU
  4. hysteresis            edge linking by connectivity                -> CPU (shared code)
Stage 4 stays on the CPU on purpose: it is a connectivity problem (an edge can snake across the
image), so it does not split into independent per-pixel work. That is the "serial part" in
Amdahl's law.

How each GPU stage runs: every line like `a + b` or `xp.where(...)` below is one CUDA kernel that
CuPy launches over ALL pixels at once - thousands of GPU threads, one pixel (or a few) per thread.

Timing method (the part to explain in a viva):
  * Warm-up passes are thrown away: at least warmup_runs AND at least 1 s of work (see canny_common.warm_up). The first GPU call pays one-off costs: CUDA context
    creation, kernel compilation, and memory-pool allocation (cudaMalloc). We do not want those
    in a steady-state measurement.
  * GPU work is ASYNCHRONOUS: Python returns before the GPU is finished. So before reading the
    clock we always call synchronize(); without it we would time only the kernel *launch*.
  * Per stage we time: host->device copy (h2d), each kernel stage, device->host copy (d2h), and
    hysteresis on the CPU. total_time_ms is a separate wall-clock around the whole pass.
  * transfer_time_ms = h2d + d2h.  kernel_time_ms = blur + sobel + nms (GPU stages only).
  * File reading is not timed. Mean and standard deviation over `repetitions` timed passes.
"""
from __future__ import annotations

import time
import warnings

import cv2
import numpy as np

from schemas import DeviceResult
from tools.canny_common import STAGE_KEYS, hysteresis, load_gray, save_png, warm_up
from tools.cpu_canny import KERNEL, SIGMA

_K1D = [float(v) for v in cv2.getGaussianKernel(KERNEL, SIGMA).flatten()]


# --------------------------------------------------------------------------------------------
# The three GPU stages. They take `xp` (the array module) so the SAME code runs on the GPU with
# CuPy and, for testing without a GPU, on the CPU with NumPy.
# --------------------------------------------------------------------------------------------
def gaussian_blur_xp(xp, gray):
    """Separable 5x5 Gaussian: horizontal pass then vertical pass. Result stays float32, rounded
    to whole grey levels like cv2.GaussianBlur's uint8 output."""
    f = gray.astype(xp.float32)
    h, w = f.shape
    r = KERNEL // 2
    p = xp.pad(f, ((0, 0), (r, r)), mode="reflect")
    tmp = _K1D[0] * p[:, 0:w]
    for i in range(1, KERNEL):
        tmp = tmp + _K1D[i] * p[:, i:i + w]
    p = xp.pad(tmp, ((r, r), (0, 0)), mode="reflect")
    out = _K1D[0] * p[0:h, :]
    for i in range(1, KERNEL):
        out = out + _K1D[i] * p[i:i + h, :]
    return xp.clip(xp.rint(out), 0, 255).astype(xp.float32)


def sobel_gradients_xp(xp, blurred):
    """3x3 Sobel built from shifted copies of the image. Each output pixel reads its 8 neighbours."""
    h, w = blurred.shape
    p = xp.pad(blurred, 1, mode="reflect")
    tl, tc, tr = p[0:h, 0:w], p[0:h, 1:w + 1], p[0:h, 2:w + 2]
    ml, mr = p[1:h + 1, 0:w], p[1:h + 1, 2:w + 2]
    bl, bc, br = p[2:h + 2, 0:w], p[2:h + 2, 1:w + 1], p[2:h + 2, 2:w + 2]
    gx = (tr - tl) + 2.0 * (mr - ml) + (br - bl)
    gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr)
    return gx, gy, xp.abs(gx) + xp.abs(gy)


def non_max_suppression_xp(xp, gx, gy, mag):
    """Same four-sector rule as the CPU version (image coordinates: y points down)."""
    angle = (xp.degrees(xp.arctan2(gy, gx)) + 180.0) % 180.0
    padded = xp.pad(mag, 1, mode="constant")
    h, w = mag.shape

    def shifted(dy, dx):
        return padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    s0 = (angle < 22.5) | (angle >= 157.5)
    s45 = (angle >= 22.5) & (angle < 67.5)
    s90 = (angle >= 67.5) & (angle < 112.5)
    n1 = xp.where(s0, shifted(0, -1), xp.where(s45, shifted(-1, -1),
         xp.where(s90, shifted(-1, 0), shifted(-1, 1))))
    n2 = xp.where(s0, shifted(0, 1), xp.where(s45, shifted(1, 1),
         xp.where(s90, shifted(1, 0), shifted(1, -1))))
    return xp.where((mag >= n1) & (mag >= n2), mag, 0.0).astype(xp.float32)


# --------------------------------------------------------------------------------------------
def _load_cupy():
    """Import CuPy and confirm a CUDA device exists. Raises if not (the caller turns it into an error)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # CuPy's harmless "CUDA path could not be detected"
        import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("no CUDA device found")
    return cp


def _one_pass(cp, gray_host, low, high):
    """One full end-to-end pass with per-stage timers. Returns (edges, device_magnitude, seconds-dict)."""
    sync = cp.cuda.Stream.null.synchronize
    t = {}
    sync()
    start = time.perf_counter()

    d_gray = cp.asarray(gray_host)                     # host -> device over PCIe
    sync(); a = time.perf_counter(); t["h2d"] = a - start

    d_blur = gaussian_blur_xp(cp, d_gray)
    sync(); b = time.perf_counter(); t["gaussian_blur"] = b - a

    d_gx, d_gy, d_mag = sobel_gradients_xp(cp, d_blur)
    sync(); c = time.perf_counter(); t["sobel_gradients"] = c - b

    d_thin = non_max_suppression_xp(cp, d_gx, d_gy, d_mag)
    sync(); d = time.perf_counter(); t["non_max_suppression"] = d - c

    thin = cp.asnumpy(d_thin)                          # device -> host (blocks until the copy is done)
    e = time.perf_counter(); t["d2h"] = e - d

    edges = hysteresis(thin, low, high)                # CPU, same code as the CPU path
    f = time.perf_counter(); t["hysteresis"] = f - e

    t["total"] = f - start
    return edges, d_mag, t


def run_gpu_canny(
    image_path: str,
    low_threshold: int = 100,
    high_threshold: int = 200,
    repetitions: int = 10,
    warmup_runs: int = 3,
    out_path: str | None = None,
    min_warmup_seconds: float | None = None,
):
    """Time the staged GPU Canny. Returns (DeviceResult, edges, gradient_magnitude).

    Never raises for a missing GPU, missing CuPy or a bad image: the problem is returned in
    DeviceResult.error so Agent 2 can fall back to CPU-only.
    """
    def failed(msg):
        return (DeviceResult(device="gpu", stage_times_ms={}, total_time_ms=0.0, total_time_std_ms=0.0,
                             repetitions=repetitions, warmup_runs=warmup_runs, error=msg), None, None)

    try:
        gray = load_gray(image_path)
    except (FileNotFoundError, ValueError) as exc:
        return failed(str(exc))
    try:
        cp = _load_cupy()
    except Exception as exc:  # ImportError, driver problems, no device ...
        return failed(f"GPU not usable: {type(exc).__name__}: {exc}")

    try:
        # discarded: context, JIT compile, pool growth, AND the GPU clock climbing back from its idle state
        warmed = warm_up(lambda: _one_pass(cp, gray, low_threshold, high_threshold), warmup_runs, min_warmup_seconds)

        runs = {k: [] for k in ("h2d", "gaussian_blur", "sobel_gradients", "non_max_suppression",
                                "d2h", "hysteresis", "total")}
        edges = d_mag = None
        for _ in range(repetitions):
            edges, d_mag, t = _one_pass(cp, gray, low_threshold, high_threshold)
            for k in runs:
                runs[k].append(t[k] * 1000.0)
        mag = cp.asnumpy(d_mag)                        # outside the timed region
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    except Exception as exc:                           # e.g. out of GPU memory on a huge image
        return failed(f"GPU run failed: {type(exc).__name__}: {exc}")

    mean = lambda k: float(np.mean(runs[k]))
    transfer = [a + b for a, b in zip(runs["h2d"], runs["d2h"])]
    kernel = [a + b + c for a, b, c in zip(runs["gaussian_blur"], runs["sobel_gradients"],
                                           runs["non_max_suppression"])]
    saved = save_png(out_path, edges) if out_path else None
    result = DeviceResult(
        device="gpu",
        stage_times_ms={k: round(mean(k), 4) for k in STAGE_KEYS},
        total_time_ms=round(mean("total"), 4),
        total_time_std_ms=round(float(np.std(runs["total"])), 4),
        transfer_time_ms=round(float(np.mean(transfer)), 4),
        kernel_time_ms=round(float(np.mean(kernel)), 4),
        repetitions=repetitions,
        warmup_runs=warmed,                            # passes actually done (at least the planned number)
        output_path=saved,
    )
    return result, edges, mag
