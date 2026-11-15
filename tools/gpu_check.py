"""Agent 2 tool: is the GPU really usable? Never raises; returns a dict describing what it found.

Checks, in order: the NVIDIA driver (nvidia-smi), then CuPy (import, device, a tiny kernel).
The agent calls this BEFORE any GPU work, and falls back to CPU-only if it fails.
"""
from __future__ import annotations

import subprocess


def check_gpu() -> dict:
    info: dict = {
        "ok": False,
        "gpu_name": None,
        "driver_version": None,
        "vram_mb": None,
        "cupy_version": None,
        "cuda_runtime": None,
        "compute_capability": None,
        "problems": [],
    }

    # 1) driver
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, driver, mem = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
            info.update(gpu_name=name, driver_version=driver, vram_mb=int(float(mem)))
        else:
            info["problems"].append(f"nvidia-smi failed: {(out.stderr or out.stdout).strip()[:200]}")
    except FileNotFoundError:
        info["problems"].append("nvidia-smi not found: NVIDIA driver is missing or not on PATH")
    except Exception as exc:  # timeout, parse error...
        info["problems"].append(f"nvidia-smi check failed: {type(exc).__name__}: {exc}")

    # 2) CuPy
    try:
        import cupy as cp
    except ImportError:
        info["problems"].append("CuPy is not installed in this environment")
        return info
    except Exception as exc:
        info["problems"].append(f"CuPy is installed but failed to load: {type(exc).__name__}: {exc}")
        return info

    try:
        info["cupy_version"] = cp.__version__
        if cp.cuda.runtime.getDeviceCount() < 1:
            info["problems"].append("CuPy sees no CUDA device")
            return info
        props = cp.cuda.runtime.getDeviceProperties(0)
        raw = props["name"]
        info["gpu_name"] = raw.decode() if isinstance(raw, bytes) else str(raw)
        info["compute_capability"] = f"{props['major']}.{props['minor']}"
        info["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        x = cp.arange(10, dtype=cp.float32)
        cp.cuda.Device().synchronize()
        if int((x * 2).sum().get()) != 90:
            info["problems"].append("a test kernel returned a wrong result")
            return info
    except Exception as exc:
        info["problems"].append(f"CuPy could not run a kernel: {type(exc).__name__}: {exc}")
        return info

    info["ok"] = not info["problems"]
    return info
