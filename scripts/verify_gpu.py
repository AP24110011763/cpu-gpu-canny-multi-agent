"""Agent 2, stage 1: can this PC run the GPU part? Run it, read the verdict.

    python scripts\verify_gpu.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.gpu_check import check_gpu  # noqa: E402


def main() -> int:
    info = check_gpu()
    print("GPU check")
    print("---------")
    for key in ("gpu_name", "driver_version", "vram_mb", "compute_capability", "cupy_version", "cuda_runtime"):
        print(f"{key:20s}: {info[key]}")
    print()
    if info["ok"]:
        print("RESULT: PASS - the GPU is ready for the benchmark.")
        return 0
    print("RESULT: FAIL")
    for p in info["problems"]:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
