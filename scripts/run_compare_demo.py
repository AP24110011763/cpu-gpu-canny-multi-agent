r"""Run the CPU and GPU Canny tools on one image and print REAL measured timings side by side.

    python scripts\run_compare_demo.py --image ..\photo2.png
    python scripts\run_compare_demo.py --image ..\photo2.png --repetitions 20
Saves cpu_edges.png and gpu_edges.png to outputs\compare_demo\.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.canny_common import STAGE_KEYS, load_gray  # noqa: E402
from tools.correctness import check_correctness  # noqa: E402
from tools.cpu_canny import run_cpu_canny  # noqa: E402
from tools.gpu_canny import run_gpu_canny  # noqa: E402


def run_single(a, out: Path) -> int:
    """One device only: timings and the edge image, no comparison and no correctness check."""
    run = run_cpu_canny if a.only == "cpu" else run_gpu_canny
    res, edges, _ = run(a.image, a.low, a.high, a.repetitions, a.warmup, str(out / f"{a.only}_edges.png"))
    if res.error:
        print(f"{a.only.upper()} error:", res.error)
        return 1
    h, w = load_gray(a.image).shape
    print(f"{a.only.upper()} only | image {w}x{h} ({w * h / 1e6:.2f} MP) | thresholds {a.low}/{a.high} | "
          f"{a.repetitions} runs after {a.warmup} warm-up runs\n")
    print(f"{'stage':<22}{'mean ms':>10}{'share':>9}")
    for k in STAGE_KEYS:
        v = res.stage_times_ms[k]
        print(f"{k:<22}{v:>10.3f}{v / res.total_time_ms:>8.0%}")
    print(f"{'TOTAL':<22}{res.total_time_ms:>10.3f}   (std {res.total_time_std_ms:.3f} ms)")
    if a.only == "gpu":
        print(f"transfer {res.transfer_time_ms:.3f} ms | kernels only {res.kernel_time_ms:.3f} ms")
    else:
        print(f"cv2.Canny reference {res.opencv_reference_time_ms:.3f} ms")
    print(f"Edge pixels: {int((edges > 0).sum())} ({(edges > 0).mean():.1%} of the image)")
    print(f"Saved: {res.output_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--low", type=int, default=100)
    ap.add_argument("--high", type=int, default=200)
    ap.add_argument("--repetitions", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--only", choices=["cpu", "gpu"], help="run just one device, no comparison")
    a = ap.parse_args()

    out = Path(__file__).resolve().parent.parent / "outputs" / "compare_demo"
    if a.only:
        return run_single(a, out)
    cpu, c_edges, c_mag = run_cpu_canny(a.image, a.low, a.high, a.repetitions, a.warmup, str(out / "cpu_edges.png"))
    if cpu.error:
        print("CPU error:", cpu.error)
        return 1
    gpu, g_edges, g_mag = run_gpu_canny(a.image, a.low, a.high, a.repetitions, a.warmup, str(out / "gpu_edges.png"))
    if gpu.error:
        print("GPU error:", gpu.error)
        return 1

    h, w = load_gray(a.image).shape
    print(f"Image {w}x{h} ({w * h / 1e6:.2f} MP) | thresholds {a.low}/{a.high} | "
          f"{a.repetitions} runs after {a.warmup} warm-up runs\n")
    print(f"{'stage':<22}{'CPU ms':>10}{'GPU ms':>10}{'speedup':>10}")
    for k in STAGE_KEYS:
        c, g = cpu.stage_times_ms[k], gpu.stage_times_ms[k]
        print(f"{k:<22}{c:>10.3f}{g:>10.3f}{c / g:>9.1f}x")
    print(f"{'TOTAL (end to end)':<22}{cpu.total_time_ms:>10.3f}{gpu.total_time_ms:>10.3f}"
          f"{cpu.total_time_ms / gpu.total_time_ms:>9.1f}x")
    print(f"   std                {cpu.total_time_std_ms:>10.3f}{gpu.total_time_std_ms:>10.3f}\n")

    cpu_kernel_part = sum(cpu.stage_times_ms[k] for k in STAGE_KEYS[:3])
    print(f"GPU transfer (host<->device) : {gpu.transfer_time_ms:.3f} ms "
          f"({gpu.transfer_time_ms / gpu.total_time_ms:.0%} of GPU total)")
    print(f"GPU kernels only             : {gpu.kernel_time_ms:.3f} ms")
    print(f"Kernel-only speedup          : {cpu_kernel_part / gpu.kernel_time_ms:.1f}x   (CPU blur+sobel+nms vs GPU kernels)")
    ratio = gpu.total_time_ms / cpu.opencv_reference_time_ms
    verdict = (f"cv2.Canny is {ratio:.1f}x FASTER than the GPU path" if ratio > 1
               else f"the GPU path is {1 / ratio:.1f}x faster than cv2.Canny")
    print(f"cv2.Canny reference (CPU)    : {cpu.opencv_reference_time_ms:.3f} ms  -> {verdict}\n")

    chk = check_correctness(c_edges, g_edges, c_mag, g_mag)
    print(f"Correctness: {'PASS' if chk.passed else 'FAIL'} | pixel agreement {chk.pixel_agreement_pct}% "
          f"(need >= {chk.threshold_pct}%) | gradient mean abs diff {chk.gradient_mean_abs_diff}")
    print(f"             {chk.note}")
    print(f"Saved: {cpu.output_path}\n       {gpu.output_path}")
    return 0 if chk.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
