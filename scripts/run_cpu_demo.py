r"""Run the CPU Canny tool on an image and print REAL measured timings.

    python scripts/run_cpu_demo.py --image ..\test.png
    python scripts/run_cpu_demo.py --image ..\photo2.png --repetitions 20
Saves the edge image to outputs\cpu_demo\cpu_edges.png.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.canny_common import STAGE_KEYS, load_gray  # noqa: E402
from tools.cpu_canny import run_cpu_canny  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--low", type=int, default=100)
    ap.add_argument("--high", type=int, default=200)
    ap.add_argument("--repetitions", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()

    out = Path(__file__).resolve().parent.parent / "outputs" / "cpu_demo" / "cpu_edges.png"
    res, edges, _ = run_cpu_canny(a.image, a.low, a.high, a.repetitions, a.warmup, str(out))
    if res.error:
        print("Error:", res.error)
        return 1
    h, w = load_gray(a.image).shape
    print(f"Image {w}x{h} ({w * h / 1e6:.2f} MP) | thresholds {a.low}/{a.high} | "
          f"{a.repetitions} runs after {a.warmup} warm-up runs\n")
    print(f"{'stage':<22}{'mean ms':>10}{'share':>9}")
    for k in STAGE_KEYS:
        v = res.stage_times_ms[k]
        print(f"{k:<22}{v:>10.3f}{v / res.total_time_ms:>8.0%}")
    print(f"{'TOTAL':<22}{res.total_time_ms:>10.3f}   (std {res.total_time_std_ms:.3f} ms)")
    print(f"cv2.Canny reference   {res.opencv_reference_time_ms:>10.3f} ms")
    print(f"Edge pixels: {int((edges > 0).sum())} ({(edges > 0).mean():.1%} of the image)")
    print(f"Edge image saved to: {res.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
