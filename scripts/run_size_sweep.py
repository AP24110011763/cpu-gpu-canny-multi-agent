r"""Time CPU and GPU Canny at several image sizes and show whether the speedup grows with size.

    python scripts\run_size_sweep.py --image ..\image1.png --note "plugged in, Best performance"
    python scripts\run_size_sweep.py --image ..\image1.png --widths 256 512 1024 2048 3072 --repetitions 10
Real measurements only. No language model is used. Results: outputs\size_sweep\sweep.html / .csv / .json
"""
import argparse
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.size_sweep import DEFAULT_WIDTHS, format_table, sweep, write_outputs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    ap.add_argument("--low", type=int, default=100)
    ap.add_argument("--high", type=int, default=200)
    ap.add_argument("--repetitions", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--note", default="", help='machine state, e.g. "plugged in, Best performance"')
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    if not Path(a.image).exists():
        print(f"image not found: {a.image}")
        return 1
    if not a.note:
        print("Tip: add --note \"plugged in, Best performance\" so the page records the conditions.\n")
    out = str(Path(__file__).resolve().parent.parent / "outputs" / "size_sweep")
    t0 = time.time()
    rows = sweep(a.image, a.widths, a.low, a.high, a.repetitions, a.warmup, out)
    files = write_outputs(rows, out, a.image, a.note)
    print("\n" + format_table(rows) + f"\n\n{files['finding']}\nTook {time.time() - t0:.0f} s. Saved: {files['html']}")
    if not a.no_open:
        webbrowser.open(Path(files["html"]).resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
