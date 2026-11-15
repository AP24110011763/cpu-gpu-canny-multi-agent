r"""Repeat the CPU-vs-GPU benchmark several times WITHOUT the language model (about 1-2 minutes for 5 rounds).

    python scripts\repeat_benchmark.py --image ..\image1.png --rounds 5 --note "plugged in, Best performance"
Stop Ollama first (ollama stop llama3.2:3b) so the model is not using the CPU. Saves outputs\repeat_<time>\final_numbers.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.aggregate import format_aggregate  # noqa: E402
from tools.repeat_benchmark import repeat_benchmark  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--low", type=int, default=100)
    ap.add_argument("--high", type=int, default=200)
    ap.add_argument("--repetitions", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--gap", type=float, default=3.0, help="quiet seconds before each timed run (as in Agent 2)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if not Path(a.image).exists():
        print(f"image not found: {a.image}")
        return 1
    root = Path(__file__).resolve().parent.parent / "outputs"
    try:
        _, agg, folder = repeat_benchmark(a.image, a.rounds, a.low, a.high, a.repetitions, a.warmup, a.gap, str(root))
    except RuntimeError as exc:
        print(f"Stopped: {exc}")
        return 1
    agg["note"] = a.note
    print("\n" + format_aggregate(agg, a.note))
    (folder / "final_numbers.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\nSaved: {folder / 'final_numbers.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
