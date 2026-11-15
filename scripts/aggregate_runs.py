r"""Mean +/- std across several full pipeline runs.

    python scripts\aggregate_runs.py --last 3 --note "plugged in, Best performance"
    python scripts\aggregate_runs.py --results outputs\run_A\results.json outputs\run_B\results.json outputs\run_C\results.json
Run main.py three times first (same image, same request). Writes outputs\final_numbers.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.aggregate import aggregate, format_aggregate, latest_result_paths, load_results  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+")
    ap.add_argument("--last", type=int, help="use the newest N runs in outputs\\")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent / "outputs"
    paths = a.results or latest_result_paths(str(root), a.last or 3)
    if not paths:
        print("no results.json files found")
        return 1
    agg = aggregate(load_results(paths))
    agg["note"] = a.note
    print(format_aggregate(agg, a.note))
    (root / "final_numbers.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\nSaved: {root / 'final_numbers.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
