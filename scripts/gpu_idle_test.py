r"""Does an idle gap before the GPU run make the GPU timing slower or noisier?  (no LLM, about 2 minutes)

    python scripts\gpu_idle_test.py --image ..\image1.png
For each idle gap (seconds) and warm-up count it waits, reads the GPU clock state with nvidia-smi (if present),
then times the GPU Canny exactly like the pipeline does. Three rounds, so you can see whether it repeats.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.gpu_canny import run_gpu_canny  # noqa: E402


def gpu_clock_state() -> str:
    """Current graphics clock, its maximum and the performance state, or '' if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,pstate", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.splitlines()[0] if out else ""
    except Exception:
        return ""


def probe(image, gaps, warmups, rounds, reps, sleep=time.sleep, run=run_gpu_canny, clock=gpu_clock_state, say=print):
    rows = []
    for rnd in range(1, rounds + 1):
        for gap in gaps:
            for warm in warmups:
                sleep(gap)
                state = clock()
                res, _, _ = run(image, 100, 200, reps, warm, None)
                if res.error:
                    say(f"GPU error: {res.error}")
                    return rows
                rows.append({"round": rnd, "gap_s": gap, "warmup": warm, "clock": state,
                             "gpu_ms": res.total_time_ms, "std_ms": res.total_time_std_ms})
                say(f"round {rnd} | idle {gap:>4g} s | warm-up {warm:>3} | clock/pstate before run: {state or 'n/a':<28}"
                    f"| GPU {res.total_time_ms:7.2f} ms (std {res.total_time_std_ms:6.2f})")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gaps", type=float, nargs="+", default=[0, 3, 10])
    ap.add_argument("--warmups", type=int, nargs="+", default=[3, 40])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repetitions", type=int, default=20)
    a = ap.parse_args()
    if not Path(a.image).exists():
        print(f"image not found: {a.image}")
        return 1
    probe(a.image, a.gaps, a.warmups, a.rounds, a.repetitions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
