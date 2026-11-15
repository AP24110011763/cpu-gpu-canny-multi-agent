r"""Try Agent 2 on its own with the real LLM, the real CPU and the real GPU.

    python scripts\run_executor.py --image ..\image1.png
    python scripts\run_executor.py --image ..\image1.png --gpu-only --repetitions 10
There is no Agent 1 here: you write the plan yourself with the flags. (Agent 1 is joined to Agent 2
in the next stage.) Saves outputs\<run_id>\results.json, the exact object Agent 3 will receive.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.executor import ExecutorError, run_executor  # noqa: E402
from llm.client import OllamaClient  # noqa: E402
from schemas import ExecutionPlan  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Agent 2 (Execution Agent) alone")
    ap.add_argument("--image", required=True)
    ap.add_argument("--low", type=int, default=100)
    ap.add_argument("--high", type=int, default=200)
    ap.add_argument("--repetitions", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    only = ap.add_mutually_exclusive_group()
    only.add_argument("--cpu-only", action="store_true")
    only.add_argument("--gpu-only", action="store_true")
    a = ap.parse_args()

    plan = ExecutionPlan(task="canny_edge_detection", image_path=str(Path(a.image).resolve()),
                         low_threshold=a.low, high_threshold=a.high, repetitions=a.repetitions,
                         warmup_runs=a.warmup, run_cpu=not a.gpu_only, run_gpu=not a.cpu_only)
    out_root = Path(__file__).resolve().parent.parent / "outputs"
    print("Agent 2 is working. The LLM runs on the CPU, then the benchmark runs; this can take a minute or two...")
    try:
        res = run_executor(plan, OllamaClient(), output_root=str(out_root))
    except ExecutorError as exc:
        print(f"Agent 2 could not run the plan: {exc}")
        return 1

    print(f"\nRun {res.plan.run_id} | image {res.image_info.width}x{res.image_info.height} "
          f"({res.image_info.megapixels} MP) | GPU: {res.gpu_name}")
    for r in (res.cpu, res.gpu):
        if r is None:
            continue
        if r.error:
            print(f"  {r.device.upper()}: ERROR {r.error}")
            continue
        extra = f" | transfer {r.transfer_time_ms} ms, kernels {r.kernel_time_ms} ms" if r.transfer_time_ms is not None else ""
        print(f"  {r.device.upper()}: total {r.total_time_ms} ms (std {r.total_time_std_ms}){extra}")
    c = res.correctness
    print(f"  Correctness: performed={c.performed} passed={c.passed} agreement={c.pixel_agreement_pct}%")
    print(f"\nAgent 2 notes: {res.agent2_notes}")

    path = out_root / res.plan.run_id / "results.json"
    path.write_text(res.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
