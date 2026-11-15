"""Repeat ONLY the benchmark (no language model) several times and aggregate: fast, trustworthy final numbers.

It calls the same CPU and GPU tools that Agent 2 calls, with the same settings and the same quiet gap before
each timed run, and the same correctness check. What it leaves out is the LLM: the agents decide the ORDER of
these calls, they do not change any measured number. The full agent pipeline is still run separately for the demo.
Each round is stored as an ordinary results.json, so tools/metrics.py and the report writer work on it unchanged.
"""
from __future__ import annotations

import time
from pathlib import Path

from schemas import ExecutionPlan, ExecutionResults, ImageInfo
from tools.aggregate import aggregate
from tools.canny_common import load_gray
from tools.correctness import check_correctness
from tools.cpu_canny import run_cpu_canny
from tools.gpu_canny import run_gpu_canny
from tools.gpu_check import check_gpu


def repeat_benchmark(image_path: str, rounds: int = 5, low: int = 100, high: int = 200, repetitions: int = 20,
                     warmup: int = 3, gap_seconds: float = 3.0, out_root: str = "outputs", tag: str | None = None,
                     cpu_fn=run_cpu_canny, gpu_fn=run_gpu_canny, gpu_check=check_gpu, sleep=time.sleep,
                     say=print) -> tuple[list[ExecutionResults], dict, Path]:
    gpu_info = gpu_check()
    if not gpu_info.get("ok"):
        raise RuntimeError("GPU not usable: " + "; ".join(gpu_info.get("problems", [])))
    h, w = load_gray(image_path).shape
    info = ImageInfo(width=w, height=h, channels=3, megapixels=round(w * h / 1e6, 4))
    tag = tag or time.strftime("%Y%m%d_%H%M%S")
    folder = Path(out_root) / f"repeat_{tag}"
    runs: list[ExecutionResults] = []
    for k in range(1, rounds + 1):
        run_id = f"repeat_{tag}_r{k}"
        out = folder / f"round_{k}"
        sleep(gap_seconds)
        cpu, c_edges, c_mag = cpu_fn(image_path, low, high, repetitions, warmup, str(out / "cpu_edges.png"))
        sleep(gap_seconds)
        gpu, g_edges, g_mag = gpu_fn(image_path, low, high, repetitions, warmup, str(out / "gpu_edges.png"))
        if cpu.error or gpu.error:
            raise RuntimeError(cpu.error or gpu.error)
        res = ExecutionResults(
            plan=ExecutionPlan(task="canny_edge_detection", image_path=image_path, run_id=run_id, low_threshold=low,
                               high_threshold=high, repetitions=repetitions, warmup_runs=warmup),
            image_info=info, gpu_name=gpu_info.get("gpu_name"), cpu=cpu, gpu=gpu,
            correctness=check_correctness(c_edges, g_edges, c_mag, g_mag),
            agent2_notes="Benchmark-only repeat (no language model); same CPU/GPU tools as Agent 2.")
        (out).mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(res.model_dump_json(indent=2), encoding="utf-8")
        runs.append(res)
        say(f"round {k}/{rounds}: CPU {cpu.total_time_ms:.2f} ms (std {cpu.total_time_std_ms:.2f}) | "
            f"GPU {gpu.total_time_ms:.2f} ms (std {gpu.total_time_std_ms:.2f}) | speedup {cpu.total_time_ms / gpu.total_time_ms:.2f}x")
    return runs, aggregate(runs), folder
