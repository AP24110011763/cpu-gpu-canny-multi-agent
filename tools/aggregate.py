"""Combine several full pipeline runs into mean +/- std across runs (real runs only).

Each run already averages many repetitions. Repeating the WHOLE run (new process, new warm-up) exposes
run-to-run variation from clocks, temperature and background load, which a single run cannot show.
All per-run metrics come from tools/metrics.py. Nothing is estimated.
"""
from __future__ import annotations

import statistics
from pathlib import Path

from schemas import ExecutionResults
from tools.metrics import compute_metrics

KEYS = ("cpu_total_ms", "gpu_total_ms", "speedup_total", "speedup_kernel_only", "transfer_overhead_pct",
        "throughput_cpu_mpix_s", "throughput_gpu_mpix_s")
STABLE_CV_PCT = 10.0     # run-to-run spread above this share of the mean = not stable enough to quote one number


def load_results(paths: list[str]) -> list[ExecutionResults]:
    return [ExecutionResults.model_validate_json(Path(p).read_text(encoding="utf-8")) for p in paths]


def latest_result_paths(output_root: str, n: int) -> list[str]:
    runs = sorted(p for p in Path(output_root).glob("run_*/results.json"))
    return [str(p) for p in runs[-n:]]


def aggregate(runs: list[ExecutionResults]) -> dict:
    if not runs:
        raise ValueError("no runs to aggregate")
    metrics = [compute_metrics(r) for r in runs]
    warnings: list[str] = []
    if any(r.plan.run_id.startswith("mock") for r in runs):
        warnings.append("Includes MOCK data. Do not quote these as results.")
    if len(runs) < 3:
        warnings.append(f"Only {len(runs)} run(s); use at least 3 so the spread means something.")
    if {m["scenario"] for m in metrics} != {"both"}:
        warnings.append("Not every run has both a CPU and a GPU result; only complete runs are averaged.")
    if len({(r.plan.image_path, r.plan.low_threshold, r.plan.high_threshold, r.plan.repetitions, r.plan.warmup_runs)
            for r in runs}) > 1:
        warnings.append("The runs used different images or settings, so averaging them is not like-for-like.")
    if any(m["winner"] == "undetermined" for m in metrics):
        warnings.append("At least one run has no verified winner (for example failed correctness).")
    good = [m for m in metrics if m["scenario"] == "both"]
    stats: dict[str, dict] = {}
    for k in KEYS:
        vals = [m[k] for m in good if m.get(k) is not None]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        stats[k] = {"mean": round(mean, 3), "median": round(statistics.median(vals), 3), "std": round(std, 3), "min": round(min(vals), 3),
                    "max": round(max(vals), 3), "n": len(vals), "cv_pct": round(100 * std / mean, 1) if mean else None}
    sp = stats.get("speedup_total")
    if sp and sp["cv_pct"] is not None and sp["cv_pct"] > STABLE_CV_PCT:
        warnings.append(f"Speedup varies {sp['cv_pct']}% between runs (above {STABLE_CV_PCT:.0f}%). "
                        "Check power mode, charger and background apps, then repeat.")
    return {"runs": [r.plan.run_id for r in runs], "n_runs": len(runs), "stats": stats, "warnings": warnings,
            "per_run_speedup": [m["speedup_total"] for m in metrics]}


def format_aggregate(agg: dict, note: str = "") -> str:
    lines = [f"{agg['n_runs']} runs: {', '.join(agg['runs'])}", f"Conditions: {note or 'not recorded'}", "",
             f"{'metric':<26}{'mean':>10}{'median':>10}{'std':>9}{'min':>10}{'max':>10}{'spread':>8}"]
    for k, s in agg["stats"].items():
        lines.append(f"{k:<26}{s['mean']:>10.3f}{s['median']:>10.3f}{s['std']:>9.3f}{s['min']:>10.3f}{s['max']:>10.3f}{s['cv_pct']:>7.1f}%")
    lines.append(f"\nSpeedup per run: {agg['per_run_speedup']}")
    lines += [f"WARNING: {w}" for w in agg["warnings"]]
    return "\n".join(lines)
