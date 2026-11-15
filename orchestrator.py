"""Joins the agents into one pipeline.

    request + image  ->  Agent 1 (Task Planner)  ->  ExecutionPlan
                     ->  Agent 2 (Execution)     ->  ExecutionResults  ->  results.json
                     ->  Agent 3 (Analysis)      ->  AnalysisReport -> report.md + report.html

This file contains no agent logic. It only passes the structured object from one agent to the next
and saves it. Each agent still decides its own steps with its own tools.

Public interface: run_pipeline(user_request, image_path, llm, output_root="outputs", agentic=True)
                  -> (ExecutionResults, path_to_results_json)
"""
from __future__ import annotations

import json
from pathlib import Path

from agents.analyst import analyze
from agents.executor import run_executor
from agents.planner import run_planner
from agents.planner_agentic import run_planner_agentic
from schemas import AnalysisReport, ExecutionResults
from tools.report_writer import open_in_browser, write_report
from tools.run_trace import get_trace, merge_saved, start_stage


def run_pipeline(user_request: str, image_path: str, llm, output_root: str = "outputs",
                 agentic: bool = True) -> tuple[ExecutionResults, str]:
    """Agent 1 -> Agent 2. Raises PlannerError / UnsupportedTaskError / ExecutorError, which the caller reports."""
    start_stage(llm, "agent1")
    plan = (run_planner_agentic if agentic else run_planner)(user_request, image_path, llm)
    start_stage(llm, "agent2")
    results = run_executor(plan, llm, output_root=output_root)
    path = Path(output_root) / results.plan.run_id / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results.model_dump_json(indent=2), encoding="utf-8")
    return results, str(path)


def run_analysis(results: ExecutionResults, llm, output_root: str = "outputs",
                 open_report: bool = False) -> tuple[AnalysisReport, dict]:
    """Agent 3 -> report files. Runs AFTER the timed benchmarks, so its LLM use cannot disturb any timing.
    Returns (AnalysisReport with report_path set, {"report_md":..., "report_html":..., "folder":...})."""
    start_stage(llm, "agent3")
    report, source = analyze(results, llm)
    trace = get_trace(llm)
    run_dir = Path(output_root) / (results.plan.run_id or "run")
    if trace is not None:
        trace = merge_saved(trace, run_dir / "trace.json")
        trace["analysis"] = {"concepts": report.coa_concepts_used, "source": source}
    files = write_report(results, report.explanation_markdown, output_root, trace=trace)
    if trace is not None:
        (Path(files["folder"]) / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    report = report.model_copy(update={"report_path": files["report_html"]})
    (Path(files["folder"]) / "analysis.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if open_report:
        open_in_browser(files["report_html"])
    return report, files


def format_summary(res: ExecutionResults) -> str:
    """A short, human-readable text summary of the measured results."""
    p = res.plan
    lines = [
        f"Run {p.run_id} | image {res.image_info.width}x{res.image_info.height} ({res.image_info.megapixels} MP) "
        f"| GPU: {res.gpu_name}",
        f"Plan: thresholds {p.low_threshold}/{p.high_threshold} | {p.repetitions} runs after {p.warmup_runs} "
        f"warm-up | cpu={p.run_cpu} gpu={p.run_gpu}",
    ]
    for r in (res.cpu, res.gpu):
        if r is None:
            continue
        if r.error:
            lines.append(f"  {r.device.upper()}: ERROR {r.error}")
            continue
        extra = (f" | transfer {r.transfer_time_ms} ms, kernels {r.kernel_time_ms} ms"
                 if r.transfer_time_ms is not None else "")
        lines.append(f"  {r.device.upper()}: total {r.total_time_ms} ms (std {r.total_time_std_ms}){extra}")
    c = res.correctness
    lines.append(f"  Correctness: performed={c.performed} passed={c.passed} agreement={c.pixel_agreement_pct}%")
    return "\n".join(lines)
