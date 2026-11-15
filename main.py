r"""Run the whole pipeline from one command.

    python main.py --image ..\image1.png
    python main.py --image ..\image1.png --request "Compare CPU and GPU, use thresholds 50 and 150, 10 times"
    python main.py --image ..\image1.png --request "Run edge detection on the GPU only"
Agent 1 turns the request into a plan, Agent 2 runs it on the CPU and GPU, Agent 3 explains the result
and opens a report page in your browser. Both use the local LLM
on the CPU, so expect a few minutes. Results are saved to outputs\<run_id>\results.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.executor import ExecutorError  # noqa: E402
from agents.planner import PlannerError  # noqa: E402  (UnsupportedTaskError is a PlannerError)
from llm.client import OllamaClient  # noqa: E402
from tools.run_trace import TracingLLM  # noqa: E402
from orchestrator import format_summary, run_analysis, run_pipeline  # noqa: E402

DEFAULT_REQUEST = "Apply edge detection to this image and compare CPU and GPU performance"


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU vs GPU Canny analysis with agents")
    ap.add_argument("--image", required=True)
    ap.add_argument("--request", default=DEFAULT_REQUEST)
    ap.add_argument("--classic-planner", action="store_true",
                    help="use the fixed-steps planner instead of the autonomous one")
    ap.add_argument("--no-report", action="store_true", help="stop after Agent 2 (no Agent 3, no report)")
    ap.add_argument("--no-open", action="store_true", help="write the report but do not open the browser")
    a = ap.parse_args()

    out_root = Path(__file__).resolve().parent / "outputs"
    print(f'Request: "{a.request}"\nAgent 1 is planning, then Agent 2 will run it. This can take a few minutes...\n')
    try:
        llm = TracingLLM(OllamaClient())
        res, path = run_pipeline(a.request, a.image, llm, str(out_root), agentic=not a.classic_planner)
    except (PlannerError, ExecutorError) as exc:
        print(f"Stopped: {exc}")
        return 1

    print("Agent 1 notes:", res.plan.notes, "\n")
    print(format_summary(res))
    print("\nAgent 2 notes:", res.agent2_notes)
    print(f"\nSaved: {path}")
    if a.no_report:
        return 0

    print("\nAgent 3 is analysing the results and building the report...")
    report, files = run_analysis(res, llm, str(out_root), open_report=not a.no_open)
    print(f"Winner: {report.winner} | speedup {report.speedup_total} | concepts: {', '.join(report.coa_concepts_used) or 'none'}")
    print(f"Report: {files['report_html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
