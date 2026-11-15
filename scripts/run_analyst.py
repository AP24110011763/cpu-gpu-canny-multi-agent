r"""Run Agent 3 (the real language model) on a saved results.json and open the report.

    python scripts\run_analyst.py --results outputs\run_20261106_133136\results.json
No benchmarks are run, so this is quick to try. Add --no-open to skip the browser.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.client import OllamaClient  # noqa: E402
from orchestrator import run_analysis  # noqa: E402
from schemas import ExecutionResults  # noqa: E402
from tools.run_trace import TracingLLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent 3 on a saved results.json")
    ap.add_argument("--results", required=True)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    path = Path(a.results)
    if not path.exists():
        print(f"results file not found: {path}")
        return 1
    results = ExecutionResults.model_validate_json(path.read_text(encoding="utf-8"))
    report, files = run_analysis(results, TracingLLM(OllamaClient()), str(Path(__file__).resolve().parent.parent / "outputs"),
                                 open_report=not a.no_open)
    print(f"Winner: {report.winner} | speedup {report.speedup_total} | concepts: {', '.join(report.coa_concepts_used) or 'none'}")
    print(report.explanation_markdown)
    print(f"\nReport: {files['report_html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
