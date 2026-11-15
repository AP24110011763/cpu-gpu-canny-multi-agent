r"""Build the report page from a saved results.json (written by main.py or run_executor.py).

    python scripts\make_report.py --results outputs\run_20261106_132934\results.json
Until Agent 3's language model exists (next stage), the explanation section is a clearly marked placeholder.
The numbers, tables, charts and images are real and come from the results file.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import ExecutionResults  # noqa: E402
from tools.report_writer import open_in_browser, write_report  # noqa: E402

PLACEHOLDER = ("(Placeholder. Agent 3's language model will write the explanation here in the next stage. "
               "Everything above this section is computed from the measured results.)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Make report.md and report.html from a results.json")
    ap.add_argument("--results", required=True)
    ap.add_argument("--no-open", action="store_true", help="do not open the page in the browser")
    a = ap.parse_args()

    path = Path(a.results)
    if not path.exists():
        print(f"results file not found: {path}")
        return 1
    results = ExecutionResults.model_validate_json(path.read_text(encoding="utf-8"))
    out = write_report(results, PLACEHOLDER, out_root=str(Path(__file__).resolve().parent.parent / "outputs"))
    print(f"Report written:\n  {out['report_html']}\n  {out['report_md']}")
    if not a.no_open:
        open_in_browser(out["report_html"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
