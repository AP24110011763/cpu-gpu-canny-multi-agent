"""Try Agent 1 on its own with the real LLM.

    python scripts/run_planner.py --image path\\to\\photo.jpg --request "Apply edge detection and compare CPU and GPU"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.planner import PlannerError, run_planner  # noqa: E402
from agents.planner_agentic import run_planner_agentic  # noqa: E402
from llm.client import OllamaClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Agent 1 (Task Planner) alone")
    ap.add_argument("--image", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--agentic", action="store_true", help="let the agent choose its own steps (tool loop)")
    args = ap.parse_args()

    print("Planning on CPU, this can take 1-3 minutes..." + (" [agentic mode]" if args.agentic else ""))
    try:
        planner = run_planner_agentic if args.agentic else run_planner
        plan = planner(args.request, args.image, OllamaClient())
    except PlannerError as exc:
        print(f"Planner could not make a plan: {exc}")
        return 1
    print(plan.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
