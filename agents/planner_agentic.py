"""Agent 1 - Task Planner, AUTONOMOUS version.

Difference from agents/planner.py:
  planner.py          : the code fixes the order of steps (inspect -> draft -> verify).
  planner_agentic.py  : the LLM gets a GOAL and four tools and chooses its own steps:

      get_supported_tasks -> inspect_image -> validate_plan (fix errors, retry) -> submit_plan
                                                                       or  -> reject_request

The agent can check the plan, see errors, and correct itself before submitting. This is the
loop that makes it an agent: observe the tool result, decide the next action.

Safety is unchanged: whatever the agent submits is still verified by the same code checks
(explicit user numbers win, unmentioned settings return to defaults, values are clamped), and
if the agent never submits, the classic planner.py runs as a fallback.

Public interface (frozen): run_planner_agentic(user_request, image_path, llm) -> ExecutionPlan
"""
from __future__ import annotations

from pathlib import Path

from agents.planner import (
    DEFAULT_HIGH,
    DEFAULT_LOW,
    PlanDraft,
    PlannerError,
    UnsupportedTaskError,
    _clamp,
    _verify_against_request,
    run_planner,
)
from llm.agent_loop import run_tool_loop
from schemas import ExecutionPlan
from tools.image_inspect import inspect_image

SUPPORTED_TASKS = [
    {
        "task": "canny_edge_detection",
        "description": "Canny edge detection, timed on CPU and GPU.",
        "parameters": "low_threshold, high_threshold (0-255, low < high), run_cpu, run_gpu, "
        "repetitions (1-100), warmup_runs (1-10)",
    }
]


def get_supported_tasks() -> dict:
    return {"tasks": SUPPORTED_TASKS}


def validate_plan(
    low_threshold: int = DEFAULT_LOW,
    high_threshold: int = DEFAULT_HIGH,
    run_cpu: bool = True,
    run_gpu: bool = True,
    repetitions: int = 10,
    warmup_runs: int = 3,
    **_ignored,
) -> dict:
    """Checks a draft plan and tells the agent what is wrong. Never changes anything."""
    errors: list[str] = []
    try:
        low, high = int(low_threshold), int(high_threshold)
        reps, warm = int(repetitions), int(warmup_runs)
    except (TypeError, ValueError):
        return {"ok": False, "errors": ["thresholds, repetitions and warmup_runs must be integers"]}
    if not (0 <= low <= 255 and 0 <= high <= 255):
        errors.append("thresholds must be between 0 and 255")
    if low >= high:
        errors.append("low_threshold must be smaller than high_threshold")
    if not (1 <= reps <= 100):
        errors.append("repetitions must be between 1 and 100")
    if not (1 <= warm <= 10):
        errors.append("warmup_runs must be between 1 and 10")
    if not (bool(run_cpu) or bool(run_gpu)):
        errors.append("at least one of run_cpu and run_gpu must be true")
    return {"ok": not errors, "errors": errors}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_supported_tasks",
            "description": "List the tasks this system can run, with their parameters.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_image",
            "description": "Read the image: width, height, megapixels, size_class and suggested_repetitions.",
            "parameters": {
                "type": "object",
                "properties": {"image_path": {"type": "string"}},
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_plan",
            "description": "Check a draft plan. Returns ok and a list of errors. Call it before submit_plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "low_threshold": {"type": "integer"},
                    "high_threshold": {"type": "integer"},
                    "run_cpu": {"type": "boolean"},
                    "run_gpu": {"type": "boolean"},
                    "repetitions": {"type": "integer"},
                    "warmup_runs": {"type": "integer"},
                },
                "required": ["low_threshold", "high_threshold", "run_cpu", "run_gpu", "repetitions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "Submit the final plan. Use only after validate_plan returned ok.",
            "parameters": {
                "type": "object",
                "properties": {
                    "low_threshold": {"type": "integer"},
                    "high_threshold": {"type": "integer"},
                    "run_cpu": {"type": "boolean"},
                    "run_gpu": {"type": "boolean"},
                    "repetitions": {"type": "integer"},
                    "warmup_runs": {"type": "integer"},
                    "reasoning": {"type": "string", "description": "One sentence explaining the choices"},
                },
                "required": ["low_threshold", "high_threshold", "run_cpu", "run_gpu", "repetitions", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject_request",
            "description": "Use when the request asks for something this system cannot do.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are the Task Planner agent of a CPU-vs-GPU image-processing benchmark.
GOAL: turn the user's request into a valid execution plan, or reject it if it is not supported.
Work like this:
1. Call get_supported_tasks to see what the system can do.
2. Call inspect_image with the image path you are given.
3. Decide the plan. Defaults: thresholds 100 and 200, run_cpu true, run_gpu true, warmup_runs 3,
   repetitions = suggested_repetitions from the image. Change a default ONLY if the user asked.
4. Call validate_plan. If it returns errors, fix them and call validate_plan again.
5. Call submit_plan with the validated plan. If the task is not supported, call reject_request instead.
Never invent numbers the user did not ask for."""


NUDGE_PROMPT = """You are the Task Planner agent. You have already inspected the image. Do NOT explain anything.
Call validate_plan with the plan, then call submit_plan.
Defaults: thresholds 100 and 200, run_cpu true, run_gpu true, warmup_runs 3, repetitions = suggested_repetitions.
Change a default ONLY if the user asked. If the task is not supported, call reject_request instead."""

NUDGE_SCHEMAS = [s for s in TOOL_SCHEMAS if s["function"]["name"] in ("validate_plan", "submit_plan", "reject_request")]


def _clean_bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")


def run_planner_agentic(user_request: str, image_path: str, llm) -> ExecutionPlan:
    """Public interface: user text + image path -> ExecutionPlan, with the agent choosing its steps."""
    if not user_request or not user_request.strip():
        raise PlannerError("The request is empty. Describe what to do with the image.")

    real_path = str(Path(image_path).expanduser().resolve())
    seen: dict = {"validated": 0}
    corrections: list[str] = []

    def _inspect(image_path: str = "", **_ignored) -> dict:
        seen["info"] = inspect_image(real_path)  # the user's path is authoritative
        return seen["info"]

    def _validate(**kwargs) -> dict:
        seen["validated"] += 1
        return validate_plan(**kwargs)

    def _submit(**kwargs) -> dict:
        seen["submitted"] = kwargs
        return {"accepted": True}

    def _reject(reason: str = "", **_ignored) -> dict:
        seen["rejected"] = reason or "no reason given"
        return {"rejected": True}

    registry = {
        "get_supported_tasks": get_supported_tasks,
        "inspect_image": _inspect,
        "validate_plan": _validate,
        "submit_plan": _submit,
        "reject_request": _reject,
    }
    terminal = {"submit_plan", "reject_request"}
    loop = run_tool_loop(
        llm,
        SYSTEM_PROMPT,
        f'User request: "{user_request}"\nImage path: {real_path}',
        registry,
        TOOL_SCHEMAS,
        max_steps=10,
        stop_after=terminal,
    )
    trace_tools = list(loop.tools_called)
    steps = loop.steps

    def _raise_if_rejected() -> None:
        if "rejected" in seen and "submitted" not in seen:
            raise UnsupportedTaskError(f"Only Canny edge detection is supported. Agent reasoning: {seen['rejected']}")

    _raise_if_rejected()

    info = seen.get("info") or inspect_image(real_path)
    if not info.get("ok"):
        raise PlannerError(info.get("error", "image could not be inspected"))

    # A small model sometimes STOPS TO TALK ("I will now create a plan") instead of calling the next
    # tool, and the loop reads plain text as "finished". Give it ONE focused nudge: the image facts,
    # only the three tools it still needs, and an instruction not to explain.
    nudged = False
    if "submitted" not in seen and "rejected" not in seen and not loop.hit_step_limit:
        nudged = True
        facts = {k: info[k] for k in ("width", "height", "megapixels", "size_class", "suggested_repetitions")}
        loop = run_tool_loop(
            llm,
            NUDGE_PROMPT,
            f'User request: "{user_request}"\nImage facts: {facts}\nNow call validate_plan, then submit_plan.',
            registry,
            NUDGE_SCHEMAS,
            max_steps=6,
            stop_after=terminal,
        )
        trace_tools += loop.tools_called
        steps += loop.steps
        _raise_if_rejected()

    if "submitted" not in seen:
        # Still no plan: fall back to the classic planner (fixed steps).
        why = f"tools called: {', '.join(trace_tools) or 'none'}; "
        why += "step limit reached" if loop.hit_step_limit else f"it replied in plain text: '{loop.final_text[:80]}'"
        if nudged:
            why += "; a nudge did not help either"
        plan = run_planner(user_request, image_path, llm)
        plan.notes = f"Agent did not submit a plan ({why}), so the classic planner ran. " + plan.notes
        return plan

    raw = seen["submitted"]
    try:
        draft = PlanDraft(
            low_threshold=int(raw.get("low_threshold", DEFAULT_LOW)),
            high_threshold=int(raw.get("high_threshold", DEFAULT_HIGH)),
            run_cpu=_clean_bool(raw.get("run_cpu", True)),
            run_gpu=_clean_bool(raw.get("run_gpu", True)),
            repetitions=int(raw.get("repetitions", info["suggested_repetitions"])),
            warmup_runs=int(raw.get("warmup_runs", 3)),
            reasoning=str(raw.get("reasoning", "")),
        )
    except (TypeError, ValueError):
        plan = run_planner(user_request, image_path, llm)
        plan.notes = "Agent submitted unreadable values, so the classic planner ran. " + plan.notes
        return plan

    # same code-side safety net as the classic planner
    _verify_against_request(draft, user_request, corrections, info["suggested_repetitions"])
    _clamp(draft, corrections)

    notes = f"Agentic planner ({steps} steps; tools: {', '.join(trace_tools)}"
    notes += "; nudged once after it stopped to talk" if nudged else ""
    notes += f"): {draft.reasoning or 'no reasoning given'}"
    if seen["validated"] == 0:
        corrections.append("agent submitted without calling validate_plan")
    if corrections:
        notes += " | Code corrections: " + "; ".join(corrections)

    return ExecutionPlan(
        task="canny_edge_detection",
        image_path=real_path,
        low_threshold=draft.low_threshold,
        high_threshold=draft.high_threshold,
        run_cpu=draft.run_cpu,
        run_gpu=draft.run_gpu,
        repetitions=draft.repetitions,
        warmup_runs=draft.warmup_runs,
        run_id=None,
        notes=notes,
    )
