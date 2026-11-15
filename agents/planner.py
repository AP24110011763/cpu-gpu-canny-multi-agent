"""Agent 1 - Task Planner.

Input : user's natural-language request + image path
Output: a validated ExecutionPlan (schemas.py) for Agent 2

How it works (three LLM calls, each with one clear job):
  1. REASON + USE A TOOL : the LLM is told to call inspect_image and then summarise
                           what it learned about the image and the request.
  2. REASON -> STRUCTURE : the LLM turns the request + image facts into a JSON
                           "PlanDraft". Ollama constrains the output to the schema.
  3. CODE VERIFIES       : plain Python checks the draft against the request,
                           clamps values to safe ranges, and builds the ExecutionPlan.

The LLM proposes, the code verifies. A 3B model can misread a number or return an
out-of-range value, and that must never reach Agent 2.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from llm.agent_loop import run_tool_loop
from schemas import ExecutionPlan
from tools.image_inspect import inspect_image


class PlannerError(Exception):
    """The request cannot be turned into a valid plan."""


class UnsupportedTaskError(PlannerError):
    """The user asked for something other than Canny edge detection."""


class PlanDraft(BaseModel):
    """What the LLM is asked to produce. Internal to Agent 1 (not part of the frozen contract)."""

    task: Literal["canny_edge_detection", "unsupported"] = "canny_edge_detection"
    low_threshold: int = 100
    high_threshold: int = 200
    run_cpu: bool = True
    run_gpu: bool = True
    repetitions: int = 10
    warmup_runs: int = 3
    reasoning: str = ""


# ----------------------------------------------------------------------------- prompts

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_image",
            "description": "Read the image file and return its width, height, megapixels, "
            "size class and a suggested number of repetitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path of the image file"}
                },
                "required": ["image_path"],
            },
        },
    }
]

SYSTEM_PROMPT_INSPECT = (
    "You are the Task Planner agent of a CPU-vs-GPU image-processing benchmark. "
    "The only supported task is Canny edge detection. "
    "Step 1: call the inspect_image tool with the image path you are given. "
    "Step 2: after you get its result, reply in 2 or 3 short sentences: the image size, "
    "and anything specific the user asked for (number of runs, thresholds, CPU only, GPU only)."
)

SYSTEM_PROMPT_PLAN = """You create execution plans for a CPU-vs-GPU benchmark of Canny edge detection.
Return ONLY a JSON object with these fields:
  task: "canny_edge_detection", or "unsupported" if the user clearly wants a different
        algorithm (blur, sharpen, face detection, ...). Requests about comparing CPU and
        GPU on an image, or about edges, are canny_edge_detection.
  low_threshold, high_threshold: integers 0-255, low < high. Default 100 and 200.
  run_cpu, run_gpu: booleans. Both true by default. Only set one false if the user
        explicitly asks for CPU only or GPU only.
  repetitions: number of timed runs. Use the user's number if they gave one,
        otherwise use suggested_repetitions from the image facts.
  warmup_runs: default 3.
  reasoning: one short sentence explaining your choices.

Example 1
Request: "Apply edge detection to this image and compare CPU and GPU performance."
Image facts: megapixels 0.3, suggested_repetitions 20
JSON: {"task":"canny_edge_detection","low_threshold":100,"high_threshold":200,"run_cpu":true,"run_gpu":true,"repetitions":20,"warmup_runs":3,"reasoning":"Default thresholds, small image so 20 runs."}

Example 2
Request: "Run canny on the GPU only, 30 times, thresholds 50 and 150"
Image facts: megapixels 2.1, suggested_repetitions 5
JSON: {"task":"canny_edge_detection","low_threshold":50,"high_threshold":150,"run_cpu":false,"run_gpu":true,"repetitions":30,"warmup_runs":3,"reasoning":"User asked for GPU only, 30 runs and thresholds 50/150."}
"""

# ----------------------------------------------------------------------------- helpers

_REPS_RE = re.compile(r"(\d{1,4})\s*(?:times|runs|repetitions|iterations|repeats)\b", re.I)
_THRESH_PAIR_RE = re.compile(
    r"thresholds?\D{0,20}?(\d{1,3})\s*(?:and|to|,|&|-|/)\s*(\d{1,3})", re.I
)
_LOW_RE = re.compile(r"\blow(?:er)?\s+threshold\D{0,10}?(\d{1,3})", re.I)
_HIGH_RE = re.compile(r"\bhigh(?:er)?\s+threshold\D{0,10}?(\d{1,3})", re.I)
_CPU_ONLY_RE = re.compile(r"\bcpu[\s-]*only\b|\bonly\s+(?:on\s+)?(?:the\s+)?cpu\b", re.I)
_GPU_ONLY_RE = re.compile(r"\bgpu[\s-]*only\b|\bonly\s+(?:on\s+)?(?:the\s+)?gpu\b", re.I)
# words that make a threshold change plausible, so the LLM's choice is allowed to stand
_SENSITIVITY_RE = re.compile(
    r"threshold|sensitiv|strong edge|weak edge|fine detail|detail|noise|more edges|fewer edges|less edges", re.I
)
# words that make skipping a device plausible ("just the GPU", "skip CPU", "without the GPU", ...)
_DEVICE_WORDS_RE = re.compile(r"\bonly\b|\bjust\b|\bskip\b|\bwithout\b|\bexclude\b|\bno\s+(?:cpu|gpu)\b|\bdon'?t\b", re.I)
DEFAULT_LOW, DEFAULT_HIGH = 100, 200


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in LLM reply")
    return text[start : end + 1]


def _ask_for_draft(llm, user_request: str, info: dict, summary: str):
    """LLM call #2. Retries once if the JSON is invalid. Returns (draft or None, note)."""
    facts = {k: info[k] for k in ("width", "height", "megapixels", "size_class", "suggested_repetitions")}
    user_msg = (
        f'Request: "{user_request}"\n'
        f"Image facts: {facts}\n"
        f"Your earlier summary: {summary or '(none)'}\n"
        "JSON:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PLAN},
        {"role": "user", "content": user_msg},
    ]
    schema = PlanDraft.model_json_schema()
    last_error = ""
    for _ in range(2):
        reply = llm.chat(messages, json_schema=schema)
        text = reply.get("content", "") or ""
        try:
            return PlanDraft.model_validate_json(_extract_json(text)), ""
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            last_error = str(exc).splitlines()[0][:120]
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That was not valid JSON for the schema ({last_error}). "
                                            "Return only the JSON object."},
            ]
    return None, f"LLM output was invalid twice ({last_error}); safe defaults were used."


def _verify_against_request(
    draft: PlanDraft, request: str, corrections: list[str], suggested_reps: int | None = None
) -> None:
    """Code-side check.

    Rule 1: explicit things the user wrote beat the LLM's reading of them.
    Rule 2: if the user said NOTHING about a setting, the LLM may not invent a value for it.
            (A 3B model sometimes copies numbers from the prompt's examples.)
    """
    m = _REPS_RE.search(request)
    if m and int(m.group(1)) != draft.repetitions:
        corrections.append(f"repetitions set to {m.group(1)} from the request text (LLM said {draft.repetitions})")
        draft.repetitions = int(m.group(1))
    elif not m and suggested_reps is not None and draft.repetitions != suggested_reps:
        corrections.append(
            f"request gave no run count; repetitions {draft.repetitions} reset to the image-based {suggested_reps}"
        )
        draft.repetitions = suggested_reps

    pair = _THRESH_PAIR_RE.search(request)
    low_m, high_m = _LOW_RE.search(request), _HIGH_RE.search(request)
    wanted_low = wanted_high = None
    if pair:
        wanted_low, wanted_high = int(pair.group(1)), int(pair.group(2))
    else:
        wanted_low = int(low_m.group(1)) if low_m else None
        wanted_high = int(high_m.group(1)) if high_m else None
    if wanted_low is not None and wanted_low != draft.low_threshold:
        corrections.append(f"low_threshold set to {wanted_low} from the request text")
        draft.low_threshold = wanted_low
    if wanted_high is not None and wanted_high != draft.high_threshold:
        corrections.append(f"high_threshold set to {wanted_high} from the request text")
        draft.high_threshold = wanted_high
    if wanted_low is None and wanted_high is None and not _SENSITIVITY_RE.search(request):
        if (draft.low_threshold, draft.high_threshold) != (DEFAULT_LOW, DEFAULT_HIGH):
            corrections.append(
                f"request did not mention thresholds; {draft.low_threshold}/{draft.high_threshold} "
                f"reset to defaults {DEFAULT_LOW}/{DEFAULT_HIGH}"
            )
            draft.low_threshold, draft.high_threshold = DEFAULT_LOW, DEFAULT_HIGH

    if not _CPU_ONLY_RE.search(request) and not _GPU_ONLY_RE.search(request) and not _DEVICE_WORDS_RE.search(request):
        if not (draft.run_cpu and draft.run_gpu):
            corrections.append("request did not ask to skip a device; run_cpu and run_gpu both enabled")
            draft.run_cpu = draft.run_gpu = True

    if _CPU_ONLY_RE.search(request) and (draft.run_gpu or not draft.run_cpu):
        corrections.append("request says CPU only; run_cpu=true, run_gpu=false")
        draft.run_cpu, draft.run_gpu = True, False
    elif _GPU_ONLY_RE.search(request) and (draft.run_cpu or not draft.run_gpu):
        corrections.append("request says GPU only; run_cpu=false, run_gpu=true")
        draft.run_cpu, draft.run_gpu = False, True


def _clamp(draft: PlanDraft, corrections: list[str]) -> None:
    """Make every value safe for Agent 2, whatever the LLM or the user produced."""

    def clamp(name: str, value: int, lo: int, hi: int) -> int:
        new = max(lo, min(hi, value))
        if new != value:
            corrections.append(f"{name} {value} out of range, clamped to {new}")
        return new

    draft.low_threshold = clamp("low_threshold", draft.low_threshold, 0, 255)
    draft.high_threshold = clamp("high_threshold", draft.high_threshold, 0, 255)
    if draft.low_threshold > draft.high_threshold:
        corrections.append("low_threshold was above high_threshold; swapped")
        draft.low_threshold, draft.high_threshold = draft.high_threshold, draft.low_threshold
    if draft.low_threshold == draft.high_threshold:
        corrections.append("thresholds were equal; widened the gap")
        if draft.high_threshold < 255:
            draft.high_threshold = min(255, draft.low_threshold + 50)
        else:
            draft.low_threshold = 205
    draft.repetitions = clamp("repetitions", draft.repetitions, 1, 100)
    draft.warmup_runs = clamp("warmup_runs", draft.warmup_runs, 1, 10)  # 0 would time GPU start-up cost
    if not draft.run_cpu and not draft.run_gpu:
        corrections.append("both devices were disabled; enabled both")
        draft.run_cpu = draft.run_gpu = True


# ----------------------------------------------------------------------------- the agent

def run_planner(user_request: str, image_path: str, llm) -> ExecutionPlan:
    """Public interface (frozen): user text + image path -> ExecutionPlan."""
    if not user_request or not user_request.strip():
        raise PlannerError("The request is empty. Describe what to do with the image.")

    real_path = str(Path(image_path).expanduser().resolve())
    corrections: list[str] = []
    seen: dict = {}

    def _inspect(image_path: str = "", **_ignored) -> dict:
        # The user's path is authoritative; a small model may mangle the argument it passes.
        seen["info"] = inspect_image(real_path)
        return seen["info"]

    # --- LLM call(s) #1: reasoning + tool use -----------------------------------------
    loop = run_tool_loop(
        llm,
        SYSTEM_PROMPT_INSPECT,
        f'User request: "{user_request}"\nImage path: {real_path}',
        {"inspect_image": _inspect},
        TOOL_SCHEMAS,
        max_steps=4,
    )
    if "info" not in seen:
        corrections.append("LLM did not call inspect_image, so the planner ran it itself")
        _inspect()
    info = seen["info"]
    if not info.get("ok"):
        raise PlannerError(info.get("error", "image could not be inspected"))

    # --- LLM call #2: reasoning -> structured draft -----------------------------------
    draft, note = _ask_for_draft(llm, user_request, info, loop.final_text)
    if draft is None:
        corrections.append(note)
        draft = PlanDraft(repetitions=info["suggested_repetitions"])
    if draft.task == "unsupported":
        raise UnsupportedTaskError(
            "Only Canny edge detection is supported in this version. "
            f"Planner reasoning: {draft.reasoning or 'n/a'}"
        )

    # --- code: verify against the request, clamp, build the contract object -----------
    _verify_against_request(draft, user_request, corrections, info["suggested_repetitions"])
    _clamp(draft, corrections)

    notes = f"Planner: {draft.reasoning}".strip() if draft.reasoning else "Planner: (no reasoning given)"
    if corrections:
        notes += " | Code corrections: " + "; ".join(corrections)

    return ExecutionPlan(
        task="canny_edge_detection",  # fixed by code, never taken from LLM text
        image_path=real_path,
        low_threshold=draft.low_threshold,
        high_threshold=draft.high_threshold,
        run_cpu=draft.run_cpu,
        run_gpu=draft.run_gpu,
        repetitions=draft.repetitions,
        warmup_runs=draft.warmup_runs,
        run_id=None,  # assigned by the orchestrator
        notes=notes,
    )
