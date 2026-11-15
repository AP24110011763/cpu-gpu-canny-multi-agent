"""Agent 2 - Execution Agent.

GOAL given to the LLM: carry out an ExecutionPlan on the CPU and the GPU and say what happened.

What is reasoning, what is tool use, what is communication
  * Reasoning     : the LLM decides the ORDER and whether to skip something. For example, if
                    check_gpu says the GPU is unusable, it must not call run_gpu and should run
                    the CPU only.
  * Tool use      : it calls check_gpu, run_cpu, run_gpu, check_correctness and finish.
  * Communication : it receives an ExecutionPlan (from Agent 1) and returns an ExecutionResults
                    (to Agent 3). Both are Pydantic objects, never free text.

"LLM proposes, code verifies":
  * The tools take NO settings. Thresholds, repetitions and warm-up come from the plan inside the
    closures, so the LLM cannot change them and no measured number ever passes through its text.
  * The results are assembled by code from what the tools actually returned.
  * If the LLM skips a required step (small models sometimes do), a code-side safety net runs it
    and says so in agent2_notes.

Public interface (frozen): run_executor(plan, llm, output_root="outputs") -> ExecutionResults
"""
from __future__ import annotations

import time
from pathlib import Path

from llm.agent_loop import run_tool_loop
from schemas import CorrectnessCheck, DeviceResult, ExecutionPlan, ExecutionResults, ImageInfo
from tools.correctness import check_correctness
from tools.cpu_canny import run_cpu_canny
from tools.gpu_canny import run_gpu_canny
from tools.gpu_check import check_gpu
from tools.image_inspect import inspect_image


class ExecutorError(Exception):
    """The plan cannot be executed at all (for example, the image is unreadable)."""


SYSTEM_PROMPT = """You are the Execution agent of a CPU-vs-GPU image-processing benchmark.
GOAL: carry out the plan you are given, then report. The plan fixes every setting; you cannot change them.
Follow these steps:
1. If the plan runs the GPU, call check_gpu first.
2. If the plan runs the CPU, call run_cpu.
3. If the plan runs the GPU AND check_gpu said ok, call run_gpu. If the GPU is not ok, do NOT call run_gpu.
4. If both run_cpu and run_gpu succeeded, call check_correctness.
5. Call finish with ONE short sentence saying what ran and anything that went wrong.
Never invent numbers. Never call a tool twice."""

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "check_gpu",
        "description": "Check that the NVIDIA GPU, driver and CuPy work. Returns ok and the GPU name.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "run_cpu",
        "description": "Run the timed CPU Canny edge detection using the plan's settings.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "run_gpu",
        "description": "Run the timed GPU Canny edge detection using the plan's settings. Only after check_gpu was ok.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "check_correctness",
        "description": "Compare the CPU and GPU edge images. Only after both runs succeeded.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "End the job with one short sentence about what ran and what went wrong, if anything.",
        "parameters": {"type": "object",
                       "properties": {"summary": {"type": "string"}},
                       "required": ["summary"]}}},
]


def _brief(res: DeviceResult) -> dict:
    """A short summary for the LLM. It never needs the full set of numbers."""
    if res.error:
        return {"ok": False, "error": res.error}
    out = {"ok": True, "total_ms": res.total_time_ms}
    if res.transfer_time_ms is not None:
        out.update(transfer_ms=res.transfer_time_ms, kernel_ms=res.kernel_time_ms)
    return out


def run_executor(plan: ExecutionPlan, llm, output_root: str = "outputs",
                 settle_seconds: float = 3.0) -> ExecutionResults:
    """settle_seconds: quiet time before each benchmark. The LLM has just used every CPU core, so we
    unload it from RAM (if the client can) and wait, so its heat and memory use do not leak into the timings."""
    info = inspect_image(plan.image_path)
    if not info.get("ok"):
        raise ExecutorError(info.get("error", "image could not be read"))

    run_id = plan.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    plan = plan.model_copy(update={"run_id": run_id})
    out_dir = Path(output_root) / run_id

    st: dict = {"gpu_info": None, "cpu": None, "gpu": None, "corr": None, "note": ""}
    safety: list[str] = []

    def _settle() -> None:
        unload = getattr(llm, "unload", None)
        if callable(unload) and unload():
            st["unloaded"] = True
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        st["settled"] = st.get("settled", 0) + 1

    # ---- the tools. They use the plan's settings; they accept (and ignore) any arguments the LLM adds.
    def _check_gpu(**_ignored) -> dict:
        st["gpu_info"] = check_gpu()
        g = st["gpu_info"]
        return {"ok": g["ok"], "gpu_name": g.get("gpu_name"), "problems": g.get("problems", [])}

    def _run_cpu(**_ignored) -> dict:
        _settle()
        res, edges, mag = run_cpu_canny(plan.image_path, plan.low_threshold, plan.high_threshold,
                                        plan.repetitions, plan.warmup_runs, str(out_dir / "cpu_edges.png"))
        st["cpu"], st["cpu_edges"], st["cpu_mag"] = res, edges, mag
        return _brief(res)

    def _run_gpu(**_ignored) -> dict:
        if st["gpu_info"] is not None and not st["gpu_info"]["ok"]:
            return {"ok": False, "error": "GPU check failed, so the GPU run was refused"}
        _settle()
        res, edges, mag = run_gpu_canny(plan.image_path, plan.low_threshold, plan.high_threshold,
                                        plan.repetitions, plan.warmup_runs, str(out_dir / "gpu_edges.png"))
        st["gpu"], st["gpu_edges"], st["gpu_mag"] = res, edges, mag
        return _brief(res)

    def _check_correctness(**_ignored) -> dict:
        c, g = st["cpu"], st["gpu"]
        if c is None or g is None or c.error or g.error:
            return {"ok": False, "error": "both the CPU and the GPU run must have succeeded first"}
        st["corr"] = check_correctness(st["cpu_edges"], st["gpu_edges"], st["cpu_mag"], st["gpu_mag"])
        return {"ok": True, "passed": st["corr"].passed, "pixel_agreement_pct": st["corr"].pixel_agreement_pct}

    def _finish(summary: str = "", **_ignored) -> dict:
        st["note"] = str(summary)
        return {"done": True}

    registry = {"check_gpu": _check_gpu, "run_cpu": _run_cpu, "run_gpu": _run_gpu,
                "check_correctness": _check_correctness, "finish": _finish}
    todo = (f"Plan for image {plan.image_path}: run_cpu={plan.run_cpu}, run_gpu={plan.run_gpu}, "
            f"thresholds {plan.low_threshold}/{plan.high_threshold}, {plan.repetitions} repetitions, "
            f"{plan.warmup_runs} warm-up runs.")
    loop = run_tool_loop(llm, SYSTEM_PROMPT, todo, registry, TOOL_SCHEMAS, max_steps=10, stop_after={"finish"})

    # ---- code-side safety net: whatever the plan requires must have happened.
    if plan.run_gpu and st["gpu_info"] is None:
        _check_gpu()
        safety.append("agent did not call check_gpu, so code did")
    if plan.run_cpu and st["cpu"] is None:
        _run_cpu()
        safety.append("agent did not run the CPU, so code did")
    gpu_usable = bool(st["gpu_info"] and st["gpu_info"]["ok"])
    if plan.run_gpu and st["gpu"] is None and gpu_usable:
        _run_gpu()
        safety.append("agent did not run the GPU, so code did")
    if st["corr"] is None and st["cpu"] and st["gpu"] and not st["cpu"].error and not st["gpu"].error:
        _check_correctness()
        safety.append("agent did not check correctness, so code did")

    gpu_missing = plan.run_gpu and st["gpu"] is None
    if gpu_missing:
        problems = "; ".join((st["gpu_info"] or {}).get("problems", [])) or "GPU not usable"
        safety.append(f"GPU was planned but is not usable ({problems}); results are CPU-only")

    correctness = st["corr"] or CorrectnessCheck(
        performed=False,
        note="Not performed: needs a successful CPU run and a successful GPU run.")

    notes = f"Agent 2 ({loop.steps} steps; tools: {', '.join(loop.tools_called) or 'none'})"
    if st["note"]:
        notes += f": {st['note']}"
    if st.get("settled"):
        notes += (f" | Quiet start before each benchmark: {'LLM unloaded, ' if st.get('unloaded') else ''}"
                  f"waited {settle_seconds:g} s")
    if safety:
        notes += " | Code safety net: " + "; ".join(safety)

    return ExecutionResults(
        plan=plan,
        image_info=ImageInfo(width=info["width"], height=info["height"],
                             channels=info["channels"], megapixels=info["megapixels"]),
        gpu_name=(st["gpu_info"] or {}).get("gpu_name") if plan.run_gpu else None,
        cpu=st["cpu"] if plan.run_cpu else None,
        gpu=st["gpu"] if plan.run_gpu else None,
        correctness=correctness,
        agent2_notes=notes,
    )
