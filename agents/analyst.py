"""Agent 3 - Comparison & Analysis Agent.

GOAL given to the LLM: look at the measured results and explain WHY the CPU or GPU was faster, using
Computer Organization and Architecture (COA) concepts.

What is reasoning, what is tool use, what is communication
  * Reasoning     : the LLM decides which situation this is (GPU wins, CPU wins, tie, GPU missing, correctness
                    failed...), which COA concepts explain it, and how to word the explanation.
  * Tool use      : get_metrics, get_stage_breakdown, check_correctness_status (the calculator from
                    tools/metrics.py), lookup_coa_concept (a fixed, checked list of concept notes), and
                    submit_explanation.
  * Communication : it receives an ExecutionResults (from Agent 2) and returns an AnalysisReport.

"LLM proposes, code verifies":
  * Every NUMBER field of the AnalysisReport is filled by code from tools/metrics.py. The LLM only writes
    the explanation text.
  * Code then checks that every number in that text really exists in the measured data. A wrong number or a
    claim we never measured (cache miss rate, occupancy...) rejects the text once; the LLM gets one retry.
  * If the LLM fails twice (or never submits), a deterministic template explanation is used and the report
    says so. The pipeline never ends without an explanation.

Public interface: run_analyst(results, llm) -> AnalysisReport
                  analyze(results, llm) -> (AnalysisReport, notes)
"""
from __future__ import annotations

import re

from llm.agent_loop import run_tool_loop
from schemas import AnalysisReport, ExecutionResults
from tools.metrics import check_correctness_status, compute_metrics, get_stage_breakdown

# --------------------------------------------------------------------------------------------------------
# The fixed COA concept notes. They contain no measured numbers on purpose: numbers come only from tools.
# --------------------------------------------------------------------------------------------------------
CONCEPTS: dict[str, dict] = {
    "cpu_vs_gpu_architecture": {
        "note": "A CPU has a few powerful cores with deep caches, branch prediction and out-of-order execution, "
                "built to finish one thread's work with low latency. A GPU has many simple cores grouped into "
                "streaming multiprocessors that run threads in lockstep groups (SIMT). It spends its transistors "
                "on arithmetic units instead of control logic.",
        "evidence": "architecture fact, not measured here",
    },
    "sequential_vs_parallel": {
        "note": "Gaussian blur, Sobel gradients and non-maximum suppression compute every pixel independently from "
                "its neighbours, so thousands of GPU threads can each take one pixel (data parallelism). "
                "Hysteresis follows edge chains pixel to pixel, so it is inherently sequential and runs on the CPU "
                "in both paths.",
        "evidence": "architecture fact about the algorithm, not measured here",
    },
    "memory_hierarchy_and_pcie": {
        "note": "The image starts in host RAM. To use the GPU it must cross the PCIe bus into the GPU's own memory "
                "(VRAM), and the result must come back. That bus is far slower than on-card memory, and the copy "
                "time is paid on every run. A small image does little arithmetic per byte copied, so the transfer "
                "can cost more than the work saved. The metrics tool reports the measured transfer time.",
        "evidence": "architecture fact; the transfer time itself IS measured",
    },
    "throughput_vs_latency": {
        "note": "A CPU is designed for low latency on a single task; a GPU is designed for high throughput over "
                "many tasks at once. A GPU only shows its strength when there is enough parallel work to keep all "
                "its cores busy, so larger images favour it. The throughput numbers (megapixels per second) are "
                "computed from measured times.",
        "evidence": "architecture fact; throughput is computed from measured times",
    },
    "amdahl_law": {
        "note": "Amdahl's law: if a fraction of the work cannot be parallelised, the overall speedup is capped by "
                "total time divided by that serial time, however fast the parallel part becomes. Here the serial "
                "part is hysteresis. The metrics tool reports the serial fraction and the resulting limit, "
                "computed from the measured CPU stage times.",
        "evidence": "architecture law; the serial fraction and limit are computed from measurements",
    },
    "kernel_launch_overhead": {
        "note": "Each GPU step is launched as a separate kernel, which has a fixed start-up cost. When a stage "
                "does little arithmetic, or is built from many small kernels, that fixed cost can exceed the time "
                "the CPU needs for the whole stage, so a stage can be slower on the GPU. This is a hypothesis for "
                "such stages; launch cost was not measured separately here.",
        "evidence": "hypothesis, NOT measured here",
    },
}
CONCEPT_NAMES = list(CONCEPTS)
MIN_LOOKUPS = 2          # the agent must ground its "Why" section in at least this many checked concept notes

SYSTEM_PROMPT = """You are the Analysis agent of a CPU-vs-GPU benchmark for Canny edge detection.
GOAL: explain the measured results using Computer Organization and Architecture concepts.
Follow these steps:
1. Call get_metrics, get_stage_breakdown and check_correctness_status (no arguments).
2. REQUIRED: look at metrics.scenario and metrics.winner. Call lookup_coa_concept for 2 to 4 concepts that explain THIS situation.
   Concepts: cpu_vs_gpu_architecture, sequential_vs_parallel, memory_hierarchy_and_pcie,
   throughput_vs_latency, amdahl_law, kernel_launch_overhead.
3. Call submit_explanation ONCE with short markdown in exactly this layout:
## What was measured
- 3 to 5 bullets quoting numbers from the tools
## Why (architecture explanation, not measured here)
- 2 to 4 bullets, each naming a COA concept
## Caveats
- 1 to 3 bullets (warnings from the tools, noise, GPU missing, failed correctness, what was not measured)
RULES:
- Use ONLY numbers that appear in the tool results. Copy them exactly. Never calculate new numbers.
- If metrics.winner is "undetermined", do not name a winner. Say why from metrics.warnings.
- If the CPU is faster or it is a tie, say so plainly and explain why that is plausible.
- Hysteresis is serial and runs on the CPU in BOTH paths. Never say the GPU speeds it up.
- Do not mention cache miss rates, cache hit rates, occupancy or warp divergence: none of these were measured.
- Say clearly which statements are measured and which are architecture explanations.
Example of the style, with placeholders instead of numbers:
## What was measured
- GPU total <gpu_total_ms> ms versus CPU total <cpu_total_ms> ms, a speedup of <speedup_total>x.
- PCIe transfer was <transfer_overhead_pct>% of the GPU time.
## Why (architecture explanation, not measured here)
- Memory hierarchy: the image crosses PCIe into VRAM, a cost the CPU does not pay.
## Caveats
- <copy any metrics.warnings here>"""

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_metrics",
        "description": "Calculated metrics: scenario, winner, speedups, throughput, transfer share, Amdahl limit, warnings.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_stage_breakdown",
        "description": "Per-stage CPU and GPU times, speedups and shares of total time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "check_correctness_status",
        "description": "Whether the CPU and GPU outputs agreed, as a plain verdict.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "lookup_coa_concept",
        "description": "Get a checked note about one COA concept.",
        "parameters": {"type": "object",
                       "properties": {"concept": {"type": "string", "enum": CONCEPT_NAMES}},
                       "required": ["concept"]}}},
    {"type": "function", "function": {
        "name": "submit_explanation",
        "description": "Submit the final markdown explanation. Call once, last.",
        "parameters": {"type": "object",
                       "properties": {"explanation_markdown": {"type": "string"}},
                       "required": ["explanation_markdown"]}}},
]

# --------------------------------------------------------------------------------------------------------
# Code-side verification of the LLM's text
# --------------------------------------------------------------------------------------------------------
_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)(\s?(?:x|×|%))?")
_UNMEASURED_RE = re.compile(
    r"cache[- ]?(?:miss|hit)|hit[- ]?rate|miss[- ]?rate|occupancy|warp[- ]?divergence|bank[- ]?conflict|register[- ]?spill",
    re.I)
_WINNER_CLAIM_RE = re.compile(r"\b(cpu|gpu)\b[^.\n]{0,25}\b(?:is|was|were|are)\b[^.\n]{0,12}\b(?:faster|the winner|quicker)", re.I)


_HYST_GOOD_RE = re.compile(r"serial|sequential|cpu|limit|cap\b|caps\b|bottleneck|amdahl|same|unchanged", re.I)
_HYST_BENEFIT_RE = re.compile(r"benefit|accelerat|parallel|speed(?:s|ed)? up|speedup|faster|contribut", re.I)


def _hysteresis_claims(text: str) -> bool:
    """True if a sentence suggests hysteresis is sped up by the GPU. It is serial and runs on the CPU in both paths."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if "hysteresis" in sentence.lower() and _HYST_BENEFIT_RE.search(sentence) and not _HYST_GOOD_RE.search(sentence):
            return True
    return False


def _flatten_numbers(obj, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, str):
        out.extend(float(x.replace(",", "")) for x in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", obj))


def allowed_numbers(results: ExecutionResults, metrics: dict, breakdown: dict, verdict: dict) -> list[float]:
    nums: list[float] = []
    for part in (results.model_dump(), metrics, breakdown, verdict):
        _flatten_numbers(part, nums)
    return nums


def find_problems(text: str, allowed: list[float], winner: str) -> list[str]:
    """Return a list of reasons the explanation must be rejected (empty = accepted)."""
    problems: list[str] = []
    if not text or len(text.strip()) < 40:
        return ["the explanation is empty or far too short"]
    bad: list[str] = []
    for m in _NUM_RE.finditer(text):
        token, suffix = m.group(1), m.group(2)
        value = float(token.replace(",", ""))
        decimals = len(token.split(".")[1]) if "." in token else 0
        if decimals == 0 and not suffix and value <= 5:        # counting words such as "3 stages"
            continue
        tol = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(a - value) <= tol for a in allowed):
            bad.append(token + (suffix or "").strip())
    if bad:
        problems.append("these numbers are not in the measured data: " + ", ".join(sorted(set(bad))[:8]))
    u = _UNMEASURED_RE.search(text)
    if u:
        problems.append(f"it mentions '{u.group(0)}', which was never measured")
    if _hysteresis_claims(text):
        problems.append("it says hysteresis benefits from the GPU, but hysteresis is serial and runs on the CPU in both paths")
    if winner in ("undetermined", "tie") and _WINNER_CLAIM_RE.search(text):
        problems.append(f"it names a faster device, but the verified result is '{winner}'")
    return problems


# --------------------------------------------------------------------------------------------------------
# Deterministic fallback explanation (also the reference for what a good answer looks like)
# --------------------------------------------------------------------------------------------------------
def template_explanation(m: dict, b: dict, verdict: dict) -> tuple[str, list[str]]:
    used: list[str] = []
    measured: list[str] = []
    why: list[str] = []
    caveats: list[str] = [w for w in m.get("warnings", [])]
    sc, winner = m["scenario"], m["winner"]

    def use(name: str, line: str) -> None:
        used.append(name)
        why.append(f"- **{name.replace('_', ' ').title()}**: {line}")

    if sc == "no_data":
        return ("## What was measured\n- Neither the CPU nor the GPU produced a usable result, so there is nothing to explain.\n"
                "## Caveats\n" + "\n".join(f"- {w}" for w in caveats)), used

    if m.get("cpu_total_ms") is not None:
        measured.append(f"CPU total {m['cpu_total_ms']} ms ({m['throughput_cpu_mpix_s']} MP/s).")
    if m.get("gpu_total_ms") is not None:
        measured.append(f"GPU total {m['gpu_total_ms']} ms ({m['throughput_gpu_mpix_s']} MP/s).")
    if sc == "both":
        measured.append(f"Total speedup {m['speedup_total']}x" +
                        (f"; kernels only {m['speedup_kernel_only']}x." if m.get("speedup_kernel_only") else "."))
        if m.get("transfer_overhead_pct") is not None:
            measured.append(f"PCIe transfer was {m['transfer_overhead_pct']}% of the GPU time.")
        if m.get("gpu_hysteresis_share_pct") is not None:
            measured.append(f"Hysteresis (serial, on the CPU) was {m['gpu_hysteresis_share_pct']}% of the GPU path.")
        measured.append(f"Result: {'no winner declared' if winner == 'undetermined' else winner.upper() + ' faster' if winner in ('cpu', 'gpu') else 'tie'}.")
    measured.append(f"Correctness: {verdict['verdict']}")

    # --- architecture explanations, chosen by situation
    if sc == "both" and winner in ("gpu", "cpu", "tie"):
        use("sequential_vs_parallel", "blur, Sobel and non-maximum suppression handle each pixel independently, so the GPU can "
            "give each pixel its own thread. Hysteresis is sequential and stays on the CPU.")
        if m.get("amdahl_max_speedup"):
            use("amdahl_law", f"the serial hysteresis step caps the overall speedup at about {m['amdahl_max_speedup']}x "
                "however fast the parallel stages become (limit computed from the measured CPU times).")
        use("memory_hierarchy_and_pcie", "the GPU path must copy the image over PCIe into VRAM and copy the result back; the CPU "
            "works on data already in RAM. The measured transfer share above is that cost.")
        if winner == "cpu" or (m.get("speedup_total") and m["speedup_total"] < 1.5):
            use("throughput_vs_latency", "a GPU wins on throughput only when there is enough parallel work to keep its cores busy; "
                "with little work per byte copied, its fixed costs dominate.")
        slow = [s for s, v in m.get("per_stage_speedup", {}).items() if v < 1 and s != "hysteresis"]
        if slow:
            use("kernel_launch_overhead", f"{', '.join(slow)} ran slower on the GPU. A plausible reason is the fixed launch "
                "cost of each kernel. This is a hypothesis; launch cost was not measured separately.")
    elif sc == "both":                                          # correctness failed
        use("cpu_vs_gpu_architecture", "the two devices should compute the same edges; different floating-point rounding can "
            "shift thin edges slightly, but a large disagreement means the speed comparison is not trustworthy.")
    else:
        use("cpu_vs_gpu_architecture", "no CPU-vs-GPU comparison was possible in this run, so only the architecture background applies.")
        if m.get("amdahl_max_speedup"):
            use("amdahl_law", f"from the CPU stage times alone, the serial hysteresis step would cap any GPU speedup at about "
                f"{m['amdahl_max_speedup']}x.")

    if sc == "both" and m.get("opencv_faster_than_gpu_by"):
        caveats.append(f"OpenCV's own optimised cv2.Canny ran {m['opencv_faster_than_gpu_by']}x faster than this project's GPU path, "
                       "so this GPU code is a teaching implementation, not the fastest possible.")
    if not caveats:
        caveats.append("No warnings were raised by the metrics tool.")
    md = ("## What was measured\n" + "\n".join(f"- {x}" for x in measured) +
          "\n## Why (architecture explanation, not measured here)\n" + "\n".join(why) +
          "\n## Caveats\n" + "\n".join(f"- {x}" for x in caveats))
    return md, used


# --------------------------------------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------------------------------------
def _compact(d: dict) -> dict:
    """Drop empty entries so the small model has less to read."""
    return {k: v for k, v in d.items() if v is not None and v != {} and v != []}


def _attempt(llm, results, metrics, breakdown, verdict, allowed, extra: str = ""):
    seen: dict = {"looked_up": [], "submitted": None}

    def _lookup(concept: str = "", **_ignored) -> dict:
        if concept not in CONCEPTS:
            return {"error": f"unknown concept; choose one of {CONCEPT_NAMES}"}
        if concept not in seen["looked_up"]:
            seen["looked_up"].append(concept)
        return {"concept": concept, **CONCEPTS[concept]}

    def _submit(explanation_markdown: str = "", **_ignored) -> dict:
        seen["submitted"] = str(explanation_markdown)
        return {"received": True}

    registry = {
        "get_metrics": lambda **_: _compact(metrics),
        "get_stage_breakdown": lambda **_: _compact(breakdown),
        "check_correctness_status": lambda **_: verdict,
        "lookup_coa_concept": _lookup,
        "submit_explanation": _submit,
    }
    task = (f"Analyse run {results.plan.run_id}: image {results.image_info.width}x{results.image_info.height}, "
            f"GPU {results.gpu_name or 'none'}. Use the tools, then submit." + extra)
    loop = run_tool_loop(llm, SYSTEM_PROMPT, task, registry, TOOL_SCHEMAS, max_steps=10, stop_after={"submit_explanation"})
    text = seen["submitted"] if seen["submitted"] is not None else loop.final_text
    return text, seen["looked_up"], loop


def analyze(results: ExecutionResults, llm) -> tuple[AnalysisReport, str]:
    metrics = compute_metrics(results)
    breakdown = get_stage_breakdown(results)
    verdict = check_correctness_status(results)
    allowed = allowed_numbers(results, metrics, breakdown, verdict)

    explanation, concepts, source = None, [], ""
    log: list[str] = []
    extra = ""
    for attempt in (1, 2):
        try:
            text, looked, loop = _attempt(llm, results, metrics, breakdown, verdict, allowed, extra)
        except Exception as exc:                                   # Ollama down, bad reply, etc.
            log.append(f"attempt {attempt} failed to run ({type(exc).__name__}: {exc})")
            break
        problems = find_problems(text, allowed, metrics["winner"])
        if len(looked) < MIN_LOOKUPS:
            problems.append(f"you looked up only {len(looked)} COA concepts; call lookup_coa_concept for at least {MIN_LOOKUPS}")
        if not problems:
            explanation, concepts = text.strip(), looked
            source = (f"written by Agent 3's language model ({loop.steps} steps; tools: {', '.join(loop.tools_called)}); "
                      "every number in it was checked by code against the measured data")
            if attempt == 2:
                source += " (accepted on the second try)"
            break
        # the log goes into the report, so it must not repeat the rejected (possibly invented) numbers
        log.append(f"attempt {attempt} rejected: {'; '.join(re.sub(r':.*', '', q) for q in problems)}")
        notes = "".join(f"\n- {n}: {CONCEPTS[n]['note']}" for n in template_explanation(metrics, breakdown, verdict)[1])
        extra = ("\nYour previous explanation was rejected because " + "; ".join(problems) +
                 ". Use only numbers copied from the tool results. Hysteresis is serial and runs on the CPU in both paths. "
                 "Still call lookup_coa_concept for at least two concepts. Checked notes you may rely on:" + notes)

    if explanation is None:
        explanation, concepts = template_explanation(metrics, breakdown, verdict)
        source = "built by code from the measured results because the language model's text could not be verified (" + \
                 ("; ".join(log) or "no usable reply") + ")"

    footer = f"\n\nExplanation source: {source}."
    report = AnalysisReport(
        run_id=results.plan.run_id,
        winner=metrics["winner"],
        speedup_total=metrics["speedup_total"],
        speedup_kernel_only=metrics["speedup_kernel_only"],
        transfer_overhead_pct=metrics["transfer_overhead_pct"],
        throughput_cpu_mpix_s=metrics["throughput_cpu_mpix_s"],
        throughput_gpu_mpix_s=metrics["throughput_gpu_mpix_s"],
        per_stage_speedup=metrics["per_stage_speedup"],
        correctness_verdict=verdict["verdict"],
        explanation_markdown=explanation + footer,
        coa_concepts_used=[c for c in concepts if c in CONCEPTS],
    )
    return report, source


def run_analyst(results: ExecutionResults, llm) -> AnalysisReport:
    return analyze(results, llm)[0]
