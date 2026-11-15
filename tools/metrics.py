"""Agent 3's calculator: every number in the final report comes from here, never from the LLM.

A 3B language model is unreliable at arithmetic and at copying numbers. So these are plain,
deterministic functions. The LLM may CALL them and then explain what they return.

Three tools (the LLM chooses which to call):
  compute_metrics(results)        speedups, transfer share, throughput, Amdahl estimate, winner, warnings
  get_stage_breakdown(results)    per-stage CPU time, GPU time, speedup and share of total
  check_correctness_status(results)  one-line verdict on whether the CPU and GPU agree

Every function copes with missing devices (CPU-only, GPU unavailable, a device that errored).

Definitions (they match the comments in schemas.AnalysisReport):
  speedup_total          = cpu total / gpu total                         (> 1 means the GPU is faster)
  speedup_kernel_only    = cpu (blur+sobel+nms) / gpu kernel time         (compute only, no transfer, no hysteresis)
  transfer_overhead_pct  = gpu transfer / gpu total * 100
  throughput             = megapixels / seconds
  amdahl_max_speedup     = 1 / (cpu hysteresis share). Hysteresis is the serial stage, so even a GPU that
                           made every parallel stage free could not beat this (it ignores transfer cost).
"""
from __future__ import annotations

from schemas import DeviceResult, ExecutionResults

PARALLEL_STAGES = ["gaussian_blur", "sobel_gradients", "non_max_suppression"]
STAGES = PARALLEL_STAGES + ["hysteresis"]
TIE_BAND = 0.05          # speedup within 0.95..1.05 counts as a tie
NOISY_PCT = 15.0         # std above 15% of the mean = an unreliable measurement


def _usable(dev: DeviceResult | None) -> bool:
    return dev is not None and dev.error is None and dev.total_time_ms > 0


def _r(x: float | None, n: int = 3) -> float | None:
    return None if x is None else round(float(x), n)


def check_correctness_status(results: ExecutionResults) -> dict:
    c = results.correctness
    if not c.performed:
        verdict = f"NOT PERFORMED: {c.note or 'the CPU and GPU outputs were not both available'}"
    elif c.passed:
        verdict = (f"PASSED: CPU and GPU edge maps agree ({c.pixel_agreement_pct}% of pixels; "
                   f"needed {c.threshold_pct}%). {c.note}").strip()
    else:
        verdict = (f"FAILED: CPU and GPU edge maps disagree ({c.pixel_agreement_pct}% of pixels; "
                   f"needed {c.threshold_pct}%). {c.note}").strip()
    return {"performed": c.performed, "passed": c.passed, "pixel_agreement_pct": c.pixel_agreement_pct,
            "gradient_mean_abs_diff": c.gradient_mean_abs_diff, "verdict": verdict}


def get_stage_breakdown(results: ExecutionResults) -> dict:
    cpu = results.cpu if _usable(results.cpu) else None
    gpu = results.gpu if _usable(results.gpu) else None
    rows = []
    for s in STAGES:
        c = cpu.stage_times_ms.get(s) if cpu else None
        g = gpu.stage_times_ms.get(s) if gpu else None
        rows.append({
            "stage": s,
            "runs_on_gpu": s in PARALLEL_STAGES,
            "cpu_ms": _r(c), "gpu_ms": _r(g),
            "speedup": _r(c / g) if c and g else None,
            "cpu_share_pct": _r(100 * c / cpu.total_time_ms) if c is not None and cpu else None,
            "gpu_share_pct": _r(100 * g / gpu.total_time_ms) if g is not None and gpu else None,
        })
    out = {"stages": rows, "gpu_transfer_ms": None, "gpu_transfer_share_pct": None}
    if gpu and gpu.transfer_time_ms is not None:
        out["gpu_transfer_ms"] = _r(gpu.transfer_time_ms)
        out["gpu_transfer_share_pct"] = _r(100 * gpu.transfer_time_ms / gpu.total_time_ms)
    return out


def compute_metrics(results: ExecutionResults) -> dict:
    cpu = results.cpu if _usable(results.cpu) else None
    gpu = results.gpu if _usable(results.gpu) else None
    mp = results.image_info.megapixels
    warnings: list[str] = []
    m: dict = {k: None for k in (
        "speedup_total", "speedup_kernel_only", "transfer_overhead_pct", "throughput_cpu_mpix_s",
        "throughput_gpu_mpix_s", "amdahl_serial_fraction", "amdahl_max_speedup",
        "gpu_hysteresis_share_pct", "opencv_reference_ms", "opencv_faster_than_gpu_by")}
    m["per_stage_speedup"] = {}

    # ---- which situation are we in?
    if cpu is None and gpu is None:
        scenario = "no_data"
    elif cpu is not None and gpu is None:
        scenario = "gpu_error" if results.gpu is not None and results.gpu.error else "cpu_only"
        if scenario == "cpu_only" and results.plan.run_gpu:
            scenario = "gpu_missing"
    elif cpu is None:
        scenario = "cpu_error" if results.cpu is not None and results.cpu.error else "gpu_only"
    else:
        scenario = "both"
    for dev, name in ((results.cpu, "CPU"), (results.gpu, "GPU")):
        if dev is not None and dev.error:
            warnings.append(f"{name} run failed: {dev.error}")

    # ---- per-device numbers
    if cpu:
        m["throughput_cpu_mpix_s"] = _r(mp / (cpu.total_time_ms / 1000.0))
        m["cpu_total_ms"] = _r(cpu.total_time_ms)
        m["cpu_std_pct_of_mean"] = _r(100 * cpu.total_time_std_ms / cpu.total_time_ms, 1)
        hyst = cpu.stage_times_ms.get("hysteresis")
        if hyst is not None:
            m["amdahl_serial_fraction"] = _r(hyst / cpu.total_time_ms, 4)
            m["amdahl_max_speedup"] = _r(cpu.total_time_ms / hyst, 2) if hyst > 0 else None
        if cpu.opencv_reference_time_ms:
            m["opencv_reference_ms"] = _r(cpu.opencv_reference_time_ms)
    if gpu:
        m["throughput_gpu_mpix_s"] = _r(mp / (gpu.total_time_ms / 1000.0))
        m["gpu_total_ms"] = _r(gpu.total_time_ms)
        m["gpu_std_pct_of_mean"] = _r(100 * gpu.total_time_std_ms / gpu.total_time_ms, 1)
        if gpu.transfer_time_ms is not None:
            m["transfer_overhead_pct"] = _r(100 * gpu.transfer_time_ms / gpu.total_time_ms)
        h = gpu.stage_times_ms.get("hysteresis")
        if h is not None:
            m["gpu_hysteresis_share_pct"] = _r(100 * h / gpu.total_time_ms)

    # ---- comparison (needs both)
    if cpu and gpu:
        m["speedup_total"] = _r(cpu.total_time_ms / gpu.total_time_ms)
        if gpu.kernel_time_ms:
            cpu_parallel = sum(cpu.stage_times_ms.get(s, 0.0) for s in PARALLEL_STAGES)
            m["speedup_kernel_only"] = _r(cpu_parallel / gpu.kernel_time_ms)
        m["per_stage_speedup"] = {
            s: _r(cpu.stage_times_ms[s] / gpu.stage_times_ms[s])
            for s in STAGES if cpu.stage_times_ms.get(s) and gpu.stage_times_ms.get(s)}
        if m["opencv_reference_ms"]:
            m["opencv_faster_than_gpu_by"] = _r(gpu.total_time_ms / m["opencv_reference_ms"], 2)

    # ---- winner
    corr = results.correctness
    if scenario != "both":
        winner = "undetermined"
    elif corr.performed and corr.passed is False:
        winner = "undetermined"
        warnings.append("Correctness FAILED: the CPU and GPU outputs differ, so speed alone cannot declare a winner.")
    elif 1 - TIE_BAND <= m["speedup_total"] <= 1 + TIE_BAND:
        winner = "tie"
    else:
        winner = "gpu" if m["speedup_total"] > 1 else "cpu"
    if scenario == "both" and not corr.performed:
        warnings.append("Correctness was not checked for this run.")

    # ---- warnings
    if scenario in ("cpu_only", "gpu_missing"):
        warnings.append("Only the CPU produced results, so no CPU-vs-GPU comparison is possible.")
    for name, key in (("CPU", "cpu_std_pct_of_mean"), ("GPU", "gpu_std_pct_of_mean")):
        if m.get(key) is not None and m[key] > NOISY_PCT:
            warnings.append(f"{name} timing is noisy (std is {m[key]}% of the mean); repeat the run before trusting the speedup.")

    m.update(scenario=scenario, winner=winner, warnings=warnings,
             image_megapixels=_r(mp, 4), run_id=results.plan.run_id)
    return m
