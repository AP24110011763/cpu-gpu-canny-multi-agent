"""Agent 2 (executor) tests. No Ollama and no GPU needed: the LLM is scripted and the GPU is faked.

The fake GPU runs the real CPU tool and relabels the result, so the correctness check has real
edge images to compare.
"""
import cv2
import numpy as np
import pytest

import agents.executor as ex
from agents.executor import ExecutorError, run_executor
from llm.fake_client import ScriptedLLM
from schemas import DeviceResult, ExecutionPlan
from tools.cpu_canny import run_cpu_canny


def call(name, **args):
    return {"content": "", "tool_calls": [{"name": name, "arguments": args}]}


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """The executor waits before each benchmark; tests must not really wait. Records the waits."""
    waited = []
    monkeypatch.setattr(ex.time, "sleep", lambda s: waited.append(s))
    return waited


@pytest.fixture
def img(tmp_path):
    a = np.full((120, 120), 40, np.uint8)
    a[30:90, 30:90] = 220
    p = tmp_path / "sq.png"
    assert cv2.imwrite(str(p), a)
    return str(p)


def make_plan(img, **over):
    base = dict(task="canny_edge_detection", image_path=img, low_threshold=100, high_threshold=200,
                repetitions=2, warmup_runs=1)
    return ExecutionPlan(**{**base, **over})


GPU_OK = {"ok": True, "gpu_name": "Fake GTX", "problems": []}
GPU_BAD = {"ok": False, "gpu_name": None, "problems": ["CuPy is not installed in this environment"]}


@pytest.fixture
def fake_gpu(monkeypatch):
    """GPU available: check_gpu is ok and run_gpu_canny returns the CPU result relabelled as GPU."""
    monkeypatch.setattr(ex, "check_gpu", lambda: dict(GPU_OK))

    def fake_run(path, low, high, reps, warm, out):
        res, edges, mag = run_cpu_canny(path, low, high, reps, warm, out)
        gpu = res.model_copy(update={"device": "gpu", "transfer_time_ms": 1.0, "kernel_time_ms": 2.0,
                                     "opencv_reference_time_ms": None})
        return gpu, edges, mag
    monkeypatch.setattr(ex, "run_gpu_canny", fake_run)


HAPPY = [call("check_gpu"), call("run_cpu"), call("run_gpu"), call("check_correctness"),
         call("finish", summary="Both ran and agree.")]


def test_happy_path_builds_complete_results(img, tmp_path, fake_gpu):
    res = run_executor(make_plan(img), ScriptedLLM(HAPPY), output_root=str(tmp_path / "out"))
    assert res.cpu.error is None and res.gpu.error is None and res.gpu_name == "Fake GTX"
    assert res.gpu.transfer_time_ms == 1.0
    assert res.correctness.performed and res.correctness.passed
    assert res.plan.run_id and res.plan.run_id.startswith("run_")
    assert (res.image_info.width, res.image_info.height) == (120, 120)
    assert "Both ran and agree." in res.agent2_notes and "safety net" not in res.agent2_notes
    assert (tmp_path / "out" / res.plan.run_id / "cpu_edges.png").exists()


def test_cpu_only_plan_never_touches_the_gpu(img, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("GPU must not be used")
    monkeypatch.setattr(ex, "check_gpu", boom)
    monkeypatch.setattr(ex, "run_gpu_canny", boom)
    llm = ScriptedLLM([call("run_cpu"), call("finish", summary="CPU only.")])
    res = run_executor(make_plan(img, run_gpu=False), llm, output_root=str(tmp_path))
    assert res.cpu and res.gpu is None and res.gpu_name is None
    assert res.correctness.performed is False and "safety net" not in res.agent2_notes


def test_unusable_gpu_gives_cpu_only_results_and_says_why(img, tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "check_gpu", lambda: dict(GPU_BAD))
    monkeypatch.setattr(ex, "run_gpu_canny", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    # the agent wrongly tries run_gpu anyway: the tool refuses
    llm = ScriptedLLM([call("check_gpu"), call("run_cpu"), call("run_gpu"), call("finish", summary="GPU missing.")])
    res = run_executor(make_plan(img), llm, output_root=str(tmp_path))
    assert res.cpu and res.gpu is None and res.correctness.performed is False
    assert "CuPy is not installed" in res.agent2_notes


def test_llm_cannot_change_the_plans_settings(img, tmp_path, fake_gpu, monkeypatch):
    seen = {}
    real = ex.run_cpu_canny

    def spy(path, low, high, reps, warm, out):
        seen.update(low=low, high=high, reps=reps, warm=warm)
        return real(path, low, high, reps, warm, out)
    monkeypatch.setattr(ex, "run_cpu_canny", spy)
    llm = ScriptedLLM([call("check_gpu"), call("run_cpu", low_threshold=1, high_threshold=2, repetitions=500),
                       call("finish", summary="x")])
    run_executor(make_plan(img, low_threshold=100, high_threshold=200, repetitions=2), llm, output_root=str(tmp_path))
    assert seen == dict(low=100, high=200, reps=2, warm=1)


@pytest.mark.parametrize("script", [
    [call("finish", summary="done")],                               # agent skips all the work
    [{"content": "All finished!", "tool_calls": []}],               # agent just talks
    [call("check_gpu"), call("run_cpu"), call("finish", summary="partial")],   # agent stops half-way
])
def test_safety_net_does_the_missing_steps(img, tmp_path, fake_gpu, script):
    res = run_executor(make_plan(img), ScriptedLLM(script), output_root=str(tmp_path))
    assert res.cpu and res.gpu and res.correctness.performed and res.correctness.passed
    assert "Code safety net" in res.agent2_notes


def test_gpu_run_failure_is_kept_as_an_error_result(img, tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "check_gpu", lambda: dict(GPU_OK))
    bad = DeviceResult(device="gpu", stage_times_ms={}, total_time_ms=0.0, total_time_std_ms=0.0,
                       repetitions=2, warmup_runs=1, error="GPU run failed: MemoryError: out of memory")
    monkeypatch.setattr(ex, "run_gpu_canny", lambda *a, **k: (bad, None, None))
    llm = ScriptedLLM([call("check_gpu"), call("run_cpu"), call("run_gpu"), call("check_correctness"),
                       call("finish", summary="GPU failed.")])
    res = run_executor(make_plan(img), llm, output_root=str(tmp_path))
    assert res.cpu.error is None and "MemoryError" in res.gpu.error
    assert res.correctness.performed is False


def test_unreadable_image_raises(tmp_path):
    with pytest.raises(ExecutorError):
        run_executor(make_plan(str(tmp_path / "ghost.png")), ScriptedLLM([]))


def test_results_roundtrip_through_json(img, tmp_path, fake_gpu):
    """Agent 3 will receive this as JSON, so it must survive serialisation."""
    from schemas import ExecutionResults
    res = run_executor(make_plan(img), ScriptedLLM(HAPPY), output_root=str(tmp_path))
    again = ExecutionResults.model_validate_json(res.model_dump_json())
    assert again == res


# ---- quiet start before each benchmark ---------------------------------------------------------
class UnloadableLLM(ScriptedLLM):
    """A scripted LLM that, like OllamaClient, can unload itself. Logs into a shared event list."""
    def __init__(self, replies, events):
        super().__init__(replies)
        self.events = events

    def unload(self):
        self.events.append("unload")
        return True


def test_llm_is_unloaded_and_machine_settles_before_each_benchmark(img, tmp_path, fake_gpu, monkeypatch, sleeps):
    events = []
    real_cpu, real_gpu = ex.run_cpu_canny, ex.run_gpu_canny
    monkeypatch.setattr(ex, "run_cpu_canny", lambda *a, **k: (events.append("cpu"), real_cpu(*a, **k))[1])
    monkeypatch.setattr(ex, "run_gpu_canny", lambda *a, **k: (events.append("gpu"), real_gpu(*a, **k))[1])
    res = run_executor(make_plan(img), UnloadableLLM(HAPPY, events), output_root=str(tmp_path), settle_seconds=2.5)
    assert events == ["unload", "cpu", "unload", "gpu"]          # quiet BEFORE each benchmark
    assert sleeps == [2.5, 2.5]
    assert "LLM unloaded, waited 2.5 s" in res.agent2_notes


def test_settling_works_with_an_llm_that_cannot_unload(img, tmp_path, fake_gpu, sleeps):
    res = run_executor(make_plan(img), ScriptedLLM(HAPPY), output_root=str(tmp_path))   # default 3 s
    assert sleeps == [3.0, 3.0] and "waited 3 s" in res.agent2_notes and "unloaded" not in res.agent2_notes


def test_no_waiting_when_settle_seconds_is_zero(img, tmp_path, fake_gpu, sleeps):
    run_executor(make_plan(img), ScriptedLLM(HAPPY), output_root=str(tmp_path), settle_seconds=0)
    assert sleeps == []


def test_refused_gpu_run_does_not_settle(img, tmp_path, monkeypatch, sleeps):
    monkeypatch.setattr(ex, "check_gpu", lambda: dict(GPU_BAD))
    llm = ScriptedLLM([call("check_gpu"), call("run_gpu"), call("run_cpu"), call("finish", summary="x")])
    run_executor(make_plan(img), llm, output_root=str(tmp_path))
    assert sleeps == [3.0]                                       # only the CPU benchmark waited
