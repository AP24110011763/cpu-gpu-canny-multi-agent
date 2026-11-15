"""Pipeline tests: Agent 1 -> Agent 2 with ONE scripted LLM serving both agents in turn.
No Ollama and no GPU needed (the GPU is faked, as in test_executor.py)."""
import cv2
import numpy as np
import pytest

import agents.executor as ex
import orchestrator
from agents.planner import UnsupportedTaskError
from llm.fake_client import ScriptedLLM
from orchestrator import format_summary, run_pipeline
from schemas import ExecutionPlan, ExecutionResults
from tools.cpu_canny import run_cpu_canny


def call(name, **args):
    return {"content": "", "tool_calls": [{"name": name, "arguments": args}]}


@pytest.fixture(autouse=True)
def no_real_waiting(monkeypatch):
    monkeypatch.setattr(ex.time, "sleep", lambda s: None)      # Agent 2 waits before each benchmark


@pytest.fixture
def img(tmp_path):
    a = np.full((120, 120), 40, np.uint8)
    a[30:90, 30:90] = 220
    p = tmp_path / "sq.png"
    assert cv2.imwrite(str(p), a)
    return str(p)


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(ex, "check_gpu", lambda: {"ok": True, "gpu_name": "Fake GTX", "problems": []})

    def fake_run(path, low, high, reps, warm, out):
        res, edges, mag = run_cpu_canny(path, low, high, reps, warm, out)
        return res.model_copy(update={"device": "gpu", "transfer_time_ms": 1.0, "kernel_time_ms": 2.0,
                                      "opencv_reference_time_ms": None}), edges, mag
    monkeypatch.setattr(ex, "run_gpu_canny", fake_run)


def planner_script(**over):
    plan = dict(low_threshold=100, high_threshold=200, run_cpu=True, run_gpu=True, repetitions=2, warmup_runs=1)
    plan.update(over)
    return [call("get_supported_tasks"), call("inspect_image", image_path="x"), call("validate_plan", **plan),
            call("submit_plan", **plan, reasoning="as requested")]


EXECUTOR_BOTH = [call("check_gpu"), call("run_cpu"), call("run_gpu"), call("check_correctness"),
                 call("finish", summary="Both ran.")]


def test_full_pipeline_produces_and_saves_results(img, tmp_path, fake_gpu):
    llm = ScriptedLLM(planner_script() + EXECUTOR_BOTH)
    res, path = run_pipeline("Compare CPU and GPU, 2 times", img, llm, str(tmp_path / "out"))
    assert res.cpu and res.gpu and res.correctness.passed
    assert res.plan.run_id and "Agentic planner" in res.plan.notes and "Agent 2" in res.agent2_notes
    saved = ExecutionResults.model_validate_json(open(path, encoding="utf-8").read())
    assert saved == res


def test_the_plan_from_agent_1_is_what_agent_2_executes(img, tmp_path, monkeypatch):
    """Request says CPU only with thresholds 30 and 90: the GPU must never be touched."""
    def boom(*a, **k):
        raise AssertionError("GPU must not be used")
    monkeypatch.setattr(ex, "check_gpu", boom)
    monkeypatch.setattr(ex, "run_gpu_canny", boom)
    script = planner_script(low_threshold=30, high_threshold=90, run_gpu=False) + [
        call("run_cpu"), call("finish", summary="CPU only.")]
    res, _ = run_pipeline("Run edge detection on the CPU only, 2 times, thresholds 30 and 90", img,
                          ScriptedLLM(script), str(tmp_path))
    assert (res.plan.low_threshold, res.plan.high_threshold, res.plan.run_gpu) == (30, 90, False)
    assert res.cpu and res.gpu is None


def test_unsupported_request_stops_before_agent_2(img, tmp_path):
    llm = ScriptedLLM([call("get_supported_tasks"), call("reject_request", reason="blur is not supported")])
    out = tmp_path / "out"
    with pytest.raises(UnsupportedTaskError):
        run_pipeline("Blur this image", img, llm, str(out))        # Agent 2 would run out of scripted replies
    assert not out.exists()                                        # nothing was executed or saved


def test_classic_planner_option(img, tmp_path, fake_gpu, monkeypatch):
    plan = ExecutionPlan(task="canny_edge_detection", image_path=img, repetitions=2, warmup_runs=1,
                         notes="classic")
    monkeypatch.setattr(orchestrator, "run_planner", lambda req, path, llm: plan)
    res, _ = run_pipeline("x", img, ScriptedLLM(EXECUTOR_BOTH), str(tmp_path), agentic=False)
    assert res.plan.notes == "classic" and res.gpu


def test_summary_text_mentions_both_devices_and_correctness(img, tmp_path, fake_gpu):
    res, _ = run_pipeline("Compare, 2 times", img, ScriptedLLM(planner_script() + EXECUTOR_BOTH), str(tmp_path))
    text = format_summary(res)
    assert "CPU: total" in text and "GPU: total" in text and "Correctness: performed=True passed=True" in text


def test_agent_3_runs_after_agent_2_and_writes_the_report(img, tmp_path, fake_gpu, monkeypatch):
    """Agent 1 -> Agent 2 -> Agent 3 with one scripted LLM; the report is a real file on disk."""
    from agents.analyst import template_explanation
    from tools.metrics import check_correctness_status, compute_metrics, get_stage_breakdown

    llm = ScriptedLLM(planner_script() + EXECUTOR_BOTH)
    res, _ = run_pipeline("compare cpu and gpu", img, llm, str(tmp_path / "out"))
    good = template_explanation(compute_metrics(res), get_stage_breakdown(res), check_correctness_status(res))[0]
    llm.replies = [{"content": "", "tool_calls": [{"name": "get_metrics", "arguments": {}},
                                                   {"name": "lookup_coa_concept", "arguments": {"concept": "amdahl_law"}},
                                                   {"name": "lookup_coa_concept", "arguments": {"concept": "amdahl_law", "x": 1}},
                                                   {"name": "lookup_coa_concept", "arguments": {"concept": "throughput_vs_latency"}}]},
                   call("submit_explanation", explanation_markdown=good)]
    opened = []
    monkeypatch.setattr(orchestrator, "open_in_browser", lambda p: opened.append(p) or True)

    report, files = orchestrator.run_analysis(res, llm, str(tmp_path / "out"), open_report=True)

    assert report.report_path == files["report_html"] and opened == [files["report_html"]]
    assert (tmp_path / "out" / res.plan.run_id / "analysis.json").exists()
    assert "language model" in report.explanation_markdown and report.coa_concepts_used == ["amdahl_law", "throughput_vs_latency"]
    assert "language model" in open(files["report_html"], encoding="utf-8").read()


def test_report_is_not_opened_unless_asked(img, tmp_path, fake_gpu, monkeypatch):
    llm = ScriptedLLM(planner_script() + EXECUTOR_BOTH)
    res, _ = run_pipeline("compare cpu and gpu", img, llm, str(tmp_path / "out"))
    opened = []
    monkeypatch.setattr(orchestrator, "open_in_browser", lambda p: opened.append(p))
    llm.replies = []                                   # the LLM "fails", so the code template is used
    report, _ = orchestrator.run_analysis(res, llm, str(tmp_path / "out"))
    assert opened == [] and "built by code" in report.explanation_markdown


def test_full_run_saves_a_trace_with_all_three_agents_and_the_report_shows_it(img, tmp_path, fake_gpu):
    import json
    from agents.analyst import template_explanation
    from tools.metrics import check_correctness_status, compute_metrics, get_stage_breakdown
    from tools.run_trace import TracingLLM

    llm = TracingLLM(ScriptedLLM(planner_script() + EXECUTOR_BOTH))
    res, _ = run_pipeline("compare cpu and gpu", img, llm, str(tmp_path / "out"))
    good = template_explanation(compute_metrics(res), get_stage_breakdown(res), check_correctness_status(res))[0]
    llm._inner.replies = [{"content": "", "tool_calls": [{"name": "lookup_coa_concept", "arguments": {"concept": "amdahl_law"}},
                                                          {"name": "lookup_coa_concept", "arguments": {"concept": "throughput_vs_latency"}}]},
                          call("submit_explanation", explanation_markdown=good)]
    report, files = orchestrator.run_analysis(res, llm, str(tmp_path / "out"))
    tr = json.loads((tmp_path / "out" / res.plan.run_id / "trace.json").read_text(encoding="utf-8"))
    assert set(tr["stages"]) == {"agent1", "agent2", "agent3"}
    assert "submit_plan" in tr["stages"]["agent1"]["tools"] and "run_gpu" in tr["stages"]["agent2"]["tools"]
    assert tr["stages"]["agent3"]["tools"][-1] == "submit_explanation" and tr["analysis"]["concepts"] == ["amdahl_law", "throughput_vs_latency"]
    page = open(files["report_html"], encoding="utf-8").read()
    assert "How this run flowed" in page and "model calls" in page and "submit_plan" in page
