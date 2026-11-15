"""TracingLLM records real facts about each agent's stage; the report's workflow section is drawn from them."""
import json
from pathlib import Path

import pytest

from llm.fake_client import ScriptedLLM
from schemas import ExecutionResults
from tools.report_writer import write_report
from tools.run_trace import TracingLLM, get_trace, merge_saved, start_stage

FIX = Path(__file__).parent / "fixtures"


def reply(*names):
    return {"content": "", "tool_calls": [{"name": n, "arguments": {}} for n in names]}


def fake_clock():
    t = {"now": 0.0}
    def clock():
        return t["now"]
    clock.t = t
    return clock


def test_records_calls_tools_and_time_per_stage():
    clock = fake_clock()
    inner = ScriptedLLM([reply("a", "b"), reply("a"), {"content": "done", "tool_calls": []}])
    llm = TracingLLM(inner, clock)
    llm.start_stage("agent1")
    llm.chat([]); clock.t["now"] += 2.0
    llm.chat([]); clock.t["now"] += 3.0
    llm.start_stage("agent2")
    clock.t["now"] += 1.0
    llm.chat([])
    tr = llm.trace()["stages"]
    assert tr["agent1"]["llm_calls"] == 2 and tr["agent1"]["tools"] == ["a", "b", "a"] and tr["agent1"]["seconds"] == 5.0
    assert tr["agent2"]["llm_calls"] == 1 and tr["agent2"]["tools"] == [] and tr["agent2"]["seconds"] == 1.0


def test_unknown_attributes_pass_through_and_missing_ones_still_fail():
    class Inner:
        model = "m"
        def unload(self): return True
        def chat(self, messages, tools=None, json_schema=None): return {"content": ""}
    llm = TracingLLM(Inner())
    assert llm.unload() is True and llm.model == "m"
    assert getattr(llm, "nope", None) is None                      # executor probes with getattr(llm, "unload", None)


def test_a_crashing_model_is_still_counted_and_the_error_is_not_swallowed():
    class Down:
        def chat(self, *a, **k): raise ConnectionError("ollama down")
    llm = TracingLLM(Down())
    llm.start_stage("agent1")
    with pytest.raises(ConnectionError):
        llm.chat([])
    assert llm.trace()["stages"]["agent1"]["llm_calls"] == 1


def test_helpers_ignore_a_plain_llm():
    plain = ScriptedLLM([])
    start_stage(plain, "agent1")
    assert get_trace(plain) is None and merge_saved(None, Path("x")) is None


def test_merge_keeps_earlier_agents_when_only_agent3_is_rerun(tmp_path):
    (tmp_path / "trace.json").write_text(json.dumps({"stages": {"agent1": {"seconds": 9.0, "llm_calls": 3, "llm_seconds": 8.0, "tools": ["x"]},
                                                              "agent3": {"seconds": 99.0, "llm_calls": 1, "llm_seconds": 1.0, "tools": []}}}))
    new = {"stages": {"agent3": {"seconds": 5.0, "llm_calls": 2, "llm_seconds": 4.0, "tools": ["y"]}}}
    merged = merge_saved(new, tmp_path / "trace.json")["stages"]
    assert merged["agent1"]["seconds"] == 9.0 and merged["agent3"]["seconds"] == 5.0          # new data wins
    assert merge_saved({"stages": {}}, tmp_path / "missing.json") == {"stages": {}}


def results(name="mock_gpu_wins_large_image"):
    return ExecutionResults.model_validate_json((FIX / f"{name}.json").read_text(encoding="utf-8"))


TRACE = {"version": 1, "stages": {
    "agent1": {"seconds": 150.0, "llm_calls": 4, "llm_seconds": 140.0, "tools": ["get_supported_tasks", "validate_plan", "validate_plan", "submit_plan"]},
    "agent2": {"seconds": 240.5, "llm_calls": 2, "llm_seconds": 100.0, "tools": ["check_gpu", "run_cpu", "run_gpu", "check_correctness", "finish"]},
    "agent3": {"seconds": 90.0, "llm_calls": 2, "llm_seconds": 85.0, "tools": ["get_metrics", "lookup_coa_concept", "submit_explanation"]}},
    "analysis": {"concepts": ["amdahl_law", "memory_hierarchy_and_pcie"], "source": "written by Agent 3's language model (2 steps)"}}


def page(tmp_path, trace, name="mock_gpu_wins_large_image"):
    out = write_report(results(name), "x", str(tmp_path), trace=trace)
    return Path(out["report_html"]).read_text(encoding="utf-8"), Path(out["report_md"]).read_text(encoding="utf-8")


def test_workflow_shows_the_real_trace_values(tmp_path):
    html_, md = page(tmp_path, TRACE)
    assert "How this run flowed" in html_ and html_.rstrip().endswith("</html>")
    assert html_.index("How this run flowed") > html_.index("<h2>Notes</h2>")                  # at the end of the page
    assert "2 min 30 s" in html_ and "4 min 0 s" in html_
    assert "validate_plan x2" in html_ and "4 model calls" in html_
    assert "thresholds 100/200" in html_ and "CPU 131.0 ms" in html_ and "winner: gpu" in html_ and "2 COA concepts" in html_
    assert "numbers verified by code" in html_ and html_.count('class="tbar"') == 1
    assert "validate_plan x2" in md and "Handoff 3 (AnalysisReport)" in md


def test_without_a_trace_only_the_design_is_shown_and_no_timings_are_invented(tmp_path):
    html_, md = page(tmp_path, None)
    assert "No run trace was recorded" in html_ and "tools in its design" in html_ and "model calls" not in html_
    assert 'class="tbar"' not in html_ and "No run trace" in md


def test_partial_trace_marks_missing_agents_as_not_recorded(tmp_path):
    html_, _ = page(tmp_path, {"version": 1, "stages": {"agent3": TRACE["stages"]["agent3"]}})
    assert html_.count("not recorded in this trace") == 2 and "2 model calls" in html_


def test_template_fallback_is_labelled_in_the_workflow(tmp_path):
    t = {**TRACE, "analysis": {"concepts": ["amdahl_law"], "source": "built by code from the measured results because ..."}}
    html_, _ = page(tmp_path, t)
    assert "built by code" in html_ and "numbers verified by code" not in html_


def test_trace_text_cannot_inject_html(tmp_path):
    evil = {"version": 1, "stages": {"agent1": {"seconds": 1.0, "llm_calls": 1, "llm_seconds": 1.0, "tools": ["<script>alert(1)</script>"]}}}
    html_, _ = page(tmp_path, evil)
    assert "<script>alert(1)</script>" not in html_ and "&lt;script&gt;" in html_


def test_failed_run_still_has_a_workflow(tmp_path):
    html_, _ = page(tmp_path, TRACE, "mock_gpu_error")
    assert "How this run flowed" in html_ and "correctness" in html_


def test_listener_sees_stage_starts_and_model_calls_and_cannot_break_the_run():
    seen = []
    llm = TracingLLM(ScriptedLLM([reply("a"), reply("b", "c")]), fake_clock(), on_event=lambda k, d: seen.append((k, d["stage"], d["llm_calls"], list(d["tools"]))))
    llm.start_stage("agent1")
    llm.chat([])
    llm.chat([])
    assert seen == [("stage_start", "agent1", 0, []), ("model_call", "agent1", 1, ["a"]), ("model_call", "agent1", 2, ["a", "b", "c"])]

    def boom(kind, data):
        raise RuntimeError("ui crashed")
    safe = TracingLLM(ScriptedLLM([reply("a")]), fake_clock(), on_event=boom)
    safe.start_stage("agent1")
    assert safe.chat([])["tool_calls"][0]["name"] == "a"
