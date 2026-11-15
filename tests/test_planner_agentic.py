import cv2
import numpy as np
import pytest

from agents.planner import PlannerError, UnsupportedTaskError
from agents.planner_agentic import run_planner_agentic, validate_plan
from llm.fake_client import ScriptedLLM


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "t.png"
    _write_png(p, 640, 480)  # small -> 20 reps
    return str(p)


def _write_png(path, w, h, channels=3):
    """Small, highly compressible test image (a gradient plus a bright box, so Canny has edges).
    Random noise would make each PNG ~1 MB; this is a few KB, which matters on a nearly full drive."""
    x = np.tile(np.linspace(0, 200, w, dtype=np.uint8), (h, 1))
    x[h // 4: h // 2, w // 4: w // 2] = 255
    img = x if channels == 1 else np.dstack([x, x[::-1], x])
    assert cv2.imwrite(str(path), img), f"could not write test image to {path}"


def call(name, **args):
    return {"content": "", "tool_calls": [{"name": name, "arguments": args}]}


GOOD = dict(low_threshold=100, high_threshold=200, run_cpu=True, run_gpu=True, repetitions=20, warmup_runs=3)


def happy_script(**over):
    plan = {**GOOD, **over}
    return [
        call("get_supported_tasks"),
        call("inspect_image", image_path="x"),
        call("validate_plan", **plan),
        call("submit_plan", **plan, reasoning="defaults for a small image"),
    ]


def test_validate_plan_tool():
    assert validate_plan(**GOOD)["ok"]
    bad = validate_plan(low_threshold=200, high_threshold=100, run_cpu=False, run_gpu=False, repetitions=0)
    assert not bad["ok"] and len(bad["errors"]) == 3  # threshold order, repetitions range, both devices off
    assert not validate_plan(low_threshold="abc", high_threshold=1, repetitions=1)["ok"]


def test_agent_follows_its_own_steps(img):
    llm = ScriptedLLM(happy_script())
    plan = run_planner_agentic("compare cpu and gpu", img, llm)
    assert (plan.low_threshold, plan.high_threshold, plan.repetitions) == (100, 200, 20)
    assert plan.run_cpu and plan.run_gpu and plan.run_id is None
    assert "get_supported_tasks" in plan.notes and "validate_plan" in plan.notes


def test_agent_corrects_itself_after_validation_error(img):
    bad = dict(GOOD, low_threshold=200, high_threshold=100)
    script = [
        call("inspect_image", image_path="x"),
        call("validate_plan", **bad),                      # tool answers: errors
        call("validate_plan", **GOOD),                     # agent fixes it
        call("submit_plan", **GOOD, reasoning="fixed the swapped thresholds"),
    ]
    llm = ScriptedLLM(script)
    plan = run_planner_agentic("compare cpu and gpu", img, llm)
    assert plan.low_threshold < plan.high_threshold
    # the model really did see the error message before its next move
    tool_msgs = [m["content"] for m in llm.calls[2]["messages"] if m["role"] == "tool"]
    assert any("low_threshold must be smaller" in m for m in tool_msgs)


def test_agent_can_reject(img):
    llm = ScriptedLLM([call("get_supported_tasks"), call("reject_request", reason="blur is not supported")])
    with pytest.raises(UnsupportedTaskError, match="blur"):
        run_planner_agentic("blur this", img, llm)


def test_code_still_overrides_agent(img):
    llm = ScriptedLLM(happy_script(low_threshold=50, high_threshold=150, repetitions=30))
    plan = run_planner_agentic("compare cpu and gpu", img, llm)  # request mentions no numbers
    assert (plan.low_threshold, plan.high_threshold, plan.repetitions) == (100, 200, 20)
    assert "Code corrections" in plan.notes


def test_explicit_user_numbers_win(img):
    llm = ScriptedLLM(happy_script())
    plan = run_planner_agentic("GPU only, 15 times, thresholds 60 and 180", img, llm)
    assert (plan.run_cpu, plan.run_gpu, plan.repetitions) == (False, True, 15)
    assert (plan.low_threshold, plan.high_threshold) == (60, 180)


def test_submit_without_validate_is_noted(img):
    plan_args = dict(GOOD)
    llm = ScriptedLLM([call("inspect_image", image_path="x"), call("submit_plan", **plan_args, reasoning="ok")])
    plan = run_planner_agentic("compare cpu and gpu", img, llm)
    assert "without calling validate_plan" in plan.notes


def test_falls_back_to_classic_planner_if_never_submits(img):
    import json
    draft = {"task": "canny_edge_detection", "low_threshold": 100, "high_threshold": 200, "run_cpu": True,
             "run_gpu": True, "repetitions": 20, "warmup_runs": 3, "reasoning": "classic"}
    llm = ScriptedLLM([
        {"content": "I am not sure what to do", "tool_calls": []},   # agent loop ends with no submit
        {"content": "Still not sure", "tool_calls": []},               # the nudge does not help either
        call("inspect_image", image_path="x"),                        # classic planner, call 1
        {"content": "small image", "tool_calls": []},                  # classic planner, call 2
        {"content": json.dumps(draft), "tool_calls": []},              # classic planner, call 3
    ])
    plan = run_planner_agentic("compare cpu and gpu", img, llm)
    assert "classic planner ran" in plan.notes and plan.repetitions == 20
    assert "a nudge did not help" in plan.notes


def test_a_nudge_rescues_an_agent_that_stops_to_talk(img):
    """Real-model behaviour: it inspects the image, then says 'I will create a plan' without calling a tool."""
    llm = ScriptedLLM([
        call("get_supported_tasks"),
        call("inspect_image", image_path="x"),
        {"content": "Based on the image inspection, I will create a plan", "tool_calls": []},   # stops to talk
        call("validate_plan", **GOOD),                                                           # after the nudge
        call("submit_plan", **GOOD, reasoning="defaults"),
    ])
    plan = run_planner_agentic("compare cpu and gpu", img, llm)
    assert "nudged once" in plan.notes and "classic planner" not in plan.notes
    assert (plan.low_threshold, plan.high_threshold, plan.repetitions) == (100, 200, 20)
    nudge_call = llm.calls[3]                                   # the first LLM call of the second loop
    assert {t["function"]["name"] for t in nudge_call["tools"]} == {"validate_plan", "submit_plan", "reject_request"}
    assert "Image facts" in nudge_call["messages"][1]["content"]


def test_nudge_can_still_reject_an_unsupported_request(img):
    llm = ScriptedLLM([
        call("inspect_image", image_path="x"),
        {"content": "Let me think about this", "tool_calls": []},
        call("reject_request", reason="blur is not supported"),
    ])
    with pytest.raises(UnsupportedTaskError, match="blur"):
        run_planner_agentic("blur this", img, llm)


def test_missing_image_and_empty_request(tmp_path):
    with pytest.raises(PlannerError):
        run_planner_agentic("  ", str(tmp_path / "a.png"), ScriptedLLM([]))
    with pytest.raises(PlannerError, match="not found"):
        run_planner_agentic("edges", str(tmp_path / "ghost.png"), ScriptedLLM([call("inspect_image", image_path="x"), {"content": "", "tool_calls": []}]))


def test_string_values_from_small_model_are_tolerated(img):
    llm = ScriptedLLM([call("inspect_image", image_path="x"),
                       call("submit_plan", low_threshold="100", high_threshold="200", run_cpu="true",
                            run_gpu="true", repetitions="20", warmup_runs="3", reasoning="strings")])
    plan = run_planner_agentic("compare cpu and gpu", img, llm)
    assert plan.run_cpu and plan.repetitions == 20
