import json

import cv2
import numpy as np
import pytest

from agents.planner import PlanDraft, PlannerError, UnsupportedTaskError, _clamp, run_planner
from llm.fake_client import ScriptedLLM
from schemas import ExecutionPlan


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "test.png"
    _write_png(p, 640, 480)  # small image
    return str(p)


def _write_png(path, w, h, channels=3):
    """Small, highly compressible test image (a gradient plus a bright box, so Canny has edges).
    Random noise would make each PNG ~1 MB; this is a few KB, which matters on a nearly full drive."""
    x = np.tile(np.linspace(0, 200, w, dtype=np.uint8), (h, 1))
    x[h // 4: h // 2, w // 4: w // 2] = 255
    img = x if channels == 1 else np.dstack([x, x[::-1], x])
    assert cv2.imwrite(str(path), img), f"could not write test image to {path}"


def tool_call(path):
    return {"content": "", "tool_calls": [{"name": "inspect_image", "arguments": {"image_path": path}}]}


def text(t):
    return {"content": t, "tool_calls": []}


def draft(**overrides):
    base = {"task": "canny_edge_detection", "low_threshold": 100, "high_threshold": 200,
            "run_cpu": True, "run_gpu": True, "repetitions": 20, "warmup_runs": 3,
            "reasoning": "defaults"}
    base.update(overrides)
    return text(json.dumps(base))


def happy_llm(img, **overrides):
    return ScriptedLLM([tool_call(img), text("640x480 image."), draft(**overrides)])


def test_happy_path(img):
    plan = run_planner("Apply edge detection and compare CPU and GPU", img, happy_llm(img))
    assert isinstance(plan, ExecutionPlan)
    assert plan.task == "canny_edge_detection" and plan.run_cpu and plan.run_gpu
    assert plan.repetitions == 20 and plan.run_id is None
    assert plan.image_path.endswith("test.png")
    assert "defaults" in plan.notes


def test_llm_really_calls_the_tool(img):
    llm = happy_llm(img)
    run_planner("compare cpu and gpu", img, llm)
    first_reply_msgs = llm.calls[1]["messages"]  # second call already holds the tool result
    assert any(m["role"] == "tool" and "megapixels" in m["content"] for m in first_reply_msgs)


def test_code_runs_tool_if_llm_skips_it(img):
    llm = ScriptedLLM([text("no tool used"), draft()])
    plan = run_planner("compare cpu and gpu", img, llm)
    assert "ran it itself" in plan.notes


def test_explicit_numbers_in_request_beat_llm(img):
    llm = happy_llm(img, repetitions=10, low_threshold=100, high_threshold=200)
    plan = run_planner("Run canny 30 times with thresholds 50 and 150", img, llm)
    assert (plan.repetitions, plan.low_threshold, plan.high_threshold) == (30, 50, 150)
    assert "from the request text" in plan.notes


def test_threshold_then_other_number_not_confused(img):
    plan = run_planner("threshold 80, run 5 times", img, happy_llm(img))
    assert plan.repetitions == 5 and plan.low_threshold == 100  # 80/5 must not become a pair


def test_gpu_only_and_cpu_only(img):
    p1 = run_planner("GPU only please", img, happy_llm(img))
    assert (p1.run_cpu, p1.run_gpu) == (False, True)
    p2 = run_planner("run it only on the CPU", img, happy_llm(img))
    assert (p2.run_cpu, p2.run_gpu) == (True, False)


def test_out_of_range_values_are_clamped():
    d = PlanDraft(low_threshold=-5, high_threshold=999, repetitions=100000, warmup_runs=0)
    _clamp(d, [])
    assert (d.low_threshold, d.high_threshold) == (0, 255)
    assert d.repetitions == 100 and d.warmup_runs == 1


def test_swapped_thresholds_fixed():
    d = PlanDraft(low_threshold=200, high_threshold=100)
    _clamp(d, [])
    assert (d.low_threshold, d.high_threshold) == (100, 200)


def test_equal_thresholds_widened():
    d = PlanDraft(low_threshold=120, high_threshold=120)
    _clamp(d, [])
    assert d.low_threshold < d.high_threshold


def test_both_devices_off_is_repaired(img):
    plan = run_planner("edge detection", img, happy_llm(img, run_cpu=False, run_gpu=False))
    assert plan.run_cpu and plan.run_gpu


def test_garbage_llm_output_falls_back_to_defaults(img):
    llm = ScriptedLLM([tool_call(img), text("ok"), text("not json"), text("still not json")])
    plan = run_planner("edge detection", img, llm)
    assert plan.repetitions == 20  # image-size suggestion for a small image
    assert "invalid twice" in plan.notes


def test_json_wrapped_in_chatter_is_accepted(img):
    wrapped = text("Sure! Here you go:\n" + json.dumps({"reasoning": "parsed fine"}) + "\nHope that helps")
    llm = ScriptedLLM([tool_call(img), text("ok"), wrapped])
    assert "parsed fine" in run_planner("edge detection", img, llm).notes


def test_unsupported_task_raises(img):
    with pytest.raises(UnsupportedTaskError):
        run_planner("blur this image", img, happy_llm(img, task="unsupported"))


def test_missing_image_raises(tmp_path):
    ghost = str(tmp_path / "ghost.png")
    llm = ScriptedLLM([tool_call(ghost), text("x")])
    with pytest.raises(PlannerError, match="not found"):
        run_planner("edge detection", ghost, llm)


def test_empty_request_raises(img):
    with pytest.raises(PlannerError):
        run_planner("   ", img, ScriptedLLM([]))


def test_plan_survives_json_roundtrip(img):
    plan = run_planner("edge detection", img, happy_llm(img))
    assert ExecutionPlan.model_validate_json(plan.model_dump_json()) == plan


# ---- the bug seen in the first real-model run: the LLM copied 50/150 from a prompt example -----------

def test_invented_thresholds_are_reset_to_defaults(img):
    llm = happy_llm(img, low_threshold=50, high_threshold=150, reasoning="Default thresholds")
    plan = run_planner("Apply edge detection to this image and compare CPU and GPU performance", img, llm)
    assert (plan.low_threshold, plan.high_threshold) == (100, 200)
    assert "did not mention thresholds" in plan.notes


def test_llm_threshold_choice_kept_when_request_talks_about_thresholds(img):
    llm = happy_llm(img, low_threshold=120, high_threshold=240)
    plan = run_planner("Run edge detection with a higher threshold to keep only strong edges", img, llm)
    assert (plan.low_threshold, plan.high_threshold) == (120, 240)


def test_invented_repetitions_reset_to_image_based_value(img):
    plan = run_planner("compare cpu and gpu", img, happy_llm(img, repetitions=30))
    assert plan.repetitions == 20  # small image
    assert "gave no run count" in plan.notes


def test_invented_device_skip_is_undone(img):
    plan = run_planner("compare performance", img, happy_llm(img, run_cpu=False))
    assert plan.run_cpu and plan.run_gpu
    assert "did not ask to skip a device" in plan.notes


def test_skip_wording_lets_llm_choice_stand(img):
    plan = run_planner("just run it on the GPU", img, happy_llm(img, run_cpu=False))
    assert (plan.run_cpu, plan.run_gpu) == (False, True)
