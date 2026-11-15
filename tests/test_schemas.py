import pytest
from pydantic import ValidationError
from schemas import ExecutionPlan, ExecutionResults, CorrectnessCheck, ImageInfo


def test_plan_defaults():
    p = ExecutionPlan(task="canny_edge_detection", image_path="a.jpg")
    assert p.low_threshold == 100 and p.repetitions == 10 and p.warmup_runs == 3


def test_plan_json_roundtrip():
    p = ExecutionPlan(task="canny_edge_detection", image_path="a.jpg", run_id="r1")
    assert ExecutionPlan.model_validate_json(p.model_dump_json()) == p


def test_bad_task_rejected():
    with pytest.raises(ValidationError):
        ExecutionPlan(task="blur", image_path="a.jpg")


def test_results_minimal():
    plan = ExecutionPlan(task="canny_edge_detection", image_path="a.jpg")
    info = ImageInfo(width=640, height=480, channels=3, megapixels=0.3072)
    r = ExecutionResults(plan=plan, image_info=info,
                         correctness=CorrectnessCheck(performed=False))
    assert r.cpu is None and r.gpu is None
