"""Real-LLM test. Needs Ollama running with llama3.2:3b.

PowerShell:  $env:RUN_LIVE=1; pytest tests/test_planner_live.py -s
Slow on CPU (about 1-2 minutes) because the planner makes 3 LLM calls.
"""
import os

import cv2
import numpy as np
import pytest

from agents.planner import run_planner

def _write_png(path, w, h, channels=3):
    """Small, highly compressible test image (a gradient plus a bright box, so Canny has edges).
    Random noise would make each PNG ~1 MB; this is a few KB, which matters on a nearly full drive."""
    x = np.tile(np.linspace(0, 200, w, dtype=np.uint8), (h, 1))
    x[h // 4: h // 2, w // 4: w // 2] = 255
    img = x if channels == 1 else np.dstack([x, x[::-1], x])
    assert cv2.imwrite(str(path), img), f"could not write test image to {path}"


pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to run")


@pytest.fixture(scope="module")
def llm():
    from llm.client import OllamaClient
    return OllamaClient()


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "live.png"
    _write_png(p, 1280, 720)
    return str(p)


def test_live_plain_request(llm, img):
    plan = run_planner("Apply edge detection to this image and compare CPU and GPU performance.", img, llm)
    print(plan.model_dump_json(indent=2))
    assert plan.task == "canny_edge_detection" and plan.run_cpu and plan.run_gpu


def test_live_specific_request(llm, img):
    plan = run_planner("Run canny edge detection on the GPU only, 15 times, thresholds 60 and 180", img, llm)
    print(plan.model_dump_json(indent=2))
    assert (plan.run_cpu, plan.run_gpu) == (False, True)
    assert (plan.repetitions, plan.low_threshold, plan.high_threshold) == (15, 60, 180)


def test_live_agentic_plain_request(llm, img):
    from agents.planner_agentic import run_planner_agentic
    plan = run_planner_agentic("Apply edge detection to this image and compare CPU and GPU performance.", img, llm)
    print(plan.model_dump_json(indent=2))
    assert plan.run_cpu and plan.run_gpu and (plan.low_threshold, plan.high_threshold) == (100, 200)


def test_live_agentic_rejects_unsupported(llm, img):
    import pytest
    from agents.planner import UnsupportedTaskError
    from agents.planner_agentic import run_planner_agentic
    with pytest.raises(UnsupportedTaskError):
        run_planner_agentic("Blur this image and make it black and white", img, llm)
