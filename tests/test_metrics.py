"""Tests for Agent 3's calculator (no LLM, no GPU).

The main mock uses round numbers so the expected values can be worked out by hand:
  CPU stages 1 + 10 + 100 + 20 = 131 ms      GPU: kernels 2+2+2 = 6, hysteresis 20, transfer 3 -> 29 ms
"""
import json
from pathlib import Path

import pytest

from schemas import ExecutionResults
from tools.metrics import check_correctness_status, compute_metrics, get_stage_breakdown

FIXTURES = Path(__file__).parent / "fixtures"


def load(name) -> ExecutionResults:
    return ExecutionResults.model_validate_json((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def test_every_mock_is_clearly_marked_as_a_mock_and_loads():
    files = sorted(FIXTURES.glob("mock_*.json"))
    assert len(files) >= 5
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert "DEVELOPMENT-ONLY MOCK" in data["_comment"] and "NOT real" in data["_comment"]
        ExecutionResults.model_validate(data)


def test_gpu_wins_numbers_match_hand_calculation():
    m = compute_metrics(load("mock_gpu_wins_large_image"))
    assert m["scenario"] == "both" and m["winner"] == "gpu" and m["warnings"] == []
    assert m["speedup_total"] == pytest.approx(131 / 29, abs=1e-3)               # 4.517
    assert m["speedup_kernel_only"] == pytest.approx(111 / 6, abs=1e-3)           # 18.5
    assert m["transfer_overhead_pct"] == pytest.approx(300 / 29, abs=1e-3)        # 10.345
    assert m["throughput_cpu_mpix_s"] == pytest.approx(1.4384 / 0.131, abs=1e-3)
    assert m["throughput_gpu_mpix_s"] == pytest.approx(1.4384 / 0.029, abs=1e-3)
    assert m["per_stage_speedup"] == {"gaussian_blur": 0.5, "sobel_gradients": 5.0,
                                      "non_max_suppression": 50.0, "hysteresis": 1.0}
    assert m["amdahl_serial_fraction"] == pytest.approx(20 / 131, abs=1e-4)
    assert m["amdahl_max_speedup"] == pytest.approx(6.55, abs=1e-2)
    assert m["gpu_hysteresis_share_pct"] == pytest.approx(2000 / 29, abs=1e-3)    # the serial step dominates the GPU path
    assert m["opencv_faster_than_gpu_by"] == pytest.approx(29 / 4, abs=1e-2)       # the honest finding about cv2.Canny
    assert m["cpu_std_pct_of_mean"] == 5.0


def test_cpu_wins_on_a_tiny_image_and_transfer_is_a_big_share():
    m = compute_metrics(load("mock_cpu_wins_small_image"))
    assert m["winner"] == "cpu" and m["speedup_total"] == pytest.approx(0.3 / 2.16, abs=1e-3)
    assert m["transfer_overhead_pct"] == pytest.approx(900 / 21.6, abs=1e-2)     # 41.7%
    assert m["speedup_total"] < 1 and m["warnings"] == []                        # a CPU win is a finding, not an error


def test_gpu_missing_gives_cpu_numbers_only():
    m = compute_metrics(load("mock_gpu_missing"))
    assert m["scenario"] == "gpu_missing" and m["winner"] == "undetermined"
    assert m["speedup_total"] is None and m["transfer_overhead_pct"] is None and m["per_stage_speedup"] == {}
    assert m["throughput_cpu_mpix_s"] is not None and m["throughput_gpu_mpix_s"] is None
    assert m["amdahl_max_speedup"] == pytest.approx(6.55, abs=1e-2)               # CPU-only facts still available
    assert any("Only the CPU" in w for w in m["warnings"])


def test_cpu_only_plan_is_not_reported_as_a_missing_gpu():
    res = load("mock_gpu_missing")
    res = res.model_copy(update={"plan": res.plan.model_copy(update={"run_gpu": False})})
    assert compute_metrics(res)["scenario"] == "cpu_only"


def test_failed_correctness_blocks_a_winner_but_keeps_the_numbers():
    res = load("mock_correctness_failed")
    m = compute_metrics(res)
    assert m["winner"] == "undetermined" and m["speedup_total"] == pytest.approx(131 / 29, abs=1e-3)
    assert any("Correctness FAILED" in w for w in m["warnings"])
    v = check_correctness_status(res)
    assert v["passed"] is False and v["verdict"].startswith("FAILED") and "91.2" in v["verdict"]


def test_gpu_error_is_reported_and_cpu_numbers_survive():
    m = compute_metrics(load("mock_gpu_error"))
    assert m["scenario"] == "gpu_error" and m["winner"] == "undetermined"
    assert any("GPU run failed" in w and "MemoryError" in w for w in m["warnings"])
    assert m["throughput_cpu_mpix_s"] is not None and m["speedup_total"] is None


def test_close_times_are_a_tie():
    res = load("mock_gpu_wins_large_image")
    res = res.model_copy(update={"gpu": res.gpu.model_copy(update={"total_time_ms": 130.0})})   # 131 / 130 = 1.008
    assert compute_metrics(res)["winner"] == "tie"


def test_noisy_run_is_flagged():
    res = load("mock_gpu_wins_large_image")
    res = res.model_copy(update={"cpu": res.cpu.model_copy(update={"total_time_std_ms": 40.0})})
    m = compute_metrics(res)
    assert any("CPU timing is noisy" in w for w in m["warnings"]) and m["winner"] == "gpu"


def test_no_results_at_all():
    res = load("mock_gpu_error")
    res = res.model_copy(update={"cpu": None, "gpu": None})
    m = compute_metrics(res)
    assert m["scenario"] == "no_data" and m["winner"] == "undetermined"


def test_stage_breakdown_rows_and_shares():
    b = get_stage_breakdown(load("mock_gpu_wins_large_image"))
    rows = {r["stage"]: r for r in b["stages"]}
    assert list(rows) == ["gaussian_blur", "sobel_gradients", "non_max_suppression", "hysteresis"]
    nms = rows["non_max_suppression"]
    assert (nms["cpu_ms"], nms["gpu_ms"], nms["speedup"]) == (100.0, 2.0, 50.0)
    assert nms["cpu_share_pct"] == pytest.approx(10000 / 131, abs=1e-3) and nms["runs_on_gpu"] is True
    assert rows["hysteresis"]["runs_on_gpu"] is False and rows["hysteresis"]["gpu_share_pct"] == pytest.approx(2000 / 29, abs=1e-3)
    assert b["gpu_transfer_ms"] == 3.0 and b["gpu_transfer_share_pct"] == pytest.approx(300 / 29, abs=1e-3)


def test_breakdown_with_no_gpu_has_empty_gpu_columns():
    b = get_stage_breakdown(load("mock_gpu_missing"))
    assert all(r["gpu_ms"] is None and r["speedup"] is None and r["cpu_ms"] is not None for r in b["stages"])
    assert b["gpu_transfer_ms"] is None


def test_correctness_status_variants():
    assert check_correctness_status(load("mock_gpu_wins_large_image"))["verdict"].startswith("PASSED")
    assert check_correctness_status(load("mock_gpu_missing"))["verdict"].startswith("NOT PERFORMED")


@pytest.mark.parametrize("name", ["mock_gpu_wins_large_image", "mock_cpu_wins_small_image", "mock_gpu_missing",
                                  "mock_correctness_failed", "mock_gpu_error"])
def test_tool_outputs_are_json_serialisable(name):
    """Tool results are sent to the LLM as JSON, so they must survive json.dumps."""
    res = load(name)
    for out in (compute_metrics(res), get_stage_breakdown(res), check_correctness_status(res)):
        json.dumps(out)
