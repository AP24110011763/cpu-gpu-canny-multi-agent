import pytest
from pathlib import Path

from schemas import ExecutionResults
from tools.aggregate import aggregate, format_aggregate, latest_result_paths

FIX = Path(__file__).parent / "fixtures"


def load(name="mock_gpu_wins_large_image"):
    return ExecutionResults.model_validate_json((FIX / f"{name}.json").read_text(encoding="utf-8"))


def real(run_id, gpu_ms):
    r = load()
    g = r.gpu.model_copy(update={"total_time_ms": gpu_ms})
    return r.model_copy(update={"gpu": g, "plan": r.plan.model_copy(update={"run_id": run_id})})


def test_mean_and_std_across_runs_match_hand_calculation():
    agg = aggregate([real("run_1", 29.0), real("run_2", 26.2), real("run_3", 31.8)])
    s = agg["stats"]["gpu_total_ms"]
    assert s["mean"] == pytest.approx(29.0, abs=1e-3) and s["std"] == pytest.approx(2.8, abs=1e-3) and s["n"] == 3
    assert agg["stats"]["cpu_total_ms"]["std"] == 0.0 and agg["warnings"] == []


def test_unstable_speedup_is_flagged():
    agg = aggregate([real("run_1", 20.0), real("run_2", 40.0), real("run_3", 29.0)])
    assert any("varies" in w for w in agg["warnings"])


def test_mock_few_runs_and_mixed_runs_are_warned_about():
    agg = aggregate([load()])
    text = " ".join(agg["warnings"])
    assert "MOCK" in text and "at least 3" in text
    mixed = aggregate([real("run_1", 29.0), load("mock_gpu_missing"), real("run_3", 29.0)])
    assert any("both a CPU and a GPU" in w for w in mixed["warnings"])
    assert mixed["stats"]["gpu_total_ms"]["n"] == 2


def test_different_settings_are_not_averaged_silently():
    other = real("run_2", 29.0)
    other = other.model_copy(update={"plan": other.plan.model_copy(update={"repetitions": 5})})
    assert any("different images or settings" in w for w in aggregate([real("run_1", 29.0), other])["warnings"])


def test_empty_input_and_formatting_and_latest(tmp_path):
    with pytest.raises(ValueError):
        aggregate([])
    for n in ("run_20261107_100000", "run_20261107_110000", "run_20261107_120000", "compare_demo"):
        (tmp_path / n).mkdir()
        (tmp_path / n / "results.json").write_text("{}")
    got = latest_result_paths(str(tmp_path), 2)
    assert [Path(p).parent.name for p in got] == ["run_20261107_110000", "run_20261107_120000"]
    assert "speedup_total" in format_aggregate(aggregate([real("a", 29.0), real("b", 30.0)]), "plugged in")
