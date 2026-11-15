"""Dashboard data layer: saved runs, sweep, uploads and machine status. No Streamlit, no GPU, no Ollama."""
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from schemas import ExecutionResults
from tools import dashboard_data as dd

FIX = Path(__file__).parent / "fixtures"


def put_run(root, folder, run_id, fixture="mock_gpu_wins_large_image", report=False, trace=False):
    res = ExecutionResults.model_validate_json((FIX / f"{fixture}.json").read_text(encoding="utf-8"))
    res = res.model_copy(update={"plan": res.plan.model_copy(update={"run_id": run_id})})
    d = Path(root) / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(res.model_dump_json(), encoding="utf-8")
    if report:
        (d / "report.html").write_text("<html></html>", encoding="utf-8")
    if trace:
        (d / "trace.json").write_text(json.dumps({"stages": {"agent1": {"seconds": 1.0}}}), encoding="utf-8")
    return d


def test_lists_full_runs_and_benchmark_rounds_and_marks_mock_ones(tmp_path):
    put_run(tmp_path, "run_20261107_100000", "run_20261107_100000", report=True)
    put_run(tmp_path, "repeat_x/round_1", "repeat_x_r1")
    put_run(tmp_path, "run_mock", "mock_gpu_wins")
    rows, bad = dd.list_runs(str(tmp_path))
    by = {r["run_id"]: r for r in rows}
    assert bad == [] and set(by) == {"run_20261107_100000", "repeat_x_r1", "mock_gpu_wins"}
    assert by["run_20261107_100000"]["kind"] == "full run" and by["run_20261107_100000"]["report_html"].endswith("report.html")
    assert by["repeat_x_r1"]["kind"] == "benchmark round" and by["repeat_x_r1"]["report_html"] is None
    assert by["mock_gpu_wins"]["mock"] is True and by["run_20261107_100000"]["mock"] is False
    assert by["run_20261107_100000"]["speedup"] == pytest.approx(131 / 29, abs=1e-3) and by["run_20261107_100000"]["winner"] == "gpu"


def test_unreadable_files_are_reported_not_fatal_and_missing_folder_is_empty(tmp_path):
    (tmp_path / "run_bad").mkdir()
    (tmp_path / "run_bad" / "results.json").write_text("{not json", encoding="utf-8")
    put_run(tmp_path, "run_ok", "run_ok")
    rows, bad = dd.list_runs(str(tmp_path))
    assert [r["run_id"] for r in rows] == ["run_ok"] and len(bad) == 1
    assert dd.list_runs(str(tmp_path / "nope")) == ([], [])


def test_newest_first_and_speedup_series_ignores_mock_and_missing_gpu(tmp_path):
    import os
    a = put_run(tmp_path, "run_a", "run_a")
    b = put_run(tmp_path, "run_b", "run_b", fixture="mock_cpu_wins_small_image")
    put_run(tmp_path, "run_c", "run_c", fixture="mock_gpu_missing")
    put_run(tmp_path, "run_m", "mock_m")
    os.utime(a / "results.json", (1_000_000, 1_000_000))
    os.utime(b / "results.json", (2_000_000, 2_000_000))
    rows, _ = dd.list_runs(str(tmp_path))
    assert [r["run_id"] for r in rows if r["run_id"] in ("run_a", "run_b")] == ["run_b", "run_a"]
    series = dd.speedup_series(rows)
    assert len(series) == 2 and all(s["speedup"] is not None for s in series)             # no mock, no GPU-missing run
    missing = next(r for r in rows if r["run_id"] == "run_c")
    assert missing["gpu_ms"] is None and missing["speedup"] is None and missing["scenario"] == "gpu_missing"


def test_noisy_runs_are_flagged(tmp_path):
    res = ExecutionResults.model_validate_json((FIX / "mock_gpu_wins_large_image.json").read_text(encoding="utf-8"))
    res = res.model_copy(update={"cpu": res.cpu.model_copy(update={"total_time_std_ms": 50.0})})
    d = tmp_path / "run_n"
    d.mkdir()
    (d / "results.json").write_text(res.model_dump_json(), encoding="utf-8")
    assert dd.list_runs(str(tmp_path))[0][0]["noisy"] is True


def test_load_run_returns_results_metrics_and_optional_trace(tmp_path):
    put_run(tmp_path, "run_t", "run_t", trace=True)
    put_run(tmp_path, "run_u", "run_u")
    rows = {r["run_id"]: r for r in dd.list_runs(str(tmp_path))[0]}
    t, u = dd.load_run(rows["run_t"]), dd.load_run(rows["run_u"])
    assert t["metrics"]["winner"] == "gpu" and t["trace"]["stages"]["agent1"]["seconds"] == 1.0 and u["trace"] is None
    (tmp_path / "run_u" / "trace.json").write_text("garbage", encoding="utf-8")
    assert dd.load_run(rows["run_u"])["trace"] is None
    assert "MOCK" not in dd.run_label(rows["run_t"]) and "gpu" in dd.run_label(rows["run_t"])


def test_sweep_loading_and_enlarged_copy_labels(tmp_path):
    assert dd.load_sweep(str(tmp_path)) is None
    (tmp_path / "size_sweep").mkdir()
    base = dict(height=1, megapixels=0.1, cpu_ms=1.0, gpu_ms=1.0, speedup_kernel_only=2.0, transfer_pct=5.0, correct=True)
    rows = [dict(base, width=256, speedup_total=1.6, noisy=True), dict(base, width=1024, speedup_total=4.0, noisy=False),
            {"width": 2048, "height": 9, "megapixels": 6.0, "error": "MemoryError"}]
    (tmp_path / "size_sweep" / "sweep.json").write_text(json.dumps({"image": "x", "note": "n", "rows": rows}), encoding="utf-8")
    (tmp_path / "size_sweep" / "sweep.html").write_text("<html/>", encoding="utf-8")
    data = dd.load_sweep(str(tmp_path))
    assert data["note"] == "n" and data["html"].endswith("sweep.html")
    t = dd.sweep_table(data["rows"], original_width=474)
    assert t[0]["note"] == "noisy, repeat" and t[1]["note"] == "enlarged copy" and "skipped" in t[2]["note"]
    (tmp_path / "size_sweep" / "sweep.json").write_text("nope", encoding="utf-8")
    assert dd.load_sweep(str(tmp_path)) is None


def png_bytes():
    ok, buf = cv2.imencode(".png", np.full((20, 30, 3), 90, np.uint8))
    return buf.tobytes()


def test_upload_is_saved_safely(tmp_path):
    p = dd.save_upload(png_bytes(), "../../evil name!.PNG", str(tmp_path / "up"))
    assert Path(p).parent == tmp_path / "up" and Path(p).name == "evil_name.png" and Path(p).exists()
    assert dd.save_upload(png_bytes(), "../../evil name!.PNG", str(tmp_path / "up")) == p          # same bytes, same file
    other = np.full((20, 30, 3), 10, np.uint8)
    p2 = dd.save_upload(cv2.imencode(".png", other)[1].tobytes(), "evil name!.png", str(tmp_path / "up"))
    assert Path(p2).name == "evil_name_2.png"                                                       # different bytes never overwrite


def test_bad_uploads_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        dd.save_upload(b"x", "a.exe", str(tmp_path))
    with pytest.raises(ValueError, match="not a valid image"):
        dd.save_upload(b"this is not a picture", "a.png", str(tmp_path))
    with pytest.raises(ValueError, match="larger than"):
        dd.save_upload(b"0" * (dd.MAX_UPLOAD_MB * 1024 * 1024 + 1), "a.png", str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_image_summary_reuses_agent_1s_tool(tmp_path):
    p = dd.save_upload(png_bytes(), "a.png", str(tmp_path))
    s = dd.image_summary(p)
    assert s["ok"] and s["width"] == 30 and s["height"] == 20
    assert dd.image_summary("")["ok"] is False and dd.image_summary(str(tmp_path / "missing.png"))["ok"] is False


def test_ollama_status_variants(monkeypatch):
    fake = types.ModuleType("ollama")
    fake.Client = lambda: types.SimpleNamespace(list=lambda: {"models": [{"model": "llama3.2:3b"}, {"model": "other:1b"}]})
    monkeypatch.setitem(sys.modules, "ollama", fake)
    s = dd.ollama_status()
    assert s["reachable"] and s["model_installed"] and "other:1b" in s["models"]
    assert dd.ollama_status("missing:1b")["model_installed"] is False
    obj = types.SimpleNamespace(models=[types.SimpleNamespace(model="llama3.2:3b")])           # newer client returns objects
    fake.Client = lambda: types.SimpleNamespace(list=lambda: obj)
    assert dd.ollama_status()["model_installed"] is True

    def down():
        raise ConnectionError("refused")
    fake.Client = lambda: types.SimpleNamespace(list=down)
    s = dd.ollama_status()
    assert s["reachable"] is False and "ollama serve" in s["error"]
    monkeypatch.setitem(sys.modules, "ollama", None)                                             # package missing
    assert "not installed" in dd.ollama_status()["error"]


def test_system_status_survives_a_crashing_gpu_check(tmp_path):
    def boom():
        raise RuntimeError("driver exploded")
    s = dd.system_status(str(tmp_path / "out"), gpu_check=boom)
    assert s["gpu"]["ok"] is False and "driver exploded" in s["gpu"]["problems"][0]
    assert s["free_disk_gb"] is not None and (tmp_path / "out").exists()
    ok = dd.system_status(str(tmp_path), gpu_check=lambda: {"ok": True, "gpu_name": "GTX 1650", "problems": []})
    assert ok["gpu"]["gpu_name"] == "GTX 1650" and len(dd.FAIR_TIMING_CHECKLIST) == 4
