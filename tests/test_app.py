"""The dashboard page, driven by Streamlit's own test runner. No GPU, no Ollama (both are faked).
Skipped when Streamlit is not installed."""
import cv2
import json
import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import agents.executor as ex  # noqa: E402
import llm.client as llm_client  # noqa: E402
import test_orchestrator as to  # noqa: E402  (reuses its scripted agents and fake GPU)
from llm.fake_client import ScriptedLLM  # noqa: E402
from pathlib import Path  # noqa: E402
from schemas import ExecutionResults  # noqa: E402
from tools import dashboard_data as dd  # noqa: E402

APP = str(Path(__file__).parent.parent / "app.py")
FIX = Path(__file__).parent / "fixtures"
GOOD = {"gpu": {"ok": True, "gpu_name": "Fake GTX 1650", "driver_version": "1", "vram_mb": 4096, "cupy_version": "13", "problems": []},
        "ollama": {"reachable": True, "models": ["llama3.2:3b"], "model_installed": True, "error": None},
        "python": "3.12", "free_disk_gb": 50.0, "output_root": "x"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setenv("COA_OUTPUT_DIR", str(out))
    monkeypatch.setattr(dd, "system_status", lambda _out: GOOD)
    return out


def launch():
    at = AppTest.from_file(APP, default_timeout=90)
    return at.run()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def texts(at):
    return " ".join([m.value for m in at.info] + [m.value for m in at.error] + [m.value for m in at.success] +
                    [m.value for m in at.warning] + [m.value for m in at.markdown] + [c.value for c in at.caption])


def put_run(out, name, run_id, fixture="mock_gpu_wins_large_image"):
    res = ExecutionResults.model_validate_json((FIX / f"{fixture}.json").read_text(encoding="utf-8"))
    res = res.model_copy(update={"plan": res.plan.model_copy(update={"run_id": run_id})})
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(res.model_dump_json(), encoding="utf-8")


def test_page_opens_with_nothing_saved_and_no_image(env):
    at = launch()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Run the agents", "Quick benchmark", "Past runs", "Size sweep", "System and about"]
    assert button(at, "Run the three agents").disabled and "Choose an image in the sidebar first." in texts(at)
    assert "No saved runs yet" in texts(at) and "No sweep saved yet" in texts(at)


def test_missing_gpu_and_stopped_ollama_are_explained_and_block_the_buttons(env, monkeypatch):
    bad = {**GOOD, "gpu": {"ok": False, "gpu_name": None, "problems": ["CuPy is not installed in this environment"]},
           "ollama": {"reachable": False, "models": [], "model_installed": False, "error": "ConnectionError: start Ollama (run 'ollama serve') and try again"}}
    monkeypatch.setattr(dd, "system_status", lambda _out: bad)
    at = launch()
    assert not at.exception
    t = texts(at)
    assert "Ollama is not running" in t and "The GPU is not usable" in t and "CuPy is not installed" in t
    assert button(at, "Run the three agents").disabled and button(at, "Run the benchmark").disabled


def test_unusable_image_path_is_reported(env, tmp_path):
    at = launch()
    at.sidebar.text_input[0].set_value(str(tmp_path / "nope.png")).run()
    assert any("not found" in e.value for e in at.sidebar.error) and button(at, "Run the three agents").disabled


def test_past_runs_hide_mock_data_until_asked_and_label_it(env):
    put_run(env, "run_20261107_100000", "run_20261107_100000")
    put_run(env, "run_mock", "mock_gpu_wins")
    at = launch()
    assert not at.exception and len(at.selectbox) == 1
    shown = at.dataframe[0].value
    assert list(shown["run_id"]) == ["run_20261107_100000"]
    assert not any("MOCK" in e.value for e in at.error)
    at.checkbox[1].check().run()                                   # [0] = fixed-steps planner, [1] = show mock
    assert len(at.dataframe[0].value) == 2
    shown_rows = [r for r in dd.list_runs(str(env))[0]]
    at.selectbox[0].select(next(i for i, r in enumerate(shown_rows) if r["mock"])).run()
    assert not at.exception and any("MOCK" in e.value for e in at.error)


def test_saved_sweep_marks_enlarged_copies(env, tmp_path):
    img = tmp_path / "small.png"
    cv2.imwrite(str(img), np.full((50, 100, 3), 80, np.uint8))                     # 100 px wide
    base = dict(height=1, megapixels=0.1, cpu_ms=1.0, gpu_ms=1.0, speedup_kernel_only=2.0, transfer_pct=5.0, correct=True, noisy=False)
    rows = [dict(base, width=64, speedup_total=1.2), dict(base, width=256, speedup_total=3.0)]
    (env / "size_sweep").mkdir(parents=True)
    (env / "size_sweep" / "sweep.json").write_text(json.dumps({"image": str(img), "note": "plugged in", "rows": rows}), encoding="utf-8")
    at = launch()
    assert not at.exception
    table = next(d.value for d in at.dataframe if "size" in d.value.columns and "note" in d.value.columns)
    assert list(table["note"]) == ["", "enlarged copy"] and "plugged in" in texts(at)


def test_full_run_end_to_end_with_scripted_agents(env, tmp_path, monkeypatch):
    monkeypatch.setattr(ex.time, "sleep", lambda s: None)
    monkeypatch.setattr(ex, "check_gpu", lambda: {"ok": True, "gpu_name": "Fake GTX", "problems": []})

    def fake_run(path, low, high, reps, warm, out):
        res, edges, mag = to.run_cpu_canny(path, low, high, reps, warm, out)
        return res.model_copy(update={"device": "gpu", "transfer_time_ms": 1.0, "kernel_time_ms": 2.0, "opencv_reference_time_ms": None}), edges, mag
    monkeypatch.setattr(ex, "run_gpu_canny", fake_run)
    img = tmp_path / "sq.png"
    a = np.full((120, 120), 40, np.uint8)
    a[30:90, 30:90] = 220
    cv2.imwrite(str(img), a)
    # Agent 3's replies run out on purpose: the analyst must then fall back to its code template, and the run must still finish
    monkeypatch.setattr(llm_client, "OllamaClient", lambda *a, **k: ScriptedLLM(to.planner_script() + to.EXECUTOR_BOTH))

    at = launch()
    at.sidebar.text_input[0].set_value(str(img)).run()
    assert not button(at, "Run the three agents").disabled
    button(at, "Run the three agents").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Finished run_" in s.value for s in at.success)
    folders = [p for p in env.glob("run_*")]
    assert len(folders) == 1 and (folders[0] / "report.html").exists() and (folders[0] / "trace.json").exists()
    assert "built by code" in (folders[0] / "report.md").read_text(encoding="utf-8")
    assert any(m.label == "Speedup (total)" for m in at.metric)
    traced = json.loads((folders[0] / "trace.json").read_text(encoding="utf-8"))
    assert set(traced["stages"]) == {"agent1", "agent2", "agent3"}


def test_a_stopped_planner_shows_an_error_not_a_crash(env, tmp_path, monkeypatch):
    img = tmp_path / "sq.png"
    cv2.imwrite(str(img), np.full((40, 40), 90, np.uint8))
    monkeypatch.setattr(llm_client, "OllamaClient", lambda *a, **k: ScriptedLLM([]))   # the model "fails" immediately
    at = launch()
    at.sidebar.text_input[0].set_value(str(img)).run()
    button(at, "Run the three agents").click().run()
    assert not at.exception and any("ran out of scripted replies" in e.value or "Stopped" in e.value for e in at.error)


def test_quick_benchmark_shows_mean_median_and_spread(env, tmp_path, monkeypatch):
    import tools.repeat_benchmark as rb
    real = rb.repeat_benchmark

    def fake_gpu(path, low, high, reps, warm, out):
        res, edges, mag = to.run_cpu_canny(path, low, high, reps, warm, out)
        return res.model_copy(update={"device": "gpu", "total_time_ms": res.total_time_ms / 3, "transfer_time_ms": 0.1,
                                      "kernel_time_ms": 0.2, "opencv_reference_time_ms": None}), edges, mag
    monkeypatch.setattr(rb, "repeat_benchmark", lambda *a, say=print, **k: real(
        *a, say=say, gpu_fn=fake_gpu, gpu_check=lambda: {"ok": True, "gpu_name": "Fake", "problems": []}, sleep=lambda s: None, **k))
    img = tmp_path / "p.png"
    rng = np.random.default_rng(3)
    cv2.imwrite(str(img), cv2.GaussianBlur(rng.integers(0, 255, (60, 80), dtype=np.uint8), (0, 0), 2))
    at = launch()
    at.sidebar.text_input[0].set_value(str(img)).run()
    at.number_input[0].set_value(2)                                  # timed repetitions
    at.slider[0].set_value(3)                                        # rounds
    button(at, "Run the benchmark").click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Speedup (mean)" in labels and "Speedup (median)" in labels and "Spread between rounds" in labels
    assert len(list(env.glob("repeat_*/round_*/results.json"))) == 3
