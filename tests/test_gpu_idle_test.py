import importlib.util
from pathlib import Path

from schemas import DeviceResult

spec = importlib.util.spec_from_file_location("gpu_idle_test", Path(__file__).parent.parent / "scripts" / "gpu_idle_test.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fake_run(ms=12.0, error=None):
    def run(image, low, high, reps, warm, out):
        return DeviceResult(device="gpu", stage_times_ms={}, total_time_ms=ms, total_time_std_ms=0.5, repetitions=reps,
                            warmup_runs=warm, error=error), None, None
    return run


def test_probe_visits_every_combination_and_waits_each_time():
    slept, lines = [], []
    rows = mod.probe("x.png", [0, 3], [3, 40], 2, 20, sleep=slept.append, run=fake_run(), clock=lambda: "300 MHz, 1650 MHz, P8", say=lines.append)
    assert len(rows) == 8 and slept == [0, 0, 3, 3] * 2
    assert rows[0]["clock"].endswith("P8") and "idle    3 s" in lines[2]


def test_probe_stops_on_a_gpu_error_and_works_without_nvidia_smi():
    lines = []
    rows = mod.probe("x.png", [0], [3], 3, 20, sleep=lambda s: None, run=fake_run(error="GPU run failed: x"), clock=lambda: "", say=lines.append)
    assert rows == [] and "GPU error" in lines[0]
    assert mod.probe("x.png", [0], [3], 1, 20, sleep=lambda s: None, run=fake_run(), clock=lambda: "", say=lambda s: None)[0]["clock"] == ""
