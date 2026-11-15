import cv2
import numpy as np
import pytest

from tools.cpu_canny import run_cpu_canny
from tools.repeat_benchmark import repeat_benchmark


@pytest.fixture
def img(tmp_path):
    rng = np.random.default_rng(1)
    a = cv2.GaussianBlur(rng.integers(0, 255, (80, 120, 3), dtype=np.uint8), (0, 0), 2)
    a[20:60, 30:90] = 235
    p = tmp_path / "p.png"
    cv2.imwrite(str(p), a)
    return str(p)


def fake_gpu(path, low, high, reps, warm, out):
    res, edges, mag = run_cpu_canny(path, low, high, reps, warm, out)
    return res.model_copy(update={"device": "gpu", "total_time_ms": res.total_time_ms / 3, "transfer_time_ms": 0.1,
                                  "kernel_time_ms": 0.2, "opencv_reference_time_ms": None}), edges, mag


OK = lambda: {"ok": True, "gpu_name": "Fake GPU", "problems": []}  # noqa: E731


def test_every_round_is_saved_and_aggregated_with_a_gap_before_each_run(img, tmp_path):
    slept, lines = [], []
    runs, agg, folder = repeat_benchmark(img, rounds=3, repetitions=2, warmup=1, out_root=str(tmp_path / "o"), tag="t",
                                         gpu_fn=fake_gpu, gpu_check=OK, sleep=slept.append, say=lines.append)
    assert len(runs) == 3 and agg["n_runs"] == 3 and slept == [3.0] * 6 and len(lines) == 3
    assert all((folder / f"round_{k}" / "results.json").exists() for k in (1, 2, 3))
    assert agg["stats"]["speedup_total"]["median"] > 1 and "median" in agg["stats"]["cpu_total_ms"]
    assert runs[0].gpu_name == "Fake GPU" and "no language model" in runs[0].agent2_notes
    assert not any("MOCK" in w for w in agg["warnings"])


def test_unusable_gpu_or_failed_run_stops_with_a_clear_message(img, tmp_path):
    with pytest.raises(RuntimeError, match="GPU not usable"):
        repeat_benchmark(img, rounds=1, out_root=str(tmp_path), gpu_check=lambda: {"ok": False, "problems": ["no CuPy"]})

    def bad(path, low, high, reps, warm, out):
        r, e, m = fake_gpu(path, low, high, reps, warm, out)
        return r.model_copy(update={"error": "GPU run failed: x"}), e, m
    with pytest.raises(RuntimeError, match="GPU run failed"):
        repeat_benchmark(img, rounds=1, repetitions=1, warmup=0, out_root=str(tmp_path), gpu_fn=bad, gpu_check=OK, sleep=lambda s: None)
