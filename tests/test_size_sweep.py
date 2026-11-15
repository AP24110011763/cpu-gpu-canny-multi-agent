"""Size sweep tests. The GPU tool is replaced by a relabelled CPU run, so this needs no GPU."""
import csv
import json

import cv2
import numpy as np
import pytest

from tools import size_sweep as ss
from tools.cpu_canny import run_cpu_canny


@pytest.fixture
def img(tmp_path):
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.integers(0, 255, (200, 300, 3), dtype=np.uint8), (0, 0), 3)
    a[50:150, 80:220] = 230
    p = tmp_path / "pic.png"
    assert cv2.imwrite(str(p), a)
    return str(p)


def fake_gpu(path, low, high, reps, warm, out):
    res, edges, mag = run_cpu_canny(path, low, high, reps, warm, out)
    st = {k: v / 2 for k, v in res.stage_times_ms.items()}
    return res.model_copy(update={"device": "gpu", "stage_times_ms": st, "total_time_ms": res.total_time_ms / 2,
                                  "transfer_time_ms": 0.1, "kernel_time_ms": 0.2, "opencv_reference_time_ms": None}), edges, mag


def test_resize_keeps_aspect_ratio(img, tmp_path):
    out = ss.resize_to_width(img, 150, tmp_path / "r.png")
    h, w = cv2.imread(out).shape[:2]
    assert (w, h) == (150, 100)


def test_sweep_rows_use_the_shared_metrics(img, tmp_path):
    rows = ss.sweep(img, (64, 128), repetitions=2, warmup=1, out_dir=str(tmp_path / "o"), gpu_fn=fake_gpu, progress=lambda s: None)
    assert [r["width"] for r in rows] == [64, 128] and rows[0]["megapixels"] < rows[1]["megapixels"]
    for r in rows:
        assert r["speedup_total"] == pytest.approx(r["cpu_ms"] / r["gpu_ms"], rel=0.05) and r["correct"] is True
        assert r["transfer_pct"] is not None and "noisy" in r


def test_failed_gpu_row_is_skipped_not_crashed(img, tmp_path):
    def bad_gpu(path, low, high, reps, warm, out):
        r, e, m = fake_gpu(path, low, high, reps, warm, out)
        return r.model_copy(update={"error": "GPU run failed: MemoryError"}), e, m
    rows = ss.sweep(img, (64,), repetitions=2, warmup=1, out_dir=str(tmp_path / "o"), gpu_fn=bad_gpu, progress=lambda s: None)
    assert "MemoryError" in rows[0]["error"] and "skipped" in ss.format_table(rows)
    ss.write_outputs(rows, str(tmp_path / "o"), img, "")                       # no usable rows must still write files


def test_outputs_and_honest_finding(img, tmp_path):
    out = str(tmp_path / "o")
    rows = ss.sweep(img, (64, 128, 192), repetitions=2, warmup=1, out_dir=out, gpu_fn=fake_gpu, progress=lambda s: None)
    files = ss.write_outputs(rows, out, img, "plugged in, Best performance")
    data = json.loads(open(files["json"], encoding="utf-8").read())
    assert data["note"] == "plugged in, Best performance" and len(data["rows"]) == 3
    assert len(list(csv.DictReader(open(files["csv"], encoding="utf-8")))) == 3
    page = open(files["html"], encoding="utf-8").read()
    assert "<svg" in page and "plugged in, Best performance" in page and "resized for each row" in page


def test_finding_says_when_speedup_did_not_grow():
    base = dict(height=1, cpu_ms=1.0, gpu_ms=1.0, speedup_kernel_only=1.0, transfer_pct=1.0, correct=True, noisy=False)
    rows = [dict(base, width=1, megapixels=0.1, speedup_total=3.0), dict(base, width=2, megapixels=1.0, speedup_total=2.0)]
    import tempfile
    f = ss.write_outputs(rows, tempfile.mkdtemp(), "x", "")
    assert "did NOT grow" in f["finding"]


def _rows(*speedups):
    base = dict(height=1, cpu_ms=1.0, gpu_ms=1.0, speedup_kernel_only=1.0, transfer_pct=1.0, correct=True, noisy=False)
    return [dict(base, width=i + 1, megapixels=0.1 * 10 ** i, speedup_total=v) for i, v in enumerate(speedups)]


def test_finding_describes_a_plateau_and_does_not_invent_a_crossover():
    import tempfile
    f = ss.write_outputs(_rows(1.65, 3.19, 5.08, 5.05), tempfile.mkdtemp(), "x", "")["finding"]
    assert "levelled off" in f and "already faster at the smallest size" in f and "first beat" not in f


def test_finding_reports_a_real_crossover():
    import tempfile
    f = ss.write_outputs(_rows(0.5, 0.9, 1.4), tempfile.mkdtemp(), "x", "")["finding"]
    assert "first beat the CPU at 10.0 MP" in f and "it grew" in f
