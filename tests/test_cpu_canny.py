import cv2
import numpy as np
import pytest

from tools.canny_common import STAGE_KEYS, hysteresis, save_png
from tools.cpu_canny import canny_staged, run_cpu_canny


def square_image(path, size=160):
    img = np.full((size, size), 40, dtype=np.uint8)
    img[40:120, 40:120] = 220            # bright square: clear edges along its border
    assert cv2.imwrite(str(path), img)
    return img


def test_hysteresis_keeps_connected_weak_and_drops_isolated_weak():
    m = np.zeros((7, 12), dtype=np.float32)
    m[3, 1:6] = 50         # weak run ...
    m[3, 6] = 250          # ... touching a strong pixel  -> kept
    m[1, 9] = 50           # isolated weak pixel           -> dropped
    out = hysteresis(m, low=40, high=200)
    assert out[3, 1:7].all() and out[1, 9] == 0


def test_hysteresis_nothing_when_no_weak_pixels():
    assert hysteresis(np.zeros((5, 5), np.float32), 40, 200).sum() == 0


def test_staged_canny_finds_square_border(tmp_path):
    img = square_image(tmp_path / "sq.png")
    edges, mag, t = canny_staged(img, 100, 200)
    assert set(t) == set(STAGE_KEYS)
    ys, xs = np.nonzero(edges)
    assert len(ys) > 200                                  # a real outline, not noise
    assert ys.min() >= 36 and ys.max() <= 123 and xs.min() >= 36 and xs.max() <= 123  # hugs the square
    assert edges[80, 80] == 0 and edges[10, 10] == 0      # nothing inside the square or far outside


def test_staged_matches_opencv_reference_closely(tmp_path):
    img = square_image(tmp_path / "sq.png")
    ours, _, _ = canny_staged(img, 100, 200)
    ref = cv2.Canny(img, 100, 200)
    agreement = ((ours > 0) == (ref > 0)).mean()
    assert agreement > 0.98


def test_run_cpu_canny_result_fields(tmp_path):
    square_image(tmp_path / "sq.png")
    out = tmp_path / "out" / "cpu_edges.png"
    res, edges, mag = run_cpu_canny(str(tmp_path / "sq.png"), 100, 200, repetitions=4, warmup_runs=1, out_path=str(out))
    assert res.error is None and res.device == "cpu"
    assert list(res.stage_times_ms) == STAGE_KEYS and all(v > 0 for v in res.stage_times_ms.values())
    assert res.total_time_ms == pytest.approx(sum(res.stage_times_ms.values()), rel=0.01)
    assert res.total_time_std_ms >= 0 and res.opencv_reference_time_ms > 0
    assert (res.repetitions, res.warmup_runs) == (4, 1)
    assert out.exists() and edges.shape == mag.shape
    assert res.transfer_time_ms is None and res.kernel_time_ms is None   # GPU-only fields stay empty


def test_missing_and_invalid_images_return_error_not_exception(tmp_path):
    res, edges, mag = run_cpu_canny(str(tmp_path / "ghost.png"))
    assert "not found" in res.error and edges is None
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    assert "not a valid image" in run_cpu_canny(str(bad))[0].error


def test_save_png_creates_folders(tmp_path):
    p = save_png(str(tmp_path / "a" / "b" / "x.png"), np.zeros((4, 4), np.uint8))
    assert (tmp_path / "a" / "b" / "x.png").exists() and p.endswith("x.png")
