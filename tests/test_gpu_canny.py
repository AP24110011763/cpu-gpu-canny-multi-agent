"""Tests for the GPU Canny tool and the correctness check.

Most tests run WITHOUT a GPU: the stage functions take an array module `xp`, so NumPy stands in
for CuPy and we can verify the maths against the CPU tool. The last test uses the real GPU and is
skipped automatically on a machine without CuPy/CUDA.
"""
import types

import cv2
import numpy as np
import pytest

import tools.gpu_canny as gc
from tools.canny_common import STAGE_KEYS, hysteresis
from tools.correctness import check_correctness
from tools.cpu_canny import canny_staged, run_cpu_canny


def make_image(path, size=160):
    img = np.full((size, size), 40, dtype=np.uint8)
    img[40:120, 40:120] = 220
    rng = np.random.default_rng(1)
    img = np.clip(img.astype(int) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    assert cv2.imwrite(str(path), img)
    return img


class FakeCupy(types.SimpleNamespace):
    """NumPy dressed up as the few CuPy names run_gpu_canny uses. Tests the bookkeeping only."""
    def __init__(self):
        sync = types.SimpleNamespace(synchronize=lambda: None)
        pool = types.SimpleNamespace(free_all_blocks=lambda: None)
        super().__init__(cuda=types.SimpleNamespace(Stream=types.SimpleNamespace(null=sync)),
                         asarray=np.asarray, asnumpy=np.asarray, get_default_memory_pool=lambda: pool)

    def __getattr__(self, name):
        return getattr(np, name)


def test_blur_matches_opencv_within_one_grey_level(tmp_path):
    img = make_image(tmp_path / "a.png")
    ours = gc.gaussian_blur_xp(np, img)
    ref = cv2.GaussianBlur(img, (5, 5), 1.4).astype(np.float32)
    assert np.abs(ours - ref).max() <= 1.0


def test_sobel_matches_opencv(tmp_path):
    img = make_image(tmp_path / "a.png")
    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
    gx, gy, mag = gc.sobel_gradients_xp(np, blurred.astype(np.float32))
    rx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    ry = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    assert np.allclose(gx, rx, atol=1e-3) and np.allclose(gy, ry, atol=1e-3)
    assert np.allclose(mag, np.abs(rx) + np.abs(ry), atol=1e-3)


def test_gpu_stage_code_gives_same_edges_as_cpu_path(tmp_path):
    """The whole GPU pipeline (run on NumPy here) must agree with the CPU pipeline."""
    img = make_image(tmp_path / "a.png")
    cpu_edges, cpu_mag, _ = canny_staged(img, 100, 200)
    blur = gc.gaussian_blur_xp(np, img)
    gx, gy, mag = gc.sobel_gradients_xp(np, blur)
    gpu_edges = hysteresis(gc.non_max_suppression_xp(np, gx, gy, mag), 100, 200)
    chk = check_correctness(cpu_edges, gpu_edges, cpu_mag, mag)
    assert chk.passed, chk.note
    assert chk.pixel_agreement_pct > 99.0 and (cpu_edges > 0).sum() > 200


def test_run_gpu_canny_bookkeeping_with_fake_cupy(tmp_path, monkeypatch):
    make_image(tmp_path / "a.png")
    monkeypatch.setattr(gc, "_load_cupy", lambda: FakeCupy())
    out = tmp_path / "o" / "gpu.png"
    res, edges, mag = gc.run_gpu_canny(str(tmp_path / "a.png"), 100, 200, repetitions=3, warmup_runs=1, out_path=str(out))
    assert res.error is None and res.device == "gpu"
    assert list(res.stage_times_ms) == STAGE_KEYS
    assert res.transfer_time_ms > 0 and res.kernel_time_ms > 0
    parts = res.transfer_time_ms + res.kernel_time_ms + res.stage_times_ms["hysteresis"]
    assert res.total_time_ms == pytest.approx(parts, rel=0.2)          # wall total ~ sum of the parts
    assert (res.repetitions, res.warmup_runs) == (3, 1)
    assert res.opencv_reference_time_ms is None                        # that field is CPU-side only
    assert out.exists() and edges.shape == mag.shape


def test_no_gpu_returns_error_not_exception(tmp_path, monkeypatch):
    make_image(tmp_path / "a.png")

    def boom():
        raise ImportError("No module named 'cupy'")
    monkeypatch.setattr(gc, "_load_cupy", boom)
    res, edges, mag = gc.run_gpu_canny(str(tmp_path / "a.png"))
    assert res.error and "GPU not usable" in res.error and edges is None


def test_missing_image_returns_error(tmp_path):
    res, _, _ = gc.run_gpu_canny(str(tmp_path / "ghost.png"))
    assert "not found" in res.error


def test_gpu_failure_midrun_is_reported(tmp_path, monkeypatch):
    make_image(tmp_path / "a.png")
    fake = FakeCupy()

    def bad_asarray(_):
        raise MemoryError("out of memory")
    fake.asarray = bad_asarray
    monkeypatch.setattr(gc, "_load_cupy", lambda: fake)
    res, _, _ = gc.run_gpu_canny(str(tmp_path / "a.png"), repetitions=1, warmup_runs=0)
    assert "GPU run failed" in res.error and "MemoryError" in res.error


# ---- correctness check -------------------------------------------------------------------------
def test_correctness_identical_maps_pass():
    e = np.zeros((50, 50), np.uint8); e[10:40, 25] = 255
    chk = check_correctness(e, e.copy(), e.astype(np.float32), e.astype(np.float32))
    assert chk.performed and chk.passed and chk.pixel_agreement_pct == 100.0
    assert chk.gradient_mean_abs_diff == 0.0


def test_correctness_all_black_output_fails_even_though_pixel_agreement_is_high():
    e = np.zeros((100, 100), np.uint8); e[10:90, 50] = 255      # ~0.8% edge pixels
    chk = check_correctness(e, np.zeros_like(e))
    assert chk.pixel_agreement_pct > 98.0 and chk.passed is False   # the loophole is closed


def test_correctness_tolerates_one_pixel_shift_but_not_three():
    e = np.zeros((100, 100), np.uint8); e[10:90, 50] = 255
    shifted1 = np.roll(e, 1, axis=1)
    shifted3 = np.roll(e, 3, axis=1)
    assert check_correctness(e, shifted1).passed is True
    assert check_correctness(e, shifted3).passed is False


def test_correctness_missing_or_mismatched_inputs():
    assert check_correctness(None, np.zeros((3, 3))).performed is False
    chk = check_correctness(np.zeros((3, 3)), np.zeros((4, 4)))
    assert chk.performed and chk.passed is False


# ---- real GPU (skipped when there is none) -----------------------------------------------------
def _gpu_ok():
    try:
        gc._load_cupy()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_ok(), reason="needs CuPy and a CUDA GPU")
def test_real_gpu_matches_cpu(tmp_path):
    make_image(tmp_path / "a.png", size=256)
    cpu, c_edges, c_mag = run_cpu_canny(str(tmp_path / "a.png"), 100, 200, repetitions=3, warmup_runs=1)
    gpu, g_edges, g_mag = gc.run_gpu_canny(str(tmp_path / "a.png"), 100, 200, repetitions=3, warmup_runs=1)
    assert gpu.error is None and gpu.transfer_time_ms > 0 and gpu.kernel_time_ms > 0
    chk = check_correctness(c_edges, g_edges, c_mag, g_mag)
    assert chk.passed, chk.note
