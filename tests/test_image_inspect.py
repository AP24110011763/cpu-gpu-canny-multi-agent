import cv2
import numpy as np

from tools.image_inspect import inspect_image


def _write_png(path, w, h, channels=3):
    """Small, highly compressible test image (a gradient plus a bright box, so Canny has edges).
    Random noise would make each PNG ~1 MB; this is a few KB, which matters on a nearly full drive."""
    x = np.tile(np.linspace(0, 200, w, dtype=np.uint8), (h, 1))
    x[h // 4: h // 2, w // 4: w // 2] = 255
    img = x if channels == 1 else np.dstack([x, x[::-1], x])
    assert cv2.imwrite(str(path), img), f"could not write test image to {path}"


def _make_image(path, w, h, channels=3):
    _write_png(path, w, h, channels)


def test_small_image(tmp_path):
    p = tmp_path / "small.png"
    _make_image(p, 640, 480)
    r = inspect_image(str(p))
    assert r["ok"] and (r["width"], r["height"], r["channels"]) == (640, 480, 3)
    assert r["megapixels"] == 0.3072
    assert r["size_class"] == "small" and r["suggested_repetitions"] == 20


def test_medium_and_large(tmp_path):
    m, l = tmp_path / "m.png", tmp_path / "l.png"
    _make_image(m, 1280, 720)     # 0.92 MP
    _make_image(l, 2000, 1500, 1)  # 3.0 MP, grayscale
    assert inspect_image(str(m))["size_class"] == "medium"
    big = inspect_image(str(l))
    assert big["size_class"] == "large" and big["channels"] == 1 and big["suggested_repetitions"] == 5


def test_missing_file(tmp_path):
    r = inspect_image(str(tmp_path / "nope.png"))
    assert r["ok"] is False and "not found" in r["error"]


def test_not_an_image(tmp_path):
    p = tmp_path / "fake.png"
    p.write_text("this is not an image")
    r = inspect_image(str(p))
    assert r["ok"] is False and "not a valid image" in r["error"]


def test_unsupported_extension(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("x")
    assert inspect_image(str(p))["ok"] is False


def test_path_with_spaces(tmp_path):
    folder = tmp_path / "my folder"
    folder.mkdir()
    p = folder / "pic one.png"
    _make_image(p, 100, 100)
    assert inspect_image(str(p))["ok"]
