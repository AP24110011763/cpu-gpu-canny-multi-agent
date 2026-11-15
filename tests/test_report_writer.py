"""Report writer tests: all five scenarios, real images, HTML safety, and mock labelling."""
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

import tools.report_writer as rw
from schemas import ExecutionResults
from tools.report_writer import open_in_browser, write_report

FIXTURES = Path(__file__).parent / "fixtures"
ALL = ["mock_gpu_wins_large_image", "mock_cpu_wins_small_image", "mock_gpu_missing",
       "mock_correctness_failed", "mock_gpu_error"]


def load(name) -> ExecutionResults:
    return ExecutionResults.model_validate_json((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ALL)
def test_every_scenario_produces_both_files(name, tmp_path):
    out = write_report(load(name), "Some explanation.", str(tmp_path))
    md, page = Path(out["report_md"]), Path(out["report_html"])
    assert md.exists() and page.exists() and md.parent.name == load(name).plan.run_id
    assert "## Key numbers" in md.read_text(encoding="utf-8") and "<h2>Key numbers</h2>" in page.read_text(encoding="utf-8")


def test_numbers_in_the_report_are_the_calculated_ones(tmp_path):
    out = write_report(load("mock_gpu_wins_large_image"), "x", str(tmp_path))
    md = Path(out["report_md"]).read_text(encoding="utf-8")
    assert "GPU is faster" in md and "4.52x" in md and "18.50x" in md          # 131/29 and 111/6
    assert "| non_max_suppression | GPU | 100.000 | 2.000 | 50.00x |" in md
    assert "cv2.Canny 7.25x faster" in md                                        # 29 / 4: the honest finding is shown, not hidden


def test_cpu_win_and_failed_correctness_are_stated_plainly(tmp_path):
    small = Path(write_report(load("mock_cpu_wins_small_image"), "x", str(tmp_path))["report_md"]).read_text(encoding="utf-8")
    assert "CPU is faster" in small
    bad = Path(write_report(load("mock_correctness_failed"), "x", str(tmp_path))["report_md"]).read_text(encoding="utf-8")
    assert "No winner declared" in bad and "FAILED" in bad and "Correctness FAILED" in bad


def test_missing_gpu_report_has_no_gpu_numbers_but_does_not_crash(tmp_path):
    out = write_report(load("mock_gpu_missing"), "x", str(tmp_path))
    page = Path(out["report_html"]).read_text(encoding="utf-8")
    assert "Only the CPU produced results" in page and "GPU total:" not in page
    assert "Where the GPU path spends its time" not in page                        # no GPU chart without GPU data


def test_mock_reports_are_loudly_labelled_as_mock(tmp_path):
    out = write_report(load("mock_gpu_wins_large_image"), "x", str(tmp_path))
    for f in (out["report_md"], out["report_html"]):
        assert "DEVELOPMENT-ONLY MOCK DATA" in Path(f).read_text(encoding="utf-8")


def test_real_looking_run_id_gets_no_mock_banner(tmp_path):
    res = load("mock_gpu_wins_large_image")
    res = res.model_copy(update={"plan": res.plan.model_copy(update={"run_id": "run_20261106_120000"})})
    out = write_report(res, "x", str(tmp_path))
    assert "DEVELOPMENT-ONLY" not in Path(out["report_md"]).read_text(encoding="utf-8")


def test_images_are_embedded_when_they_exist(tmp_path):
    img = np.full((60, 80), 40, np.uint8); img[10:50, 20:60] = 220
    for n in ("orig.png", "cpu.png", "gpu.png"):
        assert cv2.imwrite(str(tmp_path / n), img)
    res = load("mock_gpu_wins_large_image")
    res = res.model_copy(update={
        "plan": res.plan.model_copy(update={"image_path": str(tmp_path / "orig.png"), "run_id": "run_img"}),
        "cpu": res.cpu.model_copy(update={"output_path": str(tmp_path / "cpu.png")}),
        "gpu": res.gpu.model_copy(update={"output_path": str(tmp_path / "gpu.png")})})
    out = write_report(res, "x", str(tmp_path / "o"))
    page = Path(out["report_html"]).read_text(encoding="utf-8")
    assert page.count("data:image/png;base64,") == 3                                # one page, nothing to search for
    folder = Path(out["folder"])
    assert all((folder / f).exists() for f in ("preview_original.png", "preview_cpu_edges.png", "preview_gpu_edges.png"))
    assert "![CPU edges](preview_cpu_edges.png)" in Path(out["report_md"]).read_text(encoding="utf-8")


def test_big_images_are_scaled_down(tmp_path):
    cv2.imwrite(str(tmp_path / "big.png"), np.zeros((900, 1600), np.uint8))
    assert rw._save_preview(str(tmp_path / "big.png"), tmp_path / "p.png")
    assert cv2.imread(str(tmp_path / "p.png")).shape[1] == rw.PREVIEW_WIDTH


def test_llm_text_cannot_inject_html(tmp_path):
    evil = "Nice.\n<script>alert(1)</script>\n- **bold** point <b>x</b>"
    page = Path(write_report(load("mock_gpu_wins_large_image"), evil, str(tmp_path))["report_html"]).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert "<strong>bold</strong>" in page and "<ul>" in page


def test_explanation_is_labelled_as_llm_text(tmp_path):
    md = Path(write_report(load("mock_gpu_wins_large_image"), "My text.", str(tmp_path))["report_md"]).read_text(encoding="utf-8")
    assert "did not calculate them" in md and "My text." in md


def test_charts_are_inline_svg(tmp_path):
    page = Path(write_report(load("mock_gpu_wins_large_image"), "x", str(tmp_path))["report_html"]).read_text(encoding="utf-8")
    assert page.count("<svg") == 2 and "PCIe transfer" in page and "http" not in re.sub(r"https?://www\.w3\.org\S*", "", page).replace("http-equiv", "")


def test_open_in_browser_never_raises(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(rw.webbrowser, "open", lambda uri: seen.append(uri) or True)
    assert open_in_browser(str(tmp_path / "r.html")) is True and seen[0].startswith("file:")
    monkeypatch.setattr(rw.webbrowser, "open", lambda uri: (_ for _ in ()).throw(RuntimeError("no browser")))
    assert open_in_browser(str(tmp_path / "r.html")) is False
