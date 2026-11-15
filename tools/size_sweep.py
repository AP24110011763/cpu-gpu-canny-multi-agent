"""Size sweep: does the GPU speedup grow with image size?

The same picture is resized to several widths and the SAME CPU and GPU tools are timed on each size.
All metrics come from tools/metrics.py (the same calculator Agent 3 uses), so definitions match the report.
No language model is involved: this is a pure measurement script.

Honest limits:
  * Resizing changes the picture. Enlarging a small image makes it smoother, so it has fewer edges than a
    genuinely large photo, and hysteresis time depends on how many edges there are.
  * Every row is only as reliable as its std. Rows whose std is above NOISY_PCT of the mean are flagged.
"""
from __future__ import annotations

import csv
import html
import json
import time
from pathlib import Path

import cv2

from schemas import CorrectnessCheck, ExecutionPlan, ExecutionResults, ImageInfo
from tools.canny_common import load_gray
from tools.correctness import check_correctness
from tools.cpu_canny import run_cpu_canny
from tools.gpu_canny import run_gpu_canny
from tools.metrics import NOISY_PCT, compute_metrics

DEFAULT_WIDTHS = (256, 512, 1024, 2048)


def resize_to_width(image_path: str, width: int, out_path: Path) -> str:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {image_path}")
    h, w = img.shape[:2]
    new_h = max(1, round(h * width / w))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_CUBIC
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.resize(img, (width, new_h), interpolation=interp))
    return str(out_path)


def sweep(image_path: str, widths=DEFAULT_WIDTHS, low: int = 100, high: int = 200, repetitions: int = 10,
          warmup: int = 3, out_dir: str = "outputs/size_sweep", cpu_fn=run_cpu_canny, gpu_fn=run_gpu_canny,
          progress=print) -> list[dict]:
    out = Path(out_dir)
    rows: list[dict] = []
    for width in widths:
        small = resize_to_width(image_path, width, out / f"resized_{width}.png")
        gray = load_gray(small)
        h, w = gray.shape
        progress(f"width {width}: {w}x{h} ({w * h / 1e6:.3f} MP) ...")
        cpu, c_edges, c_mag = cpu_fn(small, low, high, repetitions, warmup, str(out / f"cpu_{width}.png"))
        gpu, g_edges, g_mag = gpu_fn(small, low, high, repetitions, warmup, str(out / f"gpu_{width}.png"))
        if cpu.error or gpu.error:
            rows.append({"width": w, "height": h, "megapixels": round(w * h / 1e6, 4),
                         "error": cpu.error or gpu.error})
            progress(f"  skipped: {cpu.error or gpu.error}")
            continue
        corr = check_correctness(c_edges, g_edges, c_mag, g_mag)
        res = ExecutionResults(
            plan=ExecutionPlan(task="canny_edge_detection", image_path=small, run_id=f"sweep_{width}",
                               low_threshold=low, high_threshold=high, repetitions=repetitions, warmup_runs=warmup),
            image_info=ImageInfo(width=w, height=h, channels=3, megapixels=round(w * h / 1e6, 4)),
            gpu_name=None, cpu=cpu, gpu=gpu, correctness=corr)
        m = compute_metrics(res)
        rows.append({
            "width": w, "height": h, "megapixels": m["image_megapixels"],
            "cpu_ms": m["cpu_total_ms"], "cpu_std_pct": m["cpu_std_pct_of_mean"],
            "gpu_ms": m["gpu_total_ms"], "gpu_std_pct": m["gpu_std_pct_of_mean"],
            "speedup_total": m["speedup_total"], "speedup_kernel_only": m["speedup_kernel_only"],
            "transfer_pct": m["transfer_overhead_pct"], "gpu_hysteresis_pct": m["gpu_hysteresis_share_pct"],
            "opencv_ms": m["opencv_reference_ms"], "winner": m["winner"], "correct": corr.passed,
            "noisy": bool((m["cpu_std_pct_of_mean"] or 0) > NOISY_PCT or (m["gpu_std_pct_of_mean"] or 0) > NOISY_PCT),
        })
        progress(f"  CPU {m['cpu_total_ms']} ms | GPU {m['gpu_total_ms']} ms | speedup {m['speedup_total']}x"
                 + (" | NOISY" if rows[-1]["noisy"] else ""))
    return rows


def format_table(rows: list[dict]) -> str:
    head = f"{'size':>11}{'MP':>8}{'CPU ms':>10}{'GPU ms':>10}{'speedup':>9}{'kernels':>9}{'transfer':>9}{'ok':>5}"
    lines = [head]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['width']}x{r['height']:<5}  skipped: {r['error']}")
            continue
        flag = "  <- noisy, repeat" if r["noisy"] else ""
        lines.append(f"{str(r['width']) + 'x' + str(r['height']):>11}{r['megapixels']:>8.3f}{r['cpu_ms']:>10.2f}{r['gpu_ms']:>10.2f}"
                     f"{r['speedup_total']:>8.2f}x{r['speedup_kernel_only']:>8.2f}x{r['transfer_pct']:>8.1f}%"
                     f"{'yes' if r['correct'] else 'NO':>5}{flag}")
    return "\n".join(lines)


def _chart_svg(rows: list[dict]) -> str:
    pts = [r for r in rows if "error" not in r]
    if len(pts) < 2:
        return ""
    W, H, L, B, T, R = 560, 300, 54, 46, 20, 20
    ymax = max(max(r["speedup_total"] for r in pts), 1.0) * 1.15
    xs = [L + i * (W - L - R) / (len(pts) - 1) for i in range(len(pts))]
    y = lambda v: H - B - v / ymax * (H - B - T)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Speedup versus image size" style="max-width:100%">',
             f'<line x1="{L}" y1="{y(1):.1f}" x2="{W - R}" y2="{y(1):.1f}" stroke="#c0392b" stroke-dasharray="5 4"/>',
             f'<text x="{W - R}" y="{y(1) - 5:.1f}" text-anchor="end" font-size="11" fill="#c0392b">1x = CPU and GPU equal</text>',
             f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="#888"/><line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#888"/>']
    for t in range(0, int(ymax) + 1):
        parts.append(f'<text x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end" font-size="11">{t}x</text>')
    parts.append(f'<polyline fill="none" stroke="#2a7fd4" stroke-width="2.5" points="' +
                 " ".join(f"{x:.1f},{y(r['speedup_total']):.1f}" for x, r in zip(xs, pts)) + '"/>')
    for x, r in zip(xs, pts):
        parts.append(f'<circle cx="{x:.1f}" cy="{y(r["speedup_total"]):.1f}" r="4.5" fill="#2a7fd4"/>')
        first = x == xs[0]
        parts.append(f'<text x="{x + (8 if first else 0):.1f}" y="{y(r["speedup_total"]) - 9:.1f}" text-anchor="{"start" if first else "middle"}" font-size="11">{r["speedup_total"]:.2f}x</text>')
        parts.append(f'<text x="{x:.1f}" y="{H - B + 16}" text-anchor="middle" font-size="11">{r["megapixels"]:.2f} MP</text>')
    parts.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle" font-size="11">image size (megapixels, equal spacing)</text></svg>')
    return "".join(parts)


def write_outputs(rows: list[dict], out_dir: str, image_path: str, note: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    good = [r for r in rows if "error" not in r]
    fields = list(good[0]) if good else ["width", "height", "megapixels", "error"]
    with open(out / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "sweep.json").write_text(json.dumps({"image": image_path, "note": note, "rows": rows}, indent=2), encoding="utf-8")
    if not good:
        finding = "No usable rows."
    else:
        first, last = good[0], good[-1]
        sp = [r["speedup_total"] for r in good]
        finding = (f"Speedup went from {first['speedup_total']}x at {first['megapixels']} MP to {last['speedup_total']}x at "
                   f"{last['megapixels']} MP")
        if len(good) >= 3 and abs(sp[-1] - sp[-2]) / sp[-2] <= 0.05 and sp[-1] > sp[0]:
            finding += f", then levelled off at about {last['speedup_total']}x (it grew with size, then stopped growing)."
        elif last["speedup_total"] > first["speedup_total"]:
            finding += " (it grew with size)."
        else:
            finding += " (it did NOT grow with size on this run)."
        if first["speedup_total"] > 1:
            finding += (f" The GPU was already faster at the smallest size tested ({first['megapixels']} MP), "
                        "so the crossover point, if there is one, is below that and was not measured.")
        else:
            cross = next((r for r in good if r["speedup_total"] > 1), None)
            finding += (f" The GPU first beat the CPU at {cross['megapixels']} MP." if cross
                        else " The GPU never beat the CPU at these sizes.")
    noisy = [r for r in good if r["noisy"]]
    page = (f"<!doctype html><meta charset='utf-8'><title>Size sweep</title><body style='font-family:sans-serif;max-width:760px;margin:16px auto;padding:0 16px'>"
            f"<h1>Does the GPU speedup grow with image size?</h1><p>{html.escape(finding)}</p>{_chart_svg(rows)}"
            f"<pre style='overflow-x:auto'>{html.escape(format_table(rows))}</pre>"
            f"<p><b>Conditions:</b> {html.escape(note or 'not recorded')}</p>"
            + (f"<p><b>Warning:</b> {len(noisy)} row(s) are noisy (std above {NOISY_PCT:.0f}% of the mean). Repeat before quoting them.</p>" if noisy else "")
            + "<p>The picture was resized for each row, so edge density changes too. Only hysteresis time depends on it.</p></body>")
    (out / "sweep.html").write_text(page, encoding="utf-8")
    return {"csv": str(out / "sweep.csv"), "json": str(out / "sweep.json"), "html": str(out / "sweep.html"), "finding": finding}
