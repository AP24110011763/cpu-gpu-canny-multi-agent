"""Agent 3's report writer: turns results + explanation into outputs/<run_id>/report.md and report.html.

Plain code, no LLM. Every NUMBER in the report is computed here from the results by tools/metrics.py.
The only LLM-written part is the explanation text that is passed in, and it is labelled as such.

  report.md    Markdown (this is AnalysisReport.report_path). Images are linked by relative name.
  report.html  ONE self-contained page: images embedded, two charts drawn as inline SVG, nothing to install.
  preview_*.png  down-scaled copies of the original and the two edge images, so the report folder is complete.

open_in_browser(path) shows the page so nobody has to go looking for the images.
"""
from __future__ import annotations

import base64
import html
import re
import webbrowser
from pathlib import Path

import cv2
import numpy as np

from schemas import ExecutionResults
from tools.canny_common import save_png
from tools.metrics import STAGES, check_correctness_status, compute_metrics, get_stage_breakdown
from tools.run_trace import AGENTS, ORDER

PREVIEW_WIDTH = 720
CPU_COLOR, GPU_COLOR = "#3b6fb6", "#2e9e6b"
SEGMENT_COLORS = {"PCIe transfer": "#d9822b", "GPU kernels": GPU_COLOR, "Hysteresis (CPU)": "#8a4fbf", "Other": "#9aa0a6"}
WINNER_TEXT = {"gpu": "GPU is faster", "cpu": "CPU is faster", "tie": "Tie (within 5%)",
               "undetermined": "No winner declared"}
EXPLANATION_NOTE = ("This text explains the numbers in this report; it did not calculate them. "
                    "The last line says who wrote it: Agent 3's language model or a code template.")


# ------------------------------------------------------------------ small helpers
def _f(x, unit: str = "", digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}{unit}"


def _load_image(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def _save_preview(src: str | None, dest: Path) -> bool:
    img = _load_image(src)
    if img is None:
        return False
    h, w = img.shape[:2]
    if w > PREVIEW_WIDTH:
        img = cv2.resize(img, (PREVIEW_WIDTH, max(1, round(h * PREVIEW_WIDTH / w))), interpolation=cv2.INTER_AREA)
    save_png(str(dest), img)
    return True


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _md_to_html(text: str) -> str:
    """A deliberately tiny Markdown subset (headings, bullets, **bold**). Text is escaped FIRST, so
    nothing the LLM writes can inject HTML."""
    out, in_list = [], False
    for raw in html.escape(text.strip()).splitlines():
        line = raw.strip()
        bullet = line[2:] if line.startswith(("- ", "* ")) else None
        if in_list and bullet is None:
            out.append("</ul>")
            in_list = False
        if bullet is not None:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_bold(bullet)}</li>")
        elif line.startswith("#"):
            out.append(f"<h3>{_bold(line.lstrip('#').strip())}</h3>")
        elif line:
            out.append(f"<p>{_bold(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _bold(s: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)


# ------------------------------------------------------------------ charts (inline SVG, no libraries)
def _stage_chart(breakdown: dict) -> str:
    rows = breakdown["stages"]
    vals = [v for r in rows for v in (r["cpu_ms"], r["gpu_ms"]) if v]
    if not vals:
        return ""
    top, left, bar_w, h = max(vals), 150, 330, 24
    parts = [f'<svg viewBox="0 0 600 {len(rows) * 62 + 30}" role="img" aria-label="Time per stage, CPU against GPU" class="chart">']
    y = 10
    for r in rows:
        parts.append(f'<text x="0" y="{y + 26}" class="lbl">{html.escape(r["stage"])}</text>')
        for key, color, name in (("cpu_ms", CPU_COLOR, "CPU"), ("gpu_ms", GPU_COLOR, "GPU")):
            v = r[key]
            if v is None:
                continue
            w = max(2.0, bar_w * v / top)
            parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{h - 4}" rx="3" fill="{color}"/>')
            parts.append(f'<text x="{left + w + 6:.1f}" y="{y + h - 9}" class="val">{name} {v:g} ms</text>')
            y += h
        y += 14
    parts.append("</svg>")
    return "".join(parts)


def _gpu_segments(results: ExecutionResults) -> list[tuple[str, float]]:
    g = results.gpu
    if g is None or g.error or g.total_time_ms <= 0 or g.transfer_time_ms is None:
        return []
    kernels = g.kernel_time_ms or 0.0
    hyst = g.stage_times_ms.get("hysteresis", 0.0)
    other = max(0.0, g.total_time_ms - g.transfer_time_ms - kernels - hyst)
    segs = [("PCIe transfer", g.transfer_time_ms), ("GPU kernels", kernels), ("Hysteresis (CPU)", hyst), ("Other", other)]
    return [(n, v) for n, v in segs if v > 0.005 * g.total_time_ms]


def _gpu_bar(results: ExecutionResults) -> str:
    segs = _gpu_segments(results)
    if not segs:
        return ""
    total = sum(v for _, v in segs)
    parts = ['<svg viewBox="0 0 600 120" role="img" aria-label="Where the GPU path spends its time" class="chart">']
    x = 0.0
    for name, v in segs:
        w = 600 * v / total
        parts.append(f'<rect x="{x:.1f}" y="10" width="{w:.1f}" height="34" fill="{SEGMENT_COLORS[name]}"/>')
        x += w
    for i, (name, v) in enumerate(segs):
        parts.append(f'<rect x="{i * 150}" y="64" width="12" height="12" rx="2" fill="{SEGMENT_COLORS[name]}"/>'
                     f'<text x="{i * 150 + 18}" y="75" class="lbl">{html.escape(name)}</text>'
                     f'<text x="{i * 150 + 18}" y="93" class="val">{v:g} ms ({100 * v / total:.0f}%)</text>')
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ the writer
def write_report(results: ExecutionResults, explanation_markdown: str, out_root: str = "outputs",
                 trace: dict | None = None) -> dict:
    """Write report.md, report.html and preview images to <out_root>/<run_id>/. Returns their paths."""
    run_id = results.plan.run_id or "run"
    out = Path(out_root) / run_id
    out.mkdir(parents=True, exist_ok=True)

    m = compute_metrics(results)
    bd = get_stage_breakdown(results)
    cv = check_correctness_status(results)
    if run_id.startswith("mock"):       # development fixtures must never pass as real measurements
        m["warnings"].insert(0, "DEVELOPMENT-ONLY MOCK DATA: these numbers were made up to test the software. "
                                "They are NOT real measurements.")

    previews = {
        "Original": ("preview_original.png", results.plan.image_path),
        "CPU edges": ("preview_cpu_edges.png", results.cpu.output_path if results.cpu else None),
        "GPU edges": ("preview_gpu_edges.png", results.gpu.output_path if results.gpu else None),
    }
    have = {label: _save_preview(src, out / name) for label, (name, src) in previews.items()}

    md = _markdown(results, m, bd, cv, explanation_markdown, previews, have, trace)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "report.html").write_text(_html(results, m, bd, cv, explanation_markdown, previews, have, out, trace), encoding="utf-8")
    return {"report_md": str(out / "report.md"), "report_html": str(out / "report.html"), "folder": str(out)}


def open_in_browser(path: str) -> bool:
    """Open the report in the default browser. Never raises."""
    try:
        return bool(webbrowser.open(Path(path).resolve().as_uri()))
    except Exception:
        return False


def _key_numbers(m: dict) -> list[tuple[str, str]]:
    return [
        ("Result", WINNER_TEXT[m["winner"]]),
        ("Speedup (total)", _f(m["speedup_total"], "x")),
        ("Speedup (kernels only)", _f(m["speedup_kernel_only"], "x")),
        ("Transfer share of GPU time", _f(m["transfer_overhead_pct"], "%", 1)),
        ("Throughput CPU", _f(m["throughput_cpu_mpix_s"], " MP/s")),
        ("Throughput GPU", _f(m["throughput_gpu_mpix_s"], " MP/s")),
        ("Amdahl limit (serial hysteresis)", _f(m["amdahl_max_speedup"], "x")),
        ("Hysteresis share of GPU time", _f(m["gpu_hysteresis_share_pct"], "%", 1)),
        ("cv2.Canny vs the GPU path", "n/a" if m["opencv_faster_than_gpu_by"] is None
         else (f"cv2.Canny {m['opencv_faster_than_gpu_by']:g}x faster" if m["opencv_faster_than_gpu_by"] > 1
               else f"GPU path {1 / m['opencv_faster_than_gpu_by']:.1f}x faster")),
    ]


def _conditions(results: ExecutionResults) -> list[str]:
    p = results.plan
    ref = results.cpu or results.gpu
    lines = [f"Image: {results.image_info.width}x{results.image_info.height} ({results.image_info.megapixels} MP)",
             f"Canny thresholds {p.low_threshold} / {p.high_threshold}",
             f"{ref.repetitions if ref else p.repetitions} timed runs after {ref.warmup_runs if ref else p.warmup_runs} discarded warm-up passes (warm-up continues until the clocks are steady: at least the planned count and at least 1 s); mean and standard deviation reported",
             "GPU timings synchronise the device before each clock read; transfer time (host to device and back) is separate from kernel time"]
    if results.gpu_name:
        lines.append(f"GPU: {results.gpu_name}")
    return lines


def _markdown(results, m, bd, cv, explanation, previews, have, trace=None) -> str:
    L = [f"# CPU vs GPU Canny edge detection: run {results.plan.run_id}", ""]
    L += [f"- {c}" for c in _conditions(results)] + [""]
    L += ["## Key numbers", "", "| Measure | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in _key_numbers(m)] + [""]
    if any(have.values()):
        L += ["## Images", "", "| " + " | ".join(l for l in previews if have[l]) + " |",
              "|" + "---|" * sum(have.values()),
              "| " + " | ".join(f"![{l}]({previews[l][0]})" for l in previews if have[l]) + " |", ""]
    L += ["## Time per stage", "", "| Stage | Runs on | CPU ms | GPU ms | Speedup | CPU share | GPU share |", "|---|---|---|---|---|---|---|"]
    for r in bd["stages"]:
        L.append(f"| {r['stage']} | {'GPU' if r['runs_on_gpu'] else 'CPU (serial)'} | {_f(r['cpu_ms'], '', 3)} | {_f(r['gpu_ms'], '', 3)} "
                 f"| {_f(r['speedup'], 'x')} | {_f(r['cpu_share_pct'], '%', 1)} | {_f(r['gpu_share_pct'], '%', 1)} |")
    if bd["gpu_transfer_ms"] is not None:
        L.append(f"| PCIe transfer | bus | n/a | {_f(bd['gpu_transfer_ms'], '', 3)} | n/a | n/a | {_f(bd['gpu_transfer_share_pct'], '%', 1)} |")
    for label, dev in (("CPU", results.cpu), ("GPU", results.gpu)):
        if dev and not dev.error:
            L.append(f"\n{label} total: {dev.total_time_ms} ms (std {dev.total_time_std_ms} ms)")
    L += ["", "## Correctness", "", cv["verdict"], ""]
    if m["warnings"]:
        L += ["## Warnings", ""] + [f"- {w}" for w in m["warnings"]] + [""]
    L += ["## Explanation", "", f"*{EXPLANATION_NOTE}*", "", explanation.strip(), ""]
    L += ["## Notes", "", f"- Planner: {results.plan.notes or 'none'}", f"- Execution agent: {results.agent2_notes or 'none'}",
          "- Numbers in this report are computed by code (tools/metrics.py) from the measured results.", ""]
    L += ["## How this run flowed", ""] + [f"- {x}" for x in _flow_lines(results, m, trace)] + [""]
    return "\n".join(L)


# ------------------------------------------------------------------ workflow section (drawn from the run's own facts)
DESIGN_TOOLS = {
    "agent1": ["get_supported_tasks", "inspect_image", "validate_plan", "submit_plan"],
    "agent2": ["check_gpu", "run_cpu", "run_gpu", "check_correctness", "finish"],
    "agent3": ["get_metrics", "get_stage_breakdown", "check_correctness_status", "lookup_coa_concept", "submit_explanation"],
}
ROLES = {
    "agent1": ("decides the task and settings from your sentence", "reads the image, validates the plan",
               "sends an ExecutionPlan to Agent 2"),
    "agent2": ("decides the order of steps; skips the GPU if it is unusable", "runs the real CPU and GPU benchmarks and the correctness check",
               "sends ExecutionResults to Agent 3"),
    "agent3": ("recognises the situation and picks the COA concepts", "metrics calculator, concept notes",
               "returns an AnalysisReport; code checks every number in its text"),
}
AGENT_COLORS = {"agent1": "#3b6fb6", "agent2": "#2e9e6b", "agent3": "#8a4fbf"}


def _dur(sec: float) -> str:
    sec = float(sec)
    return f"{sec:.1f} s" if sec < 60 else f"{int(sec // 60)} min {int(round(sec % 60))} s"


def _tool_counts(tools: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    for t in tools:
        seen[t] = seen.get(t, 0) + 1
    return [f"{t} x{n}" if n > 1 else t for t, n in seen.items()]


def _handoffs(results: ExecutionResults, m: dict, trace: dict | None) -> dict[str, str]:
    p = results.plan
    runs = " + ".join(x for x, on in (("CPU", p.run_cpu), ("GPU", p.run_gpu)) if on) or "nothing"
    plan = f"thresholds {p.low_threshold}/{p.high_threshold} | {p.repetitions} timed runs | {p.warmup_runs} warm-up | runs on {runs}"
    parts = [f"{lab} {d.total_time_ms} ms" for lab, d in (("CPU", results.cpu), ("GPU", results.gpu)) if d and not d.error]
    parts.append("correctness " + check_correctness_status(results)["verdict"].split(":")[0].lower())
    out = {"plan": plan, "results": " | ".join(parts)}
    ana = f"winner: {m['winner']}"
    if m.get("speedup_total") is not None:
        ana += f" | speedup {m['speedup_total']}x"
    info = (trace or {}).get("analysis")
    if info:
        ana += f" | {len(info.get('concepts', []))} COA concepts"
        ana += (" | explanation by the language model, numbers verified by code" if "language model" in info.get("source", "")
                else " | explanation built by code (the model's text could not be verified)")
    out["analysis"] = ana
    return out


def _flow_lines(results, m, trace) -> list[str]:
    stages = (trace or {}).get("stages", {})
    h = _handoffs(results, m, trace)
    lines = [] if stages else ["No run trace was recorded for this report, so only the design is shown: no timings or call counts."]
    for key in ORDER:
        st = stages.get(key)
        if st:
            lines.append(f"{AGENTS[key]}: {_dur(st['seconds'])}, {st['llm_calls']} model calls; tools requested: "
                         f"{', '.join(_tool_counts(st['tools'])) or 'none'}")
        else:
            lines.append(f"{AGENTS[key]}: " + ("not recorded in this trace" if stages else "tools in its design: " + ", ".join(DESIGN_TOOLS[key])))
    lines += [f"Handoff 1 (ExecutionPlan): {h['plan']}", f"Handoff 2 (ExecutionResults): {h['results']}",
              f"Handoff 3 (AnalysisReport): {h['analysis']}"]
    return lines


def _workflow_html(results: ExecutionResults, m: dict, trace: dict | None) -> str:
    e = html.escape
    stages = (trace or {}).get("stages", {})
    h = _handoffs(results, m, trace)
    intro = ("<p class='note'>Drawn from this run's own record: the time each agent took, how often it asked the model, and the tools it "
             "requested. The values on the arrows come from the saved results.</p>" if stages else
             "<p class='note'>No run trace was recorded for this report, so this shows the design only: no timings or call counts.</p>")

    def node(key: str) -> str:
        st, (reason, tool, comm) = stages.get(key), ROLES[key]
        if st:
            chips = "".join(f'<span class="chip">{e(c)}</span>' for c in _tool_counts(st["tools"])) or '<span class="note">no tools requested</span>'
            stat = f'<p class="stat">{_dur(st["seconds"])} · {st["llm_calls"]} model calls ({_dur(st["llm_seconds"])} thinking)</p>'
        else:
            label = "not recorded in this trace" if stages else "tools in its design"
            chips = f'<span class="note">{label}: </span>' + ("" if stages else "".join(f'<span class="chip">{e(t)}</span>' for t in DESIGN_TOOLS[key]))
            stat = ""
        notes = ""
        if key == "agent1" and results.plan.notes:
            notes = f'<p class="note">Agent 1 notes: {e(results.plan.notes[:260])}</p>'
        if key == "agent2" and results.agent2_notes:
            notes = f'<p class="note">Agent 2 notes: {e(results.agent2_notes[:260])}</p>'
        return (f'<div class="node" style="border-left:5px solid {AGENT_COLORS[key]}"><b>{e(AGENTS[key])}</b> <span class="note">(language model + tools)</span>'
                f'<p class="role"><u>Reasoning</u>: {e(reason)}<br><u>Tool use</u>: {e(tool)}<br><u>Communication</u>: {e(comm)}</p>{stat}<div>{chips}</div>{notes}</div>')

    edge = lambda title, text: f'<div class="edge"><b>{e(title)}</b><br>{e(text)}</div>'  # noqa: E731
    bar = ""
    done = [(k, stages[k]["seconds"]) for k in ORDER if k in stages and stages[k]["seconds"] > 0]
    if done:
        total = sum(v for _, v in done)
        segs = "".join(f'<div style="width:{100 * v / total:.2f}%;background:{AGENT_COLORS[k]}" title="{e(AGENTS[k])}: {_dur(v)}"></div>' for k, v in done)
        legend = " · ".join(f"{e(AGENTS[k])} {_dur(v)} ({100 * v / total:.0f}%)" for k, v in done)
        bar = f'<p class="note">Where the wall time of this run went (benchmarks included, report writing not):</p><div class="tbar">{segs}</div><p class="note">{legend}</p>'
    return (f'<h2>How this run flowed</h2>{intro}<div class="flow">'
            f'<div class="node io"><b>You</b><br><span class="note">a sentence and an image: {e(Path(results.plan.image_path).name)} '
            f'({results.image_info.width}x{results.image_info.height})</span></div>'
            + edge("request and image", "the text and the image path")
            + node("agent1") + edge("ExecutionPlan", h["plan"])
            + node("agent2") + edge("ExecutionResults", h["results"])
            + node("agent3") + edge("AnalysisReport", h["analysis"])
            + '<div class="node io"><b>This page</b><br><span class="note">report.html, report.md and results.json in the run folder</span></div>'
            + f'</div>{bar}')


CSS = """
:root{--bg:#fafafa;--fg:#1d2733;--mute:#5d6b7a;--card:#fff;--line:#dfe4ea;--accent:#2e9e6b}
@media(prefers-color-scheme:dark){:root{--bg:#14181d;--fg:#e6ebf0;--mute:#9aa7b4;--card:#1c232b;--line:#2c3640;--accent:#4cc38a}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.15rem;margin:32px 0 10px;border-bottom:1px solid var(--line);padding-bottom:4px}
.sub{color:var(--mute);margin:0 0 6px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.card b{display:block;font-size:1.2rem}.card span{color:var(--mute);font-size:.85rem}
.imgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.imgs figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}
.imgs img{width:100%;height:auto;display:block;border-radius:6px}.imgs figcaption{text-align:center;color:var(--mute);font-size:.9rem;padding-top:4px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--card)}th,td{border:1px solid var(--line);padding:6px 10px;text-align:left}th{font-size:.85rem;color:var(--mute)}
td.n{text-align:right;font-variant-numeric:tabular-nums}.chart{width:100%;max-width:600px;height:auto}.chart .lbl{font-size:13px;fill:var(--fg)}.chart .val{font-size:12px;fill:var(--mute)}
.warn{background:#fff4e0;color:#5a3a00;border:1px solid #f0c27a;border-radius:8px;padding:8px 14px;margin:6px 0}
.ok{color:var(--accent);font-weight:600}.bad{color:#d1493f;font-weight:600}.note{color:var(--mute);font-size:.9rem}
.flow{display:flex;flex-direction:column}.node{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;overflow-wrap:anywhere}
.node.io{text-align:center}.node .role{margin:6px 0;font-size:.92rem}.node .stat{margin:4px 0;font-weight:600}
.edge{align-self:center;text-align:center;padding:4px 12px 10px;color:var(--mute);font-size:.86rem;overflow-wrap:anywhere;max-width:100%}
.edge::before{content:"\\25BC";display:block;color:var(--accent);line-height:1.2}
.chip{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 9px;margin:2px 3px 2px 0;font-size:.8rem}
.tbar{display:flex;height:22px;border-radius:6px;overflow:hidden;border:1px solid var(--line)}
.expl{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:4px 18px}
"""


def _html(results, m, bd, cv, explanation, previews, have, out: Path, trace=None) -> str:
    e = html.escape
    cards = "".join(f'<div class="card"><span>{e(k)}</span><b>{e(v)}</b></div>' for k, v in _key_numbers(m))
    figs = "".join(f'<figure><img alt="{e(l)}" src="{_data_uri(out / previews[l][0])}"><figcaption>{e(l)}</figcaption></figure>'
                   for l in previews if have[l])
    imgs = f'<h2>Images</h2><div class="imgs">{figs}</div>' if figs else '<h2>Images</h2><p class="note">No images were available for this run.</p>'
    rows = "".join(
        f"<tr><td>{e(r['stage'])}</td><td>{'GPU' if r['runs_on_gpu'] else 'CPU (serial)'}</td>"
        + "".join(f'<td class="n">{e(v)}</td>' for v in (_f(r["cpu_ms"], "", 3), _f(r["gpu_ms"], "", 3), _f(r["speedup"], "x"),
                                                         _f(r["cpu_share_pct"], "%", 1), _f(r["gpu_share_pct"], "%", 1))) + "</tr>"
        for r in bd["stages"])
    if bd["gpu_transfer_ms"] is not None:
        rows += (f'<tr><td>PCIe transfer</td><td>bus</td><td class="n">n/a</td><td class="n">{_f(bd["gpu_transfer_ms"], "", 3)}</td>'
                 f'<td class="n">n/a</td><td class="n">n/a</td><td class="n">{_f(bd["gpu_transfer_share_pct"], "%", 1)}</td></tr>')
    totals = "".join(f"<p class='note'>{lab} total: {d.total_time_ms} ms (std {d.total_time_std_ms} ms)</p>"
                     for lab, d in (("CPU", results.cpu), ("GPU", results.gpu)) if d and not d.error)
    cls = "ok" if cv["passed"] else ("bad" if cv["performed"] else "note")
    warn = "".join(f'<div class="warn">{e(w)}</div>' for w in m["warnings"])
    conditions = "".join(f"<li>{e(c)}</li>" for c in _conditions(results))
    charts = (f"<h2>Time per stage</h2>{_stage_chart(bd)}" if _stage_chart(bd) else "") + \
             (f"<h2>Where the GPU path spends its time</h2>{_gpu_bar(results)}" if _gpu_bar(results) else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPU vs GPU Canny: run {e(str(results.plan.run_id))}</title><style>{CSS}</style></head><body><main>
<h1>CPU vs GPU Canny edge detection</h1><p class="sub">Run {e(str(results.plan.run_id))}</p><ul class="note">{conditions}</ul>
<h2>Key numbers</h2><div class="cards">{cards}</div>{warn}{imgs}{charts}
<h2>Stage table</h2><div class="scroll"><table><tr><th>Stage</th><th>Runs on</th><th>CPU ms</th><th>GPU ms</th><th>Speedup</th><th>CPU share</th><th>GPU share</th></tr>{rows}</table></div>{totals}
<h2>Correctness</h2><p class="{cls}">{e(cv['verdict'])}</p>
<h2>Explanation</h2><p class="note">{e(EXPLANATION_NOTE)}</p><div class="expl">{_md_to_html(explanation)}</div>
<h2>Notes</h2><p class="note">Planner: {e(results.plan.notes or 'none')}<br>Execution agent: {e(results.agent2_notes or 'none')}<br>
Numbers in this report are computed by code (tools/metrics.py) from the measured results.</p>
{_workflow_html(results, m, trace)}
</main></body></html>"""
