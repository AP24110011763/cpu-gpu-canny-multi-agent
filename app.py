r"""Dashboard for the CPU vs GPU multi-agent project.

    streamlit run app.py

Runs on YOUR PC (it needs your GPU and your Ollama), opens in the browser, and replaces typing commands:
run the three agents, repeat the benchmark, browse every saved run, run the size sweep, check the machine.
This file only draws. Every number comes from the same tools the command-line scripts use.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools import dashboard_data as dd  # noqa: E402
from tools.run_trace import AGENTS  # noqa: E402

OUT = Path(os.environ.get("COA_OUTPUT_DIR") or ROOT / "outputs")      # the env variable lets tests use a temporary folder
DEFAULT_REQUEST = "Apply edge detection to this image and compare CPU and GPU performance"

st.set_page_config(page_title="CPU vs GPU Agents", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1250px}
[data-testid="stMetricValue"]{font-size:1.5rem}
.agentline{padding:.45rem .8rem;border-left:5px solid #888;margin:.3rem 0;border-radius:4px;background:rgba(128,128,128,.08)}
</style>""", unsafe_allow_html=True)
COLORS = {"agent1": "#3b6fb6", "agent2": "#2e9e6b", "agent3": "#8a4fbf"}


# --------------------------------------------------------------------------------------------- helpers
def fmt(v, unit="", digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}{unit}"


def secs(s: float) -> str:
    return f"{s:.0f} s" if s < 60 else f"{int(s // 60)} min {int(s % 60)} s"


def status(refresh=False) -> dict:
    if refresh or "status" not in st.session_state:
        with st.spinner("Checking the GPU and Ollama..."):
            st.session_state["status"] = dd.system_status(str(OUT))
    return st.session_state["status"]


def show_run(folder: str, results, metrics: dict, trace: dict | None, key: str) -> None:
    """Numbers, warnings, images, and the full self-contained report page for one run."""
    cols = st.columns(5)
    cols[0].metric("Winner", {"gpu": "GPU", "cpu": "CPU", "tie": "Tie", "undetermined": "Not decided"}.get(metrics["winner"], metrics["winner"]))
    cols[1].metric("Speedup (total)", fmt(metrics.get("speedup_total"), "x"))
    cols[2].metric("CPU total", fmt(metrics.get("cpu_total_ms"), " ms"))
    cols[3].metric("GPU total", fmt(metrics.get("gpu_total_ms"), " ms"))
    cols[4].metric("Transfer share", fmt(metrics.get("transfer_overhead_pct"), "%", 1))
    for w in metrics.get("warnings", []):
        st.warning(w)
    folder = Path(folder)
    page = folder / "report.html"
    previews = [(l, folder / f) for l, f in (("Original", "preview_original.png"), ("CPU edges", "preview_cpu_edges.png"),
                                              ("GPU edges", "preview_gpu_edges.png")) if (folder / f).exists()]
    if previews:
        for col, (label, p) in zip(st.columns(len(previews)), previews):
            col.image(str(p), caption=label)
    if page.exists():
        html = page.read_text(encoding="utf-8")
        with st.expander("Full report (same page the pipeline opens)", expanded=True):
            components.html(html, height=1500, scrolling=True)
        c1, c2 = st.columns(2)
        c1.download_button("Download report.html", html, file_name=f"{results.plan.run_id}_report.html", key=f"dl_html_{key}")
        c2.download_button("Download results.json", results.model_dump_json(indent=2), file_name=f"{results.plan.run_id}_results.json", key=f"dl_json_{key}")
    else:
        st.info("This run has no report page (benchmark rounds only measure). The numbers above are from its results.json.")
    if trace and trace.get("stages"):
        st.caption("Agent timings recorded for this run: " + " | ".join(
            f"{AGENTS[k]} {secs(v['seconds'])}, {v['llm_calls']} model calls" for k, v in trace["stages"].items() if k in AGENTS))


# --------------------------------------------------------------------------------------------- sidebar
def sidebar() -> tuple[str | None, dict]:
    st.sidebar.title("CPU vs GPU Agents")
    st.sidebar.caption("Canny edge detection, measured on this PC")
    st.sidebar.subheader("Image")
    mode = st.sidebar.radio("Choose the image by", ["Path on this PC", "Upload"], horizontal=True, label_visibility="collapsed")
    path = None
    if mode == "Upload":
        up = st.sidebar.file_uploader("PNG, JPG, BMP, TIFF or WebP", type=[e.lstrip(".") for e in sorted(dd.SUPPORTED_EXT)])
        if up is not None:
            try:
                path = dd.save_upload(up.getvalue(), up.name, str(OUT / "uploads"))
            except ValueError as exc:
                st.sidebar.error(str(exc))
    else:
        typed = st.sidebar.text_input("Image path", value=st.session_state.get("typed_path", ""), placeholder=r"..\image1.png")
        st.session_state["typed_path"] = typed
        path = typed.strip().strip('"') or None
    if path:
        info = dd.image_summary(path)
        if info.get("ok"):
            st.sidebar.image(path, caption=f"{info['width']}x{info['height']} px, {info['megapixels']:.2f} MP")
        else:
            st.sidebar.error(info.get("error", "cannot use this image"))
            path = None
    stt = status()
    st.sidebar.subheader("This machine")
    g, o = stt["gpu"], stt["ollama"]
    st.sidebar.markdown(("GPU: **" + (g.get("gpu_name") or "GPU") + "** ready") if g["ok"] else "GPU: **not usable** (see the System tab)")
    st.sidebar.markdown("Ollama: **" + ("ready, model installed" if o["reachable"] and o["model_installed"] else
                                         "running, model missing" if o["reachable"] else "not running") + "**")
    a, b = st.sidebar.columns(2)
    if a.button("Refresh"):
        status(refresh=True)
        st.rerun()
    if b.button("Unload model", help="Frees the language model from memory before a benchmark"):
        try:
            from llm.client import OllamaClient
            st.sidebar.success("Model unloaded." if OllamaClient().unload() else "Could not reach Ollama.")
        except Exception as exc:
            st.sidebar.error(f"{type(exc).__name__}: {exc}")
    with st.sidebar.expander("For fair timings"):
        for line in dd.FAIR_TIMING_CHECKLIST:
            st.markdown("- " + line)
    return path, stt


# --------------------------------------------------------------------------------------------- tab 1
def tab_agents(image_path, stt) -> None:
    st.subheader("Run the three agents")
    st.caption("Agent 1 plans, Agent 2 measures on the CPU and the GPU, Agent 3 explains. The language model runs on the CPU, "
               "so this takes several minutes. Keep this tab open and leave the PC alone while it runs.")
    request = st.text_area("Your request", value=DEFAULT_REQUEST, height=80)
    classic = st.checkbox("Use the fixed-steps planner instead of the autonomous one")
    if not stt["ollama"]["reachable"]:
        st.error("Ollama is not running. Start it (run `ollama serve`) and press Refresh in the sidebar.")
    if st.button("Run the three agents", type="primary", disabled=image_path is None or not stt["ollama"]["reachable"]):
        run_full(image_path, request, classic)
    if image_path is None:
        st.info("Choose an image in the sidebar first.")
    last = st.session_state.get("last_full")
    if last:
        st.success(f"Finished {last['run_id']} in {secs(last['seconds'])}.")
        show_run(last["folder"], last["results"], last["metrics"], last["trace"], "full")


def run_full(image_path: str, request: str, classic: bool) -> None:
    from agents.executor import ExecutorError
    from agents.planner import PlannerError
    from llm.client import OllamaClient
    from orchestrator import run_analysis, run_pipeline
    from tools.metrics import compute_metrics
    from tools.run_trace import TracingLLM

    t0 = time.time()
    box = st.status("Agents working...", expanded=True)
    lines = box.empty()
    seen: dict[str, str] = {}

    def on_event(kind: str, d: dict) -> None:
        seen[d["stage"]] = (f"<div class='agentline' style='border-color:{COLORS[d['stage']]}'><b>{AGENTS[d['stage']]}</b> &nbsp; "
                            f"{d['llm_calls']} model calls &middot; {secs(d['elapsed'])}"
                            f"{' &middot; last tool: ' + d['tools'][-1] if d['tools'] else ''}</div>")
        lines.markdown("".join(seen[k] for k in AGENTS if k in seen), unsafe_allow_html=True)

    try:
        llm = TracingLLM(OllamaClient(), on_event=on_event)
        res, _ = run_pipeline(request, image_path, llm, str(OUT), agentic=not classic)
        report, files = run_analysis(res, llm, str(OUT), open_report=False)
    except (PlannerError, ExecutorError) as exc:
        box.update(label="Stopped", state="error")
        st.error(f"Stopped: {exc}")
        return
    except Exception as exc:                                       # Ollama down mid-run, etc.
        box.update(label="Failed", state="error")
        st.error(f"{type(exc).__name__}: {exc}")
        return
    box.update(label="Done", state="complete", expanded=False)
    import json
    tp = Path(files["folder"]) / "trace.json"
    st.session_state["last_full"] = {"run_id": res.plan.run_id, "folder": files["folder"], "results": res, "metrics": compute_metrics(res),
                                     "trace": json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else None, "seconds": time.time() - t0}
    st.rerun()


# --------------------------------------------------------------------------------------------- tab 2
def tab_benchmark(image_path, stt) -> None:
    st.subheader("Quick benchmark (no language model)")
    st.caption("Repeats the CPU and GPU measurement several times with the same tools Agent 2 uses, then shows mean, median and spread. "
               "About 2 minutes. This is the best way to get numbers you can quote.")
    c = st.columns(4)
    rounds = c[0].slider("Rounds", 3, 10, 5)
    reps = c[1].number_input("Timed repetitions", 5, 100, 20)
    low = c[2].number_input("Low threshold", 1, 254, 100)
    high = c[3].number_input("High threshold", 2, 255, 200)
    if not stt["gpu"]["ok"]:
        st.error("The GPU is not usable, so there is nothing to compare. See the System tab.")
    if stt["ollama"]["reachable"]:
        st.warning("Ollama is running. Press **Unload model** in the sidebar first, or the model may use the CPU while you measure.")
    if st.button("Run the benchmark", type="primary", disabled=image_path is None or not stt["gpu"]["ok"] or low >= high):
        from tools.aggregate import aggregate
        from tools.repeat_benchmark import repeat_benchmark
        bar, live = st.progress(0.0), st.empty()
        rows: list[str] = []

        def say(line: str) -> None:
            rows.append(line)
            bar.progress(min(1.0, len(rows) / rounds))
            live.code("\n".join(rows))
        try:
            runs, agg, folder = repeat_benchmark(image_path, int(rounds), int(low), int(high), int(reps), 3, 3.0, str(OUT), say=say)
        except Exception as exc:
            st.error(f"Stopped: {exc}")
        else:
            st.session_state["last_bench"] = {"agg": agg, "folder": str(folder)}
    if low >= high:
        st.error("The low threshold must be smaller than the high threshold.")
    last = st.session_state.get("last_bench")
    if last:
        agg = last["agg"]
        s = agg["stats"].get("speedup_total")
        if s:
            m = st.columns(4)
            m[0].metric("Speedup (mean)", f"{s['mean']:.2f}x", f"± {s['std']:.2f}", delta_color="off")
            m[1].metric("Speedup (median)", f"{s['median']:.2f}x")
            m[2].metric("Range", f"{s['min']:.2f}x to {s['max']:.2f}x")
            m[3].metric("Spread between rounds", f"{s['cv_pct']:.1f}%")
        for w in agg["warnings"]:
            st.warning(w)
        if not agg["warnings"]:
            st.success("No warnings: the rounds agree with each other.")
        st.dataframe(pd.DataFrame([{"metric": k, **{x: v[x] for x in ("mean", "median", "std", "min", "max")}, "spread %": v["cv_pct"]}
                                   for k, v in agg["stats"].items()]), hide_index=True)
        st.bar_chart(pd.DataFrame({"speedup": agg["per_run_speedup"]}, index=[f"round {i + 1}" for i in range(len(agg["per_run_speedup"]))]))
        st.caption(f"Each round is saved in {last['folder']}.")


# --------------------------------------------------------------------------------------------- tab 3
def tab_runs() -> None:
    st.subheader("Past runs")
    rows, bad = dd.list_runs(str(OUT))
    show_mock = st.checkbox("Also show MOCK test data (made-up numbers used only for software tests)", value=False)
    shown = [r for r in rows if show_mock or not r["mock"]]
    if bad:
        st.caption(f"{len(bad)} results file(s) could not be read and were skipped.")
    if not shown:
        st.info("No saved runs yet. Run the agents or the quick benchmark, and they appear here.")
        return
    series = dd.speedup_series(shown)
    if len(series) >= 2:
        st.caption("Speedup of each real run, oldest to newest")
        st.line_chart(pd.DataFrame(series).set_index("run"))
    st.dataframe(pd.DataFrame([{k: r[k] for k in ("modified", "run_id", "kind", "image", "cpu_ms", "gpu_ms", "speedup", "kernel_speedup",
                                                    "transfer_pct", "winner", "noisy")} for r in shown]), hide_index=True)
    pick = st.selectbox("Open a run", range(len(shown)), format_func=lambda i: dd.run_label(shown[i]))
    row = shown[pick]
    if row["mock"]:
        st.error("DEVELOPMENT-ONLY MOCK DATA: made up to test the software. These are not real measurements.")
    d = dd.load_run(row)
    show_run(row["folder"], d["results"], d["metrics"], d["trace"], f"past{pick}")


# --------------------------------------------------------------------------------------------- tab 4
def tab_sweep(image_path, stt) -> None:
    st.subheader("Does the GPU speedup grow with image size?")
    data = dd.load_sweep(str(OUT))
    if data:
        orig_w = None
        info = dd.image_summary(data.get("image", ""))
        if info.get("ok"):
            orig_w = info["width"]
        good = [r for r in data["rows"] if "error" not in r]
        st.caption(f"Last sweep of {Path(data.get('image', '')).name or 'an image'}. Conditions: {data.get('note') or 'not recorded'}.")
        if len(good) >= 2:
            st.line_chart(pd.DataFrame({"speedup": [r["speedup_total"] for r in good]}, index=[f"{r['megapixels']:.2f} MP" for r in good]))
        st.dataframe(pd.DataFrame(dd.sweep_table(data["rows"], orig_w)), hide_index=True)
        st.caption("Rows marked 'enlarged copy' were made by enlarging a smaller picture, so they are smoother than a real photo of that size.")
        if data["html"]:
            with st.expander("Sweep page"):
                components.html(Path(data["html"]).read_text(encoding="utf-8"), height=900, scrolling=True)
    else:
        st.info("No sweep saved yet. Run one below.")
    with st.expander("Run a new sweep", expanded=data is None):
        widths = st.multiselect("Widths in pixels", [128, 256, 512, 1024, 2048, 3072, 4096], default=[256, 512, 1024, 2048])
        reps = st.slider("Timed repetitions per size", 3, 30, 10)
        note = st.text_input("Conditions to record", placeholder="plugged in, Best performance")
        if not stt["gpu"]["ok"]:
            st.error("The GPU is not usable, so a sweep cannot run.")
        if st.button("Run the sweep", disabled=image_path is None or not widths or not stt["gpu"]["ok"]):
            from tools.size_sweep import sweep, write_outputs
            live, lines = st.empty(), []

            def say(line: str) -> None:
                lines.append(line)
                live.code("\n".join(lines))
            try:
                rows = sweep(image_path, sorted(widths), 100, 200, reps, 3, str(OUT / "size_sweep"), progress=say)
                write_outputs(rows, str(OUT / "size_sweep"), image_path, note)
            except Exception as exc:
                st.error(f"Stopped: {exc}")
            else:
                st.rerun()


# --------------------------------------------------------------------------------------------- tab 5
def tab_system(stt) -> None:
    st.subheader("This machine")
    g, o = stt["gpu"], stt["ollama"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**GPU**")
        st.dataframe(pd.DataFrame([("Ready", "yes" if g["ok"] else "no"), ("Name", g.get("gpu_name")), ("Driver", g.get("driver_version")),
                                   ("VRAM (MB)", g.get("vram_mb")), ("CuPy", g.get("cupy_version")), ("CUDA runtime", g.get("cuda_runtime")),
                                   ("Compute capability", g.get("compute_capability"))], columns=["item", "value"]).astype(str), hide_index=True)
        for p in g.get("problems", []):
            st.error(p)
    with c2:
        st.markdown("**Language model and system**")
        st.dataframe(pd.DataFrame([("Ollama running", "yes" if o["reachable"] else "no"), ("llama3.2:3b installed", "yes" if o["model_installed"] else "no"),
                                   ("Python", stt["python"]), ("Free disk for outputs (GB)", stt["free_disk_gb"]),
                                   ("Outputs folder", stt["output_root"])], columns=["item", "value"]).astype(str), hide_index=True)
        if o["error"]:
            st.error(o["error"])
    st.subheader("How it works")
    st.markdown(
        "| Agent | Reasoning (the model) | Tool use | Communication |\n|---|---|---|---|\n"
        "| 1. Task Planner | picks the task and settings from your sentence | `get_supported_tasks`, `inspect_image`, `validate_plan`, `submit_plan` | sends an `ExecutionPlan` |\n"
        "| 2. Execution Agent | decides the order; skips the GPU if unusable | `check_gpu`, `run_cpu`, `run_gpu`, `check_correctness`, `finish` | sends `ExecutionResults` |\n"
        "| 3. Analysis Agent | recognises the situation, picks COA concepts | `get_metrics`, `get_stage_breakdown`, `check_correctness_status`, `lookup_coa_concept`, `submit_explanation` | returns an `AnalysisReport` |\n\n"
        "The model proposes and code verifies: every number comes from measured results, and the explanation is checked by code.")
    st.markdown("**COA ideas measured:** sequential vs parallel work, CPU vs GPU architecture, PCIe transfer cost (memory hierarchy), "
                "throughput vs latency, Amdahl's law (serial hysteresis), and timing methodology (warm-up, repetitions, synchronisation).")


# --------------------------------------------------------------------------------------------- page
image_path, stt = sidebar()
st.title("CPU vs GPU Multi-Agent Performance Analyzer")
t1, t2, t3, t4, t5 = st.tabs(["Run the agents", "Quick benchmark", "Past runs", "Size sweep", "System and about"])
with t1:
    tab_agents(image_path, stt)
with t2:
    tab_benchmark(image_path, stt)
with t3:
    tab_runs()
with t4:
    tab_sweep(image_path, stt)
with t5:
    tab_system(stt)
