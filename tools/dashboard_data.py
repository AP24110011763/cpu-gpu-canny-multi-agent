"""Everything the dashboard needs to READ and PREPARE, with no Streamlit in it, so it can be tested.

The dashboard (app.py) only draws what this module returns. Numbers come from tools/metrics.py (the same
calculator Agent 3 and the report use), never from the page itself. Nothing here is estimated.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from schemas import ExecutionResults
from tools.image_inspect import SUPPORTED_EXT, inspect_image
from tools.metrics import NOISY_PCT, compute_metrics

MAX_UPLOAD_MB = 25


# ------------------------------------------------------------------ saved runs
def _is_mock(run_id: str) -> bool:
    return str(run_id).startswith("mock")


def list_runs(output_root: str) -> tuple[list[dict], list[str]]:
    """Every saved results.json under output_root, newest first, as table rows.
    Returns (rows, unreadable_paths). Runs whose id starts with 'mock' are marked, never mixed in silently."""
    root = Path(output_root)
    rows: list[dict] = []
    bad: list[str] = []
    paths = sorted(root.glob("run_*/results.json")) + sorted(root.glob("repeat_*/round_*/results.json"))
    for p in paths:
        try:
            res = ExecutionResults.model_validate_json(p.read_text(encoding="utf-8"))
            m = compute_metrics(res)
        except Exception:
            bad.append(str(p))
            continue
        folder = p.parent
        noisy = any((m.get(k) or 0) > NOISY_PCT for k in ("cpu_std_pct_of_mean", "gpu_std_pct_of_mean"))
        rows.append({
            "run_id": res.plan.run_id, "kind": "benchmark round" if folder.parent.name.startswith("repeat_") else "full run",
            "mock": _is_mock(res.plan.run_id), "folder": str(folder), "results_path": str(p),
            "report_html": str(folder / "report.html") if (folder / "report.html").exists() else None,
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)), "_mtime": p.stat().st_mtime,
            "image": Path(res.plan.image_path).name, "megapixels": res.image_info.megapixels,
            "cpu_ms": m.get("cpu_total_ms"), "gpu_ms": m.get("gpu_total_ms"), "speedup": m.get("speedup_total"),
            "kernel_speedup": m.get("speedup_kernel_only"), "transfer_pct": m.get("transfer_overhead_pct"),
            "winner": m["winner"], "scenario": m["scenario"], "warnings": len(m["warnings"]), "noisy": noisy,
        })
    rows.sort(key=lambda r: r["_mtime"], reverse=True)
    return rows, bad


def load_run(row: dict) -> dict:
    """Full detail for one saved run: results object, metrics, saved trace (or None)."""
    res = ExecutionResults.model_validate_json(Path(row["results_path"]).read_text(encoding="utf-8"))
    trace_path = Path(row["folder"]) / "trace.json"
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else None
    except ValueError:
        trace = None
    return {"results": res, "metrics": compute_metrics(res), "trace": trace}


def run_label(row: dict) -> str:
    sp = f"{row['speedup']}x" if row["speedup"] is not None else "n/a"
    return f"{row['modified']} | {row['run_id']} | {row['winner']} {sp}" + (" | MOCK" if row["mock"] else "")


def speedup_series(rows: list[dict]) -> list[dict]:
    """Real runs with both devices, oldest first, for a 'speedup over runs' chart."""
    good = [r for r in rows if not r["mock"] and r["speedup"] is not None]
    return [{"run": r["modified"], "speedup": r["speedup"]} for r in sorted(good, key=lambda r: r["_mtime"])]


# ------------------------------------------------------------------ size sweep
def load_sweep(output_root: str) -> dict | None:
    p = Path(output_root) / "size_sweep" / "sweep.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data["html"] = str(p.parent / "sweep.html") if (p.parent / "sweep.html").exists() else None
    return data


def sweep_table(rows: list[dict], original_width: int | None) -> list[dict]:
    """Table rows for the page. A row wider than the original image is an enlarged copy, and says so."""
    out = []
    for r in rows:
        if "error" in r:
            out.append({"size": f"{r['width']}x{r['height']}", "note": f"skipped: {r['error']}"})
            continue
        notes = []
        if original_width and r["width"] > original_width:
            notes.append("enlarged copy")
        if r.get("noisy"):
            notes.append("noisy, repeat")
        out.append({"size": f"{r['width']}x{r['height']}", "megapixels": r["megapixels"], "cpu_ms": r["cpu_ms"],
                    "gpu_ms": r["gpu_ms"], "speedup": r["speedup_total"], "kernel_speedup": r["speedup_kernel_only"],
                    "transfer_pct": r["transfer_pct"], "outputs_agree": r["correct"], "note": ", ".join(notes)})
    return out


# ------------------------------------------------------------------ images
def save_upload(data: bytes, filename: str, folder: str) -> str:
    """Store an uploaded image safely (basename only, known image type, must really decode) and return its path."""
    name = Path(filename or "upload").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "upload"
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"unsupported image type '{ext}'; use one of {', '.join(sorted(SUPPORTED_EXT))}")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"image is larger than {MAX_UPLOAD_MB} MB")
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED) is None:
        raise ValueError("the file is not a valid image")
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{stem}{ext}"
    n = 1
    while dest.exists() and dest.read_bytes() != data:
        n += 1
        dest = out / f"{stem}_{n}{ext}"
    dest.write_bytes(data)
    return str(dest)


def image_summary(path: str) -> dict:
    """Width, height, megapixels and size class, or {'ok': False, 'error': ...}. Same check Agent 1 uses."""
    return inspect_image(path) if path else {"ok": False, "error": "no image chosen"}


# ------------------------------------------------------------------ the machine
def ollama_status(model: str = "llama3.2:3b") -> dict:
    """Is the Ollama server reachable, and is the model installed? Never raises."""
    try:
        import ollama
        listing = ollama.Client().list()
        raw = listing["models"] if isinstance(listing, dict) else getattr(listing, "models", [])
        names = [(m.get("model") or m.get("name")) if isinstance(m, dict) else (getattr(m, "model", None) or getattr(m, "name", None))
                 for m in raw]
        names = [n for n in names if n]
        return {"reachable": True, "models": names, "model_installed": model in names or f"{model}:latest" in names, "error": None}
    except ImportError:
        return {"reachable": False, "models": [], "model_installed": False, "error": "the ollama Python package is not installed"}
    except Exception as exc:
        return {"reachable": False, "models": [], "model_installed": False,
                "error": f"{type(exc).__name__}: start Ollama (run 'ollama serve') and try again"}


def system_status(output_root: str, gpu_check=None) -> dict:
    if gpu_check is None:
        from tools.gpu_check import check_gpu as gpu_check
    try:
        gpu = gpu_check()
    except Exception as exc:                                       # the page must still open
        gpu = {"ok": False, "gpu_name": None, "problems": [f"{type(exc).__name__}: {exc}"]}
    root = Path(output_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        free_gb = round(shutil.disk_usage(root).free / 1e9, 1)
    except OSError:
        free_gb = None
    return {"gpu": gpu, "ollama": ollama_status(), "python": sys.version.split()[0], "free_disk_gb": free_gb,
            "output_root": str(root.resolve())}


FAIR_TIMING_CHECKLIST = [
    "Charger connected, Windows power mode on Best performance.",
    "Other heavy apps closed (browser tabs, video, updates).",
    "Language model unloaded before a benchmark (the button in the sidebar), so it does not use the CPU.",
    "Repeat the benchmark several times and read the spread, not one number.",
]
