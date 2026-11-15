# CPU vs GPU Multi-Agent Performance Analyzer

A Computer Organization and Architecture (COA) project. A user gives an **image and a plain-English request**, for example
*"Apply edge detection to this image and compare CPU and GPU performance"*. Three language-model agents work in sequence:
they plan the job, run **Canny edge detection** on the CPU and on an **NVIDIA GTX 1650**, measure real times, and explain the
difference using COA concepts. The result is an HTML report with the images, charts, tables and a workflow diagram of that run.

```
request + image -> Agent 1 (Task Planner) -> ExecutionPlan -> Agent 2 (Execution) -> ExecutionResults -> Agent 3 (Analysis) -> AnalysisReport -> report.html
```

## The three agents

All three use the same local model (`llama3.2:3b` through Ollama, on the CPU) with tools and typed messages (Pydantic, `schemas.py`).
The rule everywhere is **the model proposes, code verifies**: numbers never pass through model text.

| Agent | Reasoning (the model) | Tool use | Communication |
|---|---|---|---|
| **1. Task Planner** | Reads the request, picks task, thresholds, repetitions, CPU and/or GPU | `get_supported_tasks`, `inspect_image`, `validate_plan`, `submit_plan` | Sends an `ExecutionPlan`. Code overrides any setting the user wrote explicitly (for example "30 times"). |
| **2. Execution Agent** | Decides the order of steps; skips the GPU if it is unusable | `check_gpu`, `run_cpu`, `run_gpu`, `check_correctness`, `finish` | Sends `ExecutionResults`. If the model skips a step, code runs it and notes it. |
| **3. Analysis Agent** | Recognises the situation (GPU wins, CPU wins, tie, GPU missing, correctness failed) and picks COA concepts | `get_metrics`, `get_stage_breakdown`, `check_correctness_status`, `lookup_coa_concept`, `submit_explanation` | Returns an `AnalysisReport`. Code checks every number in the text and rejects unmeasured claims; a code template is the fallback. |

## What is measured

Canny has four stages: **Gaussian blur, Sobel gradients, non-maximum suppression, hysteresis**. The first three work on each pixel
independently and run in parallel on the GPU (CuPy). Hysteresis follows edge chains, is serial, and runs on the CPU in both paths.
Compared: total time, per-stage time, speedup, kernel-only speedup, PCIe transfer share, throughput (MP/s), the Amdahl's-law limit,
speedup against image size, run-to-run stability, and whether the CPU and GPU outputs agree. `cv2.Canny` is timed as an OpenCV reference.

**Timing method.** Warm-up passes are discarded (at least the planned number *and* at least 1 second, so an idle GPU can reach full clock).
Then 20 timed repetitions by default; mean and standard deviation are reported. GPU work is asynchronous, so the GPU is synchronised
before every clock read. Host-to-device and device-to-host copies are timed separately from kernels. Totals are wall-clock around the whole pass.
The input image is already in memory, so file reading is not timed.

**Correctness.** The CPU and GPU edge maps must agree on at least 98% of pixels, and edges must overlap within 1 pixel (at least 0.95),
so an all-black output cannot pass. If this fails, no winner is declared.

## Results (our laptop, GTX 1650, charger connected, Best performance)

| | |
|---|---|
| CPU total | 33.4 ± 2.0 ms |
| GPU total | 11.85 ± 0.38 ms |
| **Speedup** | **2.82x ± 0.14** (7 rounds, 474x709 image) |
| Kernels only | 5.03x |
| Transfer | 8.5% of GPU time |

The speedup grows with image size and levels off near 5x because serial hysteresis limits it (Amdahl's law). The GPU is slower on the cheap
blur stage. OpenCV's own `cv2.Canny` is about 7 times faster than this project's GPU path. Details, tables and caveats: [docs/RESULTS.md](docs/RESULTS.md).

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install cupy-cuda12x          # GPU path; needs an NVIDIA GPU and a CUDA 12 driver
ollama pull llama3.2:3b           # install Ollama first from https://ollama.com
python scripts\check_ollama.py    # the model answers?
python scripts\verify_gpu.py      # CuPy sees the GTX 1650?
pytest -q                         # tests (some are skipped without a GPU or Ollama)
```

## Run

Easiest: the dashboard (no commands to remember).

```powershell
pip install streamlit
streamlit run app.py
```

It opens in the browser with tabs to run the three agents, run a quick benchmark, browse past runs, and see the size sweep.

Or from the command line:

```powershell
python main.py --image path\to\image.png          # full pipeline, opens the report (a few minutes: the model runs on the CPU)
python main.py --image image.png --request "Compare CPU and GPU, thresholds 50 and 150, 10 times"
python main.py --image image.png --no-open        # write the report without opening the browser
python scripts\run_compare_demo.py --image image.png --repetitions 20      # quick CPU vs GPU table, no model
python scripts\repeat_benchmark.py --image image.png --rounds 7            # repeated benchmark, mean/std/median, no model
python scripts\run_size_sweep.py --image image.png                         # speedup against image size
python scripts\gpu_idle_test.py --image image.png                          # does an idle GPU clock hurt the timing?
python scripts\run_analyst.py --results outputs\<run_id>\results.json      # Agent 3 only, on a saved run
```

For trustworthy numbers: connect the charger, set Windows to Best performance, close other apps, stop Ollama
(`ollama stop llama3.2:3b`) when running the no-model benchmarks, and repeat the run.

Each run writes `outputs/<run_id>/` with `results.json`, `report.html`, `report.md`, the edge images and `trace.json`
(what each agent did and how long it took). That folder is generated and not committed.

## Repository layout

```
agents/     planner.py, planner_agentic.py (Agent 1), executor.py (Agent 2), analyst.py (Agent 3)
llm/        Ollama client, tool-calling loop, scripted test model
tools/      cpu_canny, gpu_canny, gpu_check, correctness, metrics, report_writer, aggregate, size_sweep, ...
scripts/    command-line entry points listed above
tests/      pytest suite; tests/fixtures holds clearly labelled MOCK data for testing Agent 3 (not real results)
schemas.py  the typed messages passed between agents
app.py      Streamlit dashboard (tools/dashboard_data.py feeds it)
main.py, orchestrator.py   run the three agents in order
docs/       measured results
```

## Limitations

- The model is small (3B) and runs on the CPU, so a full run takes minutes and the model sometimes needs a nudge; code checks and fallbacks cover this.
- The explanations of *why* (kernel-launch cost for the slow blur, serial hysteresis capping the speedup) are architecture reasoning;
  launch cost, cache behaviour, occupancy, power and memory use were **not** measured.
- Timings depend on the machine state (power mode, temperature, background load). Results come from one laptop and mainly one test image.
- The CPU path is a staged implementation (OpenCV for blur and Sobel, NumPy for suppression) so stages can be timed; it is not the fastest possible code.

## Team

Six members, one part each. See [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
