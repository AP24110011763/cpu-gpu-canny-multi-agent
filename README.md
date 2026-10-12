# cpu-gpu-canny-multi-agent
Autonomous multi-agent AI system for CPU vs GPU performance analysis of Canny Edge Detection. Built for B.Tech Computer Organization &amp; Architecture (COA). Features real LLM agents, hand-implemented CuPy GPU pipeline, and detailed COA-focused performance comparison.
# Autonomous Multi-Agent System for CPU vs GPU Performance Analysis

**B.Tech Computer Organization and Architecture (COA) Project**

This project implements an autonomous multi-agent AI system that compares the performance of CPU and GPU on the classic image processing task of **Canny Edge Detection**.

A user provides an image and a natural language request (e.g. “Apply edge detection to this image and compare CPU and GPU performance”). The system then works mostly autonomously using three specialized LLM-powered agents.

## System Architecture

- **Agent 1 – Task Planner**  
  Understands the natural language request and generates a structured execution plan (JSON).

- **Agent 2 – Execution Agent**  
  Runs the same Canny Edge Detection pipeline on both CPU and NVIDIA GPU, measures real execution time, verifies output correctness, and returns structured results.

- **Agent 3 – Comparison & Analysis Agent**  
  Analyzes the measured results and produces a detailed explanation using Computer Organization and Architecture concepts (parallelism, memory hierarchy, data transfer overhead, throughput, Amdahl’s Law, etc.).

The agents communicate through a lightweight custom Python orchestrator using structured JSON messages.

## Key Technical Highlights

- **CPU Implementation**: OpenCV (`cv2.Canny`)
- **GPU Implementation**: Hand-implemented stage-by-stage using CuPy  
  (Gaussian blur, Sobel, gradient magnitude, and non-maximum suppression run in parallel on GPU; hysteresis edge linking runs on CPU)
- **LLM Backend**: Ollama + Llama 3.2 3B Instruct (CPU-only inference so it does not affect GPU benchmarks)
- **Performance Measurement**:
  - Warm-up runs + multiple timed repetitions
  - Separate measurement of kernel time vs host-device transfer time
  - Correctness verification using edge pixel matching and mean absolute difference
  - Multi-resolution image size sweep to demonstrate the crossover point where transfer overhead dominates

## Project Goals

- Demonstrate real CPU vs GPU performance differences on a concrete workload
- Clearly connect experimental results to fundamental COA concepts
- Keep the system practical, explainable, and suitable for a B.Tech end-semester project
- Avoid unnecessary complexity (no databases, microservices, or deep learning models)

## Hardware Used

- NVIDIA GTX 1650 (4 GB)
- Windows PC with 8 GB system RAM

## Status

Stage-by-stage development. Currently in early implementation phase.

---

**Note**: This is an academic project focused on Computer Organization and Architecture concepts through practical CPU–GPU performance analysis.
