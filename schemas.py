"""FROZEN CONTRACT. Owned by System 1. Do not edit; request changes via handoff."""
from typing import Literal, Optional
from pydantic import BaseModel


class ExecutionPlan(BaseModel):
    task: Literal["canny_edge_detection"]
    image_path: str
    low_threshold: int = 100
    high_threshold: int = 200
    run_cpu: bool = True
    run_gpu: bool = True
    repetitions: int = 10
    warmup_runs: int = 3
    run_id: Optional[str] = None
    notes: str = ""


class ImageInfo(BaseModel):
    width: int
    height: int
    channels: int
    megapixels: float


class DeviceResult(BaseModel):
    device: Literal["cpu", "gpu"]
    # keys: gaussian_blur, sobel_gradients, non_max_suppression, hysteresis (mean ms)
    stage_times_ms: dict[str, float]
    total_time_ms: float
    total_time_std_ms: float
    transfer_time_ms: Optional[float] = None
    kernel_time_ms: Optional[float] = None
    opencv_reference_time_ms: Optional[float] = None
    repetitions: int
    warmup_runs: int
    output_path: Optional[str] = None
    error: Optional[str] = None


class CorrectnessCheck(BaseModel):
    performed: bool
    pixel_agreement_pct: Optional[float] = None
    gradient_mean_abs_diff: Optional[float] = None
    threshold_pct: float = 98.0
    passed: Optional[bool] = None
    note: str = ""


class ExecutionResults(BaseModel):
    plan: ExecutionPlan
    image_info: ImageInfo
    gpu_name: Optional[str] = None
    cpu: Optional[DeviceResult] = None
    gpu: Optional[DeviceResult] = None
    correctness: CorrectnessCheck
    agent2_notes: str = ""


class AnalysisReport(BaseModel):
    run_id: Optional[str] = None
    winner: Literal["cpu", "gpu", "tie", "undetermined"]
    speedup_total: Optional[float] = None
    speedup_kernel_only: Optional[float] = None
    transfer_overhead_pct: Optional[float] = None
    throughput_cpu_mpix_s: Optional[float] = None
    throughput_gpu_mpix_s: Optional[float] = None
    per_stage_speedup: dict[str, float] = {}
    correctness_verdict: str
    explanation_markdown: str
    coa_concepts_used: list[str] = []
    report_path: Optional[str] = None
