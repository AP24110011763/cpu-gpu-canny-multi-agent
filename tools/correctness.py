"""Agent 2 tool: do the CPU and GPU produce the same edges? (A fast wrong answer is not a result.)

Two measures, because one alone can be fooled:
  * pixel_agreement_pct - % of ALL pixels where both maps agree edge/not-edge. Required by the
    shared contract, but edges are only ~1% of a typical image, so an all-black output would
    already score ~99%.
  * edge_overlap        - edge-only agreement WITH a 1-pixel tolerance: min(precision, recall),
    where an edge pixel counts as matched if the other map has an edge within 1 pixel. This
    catches the "all black" loophole, so it must also pass (>= min_edge_overlap).
    Why a tolerance: Canny edges are 1 pixel thin, and the CPU and GPU round the blur slightly
    differently (float vs integer arithmetic). On a flat noisy area that can move an edge by one
    pixel, which is not an error. (Measured: exact overlap 0.76, within 1 pixel 1.00.)
gradient_mean_abs_diff compares the gradient magnitudes before thresholding.
"""
from __future__ import annotations

import cv2
import numpy as np

from schemas import CorrectnessCheck

_NEIGHBOURHOOD = np.ones((3, 3), np.uint8)


def _edge_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(overlap, precision, recall) of edge maps a and b with 1-pixel tolerance."""
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 and nb == 0:
        return 1.0, 1.0, 1.0
    if na == 0 or nb == 0:
        return 0.0, 0.0, 0.0
    near_a = cv2.dilate(a.astype(np.uint8), _NEIGHBOURHOOD) > 0
    near_b = cv2.dilate(b.astype(np.uint8), _NEIGHBOURHOOD) > 0
    precision = float((b & near_a).sum() / nb)     # GPU edges that the CPU also has nearby
    recall = float((a & near_b).sum() / na)        # CPU edges that the GPU also has nearby
    return min(precision, recall), precision, recall


def check_correctness(cpu_edges, gpu_edges, cpu_mag=None, gpu_mag=None,
                      threshold_pct: float = 98.0, min_edge_overlap: float = 0.95) -> CorrectnessCheck:
    if cpu_edges is None or gpu_edges is None:
        return CorrectnessCheck(performed=False, threshold_pct=threshold_pct,
                                note="Need both CPU and GPU outputs to compare; one is missing.")
    if cpu_edges.shape != gpu_edges.shape:
        return CorrectnessCheck(performed=True, passed=False, pixel_agreement_pct=0.0,
                                threshold_pct=threshold_pct,
                                note=f"Output shapes differ: {cpu_edges.shape} vs {gpu_edges.shape}.")
    a, b = cpu_edges > 0, gpu_edges > 0
    agreement = float((a == b).mean() * 100.0)
    overlap, precision, recall = _edge_overlap(a, b)
    grad = None
    if cpu_mag is not None and gpu_mag is not None and cpu_mag.shape == gpu_mag.shape:
        grad = float(np.mean(np.abs(cpu_mag.astype(np.float64) - gpu_mag.astype(np.float64))))
    passed = agreement >= threshold_pct and overlap >= min_edge_overlap
    return CorrectnessCheck(
        performed=True,
        pixel_agreement_pct=round(agreement, 3),
        gradient_mean_abs_diff=None if grad is None else round(grad, 5),
        threshold_pct=threshold_pct,
        passed=passed,
        note=f"edge overlap within 1 px {overlap:.3f} (precision {precision:.3f}, recall {recall:.3f}), "
             f"must be >= {min_edge_overlap}; edge pixels cpu={int(a.sum())} gpu={int(b.sum())}",
    )
