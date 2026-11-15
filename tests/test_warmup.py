"""Warm-up policy: at least N passes AND at least S seconds, so an idle GPU/CPU clock has time to climb."""
import cv2
import numpy as np

import tools.canny_common as cc
from tools.cpu_canny import run_cpu_canny


def test_runs_at_least_the_planned_passes_even_with_no_time_requirement():
    calls = []
    assert cc.warm_up(lambda: calls.append(1), 3, 0.0) == 3 and len(calls) == 3


def test_keeps_going_until_the_time_has_passed(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(cc.time, "perf_counter", lambda: clock["t"])

    def step():
        clock["t"] += 0.25                       # each pass "takes" 0.25 s
    assert cc.warm_up(step, 3, 1.0) == 4         # 3 planned passes are not enough: 1 s needs 4
    clock["t"] = 0.0
    assert cc.warm_up(step, 10, 1.0) == 10       # the planned count wins when it is larger


def test_zero_passes_and_zero_time_does_nothing_and_a_runaway_step_is_capped(monkeypatch):
    assert cc.warm_up(lambda: 1 / 0, 0, 0.0) == 0
    monkeypatch.setattr(cc, "MAX_WARMUP_PASSES", 7)
    monkeypatch.setattr(cc.time, "perf_counter", lambda: 0.0)       # time never advances
    assert cc.warm_up(lambda: None, 0, 5.0) == 7


def test_default_seconds_come_from_the_module_constant(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(cc.time, "perf_counter", lambda: clock["t"])
    monkeypatch.setattr(cc, "MIN_WARMUP_SECONDS", 0.5)
    def step():
        clock["t"] += 0.1
    assert cc.warm_up(step, 1) == 5


def test_result_reports_the_warmup_passes_actually_done(tmp_path, monkeypatch):
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1)))
    res, _, _ = run_cpu_canny(str(p), 100, 200, repetitions=2, warmup_runs=1, min_warmup_seconds=0.05)
    assert res.warmup_runs >= 1 and res.repetitions == 2
    res0, _, _ = run_cpu_canny(str(p), 100, 200, repetitions=2, warmup_runs=2, min_warmup_seconds=0.0)
    assert res0.warmup_runs == 2
