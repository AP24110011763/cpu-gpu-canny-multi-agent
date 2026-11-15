"""Agent 3 (analyst) tests. A scripted LLM stands in for Ollama; the checks are done by real code."""
from pathlib import Path

import pytest

from agents import analyst as an
from agents.analyst import analyze, find_problems, run_analyst, template_explanation
from llm.fake_client import ScriptedLLM
from schemas import AnalysisReport, ExecutionResults
from tools.metrics import check_correctness_status, compute_metrics, get_stage_breakdown

FIXTURES = Path(__file__).parent / "fixtures"
ALL = ["mock_gpu_wins_large_image", "mock_cpu_wins_small_image", "mock_gpu_missing",
       "mock_correctness_failed", "mock_gpu_error"]


def load(name) -> ExecutionResults:
    return ExecutionResults.model_validate_json((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def call(name, **args):
    return {"name": name, "arguments": args}


def gathering(*concepts):
    calls = [call("get_metrics"), call("get_stage_breakdown"), call("check_correctness_status")]
    calls += [call("lookup_coa_concept", concept=c) for c in concepts]
    return {"content": "", "tool_calls": calls}


def submitting(text):
    return {"content": "", "tool_calls": [call("submit_explanation", explanation_markdown=text)]}


def good_text(res):
    return template_explanation(compute_metrics(res), get_stage_breakdown(res), check_correctness_status(res))[0]


def allowed_for(res):
    return an.allowed_numbers(res, compute_metrics(res), get_stage_breakdown(res), check_correctness_status(res))


@pytest.mark.parametrize("name", ALL)
def test_template_passes_its_own_checker(name):
    res = load(name)
    assert find_problems(good_text(res), allowed_for(res), compute_metrics(res)["winner"]) == []


@pytest.mark.parametrize("name", ALL)
def test_good_llm_text_is_accepted_and_numbers_come_from_code(name):
    res = load(name)
    llm = ScriptedLLM([gathering("amdahl_law", "memory_hierarchy_and_pcie"), submitting(good_text(res))])
    rep, source = analyze(res, llm)
    m = compute_metrics(res)
    assert isinstance(rep, AnalysisReport) and rep.winner == m["winner"] and rep.speedup_total == m["speedup_total"]
    assert rep.coa_concepts_used == ["amdahl_law", "memory_hierarchy_and_pcie"]
    assert "written by Agent 3's language model" in source and "language model" in rep.explanation_markdown


def test_gpu_wins_report_fields_are_the_hand_calculated_ones():
    res = load("mock_gpu_wins_large_image")
    rep = run_analyst(res, ScriptedLLM([gathering("amdahl_law", "memory_hierarchy_and_pcie"), submitting(good_text(res))]))
    assert rep.winner == "gpu" and rep.speedup_total == pytest.approx(131 / 29, abs=1e-3)
    assert rep.per_stage_speedup["non_max_suppression"] == 50.0 and rep.correctness_verdict.startswith("PASSED")


def test_invented_number_is_rejected_then_retry_is_accepted():
    res = load("mock_gpu_wins_large_image")
    bad = "## What was measured\n- The GPU was 37.2x faster than the CPU on this image, a huge difference.\n"
    llm = ScriptedLLM([gathering("amdahl_law", "memory_hierarchy_and_pcie"), submitting(bad), gathering("amdahl_law", "memory_hierarchy_and_pcie"), submitting(good_text(res))])
    rep, source = analyze(res, llm)
    assert "second try" in source and "37.2" not in rep.explanation_markdown
    retry_task = llm.calls[2]["messages"][1]["content"]                       # the retry says why it was rejected
    assert "37.2" in retry_task and "rejected" in retry_task


def test_two_bad_attempts_fall_back_to_the_code_template():
    res = load("mock_gpu_wins_large_image")
    bad = "## What was measured\n- The GPU was 37.2x faster than the CPU on this image, a huge difference.\n"
    rep, source = analyze(res, ScriptedLLM([gathering("amdahl_law", "memory_hierarchy_and_pcie"), submitting(bad), gathering("amdahl_law", "memory_hierarchy_and_pcie"), submitting(bad)]))
    assert source.startswith("built by code") and "37.2" not in rep.explanation_markdown
    assert "not in the measured data" in source and "4.517" in rep.explanation_markdown and rep.coa_concepts_used          # template still has the real numbers


def test_llm_that_never_submits_but_answers_in_plain_text_is_still_verified():
    res = load("mock_gpu_wins_large_image")
    rep, source = analyze(res, ScriptedLLM([gathering("amdahl_law", "memory_hierarchy_and_pcie"), {"content": good_text(res), "tool_calls": []}]))
    assert "language model" in source and "Caveats" in rep.explanation_markdown


def test_llm_that_hits_the_step_limit_falls_back():
    res = load("mock_gpu_wins_large_image")
    loops = [gathering("amdahl_law", "memory_hierarchy_and_pcie") for _ in range(10)]
    rep, source = analyze(res, ScriptedLLM(loops + loops))
    assert source.startswith("built by code") and "Caveats" in rep.explanation_markdown


def test_llm_crash_does_not_crash_the_analyst():
    class Down:
        def chat(self, *a, **k):
            raise ConnectionError("ollama not running")
    rep, source = analyze(load("mock_gpu_wins_large_image"), Down())
    assert "ConnectionError" in source and rep.winner == "gpu" and rep.explanation_markdown


def test_unmeasured_claims_are_rejected():
    res = load("mock_gpu_wins_large_image")
    for phrase in ("The GPU had a low cache miss rate.", "The GPU occupancy was good.", "No warp divergence occurred."):
        assert find_problems("## Why\n- " + phrase + " " * 30, allowed_for(res), "gpu")


def test_no_winner_may_be_named_when_the_result_is_undetermined():
    res = load("mock_correctness_failed")
    text = "## Caveats\n- The outputs differ, but the GPU is faster anyway so we say it won."
    assert any("faster device" in p for p in find_problems(text, allowed_for(res), "undetermined"))
    assert find_problems(good_text(res), allowed_for(res), "undetermined") == []


def test_small_counting_numbers_are_free_but_small_speedups_are_not():
    res = load("mock_gpu_wins_large_image")
    assert find_problems("There are 4 stages and 3 of them run on the GPU, which is plenty of text.", allowed_for(res), "gpu") == []
    assert find_problems("The GPU was 13x faster, which is plenty of words to pass the length.", allowed_for(res), "gpu")


def test_rounded_real_numbers_are_accepted():
    res = load("mock_gpu_wins_large_image")                                      # speedup 4.517
    assert find_problems("The speedup was about 4.5x and roughly 5x overall, quoted from the tools output.", allowed_for(res), "gpu") == []


@pytest.mark.parametrize("name,words", [
    ("mock_cpu_wins_small_image", "CPU faster"), ("mock_gpu_missing", "Only the CPU"),
    ("mock_correctness_failed", "no winner declared"), ("mock_gpu_error", "GPU run failed")])
def test_template_states_each_situation_plainly(name, words):
    md, used = template_explanation(compute_metrics(load(name)), get_stage_breakdown(load(name)), check_correctness_status(load(name)))
    assert words in md and used


def test_template_labels_architecture_and_hypothesis_honestly():
    res = load("mock_gpu_wins_large_image")
    md, used = template_explanation(compute_metrics(res), get_stage_breakdown(res), check_correctness_status(res))
    assert "architecture explanation, not measured here" in md and "hypothesis" in md and "kernel_launch_overhead" in used
    assert "cv2.Canny ran 7.25x faster" in md


def test_concept_notes_contain_no_numbers():
    import re
    for name, c in an.CONCEPTS.items():
        assert not re.search(r"\d", c["note"]), name


def test_unknown_concept_returns_an_error_not_a_crash():
    res = load("mock_gpu_wins_large_image")
    llm = ScriptedLLM([{"content": "", "tool_calls": [call("lookup_coa_concept", concept="magic")]},
                       submitting(good_text(res))])
    rep, _ = analyze(res, llm)
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "unknown concept" in tool_msg


def test_text_without_concept_lookups_is_rejected_and_the_retry_carries_the_notes():
    res = load("mock_gpu_wins_large_image")
    llm = ScriptedLLM([gathering(), submitting(good_text(res)),                      # attempt 1: no lookups
                       gathering("amdahl_law", "memory_hierarchy_and_pcie"), submitting(good_text(res))])
    rep, source = analyze(res, llm)
    assert "second try" in source and rep.coa_concepts_used == ["amdahl_law", "memory_hierarchy_and_pcie"]
    retry = llm.calls[2]["messages"][1]["content"]
    assert "at least 2" in retry and "serial and runs on the CPU" in retry and "PCIe" in retry


def test_claim_that_the_gpu_speeds_up_hysteresis_is_rejected():
    res = load("mock_gpu_wins_large_image")
    bad = ("## Why\n- The hysteresis stage, which is not as heavy as the others, still benefits from the GPU's "
           "parallel processing, contributing to the overall speedup.")
    assert any("hysteresis" in p for p in find_problems(bad, allowed_for(res), "gpu"))
    ok = ("## Why\n- Hysteresis is serial and runs on the CPU in both paths, so it limits the overall speedup "
          "and stays the same on both devices.")
    assert find_problems(ok, allowed_for(res), "gpu") == []
