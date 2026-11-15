"""Records what really happened in a run, so the report can draw the workflow from facts instead of a picture.

TracingLLM wraps the real LLM client. It sits between the agents and the model, so it sees every call without
the agents' code changing: how many times each agent asked the model, how long the whole agent stage took, and
which tools the model asked for, in order. Anything else (unload, model name...) passes straight through.
The orchestrator marks where each agent starts; the trace is saved as outputs/<run_id>/trace.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

AGENTS = {"agent1": "Agent 1 · Task Planner", "agent2": "Agent 2 · Execution Agent", "agent3": "Agent 3 · Analysis Agent"}
ORDER = list(AGENTS)


class TracingLLM:
    def __init__(self, inner, clock=time.perf_counter) -> None:
        self._inner, self._clock = inner, clock
        self._stages: dict[str, dict] = {}
        self._current: str | None = None
        self._started = 0.0

    def __getattr__(self, name):                       # unload(), model, ... behave as on the real client
        return getattr(self._inner, name)

    def start_stage(self, key: str) -> None:
        self.end_stage()
        self._current, self._started = key, self._clock()
        self._stages.setdefault(key, {"seconds": 0.0, "llm_calls": 0, "llm_seconds": 0.0, "tools": []})

    def end_stage(self) -> None:
        if self._current:
            self._stages[self._current]["seconds"] += self._clock() - self._started
        self._current = None

    def chat(self, messages, tools=None, json_schema=None) -> dict:
        t0 = self._clock()
        try:
            reply = self._inner.chat(messages, tools=tools, json_schema=json_schema)
        finally:
            st = self._stages.get(self._current) if self._current else None
            if st is not None:
                st["llm_calls"] += 1
                st["llm_seconds"] += self._clock() - t0
        if st is not None:
            st["tools"] += [c.get("name", "?") for c in (reply.get("tool_calls") or [])]
        return reply

    def trace(self) -> dict:
        self.end_stage()
        stages = {k: {**v, "seconds": round(v["seconds"], 2), "llm_seconds": round(v["llm_seconds"], 2)}
                  for k, v in self._stages.items()}
        return {"version": 1, "stages": stages}


def start_stage(llm, key: str) -> None:
    """Mark the start of an agent if the llm records traces; a plain llm (tests) just ignores it."""
    fn = getattr(llm, "start_stage", None)
    if callable(fn):
        fn(key)


def get_trace(llm) -> dict | None:
    fn = getattr(llm, "trace", None)
    return fn() if callable(fn) else None


def merge_saved(trace: dict | None, path: Path) -> dict | None:
    """Keep the earlier agents' record when only Agent 3 is re-run on a saved run."""
    if trace is None:
        return None
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
        for k, v in old.get("stages", {}).items():
            trace["stages"].setdefault(k, v)
    except (OSError, ValueError, AttributeError):
        pass
    return trace
