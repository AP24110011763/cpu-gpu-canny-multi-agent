"""Reusable tool-calling loop. Owned by System 1. Signature is FROZEN.

The loop is what turns "an LLM" into "an agent":
    ask the LLM -> it may request a tool -> we run the tool -> give the result
    back -> ask again ... until the LLM answers in plain text, calls a tool listed in
    stop_after (optional, e.g. a "submit" tool), or max_steps is hit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class LoopResult:
    final_text: str
    trace: list[dict] = field(default_factory=list)
    steps: int = 0
    hit_step_limit: bool = False

    @property
    def tools_called(self) -> list[str]:
        return [t["tool"] for t in self.trace if "tool" in t]


def run_tool_loop(
    llm: Any,
    system_prompt: str,
    user_message: str,
    tool_registry: dict[str, Callable],
    tool_schemas: list[dict],
    max_steps: int = 8,
    stop_after: set[str] | None = None,
) -> LoopResult:
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    trace: list[dict] = []

    for step in range(1, max_steps + 1):
        reply = llm.chat(messages, tools=tool_schemas)
        content = reply.get("content", "") or ""
        calls = reply.get("tool_calls") or []

        if not calls:  # plain-text answer: the agent is done
            trace.append({"step": step, "final": content})
            return LoopResult(content, trace, step, False)

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in calls
                ],
            }
        )

        for call in calls:
            name, args = call["name"], call.get("arguments") or {}
            func = tool_registry.get(name)
            if func is None:
                result: Any = {"error": f"unknown tool '{name}'"}
            else:
                try:
                    result = func(**args)
                except Exception as exc:  # a failing tool must not crash the agent
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            trace.append({"step": step, "tool": name, "arguments": args, "result": result})
            messages.append(
                {"role": "tool", "tool_name": name, "content": json.dumps(result, default=str)}
            )

        # a "terminal" tool (e.g. submit_plan) ends the loop without another LLM call
        if stop_after and any(c["name"] in stop_after for c in calls):
            return LoopResult(content, trace, step, False)

    return LoopResult("", trace, max_steps, True)
