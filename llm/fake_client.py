"""A scripted stand-in for the LLM, used by tests (no Ollama needed).

Pass a list of replies. Each reply is a dict {"content": str, "tool_calls": [...]}
or a function(messages) -> dict. Replies are returned in order.
"""
from __future__ import annotations

import copy
from typing import Any, Callable


class ScriptedLLM:
    def __init__(self, replies: list[dict | Callable[[list[dict]], dict]]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, json_schema=None) -> dict:
        self.calls.append(
            {"messages": copy.deepcopy(messages), "tools": tools, "json_schema": json_schema}
        )
        if not self.replies:
            raise RuntimeError("ScriptedLLM ran out of scripted replies")
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply
