"""Shared LLM interface. Owned by System 1. Signatures are FROZEN.

Every agent talks to the model ONLY through LLMClient.chat(). To swap Ollama for a
cloud API later, write another class with the same chat() method; no agent changes.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

DEFAULT_MODEL = "llama3.2:3b"


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        json_schema: dict | None = None,
    ) -> dict:
        """Returns {"content": str, "tool_calls": [{"name": str, "arguments": dict}]}"""
        ...


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an object (ollama returns objects)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class OllamaClient:
    """Local LLM through Ollama, forced onto the CPU.

    num_gpu=0 means zero model layers on the GPU. The GTX 1650 stays free so the
    CPU-vs-GPU benchmark timings are never disturbed by the LLM.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        num_gpu: int = 0,
        temperature: float = 0.0,
        keep_alive: str = "10m",
        host: str | None = None,
    ) -> None:
        import ollama  # imported here so tests that use a fake LLM need no Ollama

        self._client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model
        self.num_gpu = num_gpu
        self.temperature = temperature
        self.keep_alive = keep_alive

    def unload(self) -> bool:
        """Ask Ollama to drop the model from RAM now (it reloads on the next chat call).

        Used before a benchmark so the model is not sitting in memory while we time things.
        Returns True if Ollama accepted the request. Never raises."""
        try:
            self._client.generate(model=self.model, keep_alive=0)
            return True
        except Exception:
            return False

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        json_schema: dict | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {"num_gpu": self.num_gpu, "temperature": self.temperature},
            "keep_alive": self.keep_alive,
        }
        if tools:
            kwargs["tools"] = tools
        if json_schema:
            kwargs["format"] = json_schema  # Ollama constrains output to this schema

        response = self._client.chat(**kwargs)
        message = _get(response, "message")
        content = _get(message, "content", "") or ""

        tool_calls: list[dict] = []
        for call in _get(message, "tool_calls", None) or []:
            fn = _get(call, "function")
            args = _get(fn, "arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            tool_calls.append({"name": _get(fn, "name"), "arguments": dict(args)})

        return {"content": content, "tool_calls": tool_calls}
