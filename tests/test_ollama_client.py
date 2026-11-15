"""Checks OllamaClient's response parsing and options WITHOUT needing Ollama installed."""
from types import SimpleNamespace as NS

from llm.client import OllamaClient


def make_client(response):
    c = object.__new__(OllamaClient)  # skip __init__ (it imports ollama)
    c.model, c.num_gpu, c.temperature, c.keep_alive = "llama3.2:3b", 0, 0.0, "10m"
    c.sent = {}

    def fake_chat(**kwargs):
        c.sent = kwargs
        return response

    c._client = NS(chat=fake_chat)
    return c


def test_parses_object_style_tool_call():
    resp = NS(message=NS(content="", tool_calls=[NS(function=NS(name="inspect_image", arguments={"image_path": "a.png"}))]))
    c = make_client(resp)
    out = c.chat([{"role": "user", "content": "x"}], tools=[{"t": 1}])
    assert out == {"content": "", "tool_calls": [{"name": "inspect_image", "arguments": {"image_path": "a.png"}}]}
    assert c.sent["tools"] == [{"t": 1}]


def test_parses_dict_style_plain_text():
    c = make_client({"message": {"content": "hello"}})
    assert c.chat([]) == {"content": "hello", "tool_calls": []}


def test_cpu_only_and_schema_are_sent():
    c = make_client({"message": {"content": "{}"}})
    c.chat([], json_schema={"type": "object"})
    assert c.sent["options"]["num_gpu"] == 0           # LLM must stay off the GPU
    assert c.sent["format"] == {"type": "object"}
    assert "tools" not in c.sent


def test_unload_asks_ollama_to_drop_the_model():
    c = make_client({"message": {"content": ""}})
    seen = {}
    c._client = NS(generate=lambda **kw: seen.update(kw))
    assert c.unload() is True
    assert seen == {"model": "llama3.2:3b", "keep_alive": 0}


def test_unload_never_raises():
    c = make_client({"message": {"content": ""}})

    def boom(**kw):
        raise ConnectionError("ollama is not running")
    c._client = NS(generate=boom)
    assert c.unload() is False
