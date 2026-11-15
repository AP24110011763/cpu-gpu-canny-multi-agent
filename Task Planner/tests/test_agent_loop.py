from llm.agent_loop import run_tool_loop
from llm.fake_client import ScriptedLLM


def call(name, **args):
    return {"content": "", "tool_calls": [{"name": name, "arguments": args}]}


def text(t):
    return {"content": t, "tool_calls": []}


def test_plain_answer_no_tools():
    r = run_tool_loop(ScriptedLLM([text("hello")]), "sys", "hi", {}, [])
    assert r.final_text == "hello" and r.steps == 1 and r.tools_called == []


def test_tool_is_called_and_result_returned_to_llm():
    llm = ScriptedLLM([call("add", a=2, b=3), text("sum is 5")])
    r = run_tool_loop(llm, "sys", "add", {"add": lambda a, b: {"sum": a + b}}, [{"x": 1}])
    assert r.final_text == "sum is 5" and r.tools_called == ["add"]
    assert r.trace[0]["result"] == {"sum": 5}
    # the second LLM call must have seen the tool result
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs and "5" in tool_msgs[0]["content"]


def test_unknown_tool_does_not_crash():
    r = run_tool_loop(ScriptedLLM([call("ghost"), text("ok")]), "s", "u", {}, [])
    assert "unknown tool" in r.trace[0]["result"]["error"] and r.final_text == "ok"


def test_tool_exception_is_reported_not_raised():
    def boom(**_):
        raise ValueError("bad")

    r = run_tool_loop(ScriptedLLM([call("boom"), text("recovered")]), "s", "u", {"boom": boom}, [])
    assert "ValueError" in r.trace[0]["result"]["error"] and r.final_text == "recovered"


def test_step_limit():
    llm = ScriptedLLM([call("t")] * 3)
    r = run_tool_loop(llm, "s", "u", {"t": lambda: {}}, [], max_steps=3)
    assert r.hit_step_limit and r.final_text == "" and r.steps == 3


def test_stop_after_ends_loop_without_extra_llm_call():
    llm = ScriptedLLM([call("submit", x=1)])  # only ONE scripted reply: a second call would raise
    r = run_tool_loop(llm, "s", "u", {"submit": lambda x: {"ok": x}}, [], stop_after={"submit"})
    assert r.tools_called == ["submit"] and not r.hit_step_limit and len(llm.calls) == 1
