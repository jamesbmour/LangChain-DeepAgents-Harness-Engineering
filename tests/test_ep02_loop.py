"""Ep 2 — the agentic loop, made visible.

All tests use ``FakeToolModel`` (a ``GenericFakeChatModel`` subclass that no-ops
``bind_tools``) so Deep Agents' middleware is happy. The model returns scripted
``AIMessage``+``tool_calls`` sequences — no network, no real LLM.
"""

from itertools import cycle

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from codeit.agent import build_agent, run


class FakeToolModel(GenericFakeChatModel):
    """``GenericFakeChatModel`` that accepts ``bind_tools`` (ignores them)."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ARG002
        return self


def _tc(name: str = "get_time", args: dict | None = None, cid: str = "c1") -> dict:
    return {"name": name, "args": args or {}, "id": cid, "type": "tool_call"}


def _model_calling_time_once() -> FakeToolModel:
    """Model calls ``get_time``, then returns a plain answer."""
    return FakeToolModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[_tc()]),
                AIMessage(content="The time has been retrieved."),
            ]
        )
    )


def _model_looping_forever() -> FakeToolModel:
    """Model calls ``get_time`` on every turn — never answers."""
    return FakeToolModel(messages=cycle([AIMessage(content="", tool_calls=[_tc()])]))


def _model_no_tool_call() -> FakeToolModel:
    """Model answers directly without calling a tool."""
    return FakeToolModel(messages=iter([AIMessage(content="hello there")]))


def test_tool_runs_then_loop_terminates(monkeypatch):
    """A model that calls a tool once, then answers, should terminate cleanly."""
    monkeypatch.setenv("CODEIT_MAX_ITERS", "5")
    agent = build_agent(model=_model_calling_time_once())
    state = run(agent, "what time is it?")
    assert state and "messages" in state
    last = state["messages"][-1]
    assert isinstance(last, AIMessage)
    assert not getattr(last, "tool_calls", None)  # no pending tool calls


def test_recursion_cap_enforced(monkeypatch):
    """A model that loops forever must NOT hang — ``recursion_limit`` trips it."""
    monkeypatch.setenv("CODEIT_MAX_ITERS", "2")
    agent = build_agent(model=_model_looping_forever())
    state = run(agent, "keep calling the tool")  # should print error, return {}
    # Reaching this line at all means we didn't hang. State may be {} (error path).
    assert isinstance(state, dict)


def test_no_tool_call_path(monkeypatch):
    """A model that answers directly should also terminate cleanly."""
    monkeypatch.setenv("CODEIT_MAX_ITERS", "5")
    agent = build_agent(model=_model_no_tool_call())
    state = run(agent, "just say hi")
    assert state and state["messages"]
    last = state["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content == "hello there"


@pytest.mark.live
def test_live_ollama_loop():
    """Live: the agent actually calls ``get_time`` on a real model."""
    import os
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    except OSError as exc:
        pytest.skip(f"Ollama not running: {exc}")

    from codeit.model import get_model
    from codeit.settings import Settings

    s = Settings(
        llm_provider="ollama",
        llm_model=os.getenv("LLM_MODEL", "qwen2.5-coder:7b"),
        ollama_base_url="http://localhost:11434",
        openai_api_key="",
        openai_base_url="",
        workdir="./workspace",
        auto_approve=False,
        max_iters=5,
    )
    agent = build_agent(model=get_model(s))
    state = run(agent, "What time is it? Use your tool.", thread_id="live-ep02")
    assert state and state.get("messages")
    assert state["messages"][-1].content
