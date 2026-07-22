"""Ep 1 — model factory + agent build tests.

Non-live tests never touch the network: they only exercise object
construction and error paths. Live tests (``@pytest.mark.live``) do a real
round-trip and are skipped when the provider is unavailable.
"""

import os

import pytest

from codeit.agent import build_agent
from codeit.model import get_model
from codeit.settings import Settings

OLLAMA_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
OPENAI_MODEL = os.getenv("OPENAI_LIVE_MODEL", "gpt-4o-mini")


def _base_settings(**overrides) -> Settings:
    base = dict(
        llm_provider="ollama",
        llm_model=OLLAMA_MODEL,
        ollama_base_url="http://localhost:11434",
        openai_api_key="",
        openai_base_url="",
        workdir="./workspace",
        auto_approve=False,
        max_iters=25,
    )
    base.update(overrides)
    return Settings(**base)


def test_get_model_ollama_no_network():
    """Construction only — no .invoke(), no network."""
    from langchain_ollama import ChatOllama

    model = get_model(_base_settings(llm_provider="ollama"))
    assert isinstance(model, ChatOllama)


def test_get_model_openai_missing_key_raises():
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        get_model(_base_settings(llm_provider="openai", openai_api_key=""))


def test_get_model_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_model(_base_settings(llm_provider="bedrock"))


def test_build_agent_returns_graph(fake_model):
    agent = build_agent(model=fake_model)
    assert hasattr(agent, "invoke")
    assert hasattr(agent, "stream")


@pytest.mark.live
def test_live_ollama():
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    except OSError as exc:
        pytest.skip(f"Ollama not running: {exc}")

    from codeit.agent import build_agent as _ba

    agent = _ba(model=get_model(_base_settings(llm_provider="ollama")))
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Say hello in one word."}]},
        config={"configurable": {"thread_id": "live-ollama"}},
    )
    assert result["messages"][-1].content


@pytest.mark.live
def test_live_openai():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")

    agent = build_agent(
        model=get_model(
            _base_settings(llm_provider="openai", openai_api_key=key, llm_model=OPENAI_MODEL)
        )
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Say hello in one word."}]},
        config={"configurable": {"thread_id": "live-openai"}},
    )
    assert result["messages"][-1].content
