"""
Episode 1 — Your Agent's Brain: One File, Two Providers
=======================================================

The smallest possible Deep Agent: build a chat model for either Ollama or
OpenAI (switched by an env var), wrap it in create_deep_agent, and print
the reply. No tools, no streaming, no filesystem — just prove the brain works.

Run:
    LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b python tutorial.py "Say hello."
    LLM_PROVIDER=openai  LLM_MODEL=gpt-4o-mini         python tutorial.py "Say hello."

Requires:
    pip install deepagents langchain-ollama langchain-openai python-dotenv
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# create_deep_agent is the Deep Agents harness entry point.
# Verified: `from deepagents import create_deep_agent` (deepagents-core skill).
from deepagents import create_deep_agent

# load_dotenv reads a local .env file once at import time, if present.
# It's optional — env vars on the command line also work. (python-dotenv)
from dotenv import load_dotenv

# init_chat_model is LangChain's provider-agnostic factory.
# Verified: `from langchain.chat_models import init_chat_model` (langchain>=1.0).
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Settings — a tiny typed config loaded from the environment.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)  # frozen = immutable; safe to pass anywhere
class Settings:
    llm_provider: str  # "ollama" | "openai"
    llm_model: str  # e.g. "qwen2.5-coder:7b" or "gpt-4o-mini"
    ollama_base_url: str
    openai_api_key: str
    openai_base_url: str


def get_settings() -> Settings:
    """Read settings from env, with sensible defaults for local Ollama."""
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5-coder:7b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model factory — one function, two providers, switched by env only.
# ─────────────────────────────────────────────────────────────────────────────
def get_model(settings: Settings | None = None) -> BaseChatModel:
    """Return a chat model for the configured provider.

    We pre-build the model ourselves (instead of passing a "provider:model"
    string to create_deep_agent) so WE own the error messages — e.g. a clear
    'OPENAI_API_KEY is required' hint instead of a cryptic stack trace.
    """
    s = settings or get_settings()

    if s.llm_provider == "ollama":
        # Ollama runs locally; no API key needed. Make sure `ollama serve` is up.
        return init_chat_model(
            model=s.llm_model,
            model_provider="ollama",
            base_url=s.ollama_base_url,
        )

    if s.llm_provider == "openai":
        if not s.openai_api_key:
            # Never log the key. The error tells the viewer exactly what to fix.
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai. Set it in .env or `export OPENAI_API_KEY=...`.")
        kwargs: dict = {"model": s.llm_model, "model_provider": "openai"}
        if s.openai_base_url:  # optional — for Azure/OpenAI-compatible endpoints
            kwargs["base_url"] = s.openai_base_url
        return init_chat_model(**kwargs)

    raise ValueError(f"Unknown LLM_PROVIDER={s.llm_provider!r}. Use 'ollama' or 'openai'.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agent builder — thin wrapper around create_deep_agent.
# ─────────────────────────────────────────────────────────────────────────────
def build_agent(model: BaseChatModel | None = None, system_prompt: str | None = None):
    """Build a tool-free Deep Agent. Just the brain.

    create_deep_agent returns a compiled LangGraph graph with .invoke(),
    .stream(), .astream(), .get_state(). We'll use those in later episodes.
    """
    m = model or get_model()
    # tools=[] for now — the harness still ships built-in planning/filesystem
    # middleware, but with no custom tools the agent just talks.
    return create_deep_agent(
        model=m,
        tools=[],
        system_prompt=system_prompt or "You are CodeIt, a helpful coding assistant.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI demo — send one prompt, print the reply.
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello in one sentence."

    agent = build_agent()
    # thread_id identifies this conversation. Later episodes reuse it for
    # memory and human-in-the-loop interrupts.
    config = {"configurable": {"thread_id": "demo"}}

    # invoke() runs the full agent loop (model → tool → model → ...) to completion.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    )

    # The final state holds every message; the last one is the agent's reply.
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
