"""
Episode 2 — The Agentic Loop, Made Visible
==========================================

We drive the Deep Agents graph with `agent.stream(stream_mode="updates",
version="v2")` so each step of the loop (model request → tool call → tool
result → final answer) prints live. A tiny `get_time` @tool proves the loop
fires: the model DECIDES to call it, the tool runs, the model reads the result
and answers. That's what makes it an agent, not a chatbot.

Builds on: Episode 1 (get_model, build_agent).

Run:
    LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
        python agentic-loop.py "What time is it? Use your tool."

Requires:
    pip install deepagents langchain-ollama rich
"""

from __future__ import annotations

import os
import sys

from deepagents import create_deep_agent

# Reuse the Episode 1 model factory + settings. In a real codebase these
# would live in codeit/ package; here we inline a trimmed copy so this file
# is self-contained.
from langchain.chat_models import init_chat_model

# `tool` decorator turns a plain function into a LangChain tool whose
# docstring becomes the schema description the model sees.
# Verified: `from langchain.tools import tool` (langchain>=1.0).
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from rich.console import Console
from langgraph.checkpoint.memory import MemorySaver

console = Console()


def get_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model_name = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai.")
        return init_chat_model(model=model_name, model_provider="openai")
    return init_chat_model(
        model=model_name,
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. A demo tool — the docstring is FOR THE MODEL. It tells the model WHEN
#    to call this tool and what it returns. Tool docstrings are prompts.
# ─────────────────────────────────────────────────────────────────────────────
@tool
def get_time() -> str:
    """Return the current time. Use this when the user asks for the time."""
    import datetime

    return datetime.datetime.now().isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build the agent — same as Ep 1, but now we register get_time.
# ─────────────────────────────────────────────────────────────────────────────
def build_agent():
    return create_deep_agent(
        model=get_model(),
        tools=[get_time],
        system_prompt="You are CodeIt, a helpful coding assistant. Use tools when useful.",
        checkpointer=MemorySaver(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. The streaming driver — prints each graph node event live with rich.
# ─────────────────────────────────────────────────────────────────────────────
def _config(thread_id: str) -> dict:
    # NOTE: recursion_limit is a TOP-LEVEL config key, NOT inside configurable.
    # It counts super-steps (model call + tool exec ≈ 2). We cap it so a
    # looping model can't hang the viewer's machine.
    max_iters = int(os.getenv("CODEIT_MAX_ITERS", "25"))
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_iters * 2,
    }


def _print_event(chunk) -> None:
    """Pretty-print one v2 stream chunk. v2 chunks are dicts with type/ns/data."""
    kind = chunk.get("type")
    if kind != "updates":
        return
    for _node_name, state in chunk.get("data", {}).items():
        if not isinstance(state, dict):
            continue
        for msg in state.get("messages", []):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    console.print(f"[cyan]tool call:[/cyan] {tc['name']}({tc['args']})")
            elif isinstance(msg, AIMessage) and msg.content:
                console.print(f"[green]assistant:[/green] {msg.content}")


def run(agent, prompt: str, thread_id: str = "default") -> dict:
    """Drive the agent with streaming so each node event prints live."""
    config = _config(thread_id)
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
            stream_mode="updates",
            version="v2",  # v2 = typed dicts; v1 = bare {node: state}
        ):
            _print_event(chunk)
    except Exception as e:
        # GraphRecursionError or model errors — print clearly, don't crash.
        console.print(f"[red]error:[/red] {type(e).__name__}: {e}")
    # Fetch the final state for the return value.
    return agent.get_state(config).values


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI demo.
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What time is it? Use your tool."
    agent = build_agent()
    state = run(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)


if __name__ == "__main__":
    main()
