"""
Episode 3 — Giving Your Agent Hands: Filesystem Tools
====================================================

We attach a `FilesystemBackend(root_dir=..., virtual_mode=True)` and the
built-in read-only filesystem tools — `ls`, `read_file`, `glob`, `grep` —
appear automatically. Zero hand-written tool code from us for those four.

We DO add one custom `@tool`, `read_summary`, that wraps the idea of
read_file with line truncation. It shows how to register a custom tool
alongside the built-ins.

SECURITY: `virtual_mode=True` is the sandbox. The default `virtual_mode=False`
provides NO security even with `root_dir` set. Always pass `virtual_mode=True`.

Builds on: Episodes 1-2 (get_model, build_agent, streaming).

Run:
    # Point CODEIT_WORKDIR at any small project folder, then:
    CODEIT_WORKDIR=./my_project python tutorial.py "What's in this project? What does the main file do?"

Requires:
    pip install deepagents langchain-ollama rich
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

# FilesystemBackend is the Deep Agents backend that provides ls/read_file/
# write_file/edit_file/glob/grep automatically when attached.
# Verified: `from deepagents.backends import FilesystemBackend` (deep-agents-core skill).
from deepagents.backends import FilesystemBackend
from deepagents import create_deep_agent

console = Console()


def get_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    name = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai.")
        return init_chat_model(model=name, model_provider="openai")
    return init_chat_model(
        model=name, model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. A custom tool — `read_summary`. Wraps read_file with truncation policy.
#    The docstring is FOR THE MODEL: "Use this when you want a quick overview…"
# ─────────────────────────────────────────────────────────────────────────────
@tool
def read_summary(path: str) -> str:
    """Read a file and return its first 50 lines plus a truncation note.

    Use this when you want a quick overview of a file without reading the whole thing.
    Argument: path relative to the workspace root (e.g. 'main.py' or 'src/app.py').
    """
    root = Path(os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    try:
        text = (root / path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: {path} not found in workspace."
    except Exception as e:
        return f"Error reading {path}: {type(e).__name__}: {e}"
    lines = text.splitlines()
    if len(lines) <= 50:
        return text
    return "\n".join(lines[:50]) + f"\n... [truncated, {len(lines)-50} more lines]"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build the agent — attach a FilesystemBackend with the sandbox ON.
# ─────────────────────────────────────────────────────────────────────────────
def build_agent(workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # virtual_mode=True blocks ../, ~, and absolute paths outside root.
    # NEVER set virtual_mode=False with a real root_dir. It is insecure by default.
    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    return create_deep_agent(
        model=get_model(),
        tools=[read_summary],          # our custom tool alongside the built-ins
        system_prompt="You are CodeIt, a helpful coding assistant. Explore the workspace with ls and read_file.",
        backend=backend,               # ← ls/read_file/glob/grep appear automatically
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Streaming driver (same shape as Ep 2).
# ─────────────────────────────────────────────────────────────────────────────
def _config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": int(os.getenv("CODEIT_MAX_ITERS", "25")) * 2,
    }


def _print_event(chunk) -> None:
    if chunk.get("type") != "updates":
        return
    for _node, state in chunk.get("data", {}).items():
        if not isinstance(state, dict):
            continue
        for msg in state.get("messages", []):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    console.print(f"[cyan]tool call:[/cyan] {tc['name']}({tc['args']})")
            elif isinstance(msg, AIMessage) and msg.content:
                console.print(f"[green]assistant:[/green] {msg.content}")


def run(agent, prompt: str, thread_id: str = "default") -> dict:
    config = _config(thread_id)
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config, stream_mode="updates", version="v2",
        ):
            _print_event(chunk)
    except Exception as e:
        console.print(f"[red]error:[/red] {type(e).__name__}: {e}")
    return agent.get_state(config).values


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What files are in this project?"
    agent = build_agent()
    state = run(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)


if __name__ == "__main__":
    main()