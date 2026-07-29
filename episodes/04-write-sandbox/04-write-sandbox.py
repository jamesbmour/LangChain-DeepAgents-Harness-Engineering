"""
Episode 4 — Writing Code: `write_file` + the Workspace Sandbox
==============================================================

The shortest episode. `write_file` is already provided by FilesystemBackend
— we don't write any new tool code. The work here is:

  1. Reinforce the sandbox story (virtual_mode=True earns its keep).
  2. Add `resolve_in_workspace(path) -> Path` — a helper that custom tools
     in later episodes (shell, edit wrapper, repo map) will use to share
     the backend's sandbox discipline.

The viewer asks the agent to write a file; the agent uses the built-in
`write_file`; the viewer opens the file on disk — real code, written by
the agent.

Builds on: Episode 3 (FilesystemBackend, custom tools).

Run:
    CODEIT_WORKDIR=./workspace python tutorial.py \
        "Create main.py with a FastAPI app: GET /hello returns {'msg':'hello'}."

Requires:
    pip install deepagents langchain-ollama rich
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

console = Console()


def get_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    name = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai.")
        return init_chat_model(model=name, model_provider="openai")
    return init_chat_model(
        model=name,
        model_provider="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. resolve_in_workspace — share the sandbox discipline across custom tools.
#    Mirrors what FilesystemBackend(virtual_mode=True) does internally, but
#    for OUR tools (run_shell in Ep 5, edit_file_safe in Ep 8, etc.).
# ─────────────────────────────────────────────────────────────────────────────
class PathEscapeError(PermissionError):
    """Raised when a resolved path leaves the workspace sandbox."""


def workspace_root() -> Path:
    return Path(os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()


def resolve_in_workspace(path: str | Path) -> Path:
    """Resolve `path` relative to CODEIT_WORKDIR, refusing to escape.

    Blocks `../`, `~`, and absolute paths outside the workspace root.
    Follows symlinks via .resolve(), then checks the result is still inside.
    """
    root = workspace_root()
    target = (root / path).resolve()
    try:
        target.relative_to(root)  # raises ValueError if outside root
    except ValueError:
        raise PathEscapeError(  # noqa: B904
            f"Path {path!r} resolves outside the workspace ({root}). Refusing."
        )
    return target


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build the agent — same as Ep 3. write_file is already exposed by the
#    backend; we don't add it ourselves. No new tools this episode.
# ─────────────────────────────────────────────────────────────────────────────
def build_agent(workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    return create_deep_agent(
        model=get_model(),
        tools=[],  # write_file/read_file/ls/etc. come from the backend
        system_prompt=(
            "You are CodeIt, a coding agent. Use write_file to CREATE new files. Use edit_file for changes to existing files. Keep files inside the workspace."
        ),
        backend=backend,
        checkpointer=InMemorySaver(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Streaming driver (same as Eps 2-3).
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
            config=config,
            stream_mode="updates",
            version="v2",
        ):
            _print_event(chunk)
    except Exception as e:
        console.print(f"[red]error:[/red] {type(e).__name__}: {e}")
    return agent.get_state(config).values


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Create main.py with a FastAPI app: GET /hello returns {'msg':'hello'}."
    agent = build_agent()
    state = run(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)
    # Show that the file actually landed on disk.
    out = workspace_root() / "main.py"
    if out.exists():
        print(f"\n(wrote {out} — {out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
