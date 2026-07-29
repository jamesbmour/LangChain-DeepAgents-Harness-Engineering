"""
Episode 5 — Running Commands: the Shell Tool (and why it's dangerous)
=====================================================================

The agent gains the ability to run shell commands. There is NO built-in
`run_shell` in FilesystemBackend, so we write a custom `@tool` that:

  - Runs the command with cwd = CODEIT_WORKDIR (via workspace_root()).
  - Captures stdout, stderr, and exit code.
  - Truncates long output so we don't blow the context window.
  - Returns a single readable string to the model.

NO gating this episode. The agent CAN `rm -rf` the workspace. That's the
cliffhanger — end the episode pointing at Episode 6 (the approval gate).

Be honest on camera: run_shell confines CWD, not the process. `rm -rf /`
would still try to delete the system if the user has permission. NEVER run
unsandboxed on a real repo.

Builds on: Episodes 1-4 (model, agent, FilesystemBackend, resolve_in_workspace).

Run:
    CODEIT_WORKDIR=./my_project python tutorial.py \
        "Install fastapi and uvicorn with pip, then run pytest."

Requires:
    pip install deepagents langchain-ollama rich
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

console = Console()

MAX_OUTPUT_CHARS = 20_000  # ~5k tokens; keeps the context window healthy


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


def workspace_root() -> Path:
    return Path(os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# 1. run_shell — a custom @tool. The docstring tells the model WHEN to use it
#    and WHAT the arg is. Tool docstrings are prompts.
#
#    We use shell=True so the model can pass one string with pipes/redirects.
#    This is MORE dangerous (injection) but the agent is constructing the
#    command, and Episode 6 gates it. Trade-off: simplicity vs. safety.
# ─────────────────────────────────────────────────────────────────────────────
@tool
def run_shell(command: str) -> str:
    """Run a shell command in the workspace and return stdout+stderr+exit code.

    Use this to run tests, install packages, execute scripts, or inspect git state.
    Argument: a single shell command string (e.g. 'pytest -q' or 'pip install fastapi').
    The command runs with cwd set to the workspace root.
    Output is truncated if longer than ~20k chars.
    """
    cwd = str(workspace_root())
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True,
            timeout=int(os.getenv("CODEIT_SHELL_TIMEOUT", "120")),
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out.\nCommand: {command}"
    except Exception as e:
        return f"Error launching command: {type(e).__name__}: {e}"

    out = proc.stdout or ""
    err = proc.stderr or ""
    # We return non-zero exits as a STRING, not an exception — the model reads
    # "[exit 1]" and can self-correct (Episode 11 uses this for recovery).
    combined = f"$ {command}\n[exit {proc.returncode}]\n"
    if out:
        combined += f"--- stdout ---\n{out}\n"
    if err:
        combined += f"--- stderr ---\n{err}\n"
    if len(combined) > MAX_OUTPUT_CHARS:
        combined = combined[:MAX_OUTPUT_CHARS] + (
            f"\n... [truncated, {len(combined)-MAX_OUTPUT_CHARS} more chars]"
        )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build the agent — FilesystemBackend + our run_shell.
# ─────────────────────────────────────────────────────────────────────────────
def build_agent(workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    return create_deep_agent(
        model=get_model(),
        tools=[run_shell],
        system_prompt=(
            "You are CodeIt, a coding agent. Use run_shell to run tests, installs, "
            "and git commands. The command runs in the workspace. "
            "Be careful: destructive commands (rm -rf, git push -f) can't be undone."
        ),
        backend=backend,
        checkpointer=InMemorySaver(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Streaming driver (same shape as Eps 2-4).
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
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Run: echo hello from the shell"
    agent = build_agent()
    state = run(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)
    print(
        "\n⚠️  The agent just ran real shell commands. It COULD have run `rm -rf .` "
        "and deleted the workspace. Episode 6 adds the approval gate."
    )


if __name__ == "__main__":
    main()