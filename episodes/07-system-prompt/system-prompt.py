"""
Episode 7 — The System Prompt: engineering personality & rules
==============================================================

The agent gets a REAL system prompt — personality, tool-use policy, safety
rules, editing conventions. We also load project-specific context from
`AGENTS.md` (Deep Agents' own convention) or `CODEIT.md` fallback in the
workspace, and compose the two. The harness's own middleware injects its own
tool descriptions on top of our system_prompt; we ADD policy, not replace it.

Builds on: Episodes 1-6.

Run:
    # Add an AGENTS.md to your workspace first, e.g.:
    #   echo '# My project\\nUse FastAPI. The bug is in main.py.' > workspace/AGENTS.md
    CODEIT_WORKDIR=./workspace python tutorial.py "What is this project? Fix the failing test."

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

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

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


# 1. SYSTEM_PROMPT — written for the model, not for the viewer. Imperative, short.
SYSTEM_PROMPT = """You are CodeIt, a terminal coding agent.

# Role
You help the user write, edit, and run code in their workspace. You read files,
write files, run shell commands, and plan multi-step tasks.

# Tool-use policy
- Prefer the built-in filesystem tools (ls, read_file, grep, glob) for exploration.
- Use write_file only for new files. Use edit_file for changes to existing files.
- Use run_shell for tests, installs, and git operations. The user will be asked
  to approve mutating or destructive commands — prefer the least destructive option.
- Use write_todos to plan any task that needs more than one step.

# Safety
- Never run destructive commands (rm -rf, git push -f, dd, mkfs) without explaining why.
- If a command might modify files outside the workspace, say so and stop.
- The user can reject any action. Accept rejection gracefully and try a safer approach.

# Editing rules
- For small targeted changes, use edit_file (search-replace). Never rewrite a whole
  file to change a few lines.
- After editing code that has tests, run the tests with run_shell('pytest -q') and fix failures.

# Communication
- Be concise. Say what you're about to do, do it, then summarize the result in one line.
- When a tool call fails, read the error, explain what went wrong in one sentence, and retry.
- Don't apologize; fix.
"""


# 2. load_project_context — read AGENTS.md (Deep Agents convention) or
#    CODEIT.md fallback from the workspace. Graceful: missing file → "".
def load_project_context(root: str | Path | None = None) -> str:
    """Read AGENTS.md (or CODEIT.md fallback) from the workspace. '' if absent."""
    r = Path(root or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    for name in ("AGENTS.md", "CODEIT.md"):           # AGENTS.md wins if both exist
        candidate = r / name
        if candidate.is_file():
            try:
                return f"\n\n# Project context ({name})\n\n" + candidate.read_text(encoding="utf-8")
            except Exception:
                return ""                              # degrade gracefully
    return ""


def build_system_prompt(root: str | Path | None = None) -> str:
    """Compose the harness system prompt with any project context."""
    return SYSTEM_PROMPT + load_project_context(root)


# 3. A demo tool so the agent can act (trimmed run_shell for the demo).
@tool
def run_shell(command: str) -> str:
    """Run a shell command in the workspace. Use for tests, installs, git."""
    import subprocess
    cwd = str(Path(os.getenv("CODEIT_WORKDIR", "./workspace")).resolve())
    try:
        proc = subprocess.run(command, shell=True, cwd=cwd,
                              capture_output=True, text=True, timeout=120)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    return f"$ {command}\n[exit {proc.returncode}]\n{proc.stdout or ''}{proc.stderr or ''}"


INTERRUPT_ON = {"run_shell": True, "write_file": True, "edit_file": True, "delete": True}


def build_agent(workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=get_model(),
        tools=[run_shell],
        system_prompt=build_system_prompt(root),      # ← composed prompt
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        interrupt_on=INTERRUPT_ON,
        checkpointer=MemorySaver(),
    )


# 4. Streaming + approval driver (same shape as Ep 6).
def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": int(os.getenv("CODEIT_MAX_ITERS", "25")) * 2}


def _print_event(chunk) -> None:
    if chunk.get("type") != "updates":
        return
    for _n, state in chunk.get("data", {}).items():
        if not isinstance(state, dict):
            continue
        for msg in state.get("messages", []):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    console.print(f"[cyan]tool call:[/cyan] {tc['name']}({tc['args']})")
            elif isinstance(msg, AIMessage) and msg.content:
                console.print(f"[green]assistant:[/green] {msg.content}")


def _pending_tool_call(state):
    for msg in reversed(getattr(state, "values", {}).get("messages", []) or []):
        if getattr(msg, "tool_calls", None):
            return msg.tool_calls[-1]["name"], msg.tool_calls[-1]["args"]
    return None


def run_with_approval(agent, prompt: str, thread_id: str = "default") -> dict:
    config = _config(thread_id)
    auto = os.getenv("CODEIT_AUTO_APPROVE", "false").lower() == "true"
    for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]},
                              config=config, stream_mode="updates", version="v2"):
        _print_event(chunk)
    state = agent.get_state(config)
    while state.next:
        pending = _pending_tool_call(state)
        if pending is None:
            break
        _name, args = pending
        if auto:
            cmd = Command(resume={"decisions": [{"type": "approve"}]})
        else:
            console.print(f"tool: {_name}  args: {args}")
            answer = console.input("[bold]Approve? (y/n): [/bold]").strip().lower()
            if answer == "y":
                cmd = Command(resume={"decisions": [{"type": "approve"}]})
            else:
                cmd = Command(resume={"decisions": [{"type": "reject", "message": "No."}]})
        for chunk in agent.stream(cmd, config=config, stream_mode="updates", version="v2"):
            _print_event(chunk)
        state = agent.get_state(config)
    return state.values


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is this project?"
    agent = build_agent()
    state = run_with_approval(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)


if __name__ == "__main__":
    main()