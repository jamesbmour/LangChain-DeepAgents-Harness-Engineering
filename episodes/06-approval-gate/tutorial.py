"""
Episode 6 — Permission Gating: Human-in-the-Loop
================================================

The agent can no longer mutate the workspace or run destructive commands
without asking first. We use Deep Agents' built-in HumanInTheLoopMiddleware
via `create_deep_agent(interrupt_on=..., checkpointer=...)`.

When the model calls a gated tool, the graph PAUSES. We inspect state,
prompt the viewer y/n, and resume with `Command(resume={"decisions": [...]})`.

Two new pieces:
  1. `classify(command)` — regex risk triage that labels the prompt
     red/yellow/green so the viewer sees risk at a glance.
  2. `run_with_approval(agent, prompt, thread_id)` — the full interrupt/resume
     loop: invoke → if paused, ask → resume → repeat.

Builds on: Episodes 1-5 (model, agent, FilesystemBackend, run_shell).

Run:
    CODEIT_WORKDIR=./workspace python tutorial.py \
        "Create a file then delete it with rm. Show me each step."

Requires:
    pip install deepagents langchain-ollama rich
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from rich.console import Console
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
# MemorySaver = in-process checkpointer. REQUIRED for interrupts — without it
# Command(resume=...) has nowhere to resume from.
# Verified: `from langgraph.checkpoint.memory import MemorySaver`.
from langgraph.checkpoint.memory import MemorySaver
# Command is the LangGraph resume primitive.
# Verified: `from langgraph.types import Command`.
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


# 1. A demo tool to gate — reuses Ep 5's run_shell shape (trimmed for space).
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


# 2. classify() — regex triage that LABELS the prompt. The interrupt itself
#    comes from interrupt_on; classify() is purely for display.
SAFE_PATTERNS = [r"^\s*ls\b", r"^\s*cat\b", r"^\s*pwd\b", r"^\s*echo\b",
                 r"^\s*git status\b", r"^\s*git diff\b", r"^\s*pytest\b"]
DESTRUCTIVE_PATTERNS = [r"rm\s+-rf?\b", r"git\s+push\s+.*-f", r"git\s+reset\s+--hard",
                        r"\bdd\b", r"\bmkfs\b", r"curl.*\|\s*sh", r"wget.*\|\s*sh"]


def classify(command: str) -> str:
    """Return 'safe', 'needs-approval', or 'blocked' for a shell command."""
    for pat in DESTRUCTIVE_PATTERNS:
        if re.search(pat, command):
            return "blocked"
    for pat in SAFE_PATTERNS:
        if re.search(pat, command):
            return "safe"
    return "needs-approval"


# 3. build_agent — interrupt_on + MemorySaver. Our policy: gate shell + writes.
INTERRUPT_ON = {"run_shell": True, "write_file": True, "edit_file": True, "delete": True}


def build_agent(workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=get_model(),
        tools=[run_shell],
        system_prompt="You are CodeIt, a coding agent. The user approves destructive actions.",
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        interrupt_on=INTERRUPT_ON,           # ← pause before these tools
        checkpointer=MemorySaver(),          # ← REQUIRED for interrupts to work
    )


# 4. run_with_approval — invoke → if paused, ask → resume → repeat.
def _config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": int(os.getenv("CODEIT_MAX_ITERS", "25")) * 2,
    }


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


def _pending_tool_call(state) -> tuple[str, dict] | None:
    """Read (name, args) of the tool call the model just requested."""
    for msg in reversed(getattr(state, "values", {}).get("messages", []) or []):
        if getattr(msg, "tool_calls", None):
            return msg.tool_calls[-1]["name"], msg.tool_calls[-1]["args"]
    return None


def run_with_approval(agent, prompt: str, thread_id: str = "default") -> dict:
    """invoke → if interrupted, await_approval → resume. Loops until done."""
    config = _config(thread_id)
    auto = os.getenv("CODEIT_AUTO_APPROVE", "false").lower() == "true"

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config, stream_mode="updates", version="v2",
    ):
        _print_event(chunk)

    state = agent.get_state(config)
    while state.next:                              # paused on an interrupt
        pending = _pending_tool_call(state)
        if pending is None:
            console.print("[red]Interrupt with no recognizable tool call — aborting.[/red]")
            break
        name, args = pending
        if auto:
            console.print("[yellow]--yolo: auto-approving[/yellow]")
            cmd = Command(resume={"decisions": [{"type": "approve"}]})
        else:
            risk = classify(args.get("command", "")) if name == "run_shell" else "needs-approval"
            color = {"safe": "green", "needs-approval": "yellow", "blocked": "red"}[risk]
            console.print(f"[{color}]risk: {risk}[/{color}]  tool: {name}  args: {args}")
            answer = console.input("[bold]Approve? (y/n): [/bold]").strip().lower()
            if answer == "y":
                cmd = Command(resume={"decisions": [{"type": "approve"}]})
            else:
                cmd = Command(resume={"decisions": [
                    {"type": "reject", "message": "User denied this action."}
                ]})
        for chunk in agent.stream(cmd, config=config, stream_mode="updates", version="v2"):
            _print_event(chunk)
        state = agent.get_state(config)
    return state.values


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Run: echo hello"
    agent = build_agent()
    state = run_with_approval(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)


if __name__ == "__main__":
    main()