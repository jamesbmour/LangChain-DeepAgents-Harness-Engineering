"""
Episode 15 — LangSmith Observability Appendix
=============================================

Show viewers how to see what the agent is DOING under the hood. With
LangSmith tracing on, every model call, tool call, subagent delegation,
and the full message history at each step becomes visible in a trace URL
you can open in the browser.

This is the LEAST new code of any episode — mostly docs + a tiny helper.
LangSmith is already a dep from Phase 0; no new install needed.

New pieces:
  1. `.env.example` entries: LANGSMITH_API_KEY=, LANGSMITH_TRACING=true,
     LANGSMITH_PROJECT=codeit-demo. (Set these in your .env.)
  2. `codeit/observability.py` — `trace_url_for_run(run_id)` and
     `maybe_print_trace_url(state)` helpers.
  3. The demo runs a task with tracing on and prints the trace URL at the end.

⚠️ Assumption: env var names are LANGSMITH_* (per the ecosystem-primer skill).
   Older names (LANGCHAIN_API_KEY, LANGCHAIN_TRACING) no longer work.
⚠️ The trace URL shape changes over time. The helper is best-effort; if the
   URL doesn't open, direct viewers to the LangSmith dashboard and have them
   find the run by project + timestamp.
Builds on: Episodes 1-14. Requires: pip install deepagents langchain-ollama rich langsmith.
Run:
    export LANGSMITH_API_KEY=...
    export LANGSMITH_TRACING=true
    export LANGSMITH_PROJECT=codeit-demo
    CODEIT_WORKDIR=./examples/sample_app python tutorial.py "fix the failing test"
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

console = Console()


def get_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    name = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai.")
        return init_chat_model(model=name, model_provider="openai")
    return init_chat_model(model=name, model_provider="ollama",
                           base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

def _workspace_root() -> Path:
    return Path(os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Observability helpers — build a LangSmith trace URL if tracing is on.
#    LangSmith tracing is enabled purely by env vars; no code change to the
#    agent itself is required. This helper just surfaces the URL at the end.
# ─────────────────────────────────────────────────────────────────────────────
def trace_url_for_run(run_id: str | None) -> str | None:
    """Build a LangSmith trace URL if tracing is on and a run_id is available."""
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        return None
    if not run_id:
        return None
    project = os.getenv("LANGSMITH_PROJECT", "default")
    # URL shape: https://smith.langchain.com/o/<org>/projects/p/<project>/r/<run_id>
    # The exact shape depends on the LangSmith UI; this is a best-effort helper.
    return f"https://smith.langchain.com/projects/p/{project}/r/{run_id}"


def maybe_print_trace_url(state) -> None:
    """After a run, print the trace URL to stderr if tracing is on."""
    # The run_id isn't always on the state object — it depends on the LangGraph
    # version and whether LangSmith's tracer attached one. We try a few common
    # attribute paths; if none have it, we just tell the viewer to check the dashboard.
    run_id = None
    if state is not None:
        run_id = (getattr(state, "run_id", None)
                  or (state.get("run_id") if isinstance(state, dict) else None))
    url = trace_url_for_run(run_id)
    if url:
        console.print(f"[dim]Trace: {url}[/dim]")
    elif os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        project = os.getenv("LANGSMITH_PROJECT", "default")
        console.print(f"[dim]Tracing is on. Open the LangSmith dashboard → project '{project}' "
                      f"to find this run by timestamp.[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# 2. A demo agent (trimmed from Ep 14) so we have something to trace.
# ─────────────────────────────────────────────────────────────────────────────
@tool
def run_shell(command: str) -> str:
    """Run a shell command in the workspace. Use for tests, installs, git."""
    import subprocess
    try:
        proc = subprocess.run(command, shell=True, cwd=str(_workspace_root()),
                              capture_output=True, text=True, timeout=120)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    return f"$ {command}\n[exit {proc.returncode}]\n{proc.stdout or ''}{proc.stderr or ''}"


INTERRUPT_ON = {"run_shell": True, "write_file": True, "edit_file": True, "delete": True}


def build_agent(workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=get_model(), tools=[run_shell],
        system_prompt="You are CodeIt, a coding agent. Use run_shell to run tests.",
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        interrupt_on=INTERRUPT_ON, checkpointer=MemorySaver(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Streaming driver (same shape as Eps 2-14, ungated for simplicity).
# ─────────────────────────────────────────────────────────────────────────────
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


def run(agent, prompt: str, thread_id: str = "default") -> dict:
    config = _config(thread_id)
    try:
        for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]},
                                  config=config, stream_mode="updates", version="v2"):
            _print_event(chunk)
    except Exception as e:
        console.print(f"[red]error:[/red] {type(e).__name__}: {e}")
    return agent.get_state(config).values


def main() -> None:
    # Sanity-check tracing env before we start.
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        console.print("[yellow]Tip: set LANGSMITH_TRACING=true to see a trace URL at the end.[/yellow]")
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Run: echo hello"
    agent = build_agent()
    state = run(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)
    # Surface the trace URL (or dashboard hint) after the run.
    maybe_print_trace_url(state)
if __name__ == "__main__":
    main()