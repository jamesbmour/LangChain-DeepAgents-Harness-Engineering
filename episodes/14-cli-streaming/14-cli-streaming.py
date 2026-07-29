"""
Episode 14 — Shipping CodeIt: a Real CLI with Event Streaming
==============================================================

The finale. Everything from Eps 1-13 ships as a single `codeit` CLI command
with a live `rich` streaming view. The viewer runs:
    codeit run "add a health endpoint and test it"
and watches the agent work end-to-end: tokens stream, tool calls render in
their own panel, subagent delegations appear, the todo list updates,
approval prompts fire when needed.

New piece: `codeit/cli.py` — a `typer` app with flags (--provider, --model,
--yolo, --workdir, --mcp, --skills, --approve) and a `rich` streaming view
built on Deep Agents' event-streaming API (agent.stream_events(...,
version="v3")). Falls back to v2 updates if v3 is unavailable.
⚠️ Assumption: the v3 event-streaming API may not exist in all installed
versions. We try it first and fall back to v2 — resilience across versions.
Builds on: Episodes 1-13. Requires: pip install deepagents langchain-ollama rich typer langchain-mcp-adapters.
Run: python tutorial.py run "your task" --provider ollama --model qwen2.5-coder:7b --workdir ./workspace
     python tutorial.py run "build a small FastAPI app" --mcp --skills --approve
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

console = Console()
app = typer.Typer(help="CodeIt — your terminal coding agent.")


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

# 1. A demo custom tool so the agent can act.
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

# 2. Tiny MCP loader (async) — graceful degradation. Full version in Ep 13.
async def load_mcp_tools() -> list:
    name = os.getenv("MCP_SERVER_NAME", "")
    url = os.getenv("MCP_SERVER_URL", "")
    if not name or not url:
        return []
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        client = MultiServerMCPClient({name: {"transport": os.getenv("MCP_TRANSPORT", "http"), "url": url}})
        return await client.get_tools()
    except Exception as e:
        print(f"MCP tools unavailable: {type(e).__name__}: {e}", file=sys.stderr)
        return []

# 3. Build the agent — all Eps 1-13 wiring in one place.
INTERRUPT_ON = {"run_shell": True, "write_file": True, "edit_file": True, "delete": True}

def build_agent(mcp_tools: list | None = None, skills: list | None = None):
    root = _workspace_root(); root.mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=get_model(), tools=[run_shell] + (mcp_tools or []),
        system_prompt="You are CodeIt, a terminal coding agent. Use tools when useful.",
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        skills=skills, interrupt_on=INTERRUPT_ON, checkpointer=MemorySaver(),
    )

# 4. The streaming view. Try v3 event-streaming first; fall back to v2 updates.
def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": int(os.getenv("CODEIT_MAX_ITERS", "25")) * 2}

def _render_v2_chunk(chunk):
    if chunk.get("type") != "updates": return
    for _n, state in chunk.get("data", {}).items():
        if not isinstance(state, dict): continue
        for msg in state.get("messages", []):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    console.print(Panel(Text(f"{tc['name']}({tc['args']})", style="magenta"),
                                         title="tool", border_style="magenta"))
            elif isinstance(msg, AIMessage) and msg.content:
                console.print(Panel(Text(msg.content, style="cyan"), title="assistant", border_style="blue"))

async def _run_streaming(agent, prompt: str, thread_id: str):
    """Drive the agent with event streaming; render each projection in a rich panel."""
    config = _config(thread_id)
    input_msg = {"messages": [{"role": "user", "content": prompt}]}
    try:
        stream = agent.stream_events(input_msg, version="v3", config=config)
        # v3: interleave messages and tool_calls for ordered output.
        for name, item in stream.interleave("messages", "tool_calls"):
            if name == "messages":
                console.print(Panel(Text(getattr(item, "text", str(item)), style="cyan"),
                                     title="assistant", border_style="blue"))
            elif name == "tool_calls":
                color = "green" if getattr(item, "error", None) is None else "red"
                console.print(Panel(Text(f"{item.tool_name}({item.input})", style=color),
                                     title="tool", border_style="magenta"))
    except (TypeError, AttributeError):
        # Fallback: v2 updates stream (from Ep 2).
        for chunk in agent.stream(input_msg, config=config, stream_mode="updates", version="v2"):
            _render_v2_chunk(chunk)

# 5. Approval driver (same shape as Eps 6-13) — used when --approve is set.
def _pending_tool_call(state):
    for msg in reversed(getattr(state, "values", {}).get("messages", []) or []):
        if getattr(msg, "tool_calls", None):
            return msg.tool_calls[-1]["name"], msg.tool_calls[-1]["args"]
    return None
def run_with_approval(agent, prompt: str, thread_id: str = "cli") -> dict:
    config = _config(thread_id)
    auto = os.getenv("CODEIT_AUTO_APPROVE", "false").lower() == "true"
    for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]},
                              config=config, stream_mode="updates", version="v2"):
        _render_v2_chunk(chunk)
    state = agent.get_state(config)
    while state.next:
        pending = _pending_tool_call(state)
        if not pending: break
        _name, args = pending
        if auto:
            cmd = Command(resume={"decisions": [{"type": "approve"}]})
        else:
            console.print(f"tool: {_name}  args: {args}")
            answer = console.input("[bold]Approve? (y/n): [/bold]").strip().lower()
            cmd = (Command(resume={"decisions": [{"type": "approve"}]}) if answer == "y"
                   else Command(resume={"decisions": [{"type": "reject", "message": "No."}]}))
        for chunk in agent.stream(cmd, config=config, stream_mode="updates", version="v2"):
            _render_v2_chunk(chunk)
        state = agent.get_state(config)
    return state.values

def _apply_flags(provider: str | None, model: str | None, yolo: bool, workdir: str | None):
    """Mutate env so the rest of the code reads settings as usual (Ep 1 pattern)."""
    if provider: os.environ["LLM_PROVIDER"] = provider
    if model: os.environ["LLM_MODEL"] = model
    if yolo: os.environ["CODEIT_AUTO_APPROVE"] = "true"
    if workdir: os.environ["CODEIT_WORKDIR"] = str(Path(workdir).resolve())

async def _async_main(prompt: str, mcp: bool, skills: bool, approve: bool):
    mcp_tools = await load_mcp_tools() if mcp else []
    skills_paths = ["./skills"] if skills else None
    agent = build_agent(mcp_tools=mcp_tools, skills=skills_paths)
    if approve or os.getenv("CODEIT_AUTO_APPROVE", "false").lower() == "true":
        state = run_with_approval(agent, prompt)
        last = state["messages"][-1] if state and "messages" in state else None
        if last:
            print("\n--- final answer ---")
            print(last.content if hasattr(last, "content") else last)
    else:
        await _run_streaming(agent, prompt, thread_id="cli")

@app.command()
def run(
    prompt: str = typer.Argument(..., help="The task for the agent."),
    provider: str = typer.Option(None, "--provider", "-p", help="ollama | openai"),
    model: str = typer.Option(None, "--model", "-m", help="model name"),
    yolo: bool = typer.Option(False, "--yolo", "-y", help="DANGEROUS: auto-approve all actions."),
    workdir: str = typer.Option(None, "--workdir", "-w", help="workspace directory"),
    mcp: bool = typer.Option(False, "--mcp", help="connect to MCP server from env"),
    skills: bool = typer.Option(False, "--skills", help="load skills from ./skills"),
    approve: bool = typer.Option(False, "--approve", help="enable HITL approval gate"),
):
    """Run CodeIt on a task."""
    _apply_flags(provider, model, yolo, workdir)
    asyncio.run(_async_main(prompt, mcp, skills, approve))

def main(): app()
if __name__ == "__main__": main()