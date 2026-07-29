"""
Episode 13 — MCP + Skills: Speaking the Standard Protocol
==========================================================

The agent gains two capabilities from the Deep Agents ecosystem:
  1. MCP tools — connect to an MCP server via `langchain-mcp-adapters`. The
     server's tools register alongside our custom tools, exposed automatically.
  2. Skills — a `skills/` directory with one example `SKILL.md` (YAML
     frontmatter + markdown instructions). `SkillsMiddleware` (built-in, #2
     in the default stack) loads skill summaries at startup and the agent
     reads full instructions on demand. Progressive disclosure: context
     stays small until a skill is needed.

New pieces:
  1. `build_mcp_config(settings)` + `load_mcp_tools(config)` (async) helpers.
  2. `skills/python-testing/SKILL.md` (created by the viewer; we show the format).
  3. Extended `build_agent()` — accepts `mcp_tools` and `skills` lists.

Both are OPTIONAL — the agent works without them. Graceful degradation is
the contract: if no MCP server is configured or it's down, load_mcp_tools
returns []; if no skills directory is passed, SkillsMiddleware skips.

⚠️ MCP tool loading is ASYNC — `get_tools()` is a coroutine. The demo uses
asyncio.run to wrap the async loading before calling build_agent.
Builds on: Episodes 1-12. Requires: pip install deepagents langchain-ollama rich langchain-mcp-adapters.
Run: see `python tutorial.py --help`. Demo needs an MCP server (e.g. npx
     @modelcontextprotocol/server-filesystem ./workspace) on MCP_SERVER_URL.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from rich.console import Console

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

# 1. MCP config + tool loading. MCP is OPTIONAL — graceful degradation.
def build_mcp_config() -> dict | None:
    """Build the MultiServerMCPClient config from env. None if no server configured.

    Reads MCP_SERVER_NAME, MCP_TRANSPORT, MCP_SERVER_URL from env.
    Example .env: MCP_SERVER_NAME=filesystem, MCP_TRANSPORT=http,
                  MCP_SERVER_URL=http://localhost:8000/mcp
    """
    name = os.getenv("MCP_SERVER_NAME", "")
    url = os.getenv("MCP_SERVER_URL", "")
    transport = os.getenv("MCP_TRANSPORT", "http")
    if not name or not url:
        return None
    return {name: {"transport": transport, "url": url}}

async def load_mcp_tools(config: dict | None = None) -> list:
    """Connect to the MCP server(s) and return their tools. Empty list if no config."""
    if not config:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient
    try:
        client = MultiServerMCPClient(config)
        return await client.get_tools()
    except Exception as e:
        # Server unreachable — degrade gracefully. Agent still works without MCP tools.
        print(f"MCP tools unavailable: {type(e).__name__}: {e}", file=sys.stderr)
        return []

# 2. A demo custom tool (so the agent can do something even with no MCP server).
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

# 3. Build the agent — mcp_tools + skills are optional lists.
INTERRUPT_ON = {"run_shell": True, "write_file": True, "edit_file": True, "delete": True}

def build_agent(mcp_tools: list | None = None, skills: list | None = None, workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=get_model(),
        tools=[run_shell] + (mcp_tools or []),
        system_prompt=("You are CodeIt, a coding agent. MCP tools (if present) come from an external "
                       "server. Skills load on demand when relevant."),
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        skills=skills,                  # list of directory paths; None = no skills
        interrupt_on=INTERRUPT_ON, checkpointer=MemorySaver(),
    )

# 4. Streaming + approval driver (same shape as Eps 6-12).
def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": int(os.getenv("CODEIT_MAX_ITERS", "25")) * 2}

def _print_event(chunk) -> None:
    if chunk.get("type") != "updates": return
    for _n, state in chunk.get("data", {}).items():
        if not isinstance(state, dict): continue
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
            _print_event(chunk)
        state = agent.get_state(config)
    return state.values

# 5. main — async entry point because MCP tool loading is async.
async def _async_main(prompt: str, use_mcp: bool, skills_dir: str | None):
    mcp_tools = await load_mcp_tools(build_mcp_config()) if use_mcp else []
    if use_mcp:
        console.print(f"[dim]Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}[/dim]")
    skills = [skills_dir] if skills_dir else None
    agent = build_agent(mcp_tools=mcp_tools, skills=skills)
    state = run_with_approval(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)

def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What MCP tools do you have? Use them if useful."
    use_mcp = "--mcp" in sys.argv
    skills_dir = "./skills" if "--skills" in sys.argv else None
    asyncio.run(_async_main(prompt, use_mcp, skills_dir))
if __name__ == "__main__":
    main()