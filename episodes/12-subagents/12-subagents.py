"""
Episode 12 — Sub-Agents: Specialists in Isolated Context
=========================================================

The agent learns to delegate. Deep Agents' `SubAgentMiddleware` is built-in
and the `task` tool is automatically available when subagents are configured.
We add custom subagent specs:
  1. "explorer" — read-only subagent. Tools: build_repo_map (plus the shared
     backend's ls/read_file/grep). No shell, no writes.
  2. "tester"  — run_tests + edit_file_safe. No shell, no write_file.
  3. `spawn_subagent(task, agent)` — sugar over the built-in `task` tool.

Subagents run with isolated context; only a summary returns to the parent.
⚠️ Subagents are STATELESS — give complete instructions in one task call.
⚠️ Model size: delegation needs a larger model — 32b or OpenAI.
Builds on: Episodes 1-11. Requires: pip install deepagents langchain-ollama rich.
Run: LLM_MODEL=qwen2.5-coder:32b CODEIT_WORKDIR=./my_project \
     python tutorial.py "Use the explorer subagent to map the codebase."
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

# 1. Custom tools the subagents will use (trimmed for space; same shape as Eps 5/8/9/11).
@tool
def run_tests(path: str = ".") -> str:
    """Run pytest in the workspace. Use after editing code with tests."""
    try:
        proc = subprocess.run(["pytest", path, "-q", "--tb=short", "--no-header"],
                              cwd=str(_workspace_root()), capture_output=True, text=True, timeout=180)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    return f"$ pytest {path}\n[exit {proc.returncode}]\n{proc.stdout or ''}{proc.stderr or ''}"

@tool
def edit_file_safe(path: str, search: str, replace: str) -> str:
    """Edit a file by replacing `search` with `replace` (first occurrence). For existing files only."""
    try:
        fp = _workspace_root() / path
        text = fp.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: {path} not found."
    if search not in text:
        return f"Could not find search text in {path}. Re-read and retry."
    fp.write_text(text.replace(search, replace, 1), encoding="utf-8")
    return f"Applied exact match edit to {path}."

@tool
def build_repo_map(root: str = ".") -> str:
    """Return a compact map of the codebase: file paths + top-level signatures. Use for 'where is X?'."""
    import ast
    lines = []
    for dirpath, dirnames, filenames in os.walk(_workspace_root() / root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", ".venv"}]
        for fname in sorted(filenames):
            if not fname.endswith(".py"): continue
            fp = Path(dirpath) / fname
            try:
                tree = ast.parse(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            sigs = [f"  def {n.name}()" for n in tree.body if isinstance(n, ast.FunctionDef)]
            sigs += [f"  class {n.name}" for n in tree.body if isinstance(n, ast.ClassDef)]
            lines.append(f"{fp.relative_to(_workspace_root())}:\n" + "\n".join(sigs[:20]))
    return "\n".join(lines) or "(empty workspace)"

# 2. spawn_subagent — sugar over the built-in `task` tool. Returns a STRING
#    instructing the model to call task(agent=..., instruction=...).
#    Tools can't call other tools directly — the model orchestrates.
@tool
def spawn_subagent(task: str, agent: str = "general-purpose") -> str:
    """Delegate a subtask to a specialized subagent and return its summary.

    Use this for large or isolatable subtasks to keep the main context clean.
    Argument task: a COMPLETE, self-contained description (subagents are stateless).
    Argument agent: 'explorer' (read-only codebase search), 'tester' (run tests + fix),
      or 'general-purpose' (default, same tools as you).
    """
    return (f"To delegate to the '{agent}' subagent, call the task tool with: "
            f"agent='{agent}', instruction='''{task}'''. "
            f"The subagent runs in isolated context and returns only a summary.")

# 3. Subagent specs — the built-in SubAgentMiddleware reads these.
SUBAGENTS = [
    {
        "name": "explorer",
        "description": "Read-only codebase explorer. Use for 'where is X defined?', 'map the codebase'. Returns a summary; cannot modify files or run commands.",
        "system_prompt": "You are a read-only codebase explorer. Use ls, read_file, grep, and build_repo_map. Never write, edit, or run commands. Return a concise summary.",
        "tools": [build_repo_map],
    },
    {
        "name": "tester",
        "description": "Test-runner and fixer. Use for 'make the tests pass'. Can run_tests and edit existing files, but cannot run shell commands or create new files.",
        "system_prompt": "You are a test runner. Use run_tests; if tests fail, read the failure, use edit_file_safe to fix, then run_tests again. No run_shell or write_file.",
        "tools": [run_tests, edit_file_safe],
    },
]

# 4. Build the agent — pass subagents to create_deep_agent.
INTERRUPT_ON = {"run_shell": True, "write_file": True, "edit_file": True, "edit_file_safe": True, "delete": True}

def build_agent(workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=get_model(), tools=[spawn_subagent, run_tests, edit_file_safe, build_repo_map],
        system_prompt=("You are CodeIt, a coding agent. Delegate large or isolatable subtasks "
                       "with spawn_subagent. The explorer is read-only; the tester runs tests and fixes."),
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        subagents=SUBAGENTS, interrupt_on=INTERRUPT_ON, checkpointer=MemorySaver(),
    )

# 5. Streaming + approval driver (same shape as Eps 6-11).
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

def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Use the explorer subagent to map the codebase."
    agent = build_agent()
    state = run_with_approval(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)
if __name__ == "__main__":
    main()