"""
Episode 9 — Context Management: Repo Map & Token Trimming
=========================================================

The agent gains a bird's-eye view of the codebase WITHOUT reading every file.
  1. `build_repo_map(root)` — walks the workspace (hardcoded skip list), uses
     Python's `ast` to extract top-level def/class signatures, returns a compact
     text map. Aider ranks symbols by PageRank — we explicitly DON'T.
  2. `estimate_tokens(text)` — ~4 chars/token heuristic for budgeting.
  3. `trim_history(messages, max_tokens)` — drop oldest turns to fit a cap.

Note: Deep Agents' FilesystemMiddleware already does context engineering
internally. This is a teaching layer for custom control.
Builds on: Episodes 1-8. Requires: pip install deepagents langchain-ollama rich.
Run: CODEIT_WORKDIR=./my_project python tutorial.py "Where is X defined? Use build_repo_map first."
"""

from __future__ import annotations

import ast
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

# Hardcoded skip list — avoids a new dep (pathspec) for .gitignore parsing.
SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".pytest_cache",
             ".ruff_cache", "workspace", ".codeit"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
MAX_FILES = 200
MAX_SIGS_PER_FILE = 30
MAX_MAP_CHARS = 15_000    # ~4k tokens

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

# 1. _signatures — extract top-level def/class signatures from a Python file.
def _signatures(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    sigs: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            sigs.append(f"{kind} {node.name}({', '.join(args)})")
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            sigs.append(f"class {node.name}" + (f" ({', '.join(methods)})" if methods else ""))
        if len(sigs) >= MAX_SIGS_PER_FILE:
            sigs.append("... [truncated]"); break
    return sigs

# 2. build_repo_map — the custom @tool. Walks the workspace, returns a compact map.
@tool
def build_repo_map(root: str = ".") -> str:
    """Return a compact map of the codebase: file paths + top-level def/class signatures.

    Use this to answer 'where is X defined?' across multiple files without reading them all.
    Argument root: subdirectory to map, relative to workspace (default '.').
    """
    base = (_workspace_root() / root).resolve() if root != "." else _workspace_root()
    lines, count = [], 0
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in sorted(filenames):
            if fname.endswith(tuple(SKIP_SUFFIXES)):
                continue
            fpath, rel = Path(dirpath) / fname, (Path(dirpath) / fname).relative_to(_workspace_root())
            if fpath.suffix == ".py":
                sigs = _signatures(fpath)
                lines.append(f"{rel}:")
                lines.extend(f"  {s}" for s in sigs or ["(no top-level def/class)"])
            else:
                try:
                    n = sum(1 for _ in fpath.open(encoding="utf-8", errors="ignore"))
                    lines.append(f"{rel}: ({n} lines)")
                except Exception:
                    lines.append(f"{rel}:")
            count += 1
            if count >= MAX_FILES:
                lines.append("... [repo map truncated, too many files]"); break
        if count >= MAX_FILES:
            break
    map_text = "\n".join(lines)
    if len(map_text) > MAX_MAP_CHARS:
        map_text = map_text[:MAX_MAP_CHARS] + "\n... [repo map truncated]"
    return map_text or "(empty workspace)"

# 3. estimate_tokens + trim_history — teaching-layer context controls.
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token. Good for budgeting, not billing."""
    return max(1, len(text) // 4)

def trim_history(messages: list, max_tokens: int) -> list:
    """Drop oldest messages until total is under max_tokens. Keeps the newest turn.

    Teaching layer complementing Deep Agents' built-in context engineering.
    """
    if not messages:
        return []
    def _tok(m) -> int:
        return estimate_tokens(getattr(m, "content", str(m)) or "")
    kept = list(messages)
    while kept and sum(_tok(m) for m in kept) > max_tokens and len(kept) > 1:
        kept.pop(0)
    return kept

# 4. Build the agent + streaming + approval driver (same shape as Ep 8).
INTERRUPT_ON = {"run_shell": True, "write_file": True, "edit_file": True, "edit_file_safe": True, "delete": True}

def build_agent(workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=get_model(), tools=[build_repo_map],
        system_prompt=("You are CodeIt, a coding agent. Use build_repo_map for a bird's-eye "
                       "view of the codebase, then read_file only the files you need."),
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        interrupt_on=INTERRUPT_ON, checkpointer=MemorySaver(),
    )

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
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Where is the main file? Use build_repo_map."
    agent = build_agent()
    state = run_with_approval(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)
if __name__ == "__main__":
    main()