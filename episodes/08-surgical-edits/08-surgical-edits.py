"""
Episode 8 — Surgical Edits: `edit_file` + the Fuzzy Fallback Wrapper
====================================================================

The agent graduates from whole-file rewrites to SEARCH-REPLACE edits — the
format used by Aider, Codex, and Cline. We add a teaching wrapper,
`edit_file_safe(path, search, replace)`: exact match first, difflib fuzzy
match (>=90%) on a miss, unified diff on total failure so the model
self-corrects. Goes through the Ep 6 approval gate.

Builds on: Episodes 1-7.

Run:
    CODEIT_WORKDIR=./workspace python tutorial.py \
        "In big.py, change line 50 to say 'EDITED' instead of 'some content'. Use edit_file_safe."

Requires:
    pip install deepagents langchain-ollama rich
"""

from __future__ import annotations

import difflib
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
    return init_chat_model(model=name, model_provider="ollama",
                           base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))


def _workspace_root() -> Path:
    return Path(os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()


def _resolve_in_workspace(path: str) -> Path:
    """Resolve `path` against the workspace, refusing to escape (Ep 4 helper)."""
    root = _workspace_root()
    # Strip leading slashes so '/big.py' is treated as 'big.py' relative to root
    clean = path.lstrip('/')
    target = (root / clean).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise PermissionError(f"Path {path!r} escapes workspace ({root}).")
    return target


# 1. _fuzzy_find — O(n*m) scan with difflib. Cap file size at 50k chars.
def _fuzzy_find(text: str, search: str, threshold: float = 0.90):
    """Find the substring of `text` most similar to `search`, above `threshold`."""
    if not search or len(text) > 50_000:
        return None, 0.0
    best, best_score, n = None, 0.0, len(search)
    for i in range(0, max(0, len(text) - n + 1)):
        score = difflib.SequenceMatcher(None, search, text[i:i + n]).ratio()
        if score > best_score:
            best, best_score = text[i:i + n], score
            if best_score == 1.0:
                break
    return (best, best_score) if best_score >= threshold else (None, best_score)


# 2. edit_file_safe — the custom @tool. Docstring explains the three outcomes
#    so the model learns the contract: exact / fuzzy / fail-with-diff.
@tool
def edit_file_safe(path: str, search: str, replace: str) -> str:
    """Edit a file by replacing `search` text with `replace` text (first occurrence).

    For targeted changes to existing files — never rewrite a whole file.
    path: file relative to workspace (e.g. 'main.py').
    search: exact text to find; include enough context to be unique.
    replace: new text to substitute.

    On exact match: applies. On near-miss (>=90% similar): applies with a note.
    On no match: returns a unified diff; re-read the file and retry.
    """
    try:
        text = _resolve_in_workspace(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: {path} not found in workspace."
    except Exception as e:
        return f"Error reading {path}: {type(e).__name__}: {e}"

    if search not in text:
        best, score = _fuzzy_find(text, search, threshold=0.90)
        if best is not None:
            new_text = text.replace(best, replace, 1)  # first occurrence only
            _resolve_in_workspace(path).write_text(new_text, encoding="utf-8")
            return f"Applied fuzzy match (similarity {score:.2f}). Review the result."
        # No match — return a unified diff so the model sees the current state.
        diff = difflib.unified_diff(
            text.splitlines(keepends=True),
            (text + "\n# --- proposed edit did not apply ---\n").splitlines(keepends=True),
            fromfile=f"{path} (current)", tofile=f"{path} (NOT applied)", n=3)
        return (f"Could not find the search text in {path} (best similarity {score:.2f}, "
                f"needed >=0.90). NOT modified. Re-read with read_file, then retry:\n\n"
                + "".join(diff))

    new_text = text.replace(search, replace, 1)        # first occurrence only
    _resolve_in_workspace(path).write_text(new_text, encoding="utf-8")
    return f"Applied exact match edit to {path}."


# 3. Build the agent — gate edit_file_safe too (it mutates files).
INTERRUPT_ON = {"run_shell": True, "write_file": True, "edit_file": True, "edit_file_safe": True, "delete": True}


def build_agent(workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=get_model(), tools=[edit_file_safe],
        system_prompt=("You are CodeIt, a coding agent. Use edit_file_safe for targeted "
                       "changes to existing files. If a search misses, re-read and retry."),
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        interrupt_on=INTERRUPT_ON, checkpointer=MemorySaver(),
    )


# 4. Streaming + approval driver (same shape as Eps 6-7).
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
    """invoke → if paused, ask → resume → repeat."""
    config = _config(thread_id)
    auto = os.getenv("CODEIT_AUTO_APPROVE", "false").lower() == "true"
    for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]},
                              config=config, stream_mode="updates", version="v2"):
        _print_event(chunk)
    state = agent.get_state(config)
    while state.next:
        pending = _pending_tool_call(state)
        if not pending:
            break
        name, args = pending
        if auto:
            cmd = Command(resume={"decisions": [{"type": "approve"}]})
        else:
            console.print(f"tool: {name}  args: {args}")
            answer = console.input("[bold]Approve? (y/n): [/bold]").strip().lower()
            cmd = (Command(resume={"decisions": [{"type": "approve"}]}) if answer == "y"
                   else Command(resume={"decisions": [{"type": "reject", "message": "No."}]}))
        for chunk in agent.stream(cmd, config=config, stream_mode="updates", version="v2"):
            _print_event(chunk)
        state = agent.get_state(config)
    return state.values


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "In big.py, change line 50 to 'EDITED'. Use edit_file_safe."
    agent = build_agent()
    state = run_with_approval(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)
if __name__ == "__main__":
    main()