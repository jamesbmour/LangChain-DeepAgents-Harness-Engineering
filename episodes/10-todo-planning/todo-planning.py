"""
Episode 10 — Planning: TodoList & Task Decomposition
====================================================

The agent learns to plan. Deep Agents' `TodoListMiddleware` is built-in and
always present (it's #1 in the default middleware stack). The `write_todos`
tool is automatically available — we don't add it. We add:
  1. `plan(steps)` — a custom @tool that makes planning EXPLICIT. Returns a
     string telling the model to call `write_todos` with these items as
     pending. (Tools can't call other tools directly — the model orchestrates.)
  2. `complete_todo(id)` — instructs the model to call `write_todos` again
     with item `id` marked `completed`.
  3. `render_todos(state)` — NOT a tool; a helper for the CLI to display state.

⚠️ Assumption: state key is `todos` (some versions use `todo_list` — we check both).
⚠️ Model size: planning is unreliable on ≤8B local models. Use 32b or OpenAI.
Builds on: Episodes 1-9. Requires: pip install deepagents langchain-ollama rich.
Run: LLM_MODEL=qwen2.5-coder:32b CODEIT_WORKDIR=./my_project \
     python tutorial.py "Add basic auth: plan 4 steps first."
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. plan + complete_todo — sugar over the built-in write_todos.
#    Tools can't call other tools directly. We return a STRING instructing
#    the model to call write_todos next; the model then does so.
# ─────────────────────────────────────────────────────────────────────────────
@tool
def plan(steps: list[str]) -> str:
    """Plan a multi-step task by seeding the built-in todo list with these steps.

    Use this when the task needs more than one step. Each step becomes a pending todo.
    After planning, execute the steps one by one, calling complete_todo(id) as you finish each.
    Argument steps: a list of short step descriptions (e.g. ['read main.py', 'fix the bug', 'run tests']).
    """
    todo_lines = [f"{i+1}. {s}" for i, s in enumerate(steps)]
    return ("Plan ready. Call write_todos with these items (all status='pending'):\n"
            + "\n".join(todo_lines)
            + "\n\nThen execute each step and call complete_todo(id) when done.")


@tool
def complete_todo(id: int) -> str:
    """Mark a todo item as completed by its 1-based index.

    Use this after you finish a step you planned with the plan tool.
    Argument id: the 1-based position of the todo in the list (1, 2, 3, ...).
    """
    # write_todos REPLACES the list (doesn't append). The model must pass the
    # full list back with item `id` set to 'completed'.
    return (f"Mark step {id} as completed by calling write_todos with the full list, "
            f"setting item {id}'s status to 'completed'.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. render_todos — NOT a tool. Helper for the CLI to display state.
# ─────────────────────────────────────────────────────────────────────────────
def render_todos(state: dict) -> str:
    """Pretty-print the todo list from agent state. Empty if no todos."""
    todos = (state or {}).get("todos") or (state or {}).get("todo_list") or []
    if not todos:
        return "(no todos)"
    marks = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}
    lines = []
    for i, t in enumerate(todos, 1):
        status = t.get("status", "pending")
        lines.append(f"  {marks.get(status, '[ ]')} {i}. {t.get('content', '')}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build the agent + streaming + approval driver (same shape as Ep 9).
# ─────────────────────────────────────────────────────────────────────────────
INTERRUPT_ON = {"run_shell": True, "write_file": True, "edit_file": True,
                "edit_file_safe": True, "delete": True}


def build_agent(workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=get_model(),
        tools=[plan, complete_todo],
        system_prompt=(
            "You are CodeIt, a coding agent. For any task with more than one step, "
            "call plan([...]) first to seed the todo list, then execute step by step, "
            "calling complete_todo(id) as you finish each. "
            "Use write_todos to update statuses — it REPLACES the list, not appends."
        ),
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        interrupt_on=INTERRUPT_ON, checkpointer=MemorySaver(),
    )


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
        if not pending:
            break
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
    prompt = sys.argv[1] if len(sys.argv) > 1 else \
        "Plan a 3-step task: read main.py, find a bug, run tests. Plan it first."
    agent = build_agent()
    state = run_with_approval(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)
    todos_str = render_todos(state or {})
    if todos_str != "(no todos)":
        console.print(f"\n[yellow]todos:[/yellow]\n{todos_str}")


if __name__ == "__main__":
    main()