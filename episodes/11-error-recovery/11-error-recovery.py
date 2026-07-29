"""
Episode 11 — Error Recovery: Self-Healing Loops
================================================

The "wow" episode. The agent writes code, runs it, reads the failure, and
fixes it — autonomously, up to a retry cap. Two pieces:
  1. `run_tests(path)` — a custom @tool that runs `pytest` in the workspace,
     captures failures, returns a readable string.
  2. `run_with_recovery(agent, prompt, thread_id, max_retries)` — a plain
     Python loop AROUND the Deep Agents graph. After the agent finishes, if
     run_tests reported failures, re-invoke the agent with the failure output
     appended as a user message; cap retries. Implemented AROUND the graph,
     not as a new graph node — we don't fight the framework.

⚠️ Assumption: failure-detection heuristic looks for "[exit 1]" + "FAILED" or
"Error" in a tool message. Tighten if you see false recoveries in the demo.
⚠️ Model size: recovery is unreliable on small local models. Use 32b or OpenAI.
Builds on: Episodes 1-10. Requires: pip install deepagents langchain-ollama rich pytest.
Run: CODEIT_WORKDIR=./my_project python tutorial.py \
     "Run the tests. If they fail, read the failure and fix the code. Then run tests again."
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
MAX_OUTPUT_CHARS = 15_000  # ~4k tokens; failure output can be long


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

# 1. run_tests — a custom @tool. Thin: subprocess + pytest -q --tb=short.
#    Non-zero exit returns a STRING (not an exception) so the model reads
#    "[exit 1]" and self-corrects. The docstring primes the recovery loop.
@tool
def run_tests(path: str = ".") -> str:
    """Run pytest in the workspace and return pass/fail summary + failures.

    Use this after editing code that has tests, to check your work. If tests fail,
    read the failure output and fix the code, then run_tests again.
    Argument path: test path relative to workspace (default '.' runs all tests).
    """
    try:
        proc = subprocess.run(
            ["pytest", path, "-q", "--tb=short", "--no-header"],
            cwd=str(_workspace_root()), capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return "Error: pytest timed out after 180s."
    except FileNotFoundError:
        return "Error: pytest not installed. Run: pip install pytest"
    except Exception as e:
        return f"Error launching pytest: {type(e).__name__}: {e}"
    combined = f"$ pytest {path}\n[exit {proc.returncode}]\n{proc.stdout or ''}"
    if proc.stderr:
        combined += f"\n--- stderr ---\n{proc.stderr}\n"
    if len(combined) > MAX_OUTPUT_CHARS:
        combined = combined[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(combined)-MAX_OUTPUT_CHARS} more chars]"
    return combined

# 2. Build the agent — register run_tests alongside built-ins.
INTERRUPT_ON = {"run_shell": True, "write_file": True, "edit_file": True, "edit_file_safe": True, "delete": True}

def build_agent(workdir: str | None = None):
    root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=get_model(), tools=[run_tests],
        system_prompt=("You are CodeIt, a coding agent. After editing code with tests, call "
                       "run_tests. If tests fail, read the failure, fix the code, then run_tests again."),
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        interrupt_on=INTERRUPT_ON, checkpointer=MemorySaver(),
    )

# 3. Streaming + approval driver (same shape as Eps 6-10).
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

# 4. _detect_test_failure — heuristic scan of final state's tool messages.
def _detect_test_failure(state: dict) -> str | None:
    """Inspect agent's final state for a run_tests result that reported failure.

    Only checks the LAST run_tests tool result, not all tool messages, to avoid
    false positives from earlier failures in the conversation history.
    """
    if not state: return None
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") != "tool": continue
        content = getattr(msg, "content", "") or ""
        # Only look at the most recent tool result
        if "[exit 1]" in content and ("FAILED" in content or "Error" in content):
            return content
        # If we hit any tool result that's not a failure, stop scanning
        if "[exit 0]" in content:
            return None
        return None  # First tool message found, not a failure
    return None

# 5. run_with_recovery — plain Python loop AROUND the graph (not a node).
def run_with_recovery(agent, prompt: str, thread_id: str = "default", max_retries: int = 3) -> dict:
    """Run agent; if run_tests reports failures, re-invoke with the failure appended.

    Re-invoke uses the SAME thread_id so history (code written, tests run) persists.
    The follow-up is a USER message so the model treats it as new input.
    """
    state = run_with_approval(agent, prompt, thread_id)
    for attempt in range(max_retries):
        failure = _detect_test_failure(state)
        if not failure:
            return state
        console.print(f"[yellow]Test failure detected (attempt {attempt+1}/{max_retries}). Re-invoking...[/yellow]")
        followup = (f"The tests failed. Here is the output:\n\n{failure}\n\n"
                    f"Read the failure, fix the code, then run_tests again.")
        state = run_with_approval(agent, followup, thread_id)
    console.print(f"[red]Recovery cap reached ({max_retries} retries). Stopping.[/red]")
    return state

def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else \
        "Run the tests. If they fail, read the failure and fix the code. Then run tests again."
    agent = build_agent()
    state = run_with_recovery(agent, prompt, max_retries=int(os.getenv("CODEIT_MAX_RETRIES", "3")))
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)
if __name__ == "__main__":
    main()