"""The Deep Agent brain + the streaming agentic loop, in one file.

Ep 1 landed ``build_agent()`` (tool-free). Ep 2 adds:

* ``get_time`` — a throwaway ``@tool`` that proves the loop fires.
* ``run()``   — drives the graph with ``agent.stream(stream_mode=["updates",
  "values"], version="v2")`` so each node event prints live via ``rich`` and the
  final state is captured from the last ``values`` chunk (no checkpointer needed).

Why stream_mode=["updates","values"] (a list) rather than just "updates":
"updates" gives the per-node delta (great for live printing) but not the final
state. "values" gives a full state snapshot after each super-step; the last one
is the final state. Using both in one v2 stream gets us both with a single pass.
"""

from collections.abc import Callable

from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from rich.console import Console

from codeit.model import get_model
from codeit.settings import get_settings

console = Console()


@tool
def get_time() -> str:
    """Return the current time. Use this when the user asks for the time."""
    import datetime

    return datetime.datetime.now().isoformat(timespec="seconds")


def build_agent(
    model: BaseChatModel | None = None,
    tools: list[BaseTool | Callable] | None = None,
    system_prompt: str | None = None,
):
    """Build a Deep Agent. ``get_time`` is always registered so the loop fires on camera."""
    m = model or get_model()
    _ = get_settings()  # ensure .env loaded; settings used by later episodes
    all_tools = [get_time, *(tools or [])]
    return create_deep_agent(
        model=m,
        tools=all_tools,
        system_prompt=system_prompt or "You are CodeIt, a helpful coding assistant.",
    )


def _config(thread_id: str) -> dict:
    """Build the LangGraph config.

    NOTE: ``recursion_limit`` is TOP-LEVEL, not under ``configurable``.
    """
    s = get_settings()
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": s.max_iters * 2,  # each turn ~= 2 super-steps (model + tools)
    }


def _print_event(chunk: dict) -> None:
    """Pretty-print one v2 stream chunk. Type-narrowed on ``chunk["type"]``."""
    kind = chunk.get("type")
    if kind != "updates":
        return  # "values" chunks are for state capture, not printing
    for _node_name, state in chunk.get("data", {}).items():
        if not isinstance(state, dict):
            continue
        for msg in state.get("messages", []):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    console.print(f"[cyan]tool call:[/cyan] {tc['name']}({tc['args']})")
            elif isinstance(msg, ToolMessage):
                console.print(f"[yellow]tool result:[/yellow] {msg.name} -> {msg.content[:120]}")
            elif isinstance(msg, AIMessage) and msg.content:
                console.print(f"[green]assistant:[/green] {msg.content}")


def run(agent, prompt: str, thread_id: str = "default") -> dict:
    """Drive the agent with streaming. Enforces ``max_iters`` via ``recursion_limit``.

    Returns the final state dict (last ``values`` chunk). On ``GraphRecursionError``
    or any other exception, prints a red error and returns ``{}``.
    """
    config = _config(thread_id)
    final_state: dict = {}
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
            stream_mode=["updates", "values"],
            version="v2",
        ):
            _print_event(chunk)
            if chunk.get("type") == "values" and chunk.get("data"):
                final_state = chunk["data"]
    except Exception as e:  # noqa: BLE001 — GraphRecursionError, model errors, etc.
        console.print(f"[red]error:[/red] {type(e).__name__}: {e}")
    return final_state
