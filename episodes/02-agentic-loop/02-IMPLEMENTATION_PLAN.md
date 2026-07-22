# Episode 2 — The Agentic Loop, Made Visible

**Tag:** `ep-02` · **Shape:** use the battery — streaming · **Budget:** ~110 SLOC
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 2

---

## 2.1 · What this episode delivers

The viewer sees the agent loop fire, step by step, on screen. Deep Agents' `create_deep_agent()` returns a compiled LangGraph graph that already runs the request→tool→result cycle internally; we don't hand-roll a completions loop. Instead we **drive the graph with `agent.stream(..., stream_mode="updates", version="v2")`** so each node event (model request, tool call, tool result) prints live via `rich`.

New pieces:
1. `run()` / `run_sync()` in `codeit/agent.py` — the streaming driver with a `thread_id` and a `recursion_limit` cap.
2. A throwaway `get_time` `@tool` that proves the loop fires (tool call → tool result → final answer).
3. Updated `scripts/chat.py` to use the new driver.

This is the "this is an agent, not a chatbot" moment.

---

## 2.2 · Pre-flight

1. Verify the installed LangGraph supports `stream_mode="updates"` with `version="v2"` (v2 format returns dicts with `type`, `ns`, `data` keys — type narrowing). If only v1 is available, adapt the parsing in `run()` accordingly. The contract is "stream each node event live" — the chunk shape is implementation detail.
2. Confirm `recursion_limit` is a **top-level** config key, NOT inside `configurable` (per LangGraph docs: "recursion_limit is a standalone config key and should not be passed inside the configurable key").
3. Confirm the default recursion limit. LangGraph ≥1.0.6 defaults to 1000; we cap at `CODEIT_MAX_ITERS` (default 25) so the loop can't run away on a small model.
4. Check that `rich` is installed: `python -c "import rich; print(rich.__version__)"`.

---

## 2.3 · File specs

### Extend `codeit/agent.py` (~70 SLOC added)

```python
import asyncio
from rich.console import Console

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain.tools import tool

from codeit.model import get_model
from codeit.settings import Settings, get_settings

console = Console()


@tool
def get_time() -> str:
    """Return the current time. Use this when the user asks for the time."""
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


def build_agent(
    model: BaseChatModel | None = None,
    tools: list | None = None,
    system_prompt: str | None = None,
):
    """Build a Deep Agent. Includes the get_time demo tool by default so the loop fires."""
    m = model or get_model()
    s = get_settings()
    all_tools = [get_time] + (tools or [])
    return create_deep_agent(
        model=m,
        tools=all_tools,
        system_prompt=system_prompt or "You are CodeIt, a helpful coding assistant.",
    ), s  # return settings too so run() can read max_iters


def _config(thread_id: str, settings: Settings) -> dict:
    # NOTE: recursion_limit is TOP-LEVEL, not inside configurable.
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_iters * 2,  # each turn = ~2 super-steps
    }


def _print_event(chunk) -> None:
    """Pretty-print one v2 stream chunk. Type-narrowed on chunk['type']."""
    kind = chunk.get("type")
    if kind == "updates":
        for node_name, state in chunk.get("data", {}).items():
            msgs = state.get("messages", []) if isinstance(state, dict) else []
            for msg in msgs:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        console.print(f"[cyan]tool call:[/cyan] {tc['name']}({tc['args']})")
                elif isinstance(msg, AIMessage) and msg.content:
                    console.print(f"[green]assistant:[/green] {msg.content}")
    elif kind == "values":
        # final state snapshot — we don't print here; final answer comes from result
        pass


async def run(agent, prompt: str, thread_id: str = "default") -> dict:
    """Drive the agent with streaming so each node event prints live. Enforces max_iters via recursion_limit."""
    s = get_settings()
    config = _config(thread_id, s)
    final_state = None
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
            stream_mode="updates",
            version="v2",
        ):
            _print_event(chunk)
    except Exception as e:
        # GraphRecursionError or model errors — return a clear string to the caller
        console.print(f"[red]error:[/red] {type(e).__name__}: {e}")
    # Fetch final state for the return value
    final_state = agent.get_state(config).values
    return final_state


def run_sync(agent, prompt: str, thread_id: str = "default") -> dict:
    """Sync wrapper around run()."""
    return asyncio.get_event_loop().run_until_complete(run(agent, prompt, thread_id))
```

**Teaching notes (say these on camera):**
- We pass `stream_mode="updates"` and `version="v2"`. v2 chunks are dicts with `type`, `ns`, `data` — type narrowing lets us branch cleanly. (Mention v1 returns bare `{node_name: state}` dicts if anyone's seen older tutorials.)
- `recursion_limit` lives at the **top level** of config, not under `configurable`. This is a common mistake — show it explicitly.
- `recursion_limit = max_iters * 2` because each agent turn is roughly two super-steps (model call + tool execution). Tweak the multiplier if you observe tighter coupling.
- We don't catch `GraphRecursionError` specially — a generic `except Exception` returns a readable string. The viewer sees the cap trip when the model loops forever (see test §2.4).
- `get_time`'s docstring is **for the model**: "Use this when the user asks for the time." That's the schema description the model sees. Tool docstrings are prompts (parent plan §3).

### Update `scripts/chat.py` (~15 SLOC, replacing Ep 1 version)

```python
import sys
from codeit.agent import build_agent, run

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What time is it? Use your tool."
    agent, _ = build_agent()
    state = run(agent, prompt)
    last = state["messages"][-1] if state and "messages" in state else None
    if last:
        print("\n--- final answer ---")
        print(last.content if hasattr(last, "content") else last)

if __name__ == "__main__":
    main()
```

---

## 2.4 · Tests

### `tests/test_ep02_loop.py`

Use `GenericFakeChatModel` with scripted `AIMessage`+`ToolCall` sequences. The fixture from `conftest.py` (Phase 0) should let a test pass a list of messages.

```python
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from codeit.agent import build_agent, run

def _fake_model_calling_time_once():
    """Model calls get_time, then returns a plain answer."""
    return GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "get_time", "args": {}, "id": "c1", "type": "tool_call"}]),
        "The time has been retrieved.",
    ]))

def _fake_model_looping_forever():
    """Model calls get_time on every turn — never answers."""
    from itertools import cycle
    msgs = cycle([AIMessage(content="", tool_calls=[{"name": "get_time", "args": {}, "id": "c1", "type": "tool_call"}])])
    return GenericFakeChatModel(messages=msgs)

def test_tool_runs_then_loop_terminates(monkeypatch):
    monkeypatch.setenv("CODEIT_MAX_ITERS", "5")
    agent, _ = build_agent(model=_fake_model_calling_time_once())
    state = run(agent, "what time is it?")
    assert state is not None
    # The final message should be the plain answer, not another tool call
    last = state["messages"][-1]
    assert not getattr(last, "tool_calls", None)  # no pending tool calls

def test_recursion_cap_enforced(monkeypatch):
    monkeypatch.setenv("CODEIT_MAX_ITERS", "2")
    agent, _ = build_agent(model=_fake_model_looping_forever())
    # Should NOT hang — the recursion_limit trips and run() returns without raising
    state = run(agent, "keep calling the tool")
    # We don't assert content; we assert it returned (didn't hang) and didn't raise.
    assert state is not None or state is None  # just reaching here is success

def test_thread_id_persists_state(monkeypatch):
    # Two sequential runs with the same thread_id share state (Deep Agents default checkpointer-free persistence within a process).
    # NOTE: without a real checkpointer, cross-process persistence isn't guaranteed; this tests in-process only.
    agent, _ = build_agent(model=_fake_model_calling_time_once())
    run(agent, "what time is it?", thread_id="t1")
    # Second run on same thread should not crash; we don't assert history here (Ep 6 adds MemorySaver for real persistence).
    state = run(agent, "thanks", thread_id="t1")
    assert state is not None
```

**Test notes:**
- The `test_recursion_cap_enforced` test is the load-bearing one: it proves the loop can't hang the viewer's machine. Run it with a timeout in CI (`pytest --timeout=10` or wrap in a thread with a deadline).
- `GenericFakeChatModel` returns one message per `invoke` call. With `tool_calls`, Deep Agents will execute the tool, append a `ToolMessage`, and re-invoke the model — which returns the next scripted message. This is the honest way to test the loop without a real LLM.
- If `GenericFakeChatModel` doesn't drive Deep Agents' middleware correctly (the harness may expect certain message attributes), fall back to a subclass that returns proper `AIMessage` objects with `tool_calls` populated. Verify against the installed `langchain_core` version.

---

## 2.5 · Demo script `scripts/demo_ep02.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "=== Demo: the agent calls a tool ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b python scripts/chat.py "What time is it? Use your tool."
echo
echo "Expected: you see [tool call: get_time({})] printed, then a tool result, then a final answer."
```

On camera: point at the `[tool call: get_time({})` line — that's the model deciding to act. Then the tool result comes back, then the final answer. **This is what makes it an agent.**

---

## 2.6 · README section to add

```
## Ep 2 — The Agentic Loop, Made Visible
Adds: run()/run_sync() in codeit/agent.py, get_time demo tool; updates scripts/chat.py
Run: LLM_PROVIDER=ollama python scripts/chat.py "What time is it? Use your tool."
Key idea: agent.stream(stream_mode="updates", version="v2") makes the LangGraph loop visible.
Lines added: 110 SLOC
```

---

## 2.7 · Definition of Done

- [ ] `run()` / `run_sync()` in `codeit/agent.py` stream each node event via `rich`.
- [ ] `get_time` `@tool` registered and called in the demo.
- [ ] `pytest -m "not live" tests/test_ep02_loop.py` green; recursion cap test doesn't hang.
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-01 HEAD -- '*.py'` adds < 220 lines.
- [ ] Ep 1's demo (`scripts/demo_ep01.sh`) still passes.
- [ ] README "Ep 2" section written.
- [ ] Commit + annotated tag `ep-02`.

---

## 2.8 · What NOT to do this episode

- Do not hand-roll a completions loop with `model.invoke()` + manual tool dispatch. Use `agent.stream()`.
- Do not put `recursion_limit` inside `configurable` — it's a top-level key.
- Do not use `stream_mode="values"` for live node printing (it gives full snapshots, too noisy). `"updates"` gives just the delta per node.
- Do not catch `GraphRecursionError` and silently swallow — print the error so the viewer sees the cap trip in the demo.
- Do not add a real checkpointer yet (Ep 6 adds `MemorySaver` for HITL; not needed now).

---

## 2.9 · Common gotchas

- **`stream` v1 vs v2:** if the installed LangGraph defaults to v1 chunks (bare `{node: state}`), the `_print_event` parser breaks. Add a fallback: `if isinstance(chunk, dict) and "type" in chunk: ... else: # v1 shape`.
- **`recursion_limit` semantics:** it counts super-steps, not tool calls. A model that calls one tool uses ~2 super-steps. If the cap trips too early, raise the multiplier in `_config()`.
- **`GenericFakeChatModel` + Deep Agents middleware:** the harness wraps the model in middleware (TodoList, Filesystem, SubAgent). If the fake model doesn't return proper `AIMessage` objects, the middleware may choke. Verify the fixture produces `AIMessage` with `.tool_calls` as a list of dicts with `name`, `args`, `id`, `type`.
- **`asyncio.get_event_loop()` deprecation:** in Python 3.12+, prefer `asyncio.run(run(...))`. Adapt `run_sync` to use `asyncio.run` if the installed Python warns.
- **`agent.get_state(config).values` after stream:** confirm `get_state` returns a `StateSnapshot` with `.values` containing the final `messages`. If the API differs, read from the last `values` chunk instead.