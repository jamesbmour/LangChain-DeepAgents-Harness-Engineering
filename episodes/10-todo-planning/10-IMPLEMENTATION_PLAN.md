# Episode 10 — Planning: TodoList & Task Decomposition

**Tag:** `ep-10` · **Shape:** use + customize the battery — TodoListMiddleware · **Budget:** ~90 SLOC
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 10

---

## 10.1 · What this episode delivers

The agent learns to plan. Deep Agents' `TodoListMiddleware` is built-in and always present (it's #1 in the default middleware stack). The `write_todos` tool is automatically available — we don't add it. What we add is:

1. **`plan(steps: list[str]) -> str`** — a custom `@tool` that bridges "think about the plan" and "seed the built-in todo list." It calls `write_todos` under the hood with `status="pending"` for each step and returns a confirmation. This makes planning explicit and legible on camera.
2. **`complete_todo(id: int) -> str`** — a tiny custom `@tool` that flips a todo's status to `completed`. (The model can also just call `write_todos` again with updated statuses, but a dedicated `complete_todo` is more teachable.)
3. **`render_todos(state) -> str`** — reads `state["todos"]` and pretty-prints via `rich` for the CLI (used by Ep 14).

The viewer learns: the built-in `write_todos` is the real tool; our `plan` is sugar that makes the planning step explicit; todo state lives in `result["todos"]` and the middleware injects it into context each turn.

---

## 10.2 · Pre-flight

1. Confirm `write_todos` is the built-in tool name (per the docs: "the built-in `write_todos` tool").
2. Confirm the todo item schema: `{"content": str, "status": "pending" | "in_progress" | "completed"}` (per the orchestration skill).
3. Confirm the todo state lives in `result["todos"]` (per the orchestration skill: "Access the todo list from the agent's final state after invocation. `todos = result.get("todos", [])`"). The exact key may be `todos` or `todo_list` — verify against the installed version by inspecting a real `result` from a minimal agent.
4. Confirm `TodoListMiddleware` is always present (per the customization doc: it's #1 in the default stack). We don't need to add it; we just use its tool.
5. Confirm `write_todos` accepts a list of dicts: `write_todos(todos=[{"content": "...", "status": "pending"}])`.

---

## 10.3 · File specs

### Extend `codeit/tools/__init__.py` (~70 SLOC)

```python
from langchain.tools import tool
from typing import Annotated

# ... existing tools ...

@tool
def plan(steps: list[str]) -> str:
    """Plan a multi-step task by seeding the built-in todo list with these steps.

    Use this when the task needs more than one step. Each step becomes a pending todo.
    After planning, execute the steps one by one, calling complete_todo(id) as you finish each.
    Argument steps: a list of short step descriptions (e.g. ['read main.py', 'fix the bug', 'run tests']).
    """
    # We call the built-in write_todos indirectly by returning the plan as a string
    # and instructing the model to call write_todos next. (Direct tool-to-tool calls
    # aren't supported — tools return strings to the model, which then decides.)
    # Alternative: if the installed Deep Agents exposes write_todos as a callable we
    # can invoke directly, do so. Verify during implementation.
    todo_lines = [f"{i+1}. {s}" for i, s in enumerate(steps)]
    return (
        "Plan ready. Call write_todos with these items (all status='pending'):\n"
        + "\n".join(todo_lines)
        + "\n\nThen execute each step and call complete_todo(id) when done."
    )

@tool
def complete_todo(id: int) -> str:
    """Mark a todo item as completed by its 1-based index.

    Use this after you finish a step you planned with the plan tool.
    Argument id: the 1-based position of the todo in the list (1, 2, 3, ...).
    """
    # We can't directly mutate the agent's state from a tool. The pattern is:
    # the model calls complete_todo, we return a string telling it to call write_todos
    # with the updated statuses. The model then does so.
    return (
        f"Mark step {id} as completed by calling write_todos with the full list, "
        f"setting item {id}'s status to 'completed'."
    )

def render_todos(state: dict) -> str:
    """Pretty-print the todo list from agent state. Used by the CLI (Ep 14).

    Returns a string suitable for rich console printing. Empty if no todos.
    """
    todos = state.get("todos") or state.get("todo_list") or []
    if not todos:
        return "(no todos)"
    lines = []
    for i, t in enumerate(todos, 1):
        status = t.get("status", "pending")
        content = t.get("content", "")
        mark = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}.get(status, "[ ]")
        lines.append(f"  {mark} {i}. {content}")
    return "\n".join(lines)

CUSTOM_TOOLS = [read_summary, run_shell, edit_file_safe, build_repo_map, plan, complete_todo]

def register_custom_tools(extra: list | None = None) -> list:
    return CUSTOM_TOOLS + (extra or [])
```

**Teaching notes:**
- `plan` and `complete_todo` are **sugar**. The real tool is the built-in `write_todos`. Our tools make the planning step *explicit and legible* — on camera, the viewer sees "agent calls `plan([...])`" instead of the model silently constructing a `write_todos` payload.
- **Important limitation:** tools can't call other tools directly. A `@tool` returns a string to the model; the model then decides what to call next. So `plan` returns a string that *tells* the model to call `write_todos`. This is the honest Deep Agents pattern — don't try to invoke `write_todos` from inside `plan`. (If the installed version exposes `write_todos` as a plain callable we can invoke directly, that's cleaner — verify during implementation. The contract is "seed the todo list"; the mechanism is "return a string instructing the model.")
- `render_todos` is *not* a tool — it's a helper for the CLI (Ep 14) to display state. It reads from `state["todos"]` (or `state["todo_list"]` — handle both keys defensively).
- The status marks `[x]`, `[>]`, `[ ]` are a common convention (Aider, GitHub task lists). The model sees them in the rendered output and learns the grammar.

### No change to `codeit/agent.py`

`TodoListMiddleware` is always present by default. `write_todos` is already available. We just add `plan` and `complete_todo` to `register_custom_tools`. `build_agent` doesn't change.

---

## 10.4 · Tests

### `tests/test_ep10_todo.py`

```python
import pytest
from codeit.tools import plan, complete_todo, render_todos

def test_plan_returns_step_list():
    result = plan.invoke({"steps": ["read main.py", "fix the bug", "run tests"]})
    assert "1. read main.py" in result
    assert "2. fix the bug" in result
    assert "3. run tests" in result
    assert "write_todos" in result  # instructs the model to call write_todos

def test_plan_empty_steps():
    result = plan.invoke({"steps": []})
    # Should handle gracefully — no crash
    assert isinstance(result, str)

def test_complete_todo_returns_instruction():
    result = complete_todo.invoke({"id": 2})
    assert "2" in result
    assert "write_todos" in result or "completed" in result

def test_render_todos_empty():
    assert render_todos({}) == "(no todos)"
    assert render_todos({"todos": []}) == "(no todos)"

def test_render_todos_pending():
    state = {"todos": [{"content": "read main.py", "status": "pending"}, {"content": "fix bug", "status": "pending"}]}
    out = render_todos(state)
    assert "[ ]" in out
    assert "1. read main.py" in out
    assert "2. fix bug" in out

def test_render_todos_mixed_status():
    state = {"todos": [
        {"content": "done", "status": "completed"},
        {"content": "now", "status": "in_progress"},
        {"content": "later", "status": "pending"},
    ]}
    out = render_todos(state)
    assert "[x]" in out  # completed
    assert "[>]" in out  # in_progress
    assert "[ ]" in out  # pending

def test_render_todos_alt_key():
    # Some versions use 'todo_list' instead of 'todos'
    state = {"todo_list": [{"content": "x", "status": "pending"}]}
    out = render_todos(state)
    assert "x" in out

def test_agent_seeds_todos_via_write_todos(tmp_path, monkeypatch):
    """Drive the agent to call write_todos (built-in); assert state['todos'] is populated."""
    from langchain_core.messages import AIMessage
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from codeit.agent import build_agent, run
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_todos", "args": {"todos": [{"content": "step 1", "status": "pending"}, {"content": "step 2", "status": "pending"}]}, "id": "c1", "type": "tool_call"}]),
        "I've planned the work.",
    ]))
    agent, _ = build_agent(model=fake, workdir=str(tmp_path))
    state = run(agent, "plan a 2-step task")
    # The built-in write_todos should have populated state['todos'] (or 'todo_list')
    todos = (state or {}).get("todos") or (state or {}).get("todo_list") or []
    assert len(todos) == 2
    assert todos[0]["content"] == "step 1"
```

**Test notes:**
- `test_agent_seeds_todos_via_write_todos` is the load-bearing integration test. It confirms the built-in `write_todos` populates `state["todos"]`. The exact key (`todos` vs `todo_list`) varies by version — handle both.
- `plan` and `complete_todo` are unit-tested in isolation — they're pure string-returning tools, no state mutation.
- If `write_todos`'s arg name differs (e.g. `items` instead of `todos`), adapt the test. Verify the schema by inspecting the tool during pre-flight.

---

## 10.5 · Demo script `scripts/demo_ep10.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "=== Demo: agent plans a 5-step task and executes it ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:32b \
CODEIT_WORKDIR=./examples/sample_app \
python scripts/chat.py "Add basic auth to this app: 1) read main.py, 2) add an /login endpoint, 3) add a simple token check, 4) add a test, 5) run the tests. Plan it first, then do it."
echo
echo "Expected: agent calls plan([...]) or write_todos([...]) first, then executes step by step, calling write_todos again to mark items completed."
echo "Note: use a larger model (qwen2.5-coder:32b) for planning — small models lose the thread."
```

On camera:
1. Watch the agent call `plan([...])` (or `write_todos` directly).
2. Show the todo list in the streamed output.
3. Watch the agent execute step 1, then call `write_todos` again with step 1 marked `completed`.
4. Continue through all 5 steps.
5. Use `render_todos` (off-camera, in a debug script) to show the final state.

**Model size warning:** planning is unreliable on ≤8B local models. Use `qwen2.5-coder:32b` or OpenAI for this demo. State this on camera.

---

## 10.6 · README section to add

```
## Ep 10 — Planning (TodoList + plan tool)
Adds: plan, complete_todo tools + render_todos helper in codeit/tools/__init__.py
Run: CODEIT_WORKDIR=./examples/sample_app python scripts/chat.py "Add auth: plan 5 steps then execute."
Key idea: TodoListMiddleware + write_todos are built-in; plan/complete_todo are sugar making planning explicit. State in result['todos'].
Lines added: 90 SLOC
Model note: use qwen2.5-coder:32b or OpenAI for planning — small models lose the thread.
```

---

## 10.7 · Definition of Done

- [ ] `plan`, `complete_todo` tools and `render_todos` helper in `codeit/tools/__init__.py`.
- [ ] `register_custom_tools` includes `plan`, `complete_todo`.
- [ ] `pytest -m "not live" tests/test_ep10_todo.py` green (including `test_agent_seeds_todos_via_write_todos`).
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-09 HEAD -- '*.py'` adds < 220 lines.
- [ ] Ep 1-9 demos still pass.
- [ ] README "Ep 10" section written; model-size note included.
- [ ] Commit + annotated tag `ep-10`.

---

## 10.8 · What NOT to do this episode

- Do not reimplement `TodoListMiddleware` or `write_todos`. They're built-in and always present.
- Do not try to call `write_todos` from inside `plan`. Tools return strings to the model; the model decides the next call. Return a string instructing the model.
- Do not assume the state key is `todos`. Handle both `todos` and `todo_list` defensively in `render_todos` and tests.
- Do not let `plan` accept an unbounded list. If the model passes 50 steps, that's a planning failure. (No explicit cap needed — the model self-regulates — but mention on camera that 3-7 steps is the sweet spot.)
- Do not skip the `test_agent_seeds_todos_via_write_todos` integration test. It's the only proof the built-in actually populates state.

---

## 10.9 · Common gotchas

- **`write_todos` arg schema:** may be `{"todos": [...]}` or `{"items": [...]}`. Verify during pre-flight by inspecting the built-in tool's schema. Adapt `plan`'s return string to match.
- **State key `todos` vs `todo_list`:** the orchestration skill says `result.get("todos", [])`; the customization doc says `TodoListMiddleware`. Inspect a real `result` from a minimal agent to confirm the exact key. Handle both in `render_todos`.
- **`write_todos` replaces, doesn't append:** each call to `write_todos` sets the *full* list, not appends. The model must pass the complete list with updated statuses to mark items complete. Document this in `complete_todo`'s return string (done above).
- **`thread_id` required for todo persistence:** per the orchestration skill fix: "Todo list state requires a thread_id for persistence across invocations." Our `run()` (Ep 2) always passes a `thread_id`, so this is handled. But mention it on camera — without `thread_id`, todos don't persist across turns.
- **Tools can't call tools:** the deepest gotcha. `plan` can't directly invoke `write_todos`. The model is the orchestrator; tools are functions the model calls. Return strings; let the model decide.
- **`render_todos` is not a tool:** it's a helper for the CLI. Don't register it in `register_custom_tools`. The model doesn't need to "render" todos — it reads them from state directly.