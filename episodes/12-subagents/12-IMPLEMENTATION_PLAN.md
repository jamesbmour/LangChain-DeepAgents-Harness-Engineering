# Episode 12 — Sub-Agents: Specialists in Isolated Context

**Tag:** `ep-12` · **Shape:** use + customize the battery — SubAgentMiddleware · **Budget:** ~110 SLOC
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 12

---

## 12.1 · What this episode delivers

The agent learns to delegate. Deep Agents' `SubAgentMiddleware` is built-in (it's #4 in the default stack) and the `task` tool is automatically available. We don't add the middleware — we add **custom subagent specs**:

1. **`"explorer"`** — read-only subagent with `ls`, `read_file`, `grep`, `build_repo_map`. For "map the codebase" tasks. No shell, no writes.
2. **`"tester"`** — `run_tests` + `edit_file_safe`. For "make the tests pass" tasks. No shell (can't run arbitrary commands), no `write_file` (can't create new files — only edit existing).
3. **`spawn_subagent(task, agent)`** — a thin custom `@tool` wrapper around the built-in `task` for ergonomics and to show viewers how delegation looks in code.

The main agent delegates via the built-in `task(agent="explorer", instruction="...")`. The subagent runs with isolated context; only a summary returns to the parent. Subagent messages don't leak into parent history.

---

## 12.2 · Pre-flight

1. Confirm the subagent spec schema (per the docs):
   ```python
   {"name": "researcher", "description": "...", "system_prompt": "...", "tools": [...], "model": "...optional..."}
   ```
   The keys are `name`, `description`, `system_prompt`, `tools`, and optionally `model` (defaults to the main agent's model).
2. Confirm the built-in `task` tool's arg schema: `task(agent="name", instruction="...")`. Verify the exact arg names (`agent`/`subagent_type`, `instruction`/`task`).
3. Confirm the default `"general-purpose"` subagent exists automatically (same tools/config as main) — per the orchestration skill.
4. Confirm subagents are **stateless** (per the orchestration skill: "Subagents are stateless - provide complete instructions in a single call"). Emphasize this on camera.
5. Confirm custom subagents don't inherit the main agent's skills (per the orchestration skill fix). If we want a subagent to have skills, pass them explicitly. We don't use skills this episode (Ep 13), so this is moot — but mention it.

---

## 12.3 · File specs

### Extend `codeit/agent.py` (~70 SLOC)

```python
from codeit.tools import register_custom_tools, build_repo_map
from codeit.testing import run_tests
from codeit.tools import edit_file_safe

# Subagent specs — the built-in SubAgentMiddleware reads these.
SUBAGENTS = [
    {
        "name": "explorer",
        "description": "Read-only codebase explorer. Use for 'where is X defined?', 'map the codebase', 'find all callers of Y'. Returns a summary; cannot modify files or run commands.",
        "system_prompt": "You are a read-only codebase explorer. Use ls, read_file, grep, and build_repo_map to answer questions about the codebase. Never attempt to write, edit, or run commands. Return a concise summary of what you found.",
        "tools": [],  # filled in at build time — built-in filesystem tools come from the backend, plus build_repo_map
    },
    {
        "name": "tester",
        "description": "Test-runner and fixer. Use for 'make the tests pass'. Can run_tests and edit existing files, but cannot run shell commands or create new files.",
        "system_prompt": "You are a test runner. Use run_tests to check the suite; if tests fail, read the failure, use edit_file_safe to fix the code, then run_tests again. Do not use run_shell or write_file.",
        "tools": [],  # filled in at build time — run_tests + edit_file_safe
    },
]

def _resolve_subagent_tools(spec: dict) -> dict:
    """Attach the actual tool callables to each subagent spec by name."""
    spec = dict(spec)  # shallow copy so we don't mutate the constant
    if spec["name"] == "explorer":
        # Filesystem tools (ls, read_file, grep) come from the shared backend;
        # we only add build_repo_map here.
        spec["tools"] = [build_repo_map]
    elif spec["name"] == "tester":
        spec["tools"] = [run_tests, edit_file_safe]
    return spec

def build_agent(
    model: BaseChatModel | None = None,
    tools: list | None = None,
    system_prompt: str | None = None,
    backend=None,
    workdir: str | None = None,
    interrupt_on: dict | None = None,
    checkpointer=None,
    subagents: list | None = None,
):
    """Build a Deep Agent. If subagents is None, use the default explorer+tester specs."""
    m = model or get_model()
    s = get_settings()
    wd = workdir or s.workdir
    if backend is None:
        root = Path(wd).resolve()
        root.mkdir(parents=True, exist_ok=True)
        backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    all_tools = register_custom_tools(tools)
    kwargs = dict(
        model=m,
        tools=all_tools,
        system_prompt=system_prompt or build_system_prompt(wd),
        backend=backend,
    )
    # Subagents: resolve tool callables, then pass to create_deep_agent
    subagent_specs = subagents if subagents is not None else [_resolve_subagent_tools(s) for s in SUBAGENTS]
    if subagent_specs:
        kwargs["subagents"] = subagent_specs
    if interrupt_on:
        kwargs["interrupt_on"] = interrupt_on
        kwargs["checkpointer"] = checkpointer or MemorySaver()
    return create_deep_agent(**kwargs), s
```

**Teaching notes:**
- `SUBAGENTS` is a module constant — the specs. `_resolve_subagent_tools` attaches the actual `@tool` callables (we can't put them in the constant directly because of import ordering; resolve at build time).
- The `"explorer"` subagent has only `build_repo_map` as an explicit tool — the read-only filesystem tools (`ls`, `read_file`, `grep`) come from the shared backend automatically. We don't need to list them. (Verify this is how subagents inherit filesystem tools in the installed version; if subagents need their own backend spec, add it.)
- The `"tester"` subagent has `run_tests` + `edit_file_safe`. No `run_shell`, no `write_file` — it can't create new files or run arbitrary commands. This is the principle of least privilege.
- `subagents=None` (the default) means "use our explorer+tester specs." Passing `subagents=[]` disables them (only the built-in `general-purpose` remains). Passing a custom list overrides.
- The built-in `task` tool is automatically available when subagents are configured. We don't add it.

### Extend `codeit/tools/__init__.py` (~20 SLOC)

```python
@tool
def spawn_subagent(task: str, agent: str = "general-purpose") -> str:
    """Delegate a subtask to a specialized subagent and return its summary.

    Use this for large or isolatable subtasks to keep the main context clean.
    Argument task: a complete, self-contained description of what the subagent should do.
      Subagents are stateless — include everything they need in this one call.
    Argument agent: the subagent name. Options: 'explorer' (read-only codebase search),
      'tester' (run tests + fix), 'general-purpose' (default, same tools as you).
    """
    # This is a teaching wrapper around the built-in task tool. We return a string
    # instructing the model to call task(agent=..., instruction=...). Tools can't
    # call other tools directly — the model orchestrates.
    return (
        f"To delegate to the '{agent}' subagent, call the task tool with: "
        f"agent='{agent}', instruction='''{task}'''. "
        f"The subagent runs in isolated context and returns only a summary."
    )

CUSTOM_TOOLS = [read_summary, run_shell, edit_file_safe, build_repo_map, plan, complete_todo, run_tests, spawn_subagent]
```

**Teaching notes:**
- `spawn_subagent` is **sugar** like `plan` (Ep 10). It returns a string telling the model to call the built-in `task`. The real tool is `task`; `spawn_subagent` makes delegation explicit and legible.
- The docstring tells the model the three subagent options and reminds it that subagents are **stateless** — complete instructions in one call.
- Don't try to invoke `task` from inside `spawn_subagent`. Tools return strings; the model decides.

---

## 12.4 · Tests

### `tests/test_ep12_subagents.py`

```python
import pytest
from codeit.agent import build_agent, SUBAGENTS, _resolve_subagent_tools

def test_subagent_specs_have_required_keys():
    for spec in SUBAGENTS:
        assert "name" in spec
        assert "description" in spec
        assert "system_prompt" in spec

def test_explorer_spec_resolves_tools():
    resolved = _resolve_subagent_tools(SUBAGENTS[0])
    assert resolved["name"] == "explorer"
    tool_names = [getattr(t, "name", str(t)) for t in resolved["tools"]]
    assert "build_repo_map" in tool_names

def test_tester_spec_resolves_tools():
    resolved = _resolve_subagent_tools(SUBAGENTS[1])
    assert resolved["name"] == "tester"
    tool_names = [getattr(t, "name", str(t)) for t in resolved["tools"]]
    assert "run_tests" in tool_names
    assert "edit_file_safe" in tool_names
    # tester must NOT have run_shell
    assert "run_shell" not in tool_names

def test_build_agent_includes_subagents(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    agent, _ = build_agent(model=GenericFakeChatModel(messages=iter(["hi"])), workdir=str(tmp_path))
    assert agent is not None
    # The agent should have the task tool available (built-in when subagents are set)
    # Inspecting the exact tool list requires graph introspection; the weak assertion is "built OK".

def test_build_agent_no_subagents(tmp_path, monkeypatch):
    """Passing subagents=[] should disable custom subagents (only general-purpose remains)."""
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    agent, _ = build_agent(
        model=GenericFakeChatModel(messages=iter(["hi"])),
        workdir=str(tmp_path),
        subagents=[],
    )
    assert agent is not None

def test_spawn_subagent_returns_instruction():
    from codeit.tools import spawn_subagent
    result = spawn_subagent.invoke({"task": "map the codebase", "agent": "explorer"})
    assert "task" in result
    assert "explorer" in result

def test_agent_delegates_via_task(tmp_path, monkeypatch):
    """Drive the agent to call the built-in task tool; assert the subagent ran."""
    from langchain_core.messages import AIMessage
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from codeit.agent import build_agent, run
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    # Script the model to call task(agent='explorer', instruction='...')
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "task", "args": {"agent": "explorer", "instruction": "find the main file"}, "id": "c1", "type": "tool_call"}]),
        "The explorer subagent found main.py.",
    ]))
    agent, _ = build_agent(model=fake, workdir=str(tmp_path))
    state = run(agent, "use the explorer subagent to find the main file")
    assert state is not None
    # The subagent's detailed messages should NOT appear in the parent's state messages
    # (only the summary comes back). This is the framework's isolation guarantee.
    parent_msgs = state["messages"]
    # Heuristic: parent messages should be shorter than if the subagent's full history leaked in
    # (hard to assert precisely without knowing the subagent's exact message count).
    # Strong assertion: no AIMessage in parent state has content from the subagent's internal calls.
    # This test is best-effort; the real isolation guarantee is the framework's.

def test_subagent_messages_dont_leak(tmp_path, monkeypatch):
    """The parent's final state should not contain the subagent's internal tool calls."""
    # This is the load-bearing isolation test. Implementation: script the parent to call task,
    # then inspect the parent's state['messages']. The subagent's internal ls/read_file calls
    # should NOT appear in the parent's messages — only the summary returned by task.
    # The exact assertion depends on the installed version's state shape. Fill in during implementation.
    pass
```

**Test notes:**
- `test_tester_spec_resolves_tools` enforces the principle of least privilege: `tester` has `run_tests` + `edit_file_safe` but NOT `run_shell`. This is the security story for subagents.
- `test_subagent_messages_dont_leak` is the load-bearing isolation test. The exact assertion depends on the installed version's state shape — fill in during implementation by inspecting a real `state["messages"]` after a `task` call. The contract: parent state contains the parent's messages + the `task` tool's *result* (summary), but NOT the subagent's internal `AIMessage`s or `ToolMessage`s.
- `test_agent_delegates_via_task` is the integration test. It confirms the built-in `task` tool fires when subagents are configured.

---

## 12.5 · Demo script `scripts/demo_ep12.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "=== Demo: main agent delegates to explorer subagent ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:32b \
CODEIT_WORKDIR=./examples/sample_app \
python scripts/chat.py "Use the explorer subagent to map the codebase and tell me what files exist. Don't explore yourself — delegate."
echo
echo "Expected: agent calls task(agent='explorer', instruction='...'), explorer runs in isolation, only a summary returns to the main agent."
echo
echo "=== Demo: main agent delegates to tester subagent ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:32b \
CODEIT_WORKDIR=./examples/sample_app \
python scripts/chat.py "Use the tester subagent to make the failing test pass. Don't run tests yourself — delegate."
echo
echo "Note: subagents are stateless. Give complete instructions in one task call."
echo "Use qwen2.5-coder:32b or OpenAI — small models struggle with delegation."
```

On camera:
1. Watch the main agent call `task(agent="explorer", ...)`.
2. Show the explorer's work in the streamed output (it's a subgraph — Ep 14's event streaming will render it in its own panel).
3. Show the summary returning to the main agent — the main agent's context stays small.
4. Emphasize: the explorer can't `rm -rf` — it has no shell. The tester can't create new files — no `write_file`. Principle of least privilege.

---

## 12.6 · README section to add

```
## Ep 12 — Sub-Agents (delegation + isolated context)
Adds: SUBAGENTS specs + _resolve_subagent_tools in codeit/agent.py; spawn_subagent tool in codeit/tools/__init__.py
Run: CODEIT_WORKDIR=./examples/sample_app python scripts/chat.py "Use the explorer subagent to map the codebase."
Key idea: SubAgentMiddleware + task are built-in. We add explorer (read-only) and tester (run_tests+edit) subagent specs. Subagents are stateless; only summaries return.
Lines added: 110 SLOC
Model note: delegation needs a larger model — qwen2.5-coder:32b or OpenAI.
```

---

## 12.7 · Definition of Done

- [ ] `SUBAGENTS` constant + `_resolve_subagent_tools` in `codeit/agent.py`.
- [ ] `build_agent` accepts `subagents`; defaults to explorer+tester.
- [ ] `spawn_subagent` tool in `codeit/tools/__init__.py`.
- [ ] `pytest -m "not live" tests/test_ep12_subagents.py` green (including isolation test).
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-11 HEAD -- '*.py'` adds < 220 lines.
- [ ] Ep 1-11 demos still pass.
- [ ] README "Ep 12" section written; model-size note included.
- [ ] Commit + annotated tag `ep-12`.

---

## 12.8 · What NOT to do this episode

- Do not reimplement `SubAgentMiddleware` or the `task` tool. They're built-in.
- Do not give the `explorer` subagent `run_shell` or `write_file`. Read-only is the point.
- Do not give the `tester` subagent `run_shell`. It can run tests via `run_tests` but not arbitrary commands.
- Do not try to make subagents stateful. They're ephemeral by design — complete instructions in one `task` call.
- Do not assume subagents inherit skills from the main agent. They don't (per the orchestration skill fix). If we want a subagent to have skills (Ep 13), pass them explicitly in the spec.
- Do not invoke `task` from inside `spawn_subagent`. Tools return strings; the model orchestrates.
- Do not skip the isolation test. "Subagent messages don't leak into parent history" is the framework's core promise — verify it.

---

## 12.9 · Common gotchas

- **Subagent tool resolution timing:** the `@tool` callables (`build_repo_map`, `run_tests`, `edit_file_safe`) must be importable when `_resolve_subagent_tools` runs. Watch for circular imports — `codeit/agent.py` importing from `codeit/tools/__init__.py` which imports from `codeit/agent.py`. The current layout (tools don't import agent) should be safe, but verify.
- **Subagent filesystem tools:** do subagents inherit the main agent's `FilesystemBackend` automatically, or do they need their own backend spec? The docs are ambiguous. Verify during pre-flight by inspecting a real subagent's available tools. If subagents need an explicit backend, add it to the spec.
- **`task` arg names:** may be `agent`/`instruction` or `subagent_type`/`task`. Verify against the installed version by inspecting the built-in `task` tool's schema. Adapt `spawn_subagent`'s return string to match.
- **Stateless subagents:** the biggest teaching point. If the model calls `task(agent='explorer', instruction='find X')` and then `task(agent='explorer', instruction='what did you find?')`, the second call starts fresh — the explorer doesn't remember. Tell the model this in the system prompt (Ep 7 already covers it: "Use task for large or isolatable subtasks").
- **Subagent `model` override:** each spec can have its own `model` (e.g. explorer uses a cheaper model, main agent uses a stronger one). We don't override this episode — all use the main agent's model. Mention it as an advanced option.
- **Isolation test brittleness:** the exact assertion for "messages don't leak" depends on the state shape. A robust check: count parent messages before vs. after a `task` call; the delta should be the `task` result (one `ToolMessage`), not the subagent's full internal history.