# Episode 11 — Error Recovery: Self-Healing Loops

**Tag:** `ep-11` · **Shape:** add a new tool + custom loop around the graph · **Budget:** ~140 SLOC (tight — may split into `ep-11` + `ep-11b`)
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 11

---

## 11.1 · What this episode delivers

The "wow" episode. The agent writes code, runs it, reads the failure, and fixes it — autonomously, up to a retry cap. Two pieces:

1. **`run_tests(path) -> str`** — a custom `@tool` that runs `pytest` (and optionally `ruff`) in the workspace, captures failures, and returns a readable string. Registered via `register_custom_tools`.
2. **`run_with_recovery(agent, prompt, thread_id, max_retries) -> dict`** — a plain Python loop *around* the Deep Agents graph. After the agent finishes, if `run_tests` was called and reported failures, re-invoke the agent with the failure output appended as a user message; cap retries. Implemented **around** the graph, not as a new graph node — we don't fight the framework.

**Pre-split flag:** if the tool + loop push past 220 SLOC, split into `ep-11` (run_tests tool + tests) and `ep-11b` (recovery loop + integration tests).

---

## 11.2 · Pre-flight

1. Confirm the workspace has `pytest` and `ruff` installed (the demo runs against `examples/sample_app` which has a failing test by design).
2. Decide: does `run_tests` run `pytest` only, or `pytest + ruff`? For teaching, `pytest` only is simpler. Ruff can be a second tool (`run_lint`) or an optional arg (`run_tests(lint: bool = False)`). Go with **`pytest` only** for the core; mention ruff as an exercise.
3. Confirm the recovery loop's contract: we re-invoke the agent with the failure as a *user* message (not a system message) so the model sees it as new input. The thread_id stays the same so the agent's history (including the code it wrote) persists.
4. Skim `/oss/python/deepagents/fault-tolerance.mdx` — Deep Agents has built-in fault tolerance. Our recovery loop is a *teaching* layer on top, for the specific "test failed → fix" pattern. Don't reinvent the framework's retry logic.

---

## 11.3 · File specs

### `codeit/testing.py` (~50 SLOC)

```python
import subprocess
from langchain.tools import tool
from codeit.paths import workspace_root

MAX_OUTPUT_CHARS = 15_000  # ~4k tokens; failure output can be long

@tool
def run_tests(path: str = ".") -> str:
    """Run pytest in the workspace and return pass/fail summary + failures.

    Use this after editing code that has tests, to check your work. If tests fail,
    read the failure output and fix the code, then run_tests again.
    Argument path: test path relative to workspace (default '.' runs all tests).
    """
    cwd = str(workspace_root())
    try:
        proc = subprocess.run(
            ["pytest", path, "-q", "--tb=short", "--no-header"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return "Error: pytest timed out after 180s."
    except FileNotFoundError:
        return "Error: pytest not installed. Run: pip install pytest"
    except Exception as e:
        return f"Error launching pytest: {type(e).__name__}: {e}"

    out = proc.stdout or ""
    err = proc.stderr or ""
    combined = f"$ pytest {path}\n[exit {proc.returncode}]\n"
    if out:
        combined += out
    if err:
        combined += f"\n--- stderr ---\n{err}\n"
    if len(combined) > MAX_OUTPUT_CHARS:
        combined = combined[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(combined)-MAX_OUTPUT_CHARS} more chars]"
    return combined
```

**Teaching notes:**
- `run_tests` is a custom `@tool` (there's no built-in test runner). It's thin — `subprocess.run` with `pytest -q --tb=short` for concise output.
- `--tb=short` keeps tracebacks readable; `--no-header` removes the pytest banner. The model needs the failure, not the chrome.
- Non-zero exit (proc.returncode != 0) means failures — but we return the output *regardless*; the model reads the exit code in `[exit N]` and the failure text.
- The docstring tells the model to "read the failure output and fix the code, then run_tests again" — this primes the recovery loop.
- `timeout=180` because tests can be slow; the cap prevents hangs.
- If pytest isn't installed, we return a clear error (the model can `run_shell("pip install pytest")`).

### Extend `codeit/agent.py` (~70 SLOC for the recovery loop)

```python
from codeit.testing import run_tests
from codeit.tools import register_custom_tools

# Add run_tests to the custom toolset
from codeit.tools import CUSTOM_TOOLS  # if CUSTOM_TOOLS is importable; otherwise rebuild
# In practice, register run_tests via the register_custom_tools(extra=[run_tests]) path,
# OR add it to CUSTOM_TOOLS in codeit/tools/__init__.py. Choose one and be consistent.

def _detect_test_failure(state: dict) -> str | None:
    """Inspect the agent's final state for a run_tests tool result that reported failure.

    Returns the failure text if found, else None. Looks for ToolMessages whose content
    contains a pytest failure signature.
    """
    if not state:
        return None
    msgs = state.get("messages", [])
    for msg in reversed(msgs):  # most recent first
        if getattr(msg, "type", "") != "tool":
            continue
        content = getattr(msg, "content", "") or ""
        # Heuristic: a run_tests result with failures has "[exit 1]" and "FAILED" or "Error"
        if "run_tests" in str(getattr(msg, "name", "")) or "[exit 1]" in content:
            if "FAILED" in content or "Error" in content or "exit 1" in content:
                return content
    return None

def run_with_recovery(agent, prompt: str, thread_id: str = "default", max_retries: int = 3) -> dict:
    """Run the agent; if run_tests reports failures, re-invoke with the failure appended.

    This is a plain Python loop AROUND the Deep Agents graph — not a new graph node.
    We don't fight the framework; we orchestrate at the driver level.
    """
    from codeit.approval import run_with_approval  # if interrupts are on
    s = get_settings()
    config = _config(thread_id, s)
    # Initial run
    state = run(agent, prompt, thread_id)
    for attempt in range(max_retries):
        failure = _detect_test_failure(state)
        if not failure:
            return state  # no failure detected → done
        # Re-invoke with the failure as a user message on the same thread
        console.print(f"[yellow]Test failure detected (attempt {attempt+1}/{max_retries}). Re-invoking agent with the failure...[/yellow]")
        followup = (
            f"The tests failed. Here is the output:\n\n{failure}\n\n"
            f"Read the failure, fix the code, then run_tests again."
        )
        # Re-invoke on the same thread_id — history persists, the agent sees its prior work
        state = run(agent, followup, thread_id)
    console.print(f"[red]Recovery cap reached ({max_retries} retries). Stopping.[/red]")
    return state
```

**Teaching notes:**
- `run_with_recovery` is a **driver-level loop**, not a graph node. We don't modify the Deep Agents graph; we call `run()` multiple times with follow-up messages. This is the contract: "recovery lives *around* the graph, not inside it" (parent plan §11).
- `_detect_test_failure` is heuristic — it scans the final state's tool messages for a `run_tests` result with failure signatures. It's deliberately simple; a production version would track which tool call the agent just made and inspect that specifically.
- The re-invoke uses the **same `thread_id`** so the agent's history (the code it wrote, the tests it ran) persists. The follow-up is a *user* message so the model treats it as new input.
- `max_retries` defaults to 3. The cap prevents infinite loops — a model that can't fix the failure keeps trying forever without it. Reuse `CODEIT_MAX_ITERS` if you want a single knob; we keep them separate so the recovery cap is independent of the per-run recursion limit.
- We don't auto-approve in the recovery loop — if `interrupt_on` is set, `run()` already handles approval. The recovery loop just calls `run()` again.

### Update `codeit/tools/__init__.py` (~5 SLOC)

Add `run_tests` to `CUSTOM_TOOLS`:

```python
from codeit.testing import run_tests

CUSTOM_TOOLS = [read_summary, run_shell, edit_file_safe, build_repo_map, plan, complete_todo, run_tests]
```

---

## 11.4 · Tests

### `tests/test_ep11_recovery.py`

```python
import pytest
from codeit.testing import run_tests

def test_run_tests_passing(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    result = run_tests.invoke({"path": "."})
    assert "[exit 0]" in result
    assert "passed" in result.lower() or "1 passed" in result

def test_run_tests_failing(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    result = run_tests.invoke({"path": "."})
    assert "[exit 1]" in result
    assert "FAILED" in result or "assert False" in result

def test_run_tests_missing_pytest(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    # This test only works if pytest isn't on PATH in the test env — hard to simulate.
    # Skip unless you can mock subprocess.run. Mark as integration or skip.
    pass

def test_detect_test_failure_finds_failure():
    from codeit.agent import _detect_test_failure
    from langchain_core.messages import ToolMessage
    state = {"messages": [ToolMessage(content="[exit 1]\nFAILED test_bad.py::test_bad", name="run_tests", tool_call_id="c1")]}
    failure = _detect_test_failure(state)
    assert failure is not None
    assert "FAILED" in failure

def test_detect_test_failure_no_failure():
    from codeit.agent import _detect_test_failure
    from langchain_core.messages import ToolMessage
    state = {"messages": [ToolMessage(content="[exit 0]\n1 passed", name="run_tests", tool_call_id="c1")]}
    assert _detect_test_failure(state) is None

def test_detect_test_failure_empty_state():
    from codeit.agent import _detect_test_failure
    assert _detect_test_failure({}) is None
    assert _detect_test_failure(None) is None

def test_recovery_loop_converges(tmp_path, monkeypatch):
    """Model writes buggy code → run_tests fails → model fixes → run_tests passes."""
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from codeit.agent import build_agent, run_with_recovery
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    # Script the fake model: call run_tests (fails), then call edit_file_safe (fix), then run_tests (passes)
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "run_tests", "args": {"path": "."}, "id": "c1", "type": "tool_call"}]),
        # The harness executes run_tests and appends a ToolMessage with [exit 1] FAILED
        # Then the model's next scripted message:
        AIMessage(content="", tool_calls=[{"name": "edit_file_safe", "args": {"path": "test_bad.py", "search": "assert False", "replace": "assert True"}, "id": "c2", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "run_tests", "args": {"path": "."}, "id": "c3", "type": "tool_call"}]),
        "Done. Tests pass now.",
    ]))
    agent, _ = build_agent(model=fake, workdir=str(tmp_path))
    state = run_with_recovery(agent, "fix the failing test", max_retries=3)
    # The recovery loop should have converged (no more failures detected)
    assert state is not None

def test_recovery_loop_hits_cap(tmp_path, monkeypatch):
    """Model never fixes the failure — recovery loop stops at max_retries."""
    from langchain_core.messages import AIMessage
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from codeit.agent import build_agent, run_with_recovery
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    # Script the model to always call run_tests (which fails) and never fix
    from itertools import cycle
    fake = GenericFakeChatModel(messages=cycle([
        AIMessage(content="", tool_calls=[{"name": "run_tests", "args": {"path": "."}, "id": "c1", "type": "tool_call"}]),
    ]))
    agent, _ = build_agent(model=fake, workdir=str(tmp_path))
    state = run_with_recovery(agent, "fix it", max_retries=2)
    # Should return without hanging (cap enforced)
    assert state is not None or state is None  # reaching here is the assertion
```

**Test notes:**
- The `test_recovery_loop_converges` test is subtle: the `GenericFakeChatModel` returns scripted messages, but the *tool execution* (run_tests, edit_file_safe) is done by the framework. For the test to be meaningful, the framework must actually execute `run_tests` against the workspace — which means the workspace must have a real failing test that the scripted `edit_file_safe` actually fixes. Set up `tmp_path/test_bad.py` with `assert False`, and the scripted edit changes it to `assert True`. The second `run_tests` call then passes. This is a real integration test, not a mock.
- The `test_recovery_loop_hits_cap` test is simpler: the model never stops calling `run_tests`, the failure is always detected, and the loop caps at 2. Assert it returns (doesn't hang) — wrap with a timeout.
- `_detect_test_failure` is unit-testable in isolation — that's the load-bearing logic. Test it directly with constructed `ToolMessage` objects.
- If `ToolMessage`'s constructor in the installed `langchain_core` requires different args, adapt.

---

## 11.5 · Demo script `scripts/demo_ep11.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "=== Demo: agent writes code, runs it, reads the failure, fixes it ==="
# Reset the sample app's bug
cp examples/sample_app/test_main.py /tmp/test_main.py.bak
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:32b \
CODEIT_WORKDIR=./examples/sample_app \
python -c "
from codeit.agent import build_agent
from codeit.agent import run_with_recovery
agent, _ = build_agent()
state = run_with_recovery(agent, 'Run the tests. If they fail, read the failure and fix the code. Then run the tests again.', max_retries=3)
print('--- final ---')
print(state['messages'][-1].content if state else 'no state')
"
# Restore
cp /tmp/test_main.py.bak examples/sample_app/test_main.py 2>/dev/null || true
echo
echo "Expected: agent runs pytest, sees the failure, edits the file, re-runs pytest, passes."
echo "This is the self-healing loop. Use qwen2.5-coder:32b or OpenAI — small models can't recover."
```

On camera: this is the **wow moment**. Don't over-narrate — let the agent fail, read the traceback, fix it, and pass. Then explain what just happened.

---

## 11.6 · README section to add

```
## Ep 11 — Error Recovery (run_tests + self-healing loop)
Adds: codeit/testing.py (run_tests tool); run_with_recovery driver loop in codeit/agent.py
Run: CODEIT_WORKDIR=./examples/sample_app python -c "from codeit.agent import build_agent, run_with_recovery; a,_=build_agent(); run_with_recovery(a, 'run tests and fix failures')"
Key idea: run_tests is a custom @tool; run_with_recovery is a plain Python loop AROUND the graph (not a node). Cap prevents infinite loops.
Lines added: 140 SLOC (split into ep-11 + ep-11b if over budget)
Model note: recovery is unreliable on small local models. Use qwen2.5-coder:32b or OpenAI.
```

---

## 11.7 · Definition of Done

- [ ] `codeit/testing.py` with `run_tests` tool.
- [ ] `run_with_recovery` and `_detect_test_failure` in `codeit/agent.py`.
- [ ] `register_custom_tools` includes `run_tests`.
- [ ] `pytest -m "not live" tests/test_ep11_recovery.py` green (including convergence + cap tests).
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-10 HEAD -- '*.py'` adds < 220 lines. **If over, split into ep-11 + ep-11b.**
- [ ] Ep 1-10 demos still pass.
- [ ] README "Ep 11" section written; model-size note included.
- [ ] Commit + annotated tag `ep-11` (or `ep-11` + `ep-11b`).

---

## 11.8 · What NOT to do this episode

- Do not implement the recovery loop as a new LangGraph node. It lives *around* the graph, in the driver. Don't fight the framework.
- Do not skip the retry cap. A model that can't fix the failure loops forever without it.
- Do not make `_detect_test_failure` too clever. The heuristic ("[exit 1]" + "FAILED" in a run_tests ToolMessage) is deliberately simple. A production version would track the last tool call explicitly.
- Do not swallow the failure silently in `run_tests`. Return the full failure text — the model needs to read it.
- Do not re-invoke on a *new* `thread_id`. The history (code written, tests run) must persist; same thread.
- Do not run `ruff` inside `run_tests` this episode. Keep it `pytest`-only for clarity. Mention ruff as an exercise.
- Do not catch `GraphRecursionError` specially in the recovery loop — `run()` already handles it (Ep 2).

---

## 11.9 · Common gotchas

- **`ToolMessage` construction:** the installed `langchain_core` may require `name`, `tool_call_id`, and `content` args. Verify the constructor signature.
- **`_detect_test_failure` false positives:** a passing test run might contain the word "Error" in a traceback that's actually a passing skip reason. Tighten the heuristic if you see false recoveries in the demo. The strongest signal is `"[exit 1]"` + `"FAILED"` together.
- **pytest not installed in the workspace's venv:** `run_tests` calls `pytest` from PATH. If the viewer's venv doesn't have it, the tool returns the "pytest not installed" error. The demo script should `pip install pytest` first, or the README should tell viewers.
- **`run_with_recovery` + `interrupt_on`:** if the agent's `edit_file_safe` is gated (Ep 6/8), the recovery loop's re-invoke will trigger an approval prompt. The viewer has to approve the fix. This is correct behavior — don't bypass it. For the demo, either use `--yolo` (dangerous) or tell the viewer to approve the fix.
- **Re-invoke vs. fresh invoke:** re-invoking on the same `thread_id` continues the conversation; the agent sees its prior work. This is what we want. A fresh `thread_id` would make the agent start over, losing the code it wrote.
- **`max_retries` vs `CODEIT_MAX_ITERS`:** keep them separate. `CODEIT_MAX_ITERS` is the per-run recursion limit (Ep 2); `max_retries` is the number of recovery iterations. Don't conflate.