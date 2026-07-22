# Episode 5 — Running Commands: the Shell Tool (and why it's dangerous)

**Tag:** `ep-05` · **Shape:** add a new tool · **Budget:** ~100 SLOC
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 5

---

## 5.1 · What this episode delivers

The agent gains the ability to run shell commands. There is **no built-in `run_shell`** in Deep Agents' `FilesystemBackend` (sandboxes and `LocalShellBackend` provide an `execute` tool, but those are different backends with different security models — see §5.2). We write a custom `run_shell` `@tool` that:

- Runs the command with cwd = `CODEIT_WORKDIR` (via `workspace_root()` from Ep 4).
- Captures stdout, stderr, and exit code.
- Truncates long output with a note (so we don't blow the context window).
- Returns a single readable string to the model.

**No gating this episode.** The agent can `rm -rf` the workspace. This is the cliffhanger — end the episode pointing at Ep 6. Repeat the warning: even confined to the workspace, an agent can destroy the workspace itself.

---

## 5.2 · Pre-flight: decide `run_shell` vs `LocalShellBackend`

Deep Agents offers two paths to shell access:
1. **`LocalShellBackend(root_dir=..., env=...)`** — a backend that provides an `execute` tool. **No isolation** — runs directly on the host. The docs explicitly say "use only in controlled development environments."
2. **Our custom `run_shell` `@tool`** — runs commands with cwd confined to `CODEIT_WORKDIR`, but the process itself is still a host shell. We control the cwd; we don't sandbox the process.

**Decision: write our own `run_shell`.** Reasons:
- It's *teaching* — the viewer sees exactly what runs.
- `LocalShellBackend` replaces the filesystem backend, which we don't want (we lose `FilesystemBackend`'s virtual_mode sandbox).
- We need fine-grained control over output truncation, and we want to layer the Ep 6 approval gate on top.
- The security posture is the same either way (host process), so the teaching value wins.

**Be honest on camera about the limitation:** `run_shell` confines *cwd* but not *the process*. `run_shell("rm -rf /")` would still try to delete the system if the user has permission. The Ep 6 approval gate is the real safeguard, plus telling viewers: **never run unsandboxed on a real repo.**

---

## 5.3 · File specs

### `codeit/tools/shell.py` (~50 SLOC)

```python
import subprocess
from langchain.tools import tool
from codeit.paths import workspace_root

MAX_OUTPUT_CHARS = 20_000  # ~5k tokens; keeps context window healthy

@tool
def run_shell(command: str) -> str:
    """Run a shell command in the workspace and return stdout+stderr+exit code.

    Use this to run tests, install packages, execute scripts, or inspect git state.
    Argument: a single shell command string (e.g. 'pytest -q' or 'pip install fastapi').
    The command runs with cwd set to the workspace root.
    Output is truncated if longer than ~20k chars.
    """
    cwd = str(workspace_root())
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after 120s.\nCommand: {command}"
    except Exception as e:
        return f"Error launching command: {type(e).__name__}: {e}"

    out = proc.stdout or ""
    err = proc.stderr or ""
    combined = f"$ {command}\n[exit {proc.returncode}]\n"
    if out:
        combined += f"--- stdout ---\n{out}\n"
    if err:
        combined += f"--- stderr ---\n{err}\n"
    if len(combined) > MAX_OUTPUT_CHARS:
        head = combined[:MAX_OUTPUT_CHARS]
        combined = head + f"\n... [truncated, {len(combined)-MAX_OUTPUT_CHARS} more chars]"
    return combined
```

**Teaching notes:**
- `shell=True` so the model can pass a single string with pipes/redirects. This is *more* dangerous (injection) but the agent is the one constructing the command, and Ep 6 gates it. Trade-off: simplicity vs. safety. We pick simplicity for teaching; the gate is the safety layer.
- `timeout=120` prevents the agent hanging on `python -m http.server` or a blocking read.
- `capture_output=True` + `text=True` gives us strings, not bytes.
- Truncation at ~20k chars (~5k tokens) keeps the tool result from eating the context window. The model gets enough to act on; if it needs more, it can run `head`/`tail` itself.
- The docstring tells the model **when** to use it and **what** the arg is. Tool docstrings are prompts.

### Update `codeit/tools/__init__.py` (~10 SLOC)

```python
from codeit.tools.shell import run_shell  # new

CUSTOM_TOOLS: list[Callable] = [read_summary, run_shell]

def register_custom_tools(extra: list[Callable] | None = None) -> list[Callable]:
    return CUSTOM_TOOLS + (extra or [])
```

No change to `codeit/agent.py` — `build_agent` already calls `register_custom_tools` (Ep 3).

---

## 5.4 · Tests

### `tests/test_ep05_shell.py`

```python
import os
import pytest
from codeit.tools.shell import run_shell

def test_echo_returns_output(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    result = run_shell.invoke({"command": "echo hello"})
    assert "hello" in result
    assert "[exit 0]" in result

def test_cwd_is_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    result = run_shell.invoke({"command": "pwd"})
    assert str(tmp_path.resolve()) in result

def test_exit_code_propagated(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    result = run_shell.invoke({"command": "false"})  # exits 1
    assert "[exit 1]" in result

def test_stderr_captured(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    result = run_shell.invoke({"command": "echo oops 1>&2"})
    assert "oops" in result
    assert "stderr" in result

def test_output_truncated(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    result = run_shell.invoke({"command": "python -c 'print(\"x\"*50000)'"})
    assert "truncated" in result

def test_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    result = run_shell.invoke({"command": "sleep 200"})
    assert "timed out" in result.lower()

def test_agent_runs_shell(tmp_path, monkeypatch):
    """Drive the agent to call run_shell; assert the tool executed."""
    from langchain_core.messages import AIMessage
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from codeit.agent import build_agent, run
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "run_shell", "args": {"command": "echo agent-ran-this"}, "id": "c1", "type": "tool_call"}]),
        "The command ran.",
    ]))
    agent, _ = build_agent(model=fake, workdir=str(tmp_path))
    state = run(agent, "Run: echo agent-ran-this")
    # The tool message should contain the echo output
    tool_msgs = [m for m in state["messages"] if m.type == "tool"] if state else []
    assert any("agent-ran-this" in m.content for m in tool_msgs)
```

**Test notes:**
- Use `false` (POSIX) for the exit-1 test; on Windows you'd need a different command, but we target macOS/Linux per the env.
- `test_timeout` runs `sleep 200` — make sure the test's own timeout is higher than the tool's 120s, or lower the tool timeout in the test via monkeypatch. Better: make the tool timeout configurable via env (`CODEIT_SHELL_TIMEOUT`) and set it low in the test.
- The `test_agent_runs_shell` test confirms the wiring through the full agent, not just the tool in isolation.

---

## 5.5 · Demo script `scripts/demo_ep05.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p workspace
echo "=== Demo: agent runs a shell command ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
CODEIT_WORKDIR=./examples/sample_app \
python scripts/chat.py "Install fastapi and uvicorn with pip, then run the test file with pytest, and tell me what happened."
echo
echo "=== WARNING ==="
echo "The agent just ran real shell commands in ./examples/sample_app."
echo "It could have run 'rm -rf .' and deleted the workspace."
echo "Next episode (Ep 6) adds the approval gate so you get to say yes/no."
echo "NEVER run this unsandboxed on a real repo."
```

On camera:
1. Show the agent deciding to call `run_shell("pip install ...")`.
2. Show the agent then calling `run_shell("pytest ...")`.
3. Show the test output coming back to the agent, and the agent summarizing it.
4. **Cliffhanger:** "What if the agent had run `rm -rf .`? It could have. Next episode, we make this safe."

---

## 5.6 · README section to add

```
## Ep 5 — The Shell Tool
Adds: codeit/tools/shell.py (run_shell); registered via register_custom_tools
Run: CODEIT_WORKDIR=./examples/sample_app python scripts/chat.py "Install deps and run pytest."
Key idea: custom @tool running subprocess with cwd=workspace; output truncated; no gating yet.
Lines added: 100 SLOC
SECURITY WARNING: run_shell confines cwd, not the process. The agent can rm -rf the workspace.
NEVER run unsandboxed on a real repo. Ep 6 adds the approval gate.
```

---

## 5.7 · Definition of Done

- [ ] `codeit/tools/shell.py` with `run_shell` tool.
- [ ] `register_custom_tools` includes `run_shell`.
- [ ] `pytest -m "not live" tests/test_ep05_shell.py` green.
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-04 HEAD -- '*.py'` adds < 220 lines.
- [ ] Ep 1-4 demos still pass.
- [ ] README "Ep 5" section written; **on-screen safety line** included.
- [ ] Commit + annotated tag `ep-05`.

---

## 5.8 · What NOT to do this episode

- Do not use `LocalShellBackend` — it replaces the filesystem backend and provides no virtual_mode sandbox.
- Do not add the approval gate yet (Ep 6).
- Do not let `run_shell` accept a list of args (no `shell=False` mode). The teaching simplicity of "one string" is the point; the gate handles safety.
- Do not swallow non-zero exit codes as exceptions — return them as a string so the model can read "exit 1" and self-correct (Ep 11 uses this for recovery).
- Do not remove the timeout — without it a blocking command hangs the whole agent.

---

## 5.9 · Common gotchas

- **`shell=True` injection:** the model constructs the command string. If a malicious file is read into a command, injection is possible. The Ep 6 gate is the mitigation; for now, the workspace is the blast radius.
- **Python's `subprocess.run` with `shell=True` on Windows** uses `cmd.exe`, not bash. We target macOS/Linux per the env. Document this in README.
- **Long output blows context:** the 20k char cap is a heuristic. If the model needs more, it should `head -n 200 file` or `pytest -q`. Mention this in the system prompt (Ep 7).
- **`pip install` side effects:** the demo installs packages into the *user's* environment (or whatever venv is active). For the demo, recommend viewers use a dedicated venv. The workspace confines *files*, not *installed packages*.
- **`timeout=120` vs test runtime:** if a viewer's tests take longer than 120s, `run_shell` truncates them. Make the timeout configurable (`CODEIT_SHELL_TIMEOUT`) and raise the default if needed.