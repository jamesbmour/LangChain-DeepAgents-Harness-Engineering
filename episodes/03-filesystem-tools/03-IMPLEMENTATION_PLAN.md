# Episode 3 — Giving Your Agent Hands: Filesystem Tools

**Tag:** `ep-03` · **Shape:** use the battery — FilesystemMiddleware · **Budget:** ~80 SLOC
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 3

---

## 3.1 · What this episode delivers

The agent gains read-only access to the workspace. We configure `FilesystemBackend(root_dir=CODEIT_WORKDIR, virtual_mode=True)` and the built-in filesystem tools (`ls`, `read_file`, `glob`, `grep`) appear automatically — **zero hand-written tool code**. This is the first "the framework does the work" episode.

To make it *teaching*, not just configuration, we add **one** custom `@tool` (`read_summary`) that wraps the built-in `read_file` with line truncation, plus a central `register_custom_tools()` helper used by every later episode.

The viewer learns:
1. How to attach a `FilesystemBackend` to a Deep Agent.
2. That `virtual_mode=True` is the sandbox — it blocks `../`, `~`, and absolute paths outside root. **The default `virtual_mode=False` provides no security even with `root_dir` set.** (Quote this on camera.)
3. How to register a custom tool alongside the built-ins.

---

## 3.2 · Pre-flight

1. Confirm `FilesystemBackend` import path: `from deepagents.backends import FilesystemBackend`. Verify against the installed version.
2. Confirm the built-in tool names surface automatically when a backend is set: `ls`, `read_file`, `glob`, `grep` (read-only this episode). `write_file`, `edit_file`, `delete` also surface but we don't demo them until Ep 4/8.
3. Prepare `examples/sample_app/` — a tiny 3-file FastAPI todo service. Layout:
   ```
   examples/sample_app/
     main.py          # FastAPI app with /todos endpoint (one bug)
     models.py        # Todo model
     test_main.py     # one failing test
     requirements.txt
   ```
   This is fixture, not taught code — out of the SLOC budget.
4. Set `CODEIT_WORKDIR=./examples/sample_app` for the demo (or copy in the demo script).

---

## 3.3 · File specs

### `codeit/tools/__init__.py` (~30 SLOC)

```python
from typing import Callable
from langchain.tools import tool

@tool
def read_summary(path: str) -> str:
    """Read a file and return its first 50 lines plus a truncation note.

    Use this when you want a quick overview of a file without reading the whole thing.
    Argument: path relative to the workspace root (e.g. 'main.py' or 'src/app.py').
    """
    from pathlib import Path
    from codeit.settings import get_settings
    root = Path(get_settings().workdir).resolve()
    try:
        text = (root / path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: {path} not found in workspace."
    except Exception as e:
        return f"Error reading {path}: {type(e).__name__}: {e}"
    lines = text.splitlines()
    if len(lines) <= 50:
        return text
    return "\n".join(lines[:50]) + f"\n... [truncated, {len(lines)-50} more lines]"

CUSTOM_TOOLS: list[Callable] = [read_summary]

def register_custom_tools(extra: list[Callable] | None = None) -> list[Callable]:
    """Return the list of custom tools to pass to create_deep_agent(tools=...)."""
    return CUSTOM_TOOLS + (extra or [])
```

**Teaching notes:**
- `read_summary` is a *custom* tool — we hand-write it. The built-in `read_file` is provided by the backend; we don't reimplement it. `read_summary` wraps the *idea* of read_file with truncation policy.
- The docstring is written **for the model**: "Use this when you want a quick overview…" That's the schema description the model sees. Tool docstrings are prompts (parent plan §3).
- `register_custom_tools()` is the single seam every later episode uses to add tools. Never pass a raw list to `create_deep_agent` — go through this helper so tests can substitute.

### Extend `codeit/agent.py` (~30 SLOC added)

```python
# NEW: backend wiring
from deepagents.backends import FilesystemBackend
from codeit.tools import register_custom_tools
from pathlib import Path

def build_agent(
    model: BaseChatModel | None = None,
    tools: list | None = None,
    system_prompt: str | None = None,
    backend=None,
    workdir: str | None = None,
):
    """Build a Deep Agent. Attaches a FilesystemBackend sandbox by default."""
    m = model or get_model()
    s = get_settings()
    if backend is None:
        root = Path(workdir or s.workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    all_tools = register_custom_tools(tools)
    return create_deep_agent(
        model=m,
        tools=all_tools,
        system_prompt=system_prompt or "You are CodeIt, a helpful coding assistant.",
        backend=backend,
    ), s
```

**Teaching notes:**
- `root_dir` must be **absolute** (per the docs). We `.resolve()` it.
- `virtual_mode=True` is the sandbox. **Say on camera:** "The default `virtual_mode=False` provides no security even with `root_dir` set. Always pass `virtual_mode=True`."
- We `mkdir(parents=True, exist_ok=True)` so the demo works on a fresh clone.
- We accept `backend` as an arg so tests can inject a mock backend (Ep 4+) without touching env.
- `build_agent` now returns `(agent, settings)` — Ep 2 already established this tuple. Keep it.

---

## 3.4 · Tests

### `tests/test_ep03_filesystem.py`

Test the built-in tools by *driving the agent* with `GenericFakeChatModel` scripted to call them, then assert the tool result appears in the state. Also test `read_summary` by direct call (no agent needed).

```python
import os
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from codeit.agent import build_agent, run
from codeit.tools import read_summary

def _fake_model_listing_then_reading():
    """Model calls ls, then read_file, then answers."""
    return GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "ls", "args": {"path": "."}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "main.py"}, "id": "c2", "type": "tool_call"}]),
        "I see a FastAPI app with a todos endpoint.",
    ]))

def test_builtin_ls_and_read_file(tmp_path, monkeypatch):
    # Copy sample_app into tmp workspace
    import shutil
    src = Path(__file__).parent.parent / "examples" / "sample_app"
    dst = tmp_path / "ws"
    shutil.copytree(src, dst)
    monkeypatch.setenv("CODEIT_WORKDIR", str(dst))
    agent, _ = build_agent(model=_fake_model_listing_then_reading(), workdir=str(dst))
    state = run(agent, "What's in this project?")
    assert state is not None
    # The tool calls should have executed (no pending tool_calls on the last message)
    assert not getattr(state["messages"][-1], "tool_calls", None)

def test_read_summary_truncates(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(f"line {i}" for i in range(100)))
    import os
    os.environ["CODEIT_WORKDIR"] = str(tmp_path)
    result = read_summary.invoke({"path": "big.txt"})
    assert "truncated" in result
    assert "line 0" in result
    assert "line 99" not in result  # truncated away

def test_read_summary_missing_file(tmp_path):
    import os
    os.environ["CODEIT_WORKDIR"] = str(tmp_path)
    result = read_summary.invoke({"path": "nope.py"})
    assert "not found" in result.lower()

def test_traversal_blocked(tmp_path, monkeypatch):
    # virtual_mode=True should block ../ escapes. We test via a direct backend call
    # rather than driving the model, since the model would have to choose to escape.
    from deepagents.backends import FilesystemBackend
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    # The exact API for "try to read outside root" depends on the backend's read method;
    # assert it raises or returns an error string. Inspect the installed backend's API.
    # This test is a placeholder — fill in the actual call signature during implementation.
    pass
```

**Test notes:**
- The `test_traversal_blocked` test is important for the security story. Inspect the installed `FilesystemBackend` to find the read method and call it with `../../../etc/passwd`. Assert it raises or returns an error. If the backend's public surface doesn't expose a direct read method, test via the tool layer instead (call the `read_file` tool function directly with a traversal path).
- `GenericFakeChatModel` + Deep Agents middleware: confirm the middleware passes the tool call through to the backend. If the fake model's `AIMessage` shape doesn't match what Deep Agents expects, subclass and return proper messages.

---

## 3.5 · Demo script `scripts/demo_ep03.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "=== Demo: agent reads the project ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
CODEIT_WORKDIR=./examples/sample_app \
python scripts/chat.py "What files are in this project? What does main.py do?"
echo
echo "Expected: agent calls ls (built-in), then read_file (built-in), then answers."
echo "Zero tool code was written by us — the backend provides them."
```

On camera: emphasize that `ls`, `read_file`, `glob`, `grep` came for free. We only wrote `read_summary` as a teaching example.

---

## 3.6 · README section to add

```
## Ep 3 — Filesystem Tools (read-only)
Adds: codeit/tools/__init__.py (read_summary, register_custom_tools); backend wiring in agent.py
Run: CODEIT_WORKDIR=./examples/sample_app python scripts/chat.py "What's in this project?"
Key idea: FilesystemBackend(root_dir=..., virtual_mode=True) provides ls/read_file/glob/grep. virtual_mode=True is the sandbox.
Lines added: 80 SLOC
Security: NEVER use virtual_mode=False with a real root_dir. It provides no path restrictions.
```

---

## 3.7 · Definition of Done

- [ ] `codeit/tools/__init__.py` with `read_summary` and `register_custom_tools`.
- [ ] `build_agent()` accepts `backend` and `workdir`; defaults to `FilesystemBackend(virtual_mode=True)`.
- [ ] `pytest -m "not live" tests/test_ep03_filesystem.py` green (including traversal test).
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-02 HEAD -- '*.py'` adds < 220 lines.
- [ ] Ep 1 and Ep 2 demos still pass.
- [ ] README "Ep 3" section written; on-screen safety line included.
- [ ] Commit + annotated tag `ep-03`.

---

## 3.8 · What NOT to do this episode

- Do not hand-write `ls`, `read_file`, `glob`, or `grep`. The backend provides them.
- Do not pass `virtual_mode=False` "to keep it simple." It is insecure by default.
- Do not register `read_summary` by passing it directly to `create_deep_agent(tools=[read_summary])` — go through `register_custom_tools` so the seam is consistent for later episodes.
- Do not demo `write_file` or `edit_file` yet (Ep 4 and Ep 8).
- Do not add a `paths.py` module yet — `resolve_in_workspace` comes in Ep 4 when custom tools (shell) need to share the sandbox discipline.

---

## 3.9 · Common gotchas

- **`root_dir` must be absolute:** `FilesystemBackend(root_dir="./workspace")` may silently misbehave. Always `.resolve()`.
- **Backend doesn't expose a direct read API for tests:** if you can't call `backend.read(path)` directly, drive the agent with a fake model scripted to call `read_file`, and assert the `ToolMessage` content in the resulting state.
- **`CODEIT_WORKDIR` relative paths:** the demo script sets it to `./examples/sample_app`, which is relative to CWD. Resolve it in `build_agent` before passing to the backend.
- **Built-in tools already registered:** don't add `ls`/`read_file` to `register_custom_tools` — they conflict with the built-ins. Only register *new* tool names.
- **`FilesystemPermission` is a separate, more granular layer** (see `deepagents.permissions`). We don't use it this episode — `virtual_mode=True` is sufficient for the sandbox. Mention it exists for viewers who want allow-list/deny-list path policies later.