# Episode 4 — Writing Code: `write_file` + the Workspace Sandbox

**Tag:** `ep-04` · **Shape:** use the battery — FilesystemBackend write · **Budget:** ~70 SLOC
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 4

---

## 4.1 · What this episode delivers

The shortest episode by design. `write_file` is already provided by `FilesystemBackend` — we don't write any new tool code. The work here is:

1. **Solidify the backend config** (no behavior change from Ep 3, but we make it the unambiguous default path).
2. **Add `resolve_in_workspace(path) -> Path`** — a helper that custom tools (Ep 5 shell, Ep 8 edit wrapper, Ep 9 repo map) will use to share the backend's sandbox discipline. This is the only new code.
3. **Reinforce the sandbox story on camera** — what `virtual_mode=True` actually protects against, and why `--yolo` (Ep 6) is still dangerous *even inside* the sandbox.

The viewer writes a file by asking the agent, and the agent uses the built-in `write_file`. Then the viewer runs the file outside the agent. The "agent wrote real code" moment.

---

## 4.2 · Pre-flight

1. Confirm `write_file` is among the built-in tools surfaced by `FilesystemBackend` (it should be — see the backends doc list: `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`).
2. Confirm `write_file` creates parent directories automatically. If it doesn't, the agent will need to call a `mkdir`-like tool first or we add a thin wrapper. Check the installed backend's behavior; the plan assumes auto-mkdir.
3. Prepare a clean `workspace/` dir for the demo (it's in `.gitignore`). The demo will write `main.py` there.

---

## 4.3 · File specs

### New: `codeit/paths.py` (~25 SLOC)

```python
from pathlib import Path
from codeit.settings import get_settings

class PathEscapeError(PermissionError):
    """Raised when a resolved path leaves the workspace sandbox."""

def resolve_in_workspace(path: str | Path) -> Path:
    """Resolve a path relative to CODEIT_WORKDIR, refusing to escape the workspace.

    Custom tools (run_shell, run_tests, build_repo_map) call this so they share
    the same sandbox discipline as the built-in FilesystemBackend tools.
    """
    root = Path(get_settings().workdir).resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise PathEscapeError(
            f"Path {path!r} resolves outside the workspace ({root}). Refusing."
        )
    return target

def workspace_root() -> Path:
    """Return the resolved workspace root."""
    return Path(get_settings().workdir).resolve()
```

**Teaching notes:**
- `resolve_in_workspace` mirrors what `FilesystemBackend(virtual_mode=True)` does internally, but for *our* custom tools. When Ep 5's `run_shell` runs `cd $WORKDIR && <cmd>`, it uses `workspace_root()` so the shell's cwd is the sandbox.
- `PathEscapeError` is a `PermissionError` subclass so it's caught by generic `except Exception` handlers but identifiable.
- This is the **only** new file this episode. The agent itself doesn't change — `build_agent` already wires `FilesystemBackend` from Ep 3.

### Update `codeit/tools/__init__.py` (~5 SLOC)

Import `resolve_in_workspace` so later episodes can use it without importing from `codeit.paths` directly:

```python
from codeit.paths import resolve_in_workspace, workspace_root, PathEscapeError  # re-export
```

### No change to `codeit/agent.py`

`build_agent` from Ep 3 already defaults to `FilesystemBackend(root_dir=..., virtual_mode=True)`. `write_file` is already exposed. Nothing to change.

---

## 4.4 · Tests

### `tests/test_ep04_write_sandbox.py`

```python
import os
from pathlib import Path
import pytest
from codeit.paths import resolve_in_workspace, workspace_root, PathEscapeError

def test_resolve_inside_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    p = resolve_in_workspace("sub/dir/main.py")
    assert p == (tmp_path / "sub" / "dir" / "main.py").resolve()

def test_resolve_blocks_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    with pytest.raises(PathEscapeError):
        resolve_in_workspace("../../etc/passwd")

def test_resolve_blocks_absolute_outside(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    with pytest.raises(PathEscapeError):
        resolve_in_workspace("/etc/passwd")

def test_workspace_root_returns_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    assert workspace_root() == tmp_path.resolve()

def test_write_file_round_trip(tmp_path, monkeypatch):
    """Drive the agent to write_file, then read_file, assert contents match."""
    from langchain_core.messages import AIMessage
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from codeit.agent import build_agent, run
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"path": "hello.py", "content": "print('hi')"}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "hello.py"}, "id": "c2", "type": "tool_call"}]),
        "Done. The file contains print('hi').",
    ]))
    agent, _ = build_agent(model=fake, workdir=str(tmp_path))
    state = run(agent, "Write hello.py with print('hi'), then read it back.")
    # Assert the file exists on disk (FilesystemBackend writes to disk)
    assert (tmp_path / "hello.py").read_text() == "print('hi')"

def test_write_outside_workspace_blocked(tmp_path, monkeypatch):
    """The backend's virtual_mode=True should refuse writes outside root.
    Drive the model to attempt a traversal write; assert it fails gracefully
    (error returned to model, no file created outside workspace)."""
    # Implementation: script the fake model to call write_file(path="../../../etc/pwned", content="...").
    # Assert no file appears at /etc/pwned (it won't, in a real test env) and the
    # ToolMessage returned to the model contains an error string.
    # Fill in during implementation based on how the backend surfaces the denial.
    pass
```

**Test notes:**
- The `test_write_outside_workspace_blocked` test completes the security story from Ep 3. The exact assertion depends on whether the backend raises or returns an error string to the model. Inspect the installed backend and write the assertion to match. Either is acceptable — the point is "the agent can't escape."
- `test_write_file_round_trip` proves the file lands on disk (since we use `FilesystemBackend`, not `StateBackend`). If we later switch to `StateBackend` for some tests, files live in graph state, not on disk — adjust the assertion.

---

## 4.5 · Demo script `scripts/demo_ep04.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p workspace
echo "=== Demo: agent writes a FastAPI hello-world ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
CODEIT_WORKDIR=./workspace \
python scripts/chat.py "Create a file main.py with a FastAPI app that has a GET /hello endpoint returning {'msg':'hello'}."
echo
echo "=== Now run the file yourself (the agent did NOT run it) ==="
cd workspace && pip install fastapi uvicorn --quiet 2>/dev/null || true && python -c "import main; print('imports OK')" && cd ..
echo
echo "=== Sandbox check: the agent cannot write outside ./workspace ==="
echo "Try: ask the agent to write to /tmp/pwned — it should refuse."
```

On camera:
1. Show the agent writing `main.py` via the built-in `write_file`.
2. Open `workspace/main.py` in the editor — real file on disk.
3. Run it yourself (`uvicorn main:app`) to prove it's real code.
4. **Safety moment:** try to make the agent write to `/tmp/pwned` — show it refuses. This is `virtual_mode=True` earning its keep.

---

## 4.6 · README section to add

```
## Ep 4 — write_file + the Workspace Sandbox
Adds: codeit/paths.py (resolve_in_workspace, workspace_root, PathEscapeError)
Run: CODEIT_WORKDIR=./workspace python scripts/chat.py "Create main.py with a FastAPI hello endpoint."
Key idea: write_file is built-in; codeit/paths.py gives custom tools the same sandbox discipline.
Note: whole-file write is simplest. Ep 8 makes edit_file the default edit path; write_file stays for new files.
Lines added: 70 SLOC
Security: virtual_mode=True blocks ../, ~, and absolute paths outside root. Never weaken this.
```

---

## 4.7 · Definition of Done

- [ ] `codeit/paths.py` with `resolve_in_workspace`, `workspace_root`, `PathEscapeError`.
- [ ] `codeit/tools/__init__.py` re-exports the path helpers.
- [ ] `pytest -m "not live" tests/test_ep04_write_sandbox.py` green (including traversal blocks).
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-03 HEAD -- '*.py'` adds < 220 lines.
- [ ] Ep 1, 2, 3 demos still pass.
- [ ] README "Ep 4" section written; on-screen safety line included.
- [ ] Commit + annotated tag `ep-04`.

---

## 4.8 · What NOT to do this episode

- Do not hand-write `write_file`. It's built-in.
- Do not add a `mkdir` tool — `write_file` creates parent dirs (verify, but assume yes).
- Do not weaken `virtual_mode` for the demo. If the demo needs to write outside the workspace, change `CODEIT_WORKDIR`, not the sandbox flag.
- Do not introduce `edit_file` yet (Ep 8).
- Do not add the approval gate yet (Ep 6). The agent can `rm -rf` the *workspace* this episode — that's the cliffhanger Ep 5/6 picks up.

---

## 4.9 · Common gotchas

- **`write_file` doesn't auto-mkdir:** if the installed backend doesn't create parent dirs, the agent's write to `sub/dir/main.py` fails with a confusing error. Either (a) add a tiny `ensure_dirs` helper inside `write_file` via a wrapper tool, or (b) tell the agent in the system prompt to create dirs first. Verify before assuming.
- **`StateBackend` vs `FilesystemBackend` confusion:** `StateBackend` stores files in graph state (in-memory, thread-scoped). `FilesystemBackend` writes to disk. We use `FilesystemBackend` so the viewer sees real files. Don't accidentally regress to `StateBackend`.
- **`resolve_in_workspace` vs backend's own resolution:** our helper is for *custom* tools. The built-in `write_file`/`read_file` use the backend's own (also-secure) resolution. They should agree, but they're separate code paths. Don't try to funnel built-ins through `resolve_in_workspace`.
- **Symlinks:** `resolve()` follows symlinks. If the workspace contains a symlink pointing outside, `resolve_in_workspace` will follow it and the `relative_to(root)` check fails — correctly refusing. Document this if a viewer asks.