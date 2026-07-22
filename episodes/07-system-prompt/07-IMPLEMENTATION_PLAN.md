# Episode 7 — The System Prompt: engineering personality & rules

**Tag:** `ep-07` · **Shape:** customize the battery · **Budget:** ~80 SLOC of logic + prompt text as constants
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 7

---

## 7.1 · What this episode delivers

The agent gets a real system prompt — the personality, tool-use policy, safety rules, and editing conventions that turn a generic assistant into CodeIt. We also load project-specific context from `AGENTS.md` (Deep Agents' own convention) or `CODEIT.md` fallback in the workspace, and compose the two.

New pieces:
1. `codeit/prompts.py` — `SYSTEM_PROMPT` constant, `load_project_context()`, `build_system_prompt()`.
2. `build_agent()` now calls `build_system_prompt(workdir)` instead of a hardcoded string.

This is the "same task, dramatically better behavior" episode — show a before/after on camera.

---

## 7.2 · Pre-flight

1. Confirm Deep Agents' convention for project context files. `AGENTS.md` is the LangChain/Deep Agents convention (the `dcode` CLI reads it). We also support `CODEIT.md` as a fallback so viewers can name it either way.
2. Confirm `create_deep_agent(system_prompt=...)` accepts a plain string (per the quickstart: `system_prompt="You are a helpful assistant"`). It does.
3. Skim the Deep Agents docs for any *reserved* system-prompt behavior — the harness may prepend its own instructions (e.g. for TodoList/Filesystem middleware). Our `system_prompt` is *added*, not *replacing*. Design `SYSTEM_PROMPT` accordingly: don't try to override the harness's built-in tool descriptions; just add policy.

---

## 7.3 · File specs

### `codeit/prompts.py` (~70 SLOC of logic + prompt text)

```python
from pathlib import Path
from codeit.settings import get_settings

SYSTEM_PROMPT = """You are CodeIt, a terminal coding agent.

# Role
You help the user write, edit, and run code in their workspace. You read files, write files,
run shell commands, plan multi-step tasks, and delegate to specialized subagents when useful.

# Tool-use policy
- Prefer the built-in filesystem tools (ls, read_file, grep, glob) for exploration.
- Use write_file only for new files. Use edit_file (or edit_file_safe) for changes to existing files.
- Use run_shell for tests, installs, and git operations. The user will be asked to approve
  mutating or destructive commands — when in doubt, prefer the least destructive option.
- Use write_todos to plan any task that needs more than one step. Check items off as you finish them.
- Use task (subagents) for large or isolatable subtasks, especially codebase exploration.

# Safety
- Never run destructive commands (rm -rf, git push -f, dd, mkfs) without explaining why first.
- If a command might modify files outside the workspace, say so and stop.
- The user can reject any action. Accept rejection gracefully and try a safer approach.

# Editing rules
- For small targeted changes, use edit_file (search-replace). Never rewrite a whole file
  to change a few lines.
- After editing code that has tests, run the tests with run_shell('pytest -q') and fix failures.

# Communication
- Be concise. Say what you're about to do, do it, then summarize the result in one or two lines.
- When a tool call fails, read the error, explain what went wrong in one sentence, and try again.
- Don't apologize; fix.
"""

def load_project_context(root: str | Path | None = None) -> str:
    """Read AGENTS.md (or CODEIT.md fallback) from the workspace. Return '' if absent."""
    r = Path(root or get_settings().workdir).resolve()
    for name in ("AGENTS.md", "CODEIT.md"):
        candidate = r / name
        if candidate.is_file():
            try:
                return f"\n\n# Project context ({name})\n\n" + candidate.read_text(encoding="utf-8")
            except Exception:
                return ""  # degrade gracefully
    return ""

def build_system_prompt(root: str | Path | None = None) -> str:
    """Compose the harness system prompt with any project context from the workspace."""
    return SYSTEM_PROMPT + load_project_context(root)
```

**Teaching notes:**
- `SYSTEM_PROMPT` is a module constant. Text is cheap against the line budget (parent plan §6) — keep prompt text in `prompts.py`, not in `agent.py`.
- The prompt is written **for the model**, not for the viewer. It's a document the model reads every turn. Keep it imperative and short.
- `load_project_context` is graceful: missing file → empty string → `build_system_prompt` returns just `SYSTEM_PROMPT`. No error, no crash.
- We check `AGENTS.md` first (Deep Agents convention) then `CODEIT.md` (viewer's preferred name). One or the other; first match wins.
- The harness's *own* middleware (TodoList, Filesystem) adds its own tool descriptions on top of our `system_prompt`. We don't fight that — we add policy, not tool definitions.

### Update `codeit/agent.py` (~10 SLOC)

```python
from codeit.prompts import build_system_prompt

def build_agent(
    model: BaseChatModel | None = None,
    tools: list | None = None,
    system_prompt: str | None = None,  # now defaults to build_system_prompt(workdir)
    backend=None,
    workdir: str | None = None,
    interrupt_on: dict | None = None,
    checkpointer=None,
):
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
        system_prompt=system_prompt or build_system_prompt(wd),  # NEW: composed prompt
        backend=backend,
    )
    if interrupt_on:
        kwargs["interrupt_on"] = interrupt_on
        kwargs["checkpointer"] = checkpointer or MemorySaver()
    return create_deep_agent(**kwargs), s
```

**Teaching notes:**
- `system_prompt or build_system_prompt(wd)` — if the caller passes a prompt (e.g. a test), use it; otherwise compose from `prompts.py` + workspace's `AGENTS.md`.
- We resolve `wd` once and pass to both the backend and the prompt builder so they agree on the workspace root.

---

## 7.4 · Tests

### `tests/test_ep07_prompts.py`

```python
from pathlib import Path
from codeit.prompts import SYSTEM_PROMPT, load_project_context, build_system_prompt

def test_system_prompt_is_nonempty():
    assert "CodeIt" in SYSTEM_PROMPT
    assert "Safety" in SYSTEM_PROMPT

def test_load_project_context_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# My project\nUse FastAPI.\n")
    ctx = load_project_context(tmp_path)
    assert "My project" in ctx
    assert "FastAPI" in ctx

def test_load_project_context_codeit_md_fallback(tmp_path):
    (tmp_path / "CODEIT.md").write_text("# CodeIt project\nUse Flask.\n")
    ctx = load_project_context(tmp_path)
    assert "CodeIt project" in ctx

def test_load_project_context_agents_md_takes_precedence(tmp_path):
    (tmp_path / "AGENTS.md").write_text("use AGENTS")
    (tmp_path / "CODEIT.md").write_text("use CODEIT")
    ctx = load_project_context(tmp_path)
    assert "AGENTS" in ctx
    assert "CODEIT" not in ctx

def test_load_project_context_missing_returns_empty(tmp_path):
    assert load_project_context(tmp_path) == ""

def test_build_system_prompt_composes(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Proj\nUse FastAPI.\n")
    prompt = build_system_prompt(tmp_path)
    assert "CodeIt" in prompt  # base system prompt
    assert "Proj" in prompt      # project context appended
    assert prompt.index("CodeIt") < prompt.index("Proj")  # base first

def test_build_system_prompt_no_project_file(tmp_path):
    prompt = build_system_prompt(tmp_path)
    assert prompt == SYSTEM_PROMPT  # graceful degradation

def test_agent_uses_composed_prompt(tmp_path, monkeypatch):
    """build_agent should pass build_system_prompt(workdir) to create_deep_agent."""
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    (tmp_path / "AGENTS.md").write_text("# Proj\nUse FastAPI.\n")
    from codeit.agent import build_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    agent, _ = build_agent(model=GenericFakeChatModel(messages=iter(["hi"])), workdir=str(tmp_path))
    # We can't directly inspect the system prompt the agent received without poking
    # into its graph; instead, assert the agent built successfully. The composition
    # is tested above in test_build_system_prompt_composes.
    assert agent is not None
```

**Test notes:**
- The composition tests are the load-bearing ones: they prove the workspace's `AGENTS.md` content lands in the prompt the model sees.
- `test_agent_uses_composed_prompt` is weak (can't easily inspect the agent's internal prompt) — the strong assertion is that `build_system_prompt` is called with the right `workdir`. If you want a stronger test, monkeypatch `build_system_prompt` and assert it was called with `wd`.

---

## 7.5 · Demo script `scripts/demo_ep07.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p workspace
cat > workspace/AGENTS.md <<'EOF'
# Sample project context

This is a FastAPI todo service. The bug is in main.py: the /todos endpoint
returns a list but the test expects a dict with a 'todos' key. Fix the endpoint
to return {'todos': [...]}. Run pytest after editing.
EOF

echo "=== Demo: WITHOUT project context (bad behavior) ==="
rm -f workspace/AGENTS.md
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
CODEIT_WORKDIR=./examples/sample_app \
python scripts/chat.py "Fix the failing test." 2>&1 | head -40

echo
echo "=== Demo: WITH project context (good behavior) ==="
cp workspace/AGENTS.md examples/sample_app/AGENTS.md
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
CODEIT_WORKDIR=./examples/sample_app \
python scripts/chat.py "Fix the failing test." 2>&1 | head -40

rm -f examples/sample_app/AGENTS.md
echo
echo "Expected: with AGENTS.md, the agent follows the project's specific instructions."
```

On camera:
1. Run the same task with no `AGENTS.md` — the agent flails or does the wrong thing.
2. Add `AGENTS.md` with project-specific guidance.
3. Re-run — the agent follows the instructions, fixes the right bug, runs the tests.

---

## 7.6 · README section to add

```
## Ep 7 — The System Prompt + AGENTS.md
Adds: codeit/prompts.py (SYSTEM_PROMPT, load_project_context, build_system_prompt); build_agent composes it
Run: echo '# My project\\nUse FastAPI.' > workspace/AGENTS.md && python scripts/chat.py "What is this project?"
Key idea: a structured harness prompt + project context from AGENTS.md (Deep Agents convention).
Lines added: 80 SLOC of logic + prompt text as constants
```

---

## 7.7 · Definition of Done

- [ ] `codeit/prompts.py` with `SYSTEM_PROMPT`, `load_project_context`, `build_system_prompt`.
- [ ] `build_agent` composes `build_system_prompt(wd)` by default.
- [ ] `pytest -m "not live" tests/test_ep07_prompts.py` green.
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-06 HEAD -- '*.py'` adds < 220 lines (count prompt-text SLOC too — it's long but under budget).
- [ ] Ep 1-6 demos still pass.
- [ ] README "Ep 7" section written.
- [ ] Commit + annotated tag `ep-07`.

---

## 7.8 · What NOT to do this episode

- Do not put prompt text in `agent.py`. Keep it in `prompts.py` as constants (parent plan §6).
- Do not try to override the harness's built-in tool descriptions. `create_deep_agent`'s middleware adds its own instructions for `write_todos`, `task`, filesystem tools. Our `system_prompt` is *additional* policy, not a replacement.
- Do not make `load_project_context` raise on a missing file. Graceful degradation is the contract.
- Do not read `AGENTS.md` from outside the workspace. The workspace sandbox applies; `load_project_context` reads from `CODEIT_WORKDIR`.
- Do not write a 500-line system prompt. Keep it short enough that the model actually reads it. The `SYSTEM_PROMPT` above is ~30 lines — that's plenty.

---

## 7.9 · Common gotchas

- **`AGENTS.md` is the Deep Agents convention** (the `dcode` CLI reads it). Don't invent a new name as the primary; use `AGENTS.md` first, `CODEIT.md` as a viewer-friendly fallback.
- **Prompt injection from `AGENTS.md`:** the project file is *untrusted content* the model reads as system instructions. If a malicious `AGENTS.md` says "ignore all safety rules," the model might. Mention this on camera. The mitigation is the Ep 6 approval gate — even a misled model can't `rm -rf` without the user saying yes.
- **`create_deep_agent` may prepend its own system text:** the harness's middleware (TodoList, Filesystem, SubAgent) injects instructions about the built-in tools. Our `system_prompt` is appended. Verify by inspecting a real prompt in LangSmith (Ep 15 appendix) if viewers ask where everything lands.
- **Empty workspace, empty `AGENTS.md`:** `load_project_context` returns `""` and `build_system_prompt` returns just `SYSTEM_PROMPT`. Test this case (it's `test_build_system_prompt_no_project_file`).
- **The harness's `write_todos` is described by the framework:** our `SYSTEM_PROMPT` says "use write_todos to plan" — that's policy, not a redefinition. Don't try to redefine the tool's schema in the prompt.