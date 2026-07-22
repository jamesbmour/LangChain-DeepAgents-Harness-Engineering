# Episode 8 — Surgical Edits: `edit_file` + the Fuzzy Fallback Wrapper

**Tag:** `ep-08` · **Shape:** use + customize the battery · **Budget:** ~140 SLOC (tightest episode — may split into `ep-08` + `ep-08b`)
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 8

---

## 8.1 · What this episode delivers

The agent graduates from whole-file rewrites (`write_file`) to **search-replace edits** — the industry-standard reliable edit format used by Aider, Codex, and Cline. Deep Agents' `FilesystemBackend` provides `edit_file` as a built-in, so we don't reimplement the core. What we add is a teaching wrapper:

- **`edit_file_safe(path, search, replace)`** — a custom `@tool` that:
  1. First tries the built-in `edit_file` (exact search match).
  2. On a miss, attempts a `difflib` fuzzy match (≥90% similarity) and applies it with a note.
  3. On total failure, returns a unified diff so the model can self-correct.
- Goes through the Ep 6 approval gate (`interrupt_on={"edit_file_safe": True}`).

The viewer learns: the search-replace format, why exact matches fail on small models, and how to harden the tool so the agent self-corrects instead of giving up.

**Pre-split flag:** if the fuzzy fallback + tests push past 220 SLOC, split into `ep-08` (built-in `edit_file` + format explainer + exact-match tests) and `ep-08b` (fuzzy fallback wrapper + self-correction tests). Parent plan §11.

---

## 8.2 · Pre-flight: verify the built-in `edit_file` signature

**This is critical.** The Deep Agents docs list `edit_file` as a built-in tool but don't document its exact arg schema in the markdown (it's in the reference API). The Aider/Codex/Cline convention is `search`/`replace`, but Deep Agents may use different names (e.g. `old_str`/`new_str`, or `find`/`replace`).

Before writing code:
```bash
python -c "
from deepagents.backends import FilesystemBackend
b = FilesystemBackend(root_dir='/tmp', virtual_mode=True)
# Inspect the edit_file tool's schema
import inspect
# The tool is likely registered when the backend is attached to an agent.
# Easiest: build a minimal agent and introspect its tool list.
from deepagents import create_deep_agent
agent = create_deep_agent(model='ollama:qwen2.5-coder:7b', backend=b)
for t in agent.get_graph().nodes:
    pass
# Or: look at the tool schemas directly
import deepagents
print([x for x in dir(deepagents) if 'edit' in x.lower()])
"
```
Or check the installed `deepagents` package source for the `edit_file` tool definition. The wrapper below assumes `search`/`replace`; **adapt the arg names to match the installed version.**

Also confirm: does the built-in `edit_file` already do fuzzy matching? If so, our wrapper is purely a teaching layer (still valuable — the viewer sees the format and the fallback logic). If not, our wrapper adds real robustness.

---

## 8.3 · File specs

### Extend `codeit/tools/__init__.py` (~110 SLOC added)

```python
import difflib
from langchain.tools import tool
from codeit.paths import workspace_root, resolve_in_workspace

# ... existing read_summary, run_shell, register_custom_tools ...

def _read_file_text(path: str) -> str:
    """Read a file from the workspace as text. Returns '' on missing (caller distinguishes)."""
    try:
        return resolve_in_workspace(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def _write_file_text(path: str, content: str) -> None:
    """Write text back to a workspace file."""
    resolve_in_workspace(path).write_text(content, encoding="utf-8")

def _fuzzy_find(text: str, search: str, threshold: float = 0.90):
    """Find the substring of `text` most similar to `search`, above `threshold`.

    Returns (best_text, similarity) or (None, 0.0) if nothing scores high enough.
    Simple O(n*m) scan — fine for teaching; Aider uses a more sophisticated matcher.
    """
    if not search:
        return None, 0.0
    best, best_score = None, 0.0
    search_len = len(search)
    # Scan every window of search_len; for big files this is slow — cap the file size.
    for i in range(0, max(0, len(text) - search_len + 1)):
        window = text[i:i + search_len]
        score = difflib.SequenceMatcher(None, search, window).ratio()
        if score > best_score:
            best, best_score = window, score
            if best_score == 1.0:
                break  # exact match, done
    if best_score >= threshold:
        return best, best_score
    return None, best_score

@tool
def edit_file_safe(path: str, search: str, replace: str) -> str:
    """Edit a file by replacing the `search` text with `replace` text.

    Use this for targeted changes to existing files. Do NOT rewrite the whole file.
    Argument path: file relative to workspace root (e.g. 'main.py').
    Argument search: the exact text to find. Must match uniquely. Include enough context.
    Argument replace: the new text to substitute.

    On exact match: applies the edit. On near-miss (>=90% similar): applies with a note.
    On no match: returns a unified diff and a hint so you can retry with corrected search text.
    """
    text = _read_file_text(path)
    if text is None:
        return f"Error: {path} not found in workspace."
    if text.startswith("ERROR:"):
        return text  # propagate read error
    if search not in text:
        # Try fuzzy
        best, score = _fuzzy_find(text, search, threshold=0.90)
        if best is not None:
            new_text = text.replace(best, replace, 1)
            _write_file_text(path, new_text)
            return f"Applied fuzzy match (similarity {score:.2f}). Note: search was not exact; review the result."
        # No match — return a unified diff so the model sees the current state
        diff = difflib.unified_diff(
            text.splitlines(keepends=True),
            (text + "\n# --- proposed edit did not apply ---\n").splitlines(keepends=True),
            fromfile=f"{path} (current)",
            tofile=f"{path} (proposed, NOT applied)",
            n=3,
        )
        return (
            f"Could not find the search text in {path} (best fuzzy similarity {score:.2f}, "
            f"needed >=0.90). The file was NOT modified. Here is the current content as a diff — "
            f"re-read the file with read_file, then retry with corrected search text:\n\n"
            + "".join(diff)
        )
    # Exact match
    new_text = text.replace(search, replace, 1)
    _write_file_text(path, new_text)
    return f"Applied exact match edit to {path}."

CUSTOM_TOOLS = [read_summary, run_shell, edit_file_safe]

def register_custom_tools(extra: list | None = None) -> list:
    return CUSTOM_TOOLS + (extra or [])
```

**Teaching notes:**
- The docstring is **for the model** and explains the three outcomes (exact / fuzzy / fail-with-diff). The model reads this and learns the contract.
- `difflib.SequenceMatcher` is stdlib — no new dependency. We compute similarity as `ratio()` (0.0–1.0).
- `_fuzzy_find` is O(n*m) — fine for teaching, but **cap the file size** (e.g. skip files > 50k chars) or it's slow on big files. Add a guard: `if len(text) > 50_000: return "Error: file too large for fuzzy match; use read_file + write_file."`.
- On failure, we return the **current content as a unified diff** (with a fake "proposed" side) so the model can see what's actually in the file. This is the self-correction signal — the model reads the diff, re-issues `edit_file_safe` with corrected `search`.
- `text.replace(search, replace, 1)` — replace only the **first** occurrence. If `search` matches multiple times, the model should include more context. Document this in the docstring (done above: "Must match uniquely. Include enough context.").
- We go through `resolve_in_workspace` (Ep 4) so the sandbox applies, *not* through the built-in `edit_file`. The built-in is bypassed — `edit_file_safe` does its own read/modify/write. This is deliberate: we own the fuzzy logic. (Alternative: call the built-in `edit_file` for the exact-match path and only fall back to our own logic on miss. That's cleaner if the built-in does the same sandboxing. Decide during implementation based on whether the built-in's `edit_file` callable is accessible from our tool.)

### Update `codeit/agent.py` (~5 SLOC)

Add `edit_file_safe` to the default `interrupt_on`:

```python
DEFAULT_INTERRUPT_ON = {
    "run_shell": True,
    "write_file": True,
    "edit_file": True,
    "edit_file_safe": True,  # NEW: gate our custom wrapper too
    "delete": True,
}
```

---

## 8.4 · Tests

### `tests/test_ep08_edit.py`

```python
import pytest
from codeit.tools import edit_file_safe

def _setup_file(tmp_path, content):
    (tmp_path / "target.py").write_text(content)

def test_exact_match_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    _setup_file(tmp_path, "def hello():\n    return 'hi'\n")
    result = edit_file_safe.invoke({"path": "target.py", "search": "return 'hi'", "replace": "return 'hello'"})
    assert "Applied exact match" in result
    assert (tmp_path / "target.py").read_text() == "def hello():\n    return 'hello'\n"

def test_no_match_returns_diff_not_modified(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    _setup_file(tmp_path, "def hello():\n    return 'hi'\n")
    original = (tmp_path / "target.py").read_text()
    result = edit_file_safe.invoke({"path": "target.py", "search": "return 'nonexistent'", "replace": "x"})
    assert "Could not find" in result
    assert "NOT modified" in result
    assert (tmp_path / "target.py").read_text() == original  # unchanged

def test_fuzzy_match_applies_with_note(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    _setup_file(tmp_path, "def hello():\n    return 'hi'\n")
    # 'search' is close but not exact: extra space
    result = edit_file_safe.invoke({
        "path": "target.py",
        "search": "return  'hi'",   # two spaces instead of one
        "replace": "return 'hello'",
    })
    assert "fuzzy match" in result.lower()
    assert (tmp_path / "target.py").read_text() == "def hello():\n    return 'hello'\n"

def test_fuzzy_below_threshold_does_not_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    _setup_file(tmp_path, "def hello():\n    return 'hi'\n")
    # Very different search text — should not match
    result = edit_file_safe.invoke({
        "path": "target.py",
        "search": "completely different text that does not appear",
        "replace": "x",
    })
    assert "Could not find" in result
    assert (tmp_path / "target.py").read_text() == "def hello():\n    return 'hi'\n"

def test_missing_file_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    result = edit_file_safe.invoke({"path": "nope.py", "search": "x", "replace": "y"})
    assert "not found" in result.lower()

def test_first_occurrence_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    _setup_file(tmp_path, "x = 1\nx = 1\n")
    edit_file_safe.invoke({"path": "target.py", "search": "x = 1", "replace": "x = 2"})
    # Only the first occurrence should change
    assert (tmp_path / "target.py").read_text() == "x = 2\nx = 1\n"

def test_agent_edits_via_safe_wrapper(tmp_path, monkeypatch):
    """Drive the agent to call edit_file_safe; assert the file changed."""
    from langchain_core.messages import AIMessage
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from codeit.agent import build_agent, run
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    _setup_file(tmp_path, "def hello():\n    return 'hi'\n")
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "edit_file_safe", "args": {"path": "target.py", "search": "return 'hi'", "replace": "return 'hello'"}, "id": "c1", "type": "tool_call"}]),
        "Done.",
    ]))
    agent, _ = build_agent(model=fake, workdir=str(tmp_path))
    run(agent, "change 'hi' to 'hello' in target.py")
    assert (tmp_path / "target.py").read_text() == "def hello():\n    return 'hello'\n"
```

**Test notes:**
- The fuzzy threshold test (`test_fuzzy_match_applies_with_note`) uses a search that's ~90%+ similar. Tune the exact strings during implementation to land above/below the 0.90 threshold — `difflib.SequenceMatcher` is sensitive to whitespace and length.
- The `test_fuzzy_below_threshold_does_not_apply` test must use a search *genuinely* dissimilar. A short string compared to a long file can score low even when "obviously" related to a human. Use a search that's structurally different.
- The first-occurrence test (`test_first_occurrence_only`) enforces the `replace(..., 1)` contract. If the built-in `edit_file` (which we may call for the exact path) replaces all occurrences, our wrapper must still use `str.replace(..., 1)` — verify the behavior matches.

---

## 8.5 · Demo script `scripts/demo_ep08.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
# Set up a 200-line file
mkdir -p workspace
python -c "
lines = []
for i in range(200):
    lines.append(f'line {i}: some content here')
open('workspace/big.py', 'w').write('\n'.join(lines) + '\n')
"

echo "=== Demo: change 3 lines without rewriting the file ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
CODEIT_WORKDIR=./workspace \
python scripts/chat.py "In big.py, change line 50, line 100, and line 150 to say 'EDITED' instead of 'some content'. Use edit_file_safe, not write_file."

echo
echo "=== Demo: deliberately trigger a near-miss (self-correction) ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
CODEIT_WORKDIR=./workspace \
python scripts/chat.py "In big.py, change the text 'line 99: some content here' (note the typo — it's actually 'line 99') to say 'found it'. Use edit_file_safe."
echo
echo "Expected: agent tries the wrong search text, sees the diff, re-reads the file, and retries correctly."
```

On camera:
1. Show the agent editing 3 specific lines in a 200-line file *without rewriting it*.
2. Open `big.py` in the editor — only the 3 lines changed.
3. Trigger a near-miss: ask the agent to edit text that's slightly off. Show the fuzzy match applying with a note.
4. Trigger a total miss: show the diff coming back and the agent re-reading the file to self-correct.

---

## 8.6 · README section to add

```
## Ep 8 — Surgical Edits (edit_file + fuzzy fallback)
Adds: edit_file_safe in codeit/tools/__init__.py (wraps built-in edit_file with difflib fuzzy fallback + self-correction diff)
Run: CODEIT_WORKDIR=./workspace python scripts/chat.py "Change line 50 in big.py to say EDITED."
Key idea: built-in edit_file does search-replace; edit_file_safe hardens it for small models with fuzzy match + diff feedback.
Lines added: 140 SLOC (tightest episode — split into ep-08 + ep-08b if over budget)
Note: edit_file becomes the default edit path; write_file stays for new files.
```

---

## 8.7 · Definition of Done

- [ ] `edit_file_safe` tool in `codeit/tools/__init__.py`.
- [ ] `register_custom_tools` includes `edit_file_safe`.
- [ ] `DEFAULT_INTERRUPT_ON` includes `"edit_file_safe": True`.
- [ ] `pytest -m "not live" tests/test_ep08_edit.py` green (exact, fuzzy, no-match-diff, first-occurrence, missing-file).
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-07 HEAD -- '*.py'` adds < 220 lines. **If over, split into ep-08 + ep-08b.**
- [ ] Ep 1-7 demos still pass (Ep 4's `write_file` tests must still pass — `write_file` stays for new files).
- [ ] README "Ep 8" section written.
- [ ] Commit + annotated tag `ep-08` (or `ep-08` + `ep-08b`).

---

## 8.8 · What NOT to do this episode

- Do not reimplement `edit_file` from scratch. The built-in exists; we wrap it (or bypass it for the fuzzy path). Confirm the built-in's exact behavior during pre-flight.
- Do not let `_fuzzy_find` run on huge files. Cap the size and return an error asking the model to use `read_file` + targeted edits.
- Do not replace *all* occurrences of `search`. Use `str.replace(..., 1)` (first only). The model should include enough context for a unique match.
- Do not swallow the no-match case silently. The diff feedback is the self-correction signal — it's the whole point.
- Do not remove `write_file` from the toolset. It's still the right tool for new files. The system prompt (Ep 7) already says "use write_file only for new files; use edit_file for changes."
- Do not skip the approval gate on `edit_file_safe`. Edits mutate files; they gate.

---

## 8.9 · Common gotchas

- **Built-in `edit_file` arg names:** if the installed version uses `old_str`/`new_str` or `find`/`replace` instead of `search`/`replace`, our wrapper's docstring and tests must match. The *external* contract (what the model sees) is up to us; the *internal* call to the built-in must use its real names.
- **`difflib.SequenceMatcher` is O(n*m):** for a 200-line file (~4k chars) with a 30-char search, that's ~120k comparisons — fast. For a 10k-line file (~200k chars), it's 6M — slow. Cap file size at ~50k chars in `_fuzzy_find` and ask the model to scope down with `read_file` + `grep` first.
- **Whitespace sensitivity:** `difflib` counts whitespace in the ratio. A search with a trailing newline mismatch can drop below 0.90 even when "obviously" the right block. Consider normalizing whitespace before scoring, or lowering the threshold to 0.85 if you see too many false negatives during the demo.
- **The built-in may already be fuzzy:** some `edit_file` implementations do their own near-match. If so, our wrapper is redundant for the fuzzy path but still valuable as the teaching layer (the viewer sees the logic). Decide during pre-flight whether to call the built-in for the exact path or own the whole thing.
- **`text.replace(search, replace, 1)` with multi-line `search`:** works fine, but if `search` contains regex-special chars, that's irrelevant — `str.replace` is literal. Good.
- **File written outside the sandbox:** `edit_file_safe` uses `resolve_in_workspace` (Ep 4) for both read and write, so the sandbox applies. Don't accidentally use a raw `open(path)` — always go through the helper.