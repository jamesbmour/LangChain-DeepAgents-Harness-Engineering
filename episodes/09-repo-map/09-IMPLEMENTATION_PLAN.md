# Episode 9 — Context Management: Repo Map & Token Trimming

**Tag:** `ep-09` · **Shape:** add a new tool + customize · **Budget:** ~120 SLOC (may split into `ep-09` + `ep-09b`)
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 9

---

## 9.1 · What this episode delivers

The agent gains a bird's-eye view of the codebase without reading every file. We add:

1. **`build_repo_map(root) -> str`** — a custom `@tool` that walks the workspace (respecting `.gitignore`), uses Python's `ast` module to extract top-level `def`/`class` signatures from each `.py` file, and returns a compact text map. This is the Aider repo-tree idea, stripped to its teachable essence (no PageRank ranking — that's flagged as out-of-scope advanced work).
2. **`estimate_tokens(text) -> int`** — a ~4-chars/token heuristic for budgeting.
3. **`trim_history(messages, max_tokens) -> list`** — drop/compact oldest turns to fit a token cap. *Note:* Deep Agents' `FilesystemMiddleware` already does context engineering internally (offloading, summarization — see the Deep Agents context-engineering doc). This episode teaches the *concept* and gives the viewer a visible trim function for custom control, complementing the built-in behavior.

**Pre-split flag:** if repo map + token trim push past 220 SLOC, split into `ep-09` (repo map) and `ep-09b` (token estimate + trim). Parent plan §11.

---

## 9.2 · Pre-flight

1. Confirm the Deep Agents context-engineering story so we don't reinvent it. Read `/oss/python/deepagents/context-engineering.mdx` — specifically the "offloading" and "summarization" sections. Our `trim_history` is a *teaching* layer on top of, not a replacement for, the harness's built-in context management.
2. Confirm `ast` parsing works for the target language (Python). For non-Python files, the repo map falls back to listing the filename + a line count. We scope Ep 9 to Python; multi-language is out of scope.
3. Confirm `.gitignore` parsing: use `pathspec` (a stdlib-adjacent lib) or a simple line-by-line parser. `pathspec` is a common dependency; if not already installed, add it to `pyproject.toml` (it's tiny). Alternatively, skip `.gitignore` entirely for teaching and just hardcode skip dirs (`__pycache__`, `.git`, `node_modules`, `.venv`, `workspace`). The simpler path wins for a teaching codebase — **prefer the hardcoded skip-list** to avoid a new dep.
4. Decide: does `build_repo_map` return the map as a string the model reads, or write it to a file (e.g. `workspace/.codeit/repomap.txt`) the model can `read_file`? Returning the string is simpler and avoids a write. Go with that.

---

## 9.3 · File specs

### `codeit/repomap.py` (~110 SLOC)

```python
import ast
import os
from pathlib import Path
from langchain.tools import tool
from codeit.paths import workspace_root

SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".pytest_cache", ".ruff_cache", "workspace", ".codeit"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
MAX_FILES = 200           # cap to keep the map small
MAX_SIGS_PER_FILE = 30   # don't dump 1000 funcs from one huge file
MAX_MAP_CHARS = 15_000    # ~4k tokens; the model can grep for details

def _signatures(path: Path) -> list[str]:
    """Extract top-level def/class signatures from a Python file via ast."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []  # unparseable — skip silently
    sigs: list[str] = []
    for node in tree.body:  # top-level only
        if isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            sigs.append(f"def {node.name}({', '.join(args)})")
        elif isinstance(node, ast.AsyncFunctionDef):
            args = [a.arg for a in node.args.args]
            sigs.append(f"async def {node.name}({', '.join(args)})")
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            sigs.append(f"class {node.name}" + (f"({', '.join(methods)})" if methods else ""))
        if len(sigs) >= MAX_SIGS_PER_FILE:
            sigs.append("... [truncated]")
            break
    return sigs

@tool
def build_repo_map(root: str = ".") -> str:
    """Return a compact map of the codebase: file paths + top-level def/class signatures.

    Use this to answer 'where is X defined?' across multiple files without reading them all.
    Argument root: subdirectory to map, relative to workspace (default '.').
    Python files get signatures; other files are listed by name + line count.
    Respects a hardcoded skip list (__pycache__, .git, .venv, etc.).
    """
    base = (workspace_root() / root).resolve() if root != "." else workspace_root()
    lines: list[str] = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(base):
        # prune skip dirs in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in sorted(filenames):
            if fname.endswith(tuple(SKIP_SUFFIXES)):
                continue
            fpath = Path(dirpath) / fname
            rel = fpath.relative_to(workspace_root())
            if fpath.suffix == ".py":
                sigs = _signatures(fpath)
                if sigs:
                    lines.append(f"{rel}:")
                    for s in sigs:
                        lines.append(f"  {s}")
                else:
                    lines.append(f"{rel}: (no top-level def/class)")
            else:
                try:
                    n = sum(1 for _ in fpath.open(encoding="utf-8", errors="ignore"))
                    lines.append(f"{rel}: ({n} lines)")
                except Exception:
                    lines.append(f"{rel}:")
            count += 1
            if count >= MAX_FILES:
                lines.append("... [repo map truncated, too many files]")
                break
        if count >= MAX_FILES:
            break
    map_text = "\n".join(lines)
    if len(map_text) > MAX_MAP_CHARS:
        map_text = map_text[:MAX_MAP_CHARS] + "\n... [repo map truncated]"
    return map_text or "(empty workspace)"

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token. Good enough for budgeting, not billing."""
    return max(1, len(text) // 4)

def trim_history(messages: list, max_tokens: int) -> list:
    """Drop oldest messages until the total is under max_tokens. Keeps the newest turn.

    Note: Deep Agents' FilesystemMiddleware already does context engineering (offloading
    large tool results, summarizing old turns). This function is a teaching layer for
    custom control — use it when you want explicit trimming the harness doesn't do.
    """
    if not messages:
        return []
    # Compute per-message token cost
    def _tok(m) -> int:
        content = getattr(m, "content", str(m)) or ""
        return estimate_tokens(content)
    total = sum(_tok(m) for m in messages)
    if total <= max_tokens:
        return list(messages)
    # Keep the newest message always; trim from the front
    kept = list(messages)
    while kept and sum(_tok(m) for m in kept) > max_tokens and len(kept) > 1:
        kept.pop(0)
    return kept
```

**Teaching notes:**
- `ast.parse` + `tree.body` (top-level only) is the simplest version. Aider's repo map uses PageRank to rank symbols by importance — explicitly out of scope. Say this on camera: "Aider ranks by importance; we just list. That's enough for a 200-file project."
- The skip-list (`SKIP_DIRS`) is hardcoded, not parsed from `.gitignore`. This avoids a new dep (`pathspec`) and keeps the teaching code small. Mention `.gitignore` parsing as the production approach.
- `MAX_FILES`, `MAX_SIGS_PER_FILE`, `MAX_MAP_CHARS` caps keep the map from eating the context window. The model reads the map, then uses `grep`/`read_file` for details.
- `estimate_tokens` is a heuristic (~4 chars/token). Good for budgeting; not for billing. State this.
- `trim_history` keeps the **newest** message always (so the agent doesn't lose the current user request) and trims from the front. This is the simplest defensible policy. Mention that Deep Agents' built-in summarization is more sophisticated.
- The docstring is for the model: "Use this to answer 'where is X defined?'"

### Update `codeit/tools/__init__.py` (~5 SLOC)

```python
from codeit.repomap import build_repo_map, estimate_tokens, trim_history

CUSTOM_TOOLS = [read_summary, run_shell, edit_file_safe, build_repo_map]

def register_custom_tools(extra: list | None = None) -> list:
    return CUSTOM_TOOLS + (extra or [])
```

`estimate_tokens` and `trim_history` are plain functions, not `@tool` — they're used internally by Ep 11's recovery loop and by tests, not by the model directly.

---

## 9.4 · Tests

### `tests/test_ep09_repomap.py`

```python
import os
from pathlib import Path
from codeit.repomap import build_repo_map, estimate_tokens, trim_history, _signatures

def _make_project(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text(
        "def hello(name):\n    return f'hi {name}'\n\nclass App:\n    def run(self):\n        pass\n"
    )
    (tmp_path / "app" / "utils.py").write_text("def helper(x):\n    return x * 2\n")
    (tmp_path / "README.md").write_text("# project\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_text("garbage")

def test_signatures_extracted(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    _make_project(tmp_path)
    sigs = _signatures(tmp_path / "app" / "main.py")
    assert any("def hello" in s for s in sigs)
    assert any("class App" in s for s in sigs)

def test_build_repo_map_lists_files(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    _make_project(tmp_path)
    result = build_repo_map.invoke({"root": "."})
    assert "app/main.py" in result
    assert "def hello" in result
    assert "class App" in result
    assert "app/utils.py" in result
    assert "README.md" in result

def test_build_repo_map_skips_pycache(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    _make_project(tmp_path)
    result = build_repo_map.invoke({"root": "."})
    assert "__pycache__" not in result
    assert "junk.pyc" not in result

def test_build_repo_map_empty_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    result = build_repo_map.invoke({"root": "."})
    assert "empty" in result.lower()

def test_build_repo_map_truncates_huge(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    # Create 300 files
    (tmp_path / "many").mkdir()
    for i in range(300):
        (tmp_path / "many" / f"f{i}.py").write_text(f"def f{i}():\n    pass\n")
    result = build_repo_map.invoke({"root": "."})
    assert "truncated" in result.lower()

def test_estimate_tokens():
    assert estimate_tokens("hello world") == 3   # 11 chars -> 2.75 -> 3
    assert estimate_tokens("") == 1               # min 1
    assert estimate_tokens("a" * 100) == 25

def test_trim_history_keeps_newest_when_over_cap():
    from langchain_core.messages import HumanMessage, AIMessage
    msgs = [HumanMessage(content="x" * 1000), AIMessage(content="y" * 1000), HumanMessage(content="keep me")]
    trimmed = trim_history(msgs, max_tokens=50)  # tiny cap
    assert len(trimmed) <= len(msgs)
    assert "keep me" in trimmed[-1].content  # newest preserved

def test_trim_history_no_op_when_under_cap():
    from langchain_core.messages import HumanMessage
    msgs = [HumanMessage(content="short")]
    assert trim_history(msgs, max_tokens=1000) == msgs

def test_trim_history_empty():
    assert trim_history([], max_tokens=100) == []

def test_agent_uses_repo_map(tmp_path, monkeypatch):
    """Drive the agent to call build_repo_map; assert it ran."""
    from langchain_core.messages import AIMessage
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from codeit.agent import build_agent, run
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    _make_project(tmp_path)
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "build_repo_map", "args": {"root": "."}, "id": "c1", "type": "tool_call"}]),
        "The project has a hello function in app/main.py.",
    ]))
    agent, _ = build_agent(model=fake, workdir=str(tmp_path))
    state = run(agent, "where is hello defined?")
    tool_msgs = [m for m in state["messages"] if m.type == "tool"] if state else []
    assert any("def hello" in m.content for m in tool_msgs)
```

---

## 9.5 · Demo script `scripts/demo_ep09.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "=== Demo: agent maps the codebase without reading every file ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
CODEIT_WORKDIR=./examples/sample_app \
python scripts/chat.py "Where is the Todo model defined? Don't read every file — use build_repo_map first."
echo
echo "Expected: agent calls build_repo_map, sees the map, then reads only the file that has the model."
```

On camera: emphasize the contrast with Ep 3 (where the agent called `ls` + `read_file` repeatedly). Here it calls `build_repo_map` once and knows where to look.

---

## 9.6 · README section to add

```
## Ep 9 — Repo Map + Token Trimming
Adds: codeit/repomap.py (build_repo_map tool, estimate_tokens, trim_history)
Run: CODEIT_WORKDIR=./examples/sample_app python scripts/chat.py "Where is X defined?"
Key idea: ast-based repo map (no PageRank — that's advanced); 4-chars/token heuristic; front-trim keeping newest.
Note: Deep Agents' FilesystemMiddleware already does context engineering (offloading/summarization). This is a teaching layer.
Lines added: 120 SLOC (split into ep-09 + ep-09b if over budget)
```

---

## 9.7 · Definition of Done

- [ ] `codeit/repomap.py` with `build_repo_map` tool, `estimate_tokens`, `trim_history`.
- [ ] `register_custom_tools` includes `build_repo_map`.
- [ ] `pytest -m "not live" tests/test_ep09_repomap.py` green.
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-08 HEAD -- '*.py'` adds < 220 lines. **If over, split into ep-09 + ep-09b.**
- [ ] Ep 1-8 demos still pass.
- [ ] README "Ep 9" section written.
- [ ] Commit + annotated tag `ep-09` (or `ep-09` + `ep-09b`).

---

## 9.8 · What NOT to do this episode

- Do not implement PageRank ranking. Aider does this; we explicitly don't. State the limitation on camera.
- Do not parse `.gitignore` with a new dependency. Use the hardcoded `SKIP_DIRS`. Parsing `.gitignore` correctly is a rabbit hole.
- Do not let `build_repo_map` return a 50k-char map. The caps (`MAX_FILES`, `MAX_MAP_CHARS`) exist to protect the context window.
- Do not make `trim_history` drop the newest message. The current user request must survive.
- Do not advertise `trim_history` as a replacement for the harness's context engineering. It's a teaching complement.
- Do not parse non-Python files with `ast`. List them by name + line count.

---

## 9.9 · Common gotchas

- **`ast.parse` on syntactically-invalid files:** raises `SyntaxError`. Catch it and return `[]` — don't crash the tool.
- **Encoding errors:** use `encoding="utf-8", errors="ignore"` when counting lines for non-Python files. A binary file opened as text will otherwise raise.
- **`os.walk` ordering:** not guaranteed sorted. We `sorted(filenames)` for determinism; dirs are pruned in-place via `dirnames[:] =`.
- **Symlinks in the workspace:** `os.walk` follows them by default and can loop. Add `followlinks=False` (the default) — confirm in the implementation.
- **`estimate_tokens` is a heuristic:** viewers may ask "why not use the model's tokenizer?" Answer: it's a dep, and ~4 chars/token is close enough for budgeting. The real tokenizer matters for billing, not for "is this too big?"
- **`trim_history` and Deep Agents' built-in offloading:** if both run, you might double-trim. Our `trim_history` is for *custom* use (e.g. inside Ep 11's recovery loop); the harness still does its own thing for normal runs. Don't call `trim_history` inside the agent's normal flow — only in custom drivers.