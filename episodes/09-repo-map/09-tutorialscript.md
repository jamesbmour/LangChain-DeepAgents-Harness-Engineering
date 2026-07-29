# Episode 9 — Context Management: Repo Map & Token Trimming (Tutorial Video Script)

## Overview
**Length:** ~13 minutes  
**Goal:** Show viewers how to give their agent a bird's-eye view of the codebase without reading every file, plus teach token budgeting concepts. The episode adds `build_repo_map` for codebase exploration and `estimate_tokens`/`trim_history` as teaching-layer context controls that complement Deep Agents' built-in context engineering.

---

## Scene 1: Hook & The Context Problem (0:00–1:30)

**On-screen:** Terminal showing an agent trying to understand a large codebase by reading file after file, with the context window growing rapidly.

> **Host:** "As codebases grow, our agent faces a critical problem: it can't read every file to understand the structure. If you have 50 Python files with hundreds of functions, asking 'where is X defined?' means either reading everything — which blows up your context window — or guessing."
>
> *(Show agent struggling without repo map)*

**Key points:**
- Large codebases need a bird's-eye view before diving into specifics.
- Reading every file wastes tokens and clutters context.
- We need a compact summary of the codebase structure.

---

## Scene 2: The build_repo_map Tool (1:30–4:30)

**On-screen:** Code editor showing `build_repo_map()` implementation with AST parsing, then a live demo running it on a sample workspace.

> **Host:** "Our solution is the `build_repo_map` tool — it walks the workspace using Python's `ast` module to extract top-level function and class signatures from each file."
>
> ```python
> SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", 
>              ".pytest_cache", ".ruff_cache", "workspace", ".codeit"}
> MAX_FILES = 200
> MAX_SIGS_PER_FILE = 30
> MAX_MAP_CHARS = 15_000    # ~4k tokens
> ```

**Then show the `_signatures` function:**
```python
def _signatures(path: Path) -> list[str]:
    """Extract top-level def/class signatures from a Python file via ast."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []  # unparseable — skip silently
    sigs: list[str] = []
    for node in tree.body:  # top-level only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            sigs.append(f"{kind} {node.name}({', '.join(args)})")
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body 
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            sigs.append(f"class {node.name}" + 
                        (f" ({', '.join(methods)})" if methods else ""))
        if len(sigs) >= MAX_SIGS_PER_FILE:
            sigs.append("... [truncated]"); break
    return sigs
```

**Key points:**
- Uses Python's built-in `ast` module — no external dependencies.
- Hardcoded skip list avoids needing `.gitignore` parsing (keeps it simple for teaching).
- Caps on files, signatures per file, and total characters prevent context bloat.
- Non-Python files are listed by name + line count as a fallback.

---

## Scene 3: Walking the Workspace (4:30–6:30)

**On-screen:** Code showing `build_repo_map()` walking directories with pruning, then output of running it on a sample project.

> **Host:** "The main function walks the workspace tree, prunes skip directories in-place, and builds a compact text map."
>
> ```python
> @tool
> def build_repo_map(root: str = ".") -> str:
>     """Return a compact map of the codebase: file paths + top-level signatures."""
>     base = (_workspace_root() / root).resolve() if root != "." else _workspace_root()
>     lines, count = [], 0
>     for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
>         # prune skip dirs in-place — critical for performance
>         dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
>         for fname in sorted(filenames):
>             ...
> ```

**Live demo:** Show the output of `build_repo_map` on a sample project:
```
main.py:
  def main()
  class App
utils.py:
  def helper_function(arg1, arg2)
  async def fetch_data(url)
models/user.py:
  class User
    (get_name, set_name, to_dict)
... [repo map truncated]
```

**Key points:**
- `dirnames[:] = ...` modifies in-place — this is how you prune during `os.walk`.
- The output gives the model a structural overview without reading every file.
- After seeing the repo map, the agent can use `read_file` on specific files it needs.

---

## Scene 4: Token Estimation & History Trimming (6:30–9:00)

**On-screen:** Code showing `estimate_tokens()` and `trim_history()`, then explanation of how they complement Deep Agents' built-in context engineering.

> **Host:** "Now for the teaching-layer context controls — `estimate_tokens` and `trim_history`. These are NOT replacements for Deep Agents' built-in context management, but rather a visible way to teach the concept."
>
> ```python
> def estimate_tokens(text: str) -> int:
>     """Rough token estimate: ~4 chars per token. Good for budgeting, not billing."""
>     return max(1, len(text) // 4)
> 
> def trim_history(messages: list, max_tokens: int) -> list:
>     """Drop oldest messages until total is under max_tokens. Keeps the newest turn."""
>     if not messages:
>         return []
>     def _tok(m) -> int:
>         return estimate_tokens(getattr(m, "content", str(m)) or "")
>     kept = list(messages)
>     while kept and sum(_tok(m) for m in kept) > max_tokens and len(kept) > 1:
>         kept.pop(0)  # drop oldest first
>     return kept
> ```

**Key points:**
- `estimate_tokens` uses a ~4 chars/token heuristic — good enough for budgeting.
- `trim_history` drops the OLDEST messages first, keeping recent context intact.
- Deep Agents' FilesystemMiddleware already does sophisticated context engineering internally (offloading, summarization). Our functions are a teaching layer that complements this.

---

## Scene 5: Live Demo — Using build_repo_map in Action (9:00–11:30)

**On-screen:** Terminal session showing the agent using `build_repo_map` to understand a codebase before making changes.

> **Host:** "Let's see how this works end-to-end! We'll give our agent a task that requires understanding an existing codebase."
>
> ```bash
> CODEIT_WORKDIR=./workspace python 09-repo-map.py \
>     --yolo "Where is the main function defined? Use build_repo_map first, then read only what you need."
> ```

**What viewers will see:**
1. Agent calls `build_repo_map` — gets a compact overview of all files and their signatures.
2. Agent identifies which file likely contains the target function from the map.
3. Agent uses `read_file` on just that specific file (not every file).
4. Agent provides an accurate answer based on targeted reading.

---

## Scene 6: The Aider Connection & What We Don't Do (11:30–12:30)

**On-screen:** Text comparing our simple repo map to Aider's PageRank-based symbol ranking, with a clear "out of scope" marker.

> **Host:** "Aider — the inspiration for this episode — ranks symbols by PageRank importance so you see the most relevant code first. We explicitly DON'T do that here."
>
> *(Show side-by-side comparison)*
>
> "Our repo map is simpler: it lists all files and their top-level signatures in alphabetical order. This keeps the implementation teachable at ~120 lines of Python. PageRank ranking would add significant complexity for marginal teaching value."

**Key points:**
- Aider's full feature set includes intelligent symbol ranking — we strip this out.
- Our version is a "good enough" bird's-eye view that teaches the core concept.
- The `MAX_MAP_CHARS` cap ensures the map stays within token budget regardless of codebase size.

---

## Scene 7: Wrap-up & What's Next (12:30–13:00)

**On-screen:** Summary showing all context management capabilities and preview of Episode 10.

> **Host:** "In this episode, we've given our agent the ability to understand codebases efficiently:"
>
> - **`build_repo_map`** — walks the workspace with `ast` parsing, returns compact file + signature map.
> - **`estimate_tokens`** — ~4 chars/token heuristic for budgeting context usage.
> - **`trim_history`** — drops oldest messages when over a token cap (teaching layer).

**Key takeaways:**
1. AST-based repo mapping gives the agent structural awareness without reading every file.
2. Hardcoded skip lists keep things simple — no `.gitignore` parsing dependency needed.
3. Caps on files, signatures, and total characters prevent context bloat.
4. `trim_history` is a teaching layer that complements Deep Agents' built-in context engineering — don't replace the framework's behavior.