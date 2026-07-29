# Episode 3 — Giving Your Agent Hands: Filesystem Tools (Tutorial Video Script)

## Overview
**Length:** ~12 minutes  
**Goal:** Show viewers how to give their agent read-only access to the filesystem using Deep Agents' built-in `FilesystemBackend`. The episode demonstrates that `ls`, `read_file`, `glob`, and `grep` appear automatically with zero hand-written code, plus adds a custom `read_summary` tool for truncated file reading.

---

## Scene 1: Hook & Why Filesystem Access Matters (0:00–1:30)

**On-screen:** Terminal showing an agent that can only respond to prompts without any tools — it's just a chatbot with no ability to interact with code. Then with filesystem tools, it becomes a real coding assistant.

> **Host:** "In Episodes 1 and 2, our agent was essentially a smart chatbot — it could reason about tasks but couldn't actually look at or touch any files."
>
> *(Show agent without tools being useless for code tasks)*

**Key points:**
- An agent without filesystem access can only respond from training data.
- Real coding requires reading existing files, understanding structure, and finding relevant code.
- Deep Agents' `FilesystemBackend` provides these capabilities automatically — no custom tool code needed.

---

## Scene 2: The FilesystemBackend Configuration (1:30–4:00)

**On-screen:** Code editor showing the backend configuration with detailed explanation of each parameter, then a diagram showing how tools appear from the backend.

> **Host:** "The key is `FilesystemBackend` — we attach it to our agent and four read-only filesystem tools appear automatically."
>
> ```python
> # virtual_mode=True blocks ../, ~, and absolute paths outside root.
> # NEVER set virtual_mode=False with a real root_dir. It is insecure by default.
> backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
> 
> return create_deep_agent(
>     model=get_model(),
>     tools=[read_summary],  # our custom tool alongside the built-ins
>     system_prompt="You are CodeIt, a helpful coding assistant. Explore the workspace with ls and read_file.",
>     backend=backend,  # ← ls/read_file/glob/grep appear automatically
>     checkpointer=MemorySaver(),
> )
> ```

**Key points:**
- `virtual_mode=True` is THE sandbox — it blocks path traversal attacks.
- The default `virtual_mode=False` provides NO security even with `root_dir` set. **Always pass `virtual_mode=True`.**
- Built-in tools: `ls`, `read_file`, `glob`, `grep` (read-only this episode).

---

## Scene 3: Zero Hand-Written Tool Code (4:00–5:30)

**On-screen:** Text showing the contrast — "We write ZERO code for ls, read_file, glob, grep. The framework provides them." Then a demo of each tool being used by the agent.

> **Host:** "This is what I call 'the framework does the work' episode. We don't implement `ls`, `read_file`, `glob`, or `grep` — they come from the backend automatically when we set it."
>
> *(Show agent using built-in tools)*
> - Agent calls `ls(path=".")` → lists workspace contents
> - Agent calls `read_file(file_path="main.py")` → reads file content
> - Agent calls `grep(pattern="def ", path=".")` → searches for function definitions

**Key points:**
- These are built-in tools — we just configure the backend and they appear.
- The agent discovers them through its tool list, same as custom tools.
- This is a major productivity boost — no need to implement common operations.

---

## Scene 4: The Custom read_summary Tool (5:30–8:00)

**On-screen:** Code showing `read_summary()` implementation and explanation of how it complements the built-in `read_file`.

> **Host:** "While we don't need to implement filesystem tools, we DO add one custom tool — `read_summary` — that wraps the idea of reading a file with line truncation."
>
> ```python
> @tool
> def read_summary(path: str) -> str:
>     """Read a file and return its first 50 lines plus a truncation note.
>     
>     Use this when you want a quick overview of a file without reading the whole thing.
>     Argument: path relative to the workspace root (e.g. 'main.py' or 'src/app.py').
>     """
>     root = Path(os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
>     try:
>         text = (root / path).read_text(encoding="utf-8")
>     except FileNotFoundError:
>         return f"Error: {path} not found in workspace."
>     except Exception as e:
>         return f"Error reading {path}: {type(e).__name__}: {e}"
>     
>     lines = text.splitlines()
>     if len(lines) <= 50:
>         return text
>     return "\n".join(lines[:50]) + f"\n... [truncated, {len(lines) - 50} more lines]"
> ```

**Key points:**
- The docstring is written **for the model**: "Use this when you want a quick overview…" — that's the schema description.
- Tool docstrings are prompts — they tell the model WHEN and HOW to use each tool.
- `read_summary` complements (not replaces) the built-in `read_file`.

---

## Scene 5: Live Demo — Exploring a Project (8:00–10:30)

**On-screen:** Terminal session showing the agent exploring a sample project using filesystem tools, then reading specific files.

> **Host:** "Let's see this in action! We'll point our agent at a small project and ask it to explore."
>
> ```bash
> CODEIT_WORKDIR=./workspace python 03-filesystem-tools.py \
>     --yolo "What files are in this project? What does the main file do?"
> ```

**What viewers will see:**
1. Agent calls `ls(path=".")` — lists workspace contents.
2. Agent identifies Python files and uses `read_file` or `read_summary` to examine them.
3. Agent provides a summary of what it found in the project structure.

---

## Scene 6: The Sandbox Security Story (10:30–11:30)

**On-screen:** Text explaining virtual_mode=True security guarantees and limitations, with code examples of blocked path traversal attempts.

> **Host:** "Let's be crystal clear about what `virtual_mode=True` protects against."
>
> *(Show protection matrix)*
> - ✅ Blocks `../etc/passwd` — path traversal outside workspace
> - ✅ Blocks `/etc/shadow` — absolute paths outside root
> - ✅ Blocks `~/secret.txt` — home directory expansion
> - ❌ Does NOT prevent destructive shell commands (Episode 5)
> - ❌ Does NOT prevent `rm -rf` inside the workspace

**Key points:**
- The sandbox is about path confinement, not command safety.
- Episode 6 adds approval gates for dangerous operations.
- Always use `virtual_mode=True` — the default provides no security.

---

## Scene 7: Wrap-up & What's Next (11:30–12:00)

**On-screen:** Summary showing all filesystem tool capabilities and preview of Episode 4.

> **Host:** "In this episode, we've given our agent the ability to explore the filesystem — read-only access that lets it understand any codebase."
>
> - **`FilesystemBackend(virtual_mode=True)`** — provides `ls`, `read_file`, `glob`, `grep` automatically (zero hand-written code).
> - **`read_summary`** — custom tool with line truncation for quick file overviews.

**Key takeaways:**
1. FilesystemBackend provides built-in tools automatically — no need to implement common operations.
2. `virtual_mode=True` is critical — the default (`False`) provides NO security even with root_dir set.
3. Tool docstrings are prompts — write them FOR THE MODEL, explaining when and how to use each tool.
4. The sandbox protects against path traversal but NOT destructive commands — that's Episode 6.