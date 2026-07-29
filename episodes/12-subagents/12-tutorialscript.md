# Episode 12 — Sub-Agents: Specialists in Isolated Context (Tutorial Video Script)

## Overview
**Length:** ~14 minutes  
**Goal:** Show viewers how to give their agent the ability to delegate complex subtasks to specialized subagents that run with isolated context. The main agent stays focused on orchestration while subagents handle specific domains like codebase exploration or test running.

---

## Scene 1: Hook & The Delegation Problem (0:00–1:30)

**On-screen:** Terminal showing a long, complex prompt being processed by the agent with growing context window usage.

> **Host:** "As our agent grows more capable, it faces a fundamental scaling problem: every tool call, every file read, every test run adds to its context window. Eventually, the model gets overwhelmed and starts making worse decisions."
>
> "The solution? Delegation. Instead of having one super-agent try to do everything, we give our agent the ability to spawn specialized subagents — each with their own isolated context and tailored toolset."

**Key points:**
- Context window bloat is a real problem for coding agents.
- Subagents solve this by isolating subtasks in separate contexts.
- Only a summary returns from the subagent — not its full internal history.

---

## Scene 2: How Deep Agents Handles Subagents (1:30–3:00)

**On-screen:** Diagram showing parent agent → task tool → subagent with isolated context → summary back to parent.

> **Host:** "Deep Agents has built-in `SubAgentMiddleware` — it's #4 in the default middleware stack, so we don't need to add anything ourselves. The key is defining custom subagent specs."
>
> "Each spec defines a specialized agent with its own name, description, system prompt, and toolset. When the main agent calls the built-in `task` tool, it specifies which subagent to use and gives it instructions."

**Code snippet shown:**
```python
SUBAGENTS = [
    {
        "name": "explorer",
        "description": "Read-only codebase explorer...",
        "system_prompt": "You are a read-only codebase explorer...",
        "tools": [build_repo_map],  # plus shared backend tools (ls, read_file, grep)
    },
    {
        "name": "tester",
        "description": "Test-runner and fixer...",
        "system_prompt": "You are a test runner. Use run_tests; if tests fail...",
        "tools": [run_tests, edit_file_safe],  # no shell, no write_file
    },
]
```

**Key points:**
- SubAgentMiddleware is built-in — we just provide specs.
- The `task` tool is automatically available when subagents are configured.
- Each subagent gets its own isolated context and tailored toolset.

---

## Scene 3: Custom Tools for Subagents (3:00–5:00)

**On-screen:** Code editor showing the custom tools defined in episode 12 — `run_tests`, `edit_file_safe`, `build_repo_map`.

> **Host:** "Let's look at the three custom tools we've built that subagents will use. First, `run_tests` — it runs pytest in the workspace and returns results as a string."
>
> ```python
> @tool
> def run_tests(path: str = ".") -> str:
>     """Run pytest in the workspace. Use after editing code with tests."""
>     try:
>         proc = subprocess.run(["pytest", path, "-q", "--tb=short", "--no-header"],
>                               cwd=str(_workspace_root()), capture_output=True, text=True, timeout=180)
>     except Exception as e:
>         return f"Error: {type(e).__name__}: {e}"
>     return f"$ pytest {path}\n[exit {proc.returncode}]\n{proc.stdout or ''}{proc.stderr or ''}"
> ```

**Then show `edit_file_safe`:**
```python
@tool
def edit_file_safe(path: str, search: str, replace: str) -> str:
    """Edit a file by replacing `search` with `replace`. For existing files only."""
    try:
        fp = _workspace_root() / path
        text = fp.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: {path} not found."
    if search not in text:
        return f"Could not find search text in {path}. Re-read and retry."
    fp.write_text(text.replace(search, replace, 1), encoding="utf-8")
    return f"Applied exact match edit to {path}."
```

**Key points:**
- `run_tests` runs pytest with a timeout — returns exit code + output.
- `edit_file_safe` does exact string replacement on existing files only (no creation).
- These tools are designed specifically for the tester subagent's workflow.

---

## Scene 4: The spawn_subagent Tool (5:00–6:30)

**On-screen:** Code showing the `spawn_subagent` tool and explaining how it works as sugar over the built-in `task` tool.

> **Host:** "Here's a clever pattern — we define a custom `spawn_subagent` tool that returns instructions telling the model to call the built-in `task` tool with specific arguments."
>
> ```python
> @tool
> def spawn_subagent(task: str, agent: str = "general-purpose") -> str:
>     """Delegate a subtask to a specialized subagent and return its summary.
>     
>     Use this for large or isolatable subtasks to keep the main context clean.
>     Argument task: a COMPLETE, self-contained description (subagents are stateless).
>     Argument agent: 'explorer', 'tester', or 'general-purpose'.
>     """
>     return (f"To delegate to the '{agent}' subagent, call the task tool with: "
>             f"agent='{agent}', instruction='''{task}'''.")
> ```

**Key points:**
- Tools can't call other tools directly — the model orchestrates.
- `spawn_subagent` is ergonomic sugar that guides the model to use the right subagent.
- Subagents are **stateless** — give complete instructions in one call.

---

## Scene 5: The Explorer Subagent (6:30–8:30)

**On-screen:** Terminal showing a demo of using the explorer subagent to map a codebase.

> **Host:** "Let's see the explorer subagent in action. This is a read-only agent that can use `ls`, `read_file`, `grep` (from the shared backend), plus our custom `build_repo_map` tool."
>
> ```bash
> LLM_MODEL=qwen2.5-coder:32b CODEIT_WORKDIR=./my_project \
>     python 12-subagents.py "Use the explorer subagent to map the codebase." --yolo
> ```

**What viewers will see:**
- The main agent calls `spawn_subagent` with `agent="explorer"`.
- A new isolated context is created for the explorer.
- The explorer uses its tools to analyze the codebase structure.
- Only a summary returns to the parent — no message history leaks.

**Key points:**
- Explorer has NO shell access and NO write capabilities.
- It's perfect for "where is X defined?" or "map this codebase" tasks.
- The isolation keeps the main agent's context clean.

---

## Scene 6: The Tester Subagent (8:30–10:30)

**On-screen:** Terminal showing a demo of using the tester subagent to fix failing tests.

> **Host:** "Now let's look at the tester subagent — it can run tests and edit existing files, but it cannot create new files or run arbitrary shell commands."
>
> ```bash
> python 12-subagents.py "Use the tester subagent to make all tests pass" --yolo
> ```

**What viewers will see:**
- The main agent delegates test-running and fixing to the tester.
- Tester runs `run_tests`, reads failures, uses `edit_file_safe` to fix code.
- Only a summary of what was fixed returns to the parent.

**Key points:**
- Tester has NO shell access (can't run arbitrary commands).
- Tester can only edit existing files — no new file creation.
- This prevents subagents from going rogue and making unintended changes.

---

## Scene 7: Isolation & Context Management (10:30–12:00)

**On-screen:** Diagram showing parent context vs. subagent isolated contexts, with only summaries crossing the boundary.

> **Host:** "The key architectural insight is isolation. When a subagent runs:"
>
> 1. It gets its own fresh context — no access to the parent's message history.
> 2. Only a summary of what it did returns to the parent as a single `ToolMessage`.
> 3. The parent never sees the subagent's internal reasoning or tool calls.

**Key points:**
- This prevents context bloat from complex subtasks.
- Subagents can't accidentally leak information back that confuses the parent.
- Each subagent is purpose-built with only the tools it needs for its domain.

---

## Scene 8: Model Size Considerations (12:00–13:00)

**On-screen:** Text highlighting "Model size: delegation needs a larger model — 32b or OpenAI."

> **Host:** "One critical note: subagent delegation requires a sufficiently capable model. Small models like 7B parameters struggle with the meta-reasoning needed to decide when and how to delegate."
>
> "For this episode's demos, we recommend using `qwen2.5-coder:32b` or an OpenAI model. The agent needs to understand not just the task itself, but also when it should hand off work to a specialist."

**Key points:**
- 7B models can run the code but may not delegate effectively.
- 32B+ models handle delegation decisions much better.
- This is about reasoning quality, not just raw capability.

---

## Scene 9: Wrap-up & What's Next (13:00–14:00)

**On-screen:** Summary showing all subagent capabilities and preview of Episode 13.

> **Host:** "In this episode, we've given our agent the ability to delegate — one of the most powerful patterns in multi-agent systems."
>
> "We defined two specialized subagents:"
> - **Explorer** — read-only codebase analysis with `build_repo_map` + shared filesystem tools.
> - **Tester** — test running and fixing with `run_tests` + `edit_file_safe`.
>
> "And we built a `spawn_subagent` tool that makes delegation ergonomic for the model."

**Key takeaways:**
1. SubAgentMiddleware is built-in to Deep Agents — just provide specs.
2. Each subagent gets isolated context with only tailored tools.
3. Only summaries cross the boundary — no message history leaks.
4. Subagents are stateless — give complete instructions in one call.
5. Larger models (32B+) handle delegation decisions much better than small ones.