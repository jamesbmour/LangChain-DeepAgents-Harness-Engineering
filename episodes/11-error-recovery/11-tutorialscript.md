# Episode 11 — Error Recovery: Self-Healing Loops (Tutorial Video Script)

## Overview
**Length:** ~14 minutes  
**Goal:** Show viewers how to build a self-healing agent that writes code, runs tests, reads failures, and fixes them autonomously up to a retry cap. This is the "wow" episode where the agent demonstrates autonomous debugging.

---

## Scene 1: Hook & The Self-Healing Vision (0:00–1:30)

**On-screen:** Terminal showing an agent writing code, running tests that fail, then automatically fixing and re-running — all without human intervention.

> **Host:** "Imagine this: you give your coding agent a task like 'add a health endpoint to this API.' It writes the code, runs the tests... they fail. But instead of giving up or asking for help, it reads the failure output, identifies the bug, fixes it, and re-runs the tests — all on its own."
>
> "This is what we call a self-healing loop. And in this episode, we're going to build exactly that into CodeIt."

**Key points:**
- This is the "wow" episode — autonomous debugging without human intervention.
- The agent writes code → runs tests → reads failures → fixes bugs → re-runs tests.
- A retry cap prevents infinite loops if something can't be fixed.

---

## Scene 2: The run_tests Tool (1:30–4:00)

**On-screen:** Code editor showing the `run_tests` tool implementation, then a demo of it running pytest in a workspace with failing tests.

> **Host:** "The foundation of our recovery loop is a custom `run_tests` tool. It runs pytest in the workspace and returns a readable string — not an exception."
>
> ```python
> @tool
> def run_tests(path: str = ".") -> str:
>     """Run pytest in the workspace and return pass/fail summary + failures.
>     
>     Use this after editing code that has tests, to check your work. If tests fail,
>     read the failure output and fix the code, then run_tests again.
>     Argument path: test path relative to workspace (default '.' runs all tests).
>     """
>     try:
>         proc = subprocess.run(
>             ["pytest", path, "-q", "--tb=short", "--no-header"],
>             cwd=str(_workspace_root()), capture_output=True, text=True, timeout=180)
>     except subprocess.TimeoutExpired:
>         return "Error: pytest timed out after 180s."
>     except FileNotFoundError:
>         return "Error: pytest not installed. Run: pip install pytest"
>     except Exception as e:
>         return f"Error launching pytest: {type(e).__name__}: {e}"
>     
>     combined = f"$ pytest {path}\n[exit {proc.returncode}]\n{proc.stdout or ''}"
>     if proc.stderr:
>         combined += f"\n--- stderr ---\n{proc.stderr}\n"
>     if len(combined) > MAX_OUTPUT_CHARS:
>         combined = combined[:MAX_OUTPUT_CHARS] + f"\n... [truncated]"
>     return combined
> ```

**Key points:**
- Returns a STRING, not an exception — the model reads `[exit 1]` and failure text.
- `--tb=short` keeps tracebacks readable; `--no-header` removes pytest banner chrome.
- `MAX_OUTPUT_CHARS = 15_000` truncates long output to stay within token limits.
- The docstring primes the recovery loop: "read the failure, fix the code, then run_tests again."

---

## Scene 3: Failure Detection Heuristic (4:00–6:00)

**On-screen:** Code showing `_detect_test_failure()` function with explanation of the heuristic logic.

> **Host:** "Now for the brain of our recovery loop — a failure detection heuristic that scans the agent's final state for test failures."
>
> ```python
> def _detect_test_failure(state: dict) -> str | None:
>     """Inspect agent's final state for a run_tests result that reported failure."""
>     if not state: return None
>     for msg in reversed(state.get("messages", [])):
>         if getattr(msg, "type", "") != "tool": continue
>         content = getattr(msg, "content", "") or ""
>         # Only look at the most recent tool result
>         if "[exit 1]" in content and ("FAILED" in content or "Error" in content):
>             return content
>         # If we hit any tool result that's not a failure, stop scanning
>         if "[exit 0]" in content:
>             return None
>         return None  # First tool message found, not a failure
>     return None
> ```

**Key points:**
- Scans messages in reverse order — only checks the LAST `run_tests` result.
- Looks for `[exit 1]` combined with "FAILED" or "Error" in the output.
- If it finds `[exit 0]`, tests passed — no recovery needed.
- Returns the failure content so we can feed it back to the agent.

---

## Scene 4: The Recovery Loop (6:00–9:00)

**On-screen:** Code showing `run_with_recovery()` function, then a live demo of the full self-healing loop in action.

> **Host:** "Here's where the magic happens — our recovery loop wraps around the Deep Agents graph as plain Python code."
>
> ```python
> def run_with_recovery(agent, prompt: str, thread_id: str = "default", max_retries: int = 3) -> dict:
>     """Run agent; if run_tests reports failures, re-invoke with the failure appended.
>     
>     Re-invoke uses the SAME thread_id so history (code written, tests run) persists.
>     The follow-up is a USER message so the model treats it as new input.
>     """
>     state = run_with_approval(agent, prompt, thread_id)
>     for attempt in range(max_retries):
>         failure = _detect_test_failure(state)
>         if not failure:
>             return state  # Tests passed!
>         console.print(f"[yellow]Test failure detected (attempt {attempt+1}/{max_retries}). Re-invoking...[/yellow]")
>         followup = (f"The tests failed. Here is the output:\n\n{failure}\n\n"
>                     f"Read the failure, fix the code, then run_tests again.")
>         state = run_with_approval(agent, followup, thread_id)
>     console.print(f"[red]Recovery cap reached ({max_retries} retries). Stopping.[/red]")
>     return state
> ```

**Live demo:** Show the agent writing code with a bug, running tests that fail, then automatically fixing and re-running.

**Key points:**
- The loop is AROUND the graph — not inside it. We don't fight the framework.
- Same `thread_id` means history persists across retries (code written stays in context).
- Follow-up message is a USER message so the model treats it as new input, not system instructions.
- `max_retries=3` prevents infinite loops if something can't be fixed.

---

## Scene 5: System Prompt Priming (9:00–10:30)

**On-screen:** Code showing the agent's system prompt and explaining how it primes recovery behavior.

> **Host:** "The system prompt is crucial — it tells the model what to do when tests fail."
>
> ```python
> system_prompt=("You are CodeIt, a coding agent. After editing code with tests, call "
>                "run_tests. If tests fail, read the failure, fix the code, then run_tests again."),
> ```

**Key points:**
- The prompt explicitly instructs: "read the failure, fix the code, then run_tests again."
- This primes the model to self-correct rather than giving up or asking for help.
- Combined with our recovery loop, this creates a robust autonomous debugging workflow.

---

## Scene 6: Live Demo — Full Self-Healing Cycle (10:30–12:30)

**On-screen:** Terminal session showing the complete self-healing cycle from start to finish.

> **Host:** "Let's see the full self-healing cycle in action! We'll create a simple Python file with an intentional bug, write tests for it, and watch our agent fix everything autonomously."
>
> ```bash
> CODEIT_WORKDIR=./workspace python 11-error-recovery.py \
>     "Create a function that adds two numbers. Write a test for it. Run the tests." --yolo
> ```

**What viewers will see:**
1. Agent creates `add.py` with an intentional bug (e.g., returns subtraction instead of addition).
2. Agent writes `test_add.py`.
3. Agent calls `run_tests` — tests fail.
4. Recovery loop detects failure, re-invokes agent with failure output.
5. Agent reads the failure, fixes the bug in `add.py`, runs tests again.
6. Tests pass!

---

## Scene 7: Retry Cap & Failure Handling (12:30–13:30)

**On-screen:** Terminal showing what happens when recovery fails — retry cap message appears.

> **Host:** "What if the agent can't fix the problem? Our `max_retries` parameter prevents infinite loops."
>
> ```python
> console.print(f"[red]Recovery cap reached ({max_retries} retries). Stopping.[/red]")
> ```

**Key points:**
- The retry cap is configurable via `CODEIT_MAX_RETRIES` environment variable.
- When the cap is reached, the agent stops and returns its last state.
- This prevents the agent from spinning forever on unsolvable problems.

---

## Scene 8: Wrap-up & What's Next (13:30–14:00)

**On-screen:** Summary showing all error recovery capabilities and preview of Episode 12.

> **Host:** "In this episode, we've built a self-healing agent that can autonomously debug code:"
>
> - **`run_tests` tool** — runs pytest with readable output, returns strings not exceptions.
> - **`_detect_test_failure`** — heuristic scans for `[exit 1]` + failure indicators in the last test result.
> - **`run_with_recovery`** — Python loop around the graph that re-invokes on failure with same thread_id (history persists).
> - **System prompt priming** — explicitly instructs the model to read failures and fix code.

**Key takeaways:**
1. The recovery loop is implemented AROUND the graph, not inside it — we don't fight the framework.
2. Same `thread_id` across retries means history (code written) persists in context.
3. Follow-up messages are USER messages so the model treats them as new input.
4. Retry caps prevent infinite loops on unsolvable problems.
5. The system prompt must explicitly prime recovery behavior.