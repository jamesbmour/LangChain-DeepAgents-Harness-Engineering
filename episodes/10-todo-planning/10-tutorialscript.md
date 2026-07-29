# Episode 10 — Planning: TodoList & Task Decomposition (Tutorial Video Script)

## Overview
**Length:** ~12 minutes  
**Goal:** Show viewers how to give their agent the ability to plan multi-step tasks using Deep Agents' built-in `TodoListMiddleware`. The episode adds two custom tools (`plan` and `complete_todo`) that make planning explicit and teachable, plus a `render_todos` helper for CLI display.

---

## Scene 1: Hook & Why Planning Matters (0:00–1:30)

**On-screen:** Terminal showing an agent attempting a complex multi-step task without planning — it gets confused, forgets steps, and produces incomplete results. Then the same task with planning works perfectly.

> **Host:** "When you give an AI agent a simple one-line task like 'echo hello,' it's straightforward. But what about something more complex? Something that requires multiple coordinated steps?"
>
> *(Show failed attempt without planning)*
>
> "Without explicit planning, the model tries to hold everything in its context — and often forgets steps or gets confused about where it is in the process."

**Key points:**
- Complex tasks need structured planning.
- Without a todo list, models lose track of multi-step workflows.
- Deep Agents has built-in `TodoListMiddleware` that provides this automatically.

---

## Scene 2: TodoListMiddleware — The Built-In Battery (1:30–3:00)

**On-screen:** Diagram showing the middleware stack with TodoListMiddleware at position #1, and code showing how `write_todos` is automatically available.

> **Host:** "Deep Agents has a built-in `TodoListMiddleware` — it's always present as #1 in the default middleware stack. This means the `write_todos` tool is automatically available to our agent without any extra configuration."
>
> ```python
> # The write_todos tool is BUILT-IN — we don't add it ourselves.
> # It accepts a list of dicts: [{"content": "step 1", "status": "pending"}]
> ```

**Key points:**
- `TodoListMiddleware` is always active — no setup needed.
- The `write_todos` tool manages the todo list state.
- Todo items have three statuses: `pending`, `in_progress`, and `completed`.
- State lives in `result["todos"]` (or `todo_list` depending on version).

---

## Scene 3: The plan Tool — Making Planning Explicit (3:00–5:00)

**On-screen:** Code editor showing the `plan()` tool implementation, then a demo of it being called.

> **Host:** "While `write_todos` is built-in, we want to make planning EXPLICIT and teachable on camera. So we add a custom `plan` tool that bridges 'think about the plan' with 'seed the todo list.'"
>
> ```python
> @tool
> def plan(steps: list[str]) -> str:
>     """Plan a multi-step task by seeding the built-in todo list with these steps.
>     
>     Use this when the task needs more than one step. Each step becomes a pending todo.
>     After planning, execute the steps one by one, calling complete_todo(id) as you finish each.
>     Argument steps: a list of short step descriptions (e.g. ['read main.py', 'fix the bug', 'run tests']).
>     """
>     todo_lines = [f"{i+1}. {s}" for i, s in enumerate(steps)]
>     return ("Plan ready. Call write_todos with these items (all status='pending'):\n"
>             + "\n".join(todo_lines)
>             + "\n\nThen execute each step and call complete_todo(id) when done.")
> ```

**Key points:**
- `plan` returns a STRING instructing the model to call `write_todos`.
- Tools can't call other tools directly — the model orchestrates.
- This makes the planning step visible on camera for teaching purposes.

---

## Scene 4: The complete_todo Tool (5:00–6:30)

**On-screen:** Code showing `complete_todo()` and explaining how it works with `write_todos`.

> **Host:** "Once a step is done, we need to mark it as completed. Our `complete_todo` tool instructs the model to call `write_todos` again with the updated status."
>
> ```python
> @tool
> def complete_todo(id: int) -> str:
>     """Mark a todo item as completed by its 1-based index."""
>     return (f"Mark step {id} as completed by calling write_todos with the full list, "
>             f"setting item {id}'s status to 'completed'.")
> ```

**Key points:**
- `write_todos` REPLACES the entire list — it doesn't append.
- The model must pass back the full list with one item's status changed.
- This is more teachable than having the model figure out `write_todos` directly.

---

## Scene 5: render_todos Helper (6:30–7:30)

**On-screen:** Code showing `render_todos()` function and how it pretty-prints todos for CLI display.

> **Host:** "Finally, we add a helper — not a tool, but a utility for the CLI to display the todo list nicely."
>
> ```python
> def render_todos(state: dict) -> str:
>     """Pretty-print the todo list from agent state."""
>     todos = (state or {}).get("todos") or (state or {}).get("todo_list") or []
>     if not todos:
>         return "(no todos)"
>     marks = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}
>     lines = []
>     for i, t in enumerate(todos, 1):
>         status = t.get("status", "pending")
>         lines.append(f"  {marks.get(status, '[ ]')} {i}. {t.get('content', '')}")
>     return "\n".join(lines)
> ```

**Key points:**
- Checks both `todos` and `todo_list` keys for version compatibility.
- Uses Rich-style marks: `[x]` completed, `[>]` in progress, `[ ]` pending.
- This is a helper function — not registered as a tool since the model doesn't need to call it.

---

## Scene 6: System Prompt Priming (7:30–8:30)

**On-screen:** Code showing the system prompt that instructs the agent to plan first.

> **Host:** "The system prompt is key — it tells our agent when and how to use planning."
>
> ```python
> system_prompt=(
>     "You are CodeIt, a coding agent. For any task with more than one step, "
>     "call plan([...]) first to seed the todo list, then execute step by step, "
>     "calling complete_todo(id) as you finish each. "
>     "Use write_todos to update statuses — it REPLACES the list, not appends."
> ),
> ```

**Key points:**
- Explicitly instructs: "call plan([...]) first" for multi-step tasks.
- Reminds that `write_todos` replaces the list, not appends.
- This primes the model to think in terms of structured steps.

---

## Scene 7: Live Demo — Multi-Step Task with Planning (8:30–11:00)

**On-screen:** Terminal session showing a multi-step task being planned and executed step by step.

> **Host:** "Let's see this in action! We'll give the agent a multi-step task that requires planning."
>
> ```bash
> LLM_MODEL=qwen2.5-coder:32b CODEIT_WORKDIR=./workspace \
>     python 10-todo-planning.py --yolo "Plan a 4-step task: create main.py, write tests, run them, fix any failures"
> ```

**What viewers will see:**
1. Agent calls `plan()` with the steps — visible in the streaming output.
2. The todo list appears with all items marked as pending.
3. Agent executes each step, calling `complete_todo` after finishing.
4. Final state shows completed todos rendered by `render_todos`.

---

## Scene 8: Wrap-up & What's Next (11:00–12:00)

**On-screen:** Summary showing all planning capabilities and preview of Episode 11.

> **Host:** "In this episode, we've given our agent the ability to plan — a critical capability for any complex task."
>
> - **`plan(steps)`** — custom tool that makes planning explicit by instructing the model to call `write_todos`.
> - **`complete_todo(id)`** — marks steps as completed via `write_todos` with updated status.
> - **`render_todos(state)`** — CLI helper for pretty-printing the todo list.
> - **Built-in TodoListMiddleware** — always active, provides `write_todos` automatically.

**Key takeaways:**
1. Deep Agents' `TodoListMiddleware` is built-in and always present — no setup needed.
2. The `plan` tool makes planning explicit for teaching purposes by instructing the model to call `write_todos`.
3. `write_todos` REPLACES the list, not appends — the model must pass back the full list with updated statuses.
4. Todo state lives in `result["todos"]` (or `todo_list` depending on version).
5. Planning is unreliable on small models (≤8B) — use 32B or OpenAI for demos.