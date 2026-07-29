# Episode 6 — Permission Gating: Human-in-the-Loop (Tutorial Video Script)

## Overview
**Length:** ~14 minutes  
**Goal:** Show viewers how to make their agent safe by adding human approval gates for destructive actions. The episode uses Deep Agents' built-in `HumanInTheLoopMiddleware` via `interrupt_on` and `MemorySaver`, plus a regex-based risk classifier that labels commands as safe, needs-approval, or blocked before prompting the user.

---

## Scene 1: Hook & Why Safety Matters (0:00–2:00)

**On-screen:** Terminal showing an agent running destructive commands without asking — `rm -rf`, `git push --force` — causing damage to a project. Then the same scenario with approval gates prevents disaster.

> **Host:** "Imagine you're working on an important project, and your coding agent decides to run `rm -rf node_modules` or `git push --force`. Without any safeguards, it could destroy hours of work in seconds."
>
> *(Show destructive command being blocked by approval gate)*

**Key points:**
- An autonomous agent without safety checks is dangerous.
- Destructive commands need human approval before execution.
- Deep Agents provides `HumanInTheLoopMiddleware` — we just configure it.

---

## Scene 2: The interrupt_on Configuration (2:00–4:00)

**On-screen:** Code showing the `INTERRUPT_ON` constant and how it's passed to `create_deep_agent`.

> **Host:** "The key is the `interrupt_on` parameter — a dictionary that tells Deep Agents which tools should pause for approval before executing."
>
> ```python
> # Our policy: gate shell commands + file writes/deletes.
> INTERRUPT_ON = {"run_shell": True, "write_file": True, 
>                 "edit_file": True, "delete": True}
> 
> def build_agent(workdir: str | None = None):
>     root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
>     root.mkdir(parents=True, exist_ok=True)
>     return create_deep_agent(
>         model=get_model(),
>         tools=[run_shell],
>         system_prompt="You are CodeIt, a coding agent. The user approves destructive actions.",
>         backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
>         interrupt_on=INTERRUPT_ON,           # ← pause before these tools
>         checkpointer=MemorySaver(),          # ← REQUIRED for interrupts to work
>     )
> ```

**Key points:**
- `interrupt_on` is a dict mapping tool names to booleans.
- When the model calls a gated tool, the graph PAUSES instead of executing immediately.
- `MemorySaver()` checkpointer is **REQUIRED** — without it, there's nowhere for interrupts to resume from.

---

## Scene 3: The classify() Risk Triage (4:00–6:30)

**On-screen:** Code showing the `classify()` function with safe and destructive pattern lists, then a demo of classifying different commands.

> **Host:** "Before we prompt the user, we want to give them context about what kind of command this is. Our `classify` function uses regex patterns to label commands as 'safe', 'needs-approval', or 'blocked'."
>
> ```python
> SAFE_PATTERNS = [r"^\s*ls\b", r"^\s*cat\b", r"^\s*pwd\b", r"^\s*echo\b",
>                  r"^\s*git status\b", r"^\s*git diff\b", r"^\s*pytest\b"]
> DESTRUCTIVE_PATTERNS = [r"rm\s+-rf?\b", r"git\s+push\s+.*-f", 
>                         r"git\s+reset\s+--hard", r"\bdd\b", r"\bmkfs\b",
>                         r"curl.*\|\s*sh", r"wget.*\|\s*sh"]
> 
> def classify(command: str) -> str:
>     """Return 'safe', 'needs-approval', or 'blocked' for a shell command."""
>     for pat in DESTRUCTIVE_PATTERNS:
>         if re.search(pat, command):
>             return "blocked"
>     for pat in SAFE_PATTERNS:
>         if re.search(pat, command):
>             return "safe"
>     return "needs-approval"
> ```

**Live demo:** Show classifying different commands:
```python
classify("ls -la")        # → 'safe' (green)
classify("echo hello")    # → 'safe' (green)
classify("rm -rf /tmp/*") # → 'blocked' (red)
classify("pip install x") # → 'needs-approval' (yellow)
```

**Key points:**
- Destructive patterns are checked FIRST — safety first.
- Safe commands like `ls`, `cat`, `echo` skip the interrupt entirely in some implementations.
- The classification is for DISPLAY — the actual pausing comes from `interrupt_on`.

---

## Scene 4: The Interrupt/Resume Loop (6:30–10:00)

**On-screen:** Code showing `run_with_approval()` with detailed walkthrough of each step, then a live demo.

> **Host:** "Now for the core logic — our `run_with_approval` function that handles the full interrupt/resume cycle."
>
> ```python
> def run_with_approval(agent, prompt: str, thread_id: str = "default") -> dict:
>     """invoke → if interrupted, await_approval → resume. Loops until done."""
>     config = _config(thread_id)
>     auto = os.getenv("CODEIT_AUTO_APPROVE", "false").lower() == "true"
>     
>     # Step 1: Stream the initial invocation
>     for chunk in agent.stream(
>         {"messages": [{"role": "user", "content": prompt}]},
>         config=config, stream_mode="updates", version="v2"):
>         _print_event(chunk)
>     
>     # Step 2: Check if we're paused on an interrupt
>     state = agent.get_state(config)
>     while state.next:                              # ← non-empty when paused
>         pending = _pending_tool_call(state)        # extract the tool call
>         if pending is None:
>             console.print("[red]Interrupt with no recognizable tool call — aborting.[/red]")
>             break
>         
>         name, args = pending
>         if auto:  # --yolo mode
>             console.print("[yellow]--yolo: auto-approving[/yellow]")
>             cmd = Command(resume={"decisions": [{"type": "approve"}]})
>         else:    # Interactive approval
>             risk = classify(args.get("command", "")) if name == "run_shell" else "needs-approval"
>             color = {"safe": "green", "needs-approval": "yellow", "blocked": "red"}[risk]
>             console.print(f"[{color}]risk: {risk}[/{color}]  tool: {name}  args: {args}")
>             answer = console.input("[bold]Approve? (y/n): [/bold]").strip().lower()
>             if answer == "y":
>                 cmd = Command(resume={"decisions": [{"type": "approve"}]})
>             else:
>                 cmd = Command(resume={"decisions": [
>                     {"type": "reject", "message": "User denied this action."}
>                 ]})
>         
>         # Step 3: Resume the agent with our decision
>         for chunk in agent.stream(cmd, config=config, stream_mode="updates", version="v2"):
>             _print_event(chunk)
>         state = agent.get_state(config)
>     
>     return state.values
> ```

**Live demo:** Show the approval flow with `--yolo` flag:
```bash
CODEIT_WORKDIR=./workspace python 06-approval-gate.py "Run: echo hello" --yolo
```

*(Show tool call appearing, auto-approve message, then execution)*

---

## Scene 5: Interactive Approval Demo (10:00–12:00)

**On-screen:** Terminal session showing interactive approval prompts without `--yolo`.

> **Host:** "Now let's see the interactive mode — no `--yolo` flag. The agent will pause and ask for our approval."
>
> ```bash
> CODEIT_WORKDIR=./workspace python 06-approval-gate.py \
>     "Create a file then delete it with rm. Show me each step."
> ```

**What viewers will see:**
1. Agent calls `write_file` — approval prompt appears (yellow: needs-approval).
2. Viewer types 'y' to approve.
3. Agent calls `run_shell("rm ...")` — risk labeled as "blocked" in red.
4. Viewer can reject by typing 'n'.

**Key points:**
- The color-coded risk display helps viewers make informed decisions quickly.
- Rejection sends a message back to the model: "User denied this action."
- The agent learns from rejection and tries safer approaches.

---

## Scene 6: The --yolo Flag (12:00–13:00)

**On-screen:** Code showing how `CODEIT_AUTO_APPROVE` environment variable is checked, then a demo with `--yolo`.

> **Host:** "For demos and testing, we support auto-approval via the `CODEIT_AUTO_APPROVE=true` environment variable. This skips all prompts — useful for CI/CD or when you trust the agent."
>
> ```python
> auto = os.getenv("CODEIT_AUTO_APPROVE", "false").lower() == "true"
> if auto:
>     console.print("[yellow]--yolo: auto-approving[/yellow]")
>     cmd = Command(resume={"decisions": [{"type": "approve"}]})
> ```

**Key points:**
- `--yolo` is dangerous — it bypasses all safety checks.
- Only use in trusted environments or for demos.
- The env variable pattern keeps the approval logic centralized.

---

## Scene 7: Wrap-up & What's Next (13:00–14:00)

**On-screen:** Summary showing all approval gate capabilities and preview of Episode 7.

> **Host:** "In this episode, we've made our agent safe — it can no longer run destructive commands or modify files without explicit user approval."
>
> - **`interrupt_on`** — configures which tools pause for approval (shell, writes, deletes).
> - **`MemorySaver()`** — REQUIRED checkpointer that enables interrupt/resume functionality.
> - **`classify(command)`** — regex-based risk triage: safe (green), needs-approval (yellow), blocked (red).
> - **`run_with_approval()`** — the full interrupt/resume loop with interactive prompts or `--yolo` auto-approve.

**Key takeaways:**
1. `interrupt_on` + `MemorySaver()` are required together for human-in-the-loop to work.
2. The approval gate pauses execution, shows risk classification, and waits for user input.
3. Rejection sends a message back to the model so it can try safer approaches.
4. `--yolo` auto-approval is useful for demos but dangerous in production — use carefully.