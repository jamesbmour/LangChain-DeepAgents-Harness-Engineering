# Episode 5 — Running Commands: The Shell Tool (and Why It's Dangerous) (Tutorial Video Script)

## Overview
**Length:** ~13 minutes  
**Goal:** Show viewers how to give their agent the ability to run shell commands, while being brutally honest about the security implications. This is a cliffhanger episode — no approval gating yet, ending with a warning that sets up Episode 6's safety gate.

---

## Scene 1: Hook & The Power of Shell Access (0:00–2:00)

**On-screen:** Terminal showing an agent without shell access being unable to run tests or install packages, then gaining the ability and becoming much more capable.

> **Host:** "So far, our agent can read files, write files, and edit them — but it can't actually RUN anything. It can't execute tests, install dependencies, check git status, or build projects."
>
> *(Show agent being unable to run pytest without shell access)*

**Key points:**
- Without shell access, the agent is limited to file manipulation only.
- Real coding tasks require running commands: tests, installs, builds, git operations.
- This episode adds that capability — but with a major security warning.

---

## Scene 2: The Security Warning (2:00–3:30)

**On-screen:** Text highlighting the security implications in red, then code showing the danger.

> **Host:** "Before we write any code, let's be absolutely clear about what we're doing here."
>
> *(Show warning text)*
>
> **"⚠️ WARNING: This episode has NO approval gating. The agent CAN run `rm -rf .` and delete the entire workspace. We confine CWD to CODEIT_WORKDIR, but NOT the process itself."**

**Key points:**
- `shell=True` means arbitrary command execution — pipes, redirects, everything works.
- cwd confinement doesn't prevent system-level damage if permissions allow it.
- This is a teaching tool — NEVER run unsandboxed on a real repo.
- Episode 6 adds the approval gate that makes this safe.

---

## Scene 3: The run_shell Tool (3:30–7:00)

**On-screen:** Code editor showing `run_shell()` implementation with detailed walkthrough of each part, then live demos.

> **Host:** "Let's look at our custom `run_shell` tool — it runs commands in the workspace and returns a readable string."
>
> ```python
> MAX_OUTPUT_CHARS = 20_000  # ~5k tokens; keeps context window healthy
> 
> @tool
> def run_shell(command: str) -> str:
>     """Run a shell command in the workspace and return stdout+stderr+exit code.
>     
>     Use this to run tests, install packages, execute scripts, or inspect git state.
>     Argument: a single shell command string (e.g. 'pytest -q' or 'pip install fastapi').
>     The command runs with cwd set to the workspace root.
>     Output is truncated if longer than ~20k chars.
>     """
>     cwd = str(workspace_root())
>     try:
>         proc = subprocess.run(
>             command, shell=True, cwd=cwd,
>             capture_output=True, text=True,
>             timeout=int(os.getenv("CODEIT_SHELL_TIMEOUT", "120")),
>         )
>     except subprocess.TimeoutExpired:
>         return f"Error: command timed out.\nCommand: {command}"
>     except Exception as e:
>         return f"Error launching command: {type(e).__name__}: {e}"
>     
>     out = proc.stdout or ""
>     err = proc.stderr or ""
>     combined = f"$ {command}\n[exit {proc.returncode}]\n"
>     if out:
>         combined += f"--- stdout ---\n{out}\n"
>     if err:
>         combined += f"--- stderr ---\n{err}\n"
>     if len(combined) > MAX_OUTPUT_CHARS:
>         combined = combined[:MAX_OUTPUT_CHARS] + (
>             f"\n... [truncated, {len(combined)-MAX_OUTPUT_CHARS} more chars]"
>         )
>     return combined
> ```

**Live demo:** Show running different commands:
```bash
CODEIT_WORKDIR=./workspace python 05-shell-tool.py "Run: echo hello" --yolo
# Output shows the command, exit code, and stdout/stderr clearly formatted
```

**Key points:**
- `shell=True` allows pipes and redirects — powerful but dangerous.
- Timeout is configurable via `CODEIT_SHELL_TIMEOUT` env var (default 120s).
- Non-zero exits return as strings with `[exit N]` prefix — the model reads this for self-correction.
- Output truncation prevents context window blowup on verbose commands.

---

## Scene 4: Why shell=True? The Trade-off (7:00–8:30)

**On-screen:** Text showing the trade-off between simplicity and security, with code examples of what `shell=True` enables.

> **Host:** "You might wonder — why use `shell=True`? It's more dangerous because it allows command injection."
>
> *(Show comparison)*
>
> ```python
> # shell=False (safer but limited):
> subprocess.run(["pytest", "-q"], cwd=cwd)  # No pipes, no redirects
> 
> # shell=True (powerful but dangerous):
> subprocess.run("pytest -q | grep FAILED", shell=True, cwd=cwd)  # Pipes work!
> ```

**Key points:**
- `shell=False` is safer but can't handle pipes or complex commands.
- The agent constructs the command string — it's not user input from an untrusted source.
- Episode 6 adds the approval gate that mitigates this risk.
- Trade-off: simplicity and power vs. security.

---

## Scene 5: No Gating = Cliffhanger (8:30–10:00)

**On-screen:** Terminal showing the agent running a destructive command without any prompts, then the warning message at the end of execution.

> **Host:** "Notice something critical — there's NO approval prompt when the agent calls `run_shell`. It just runs."
>
> *(Show code with no interrupt_on)*
> ```python
> return create_deep_agent(
>     model=get_model(), tools=[run_shell],
>     system_prompt="...", backend=backend,
>     checkpointer=InMemorySaver(),  # No interrupt_on!
> )
> ```

**Key points:**
- Without `interrupt_on`, the agent executes shell commands immediately.
- The warning at the end of execution is our cliffhanger: "The agent just ran real shell commands. It COULD have run `rm -rf .` and deleted the workspace."
- This sets up Episode 6 perfectly — we need safety controls.

---

## Scene 6: Live Demo — Shell in Action (10:00–12:00)

**On-screen:** Terminal session showing a more complex shell task being executed by the agent.

> **Host:** "Let's see this working with a slightly more complex task:"
>
> ```bash
> CODEIT_WORKDIR=./workspace python 05-shell-tool.py \
>     --yolo "Check what Python version is installed and list files in workspace"
> ```

**What viewers will see:**
1. Agent calls `run_shell("python --version")` — output shows Python version.
2. Agent calls `run_shell("ls -la")` — output lists workspace contents.
3. Both commands execute immediately without any approval prompt.
4. The warning message appears at the end.

---

## Scene 7: Wrap-up & What's Next (12:00–13:00)

**On-screen:** Summary showing shell tool capabilities and preview of Episode 6.

> **Host:** "In this episode, we've given our agent the power to run real shell commands — but with a critical caveat."
>
> - **`run_shell(command)`** — runs arbitrary shell commands in the workspace via `subprocess.run` with `shell=True`.
> - **Output formatting** — returns `$ command`, `[exit N]`, stdout, and stderr as a readable string.
> - **No approval gating** — this is intentional; it's our cliffhanger for Episode 6.

**Key takeaways:**
1. Shell access makes the agent dramatically more capable but also dangerous.
2. `shell=True` enables pipes and redirects but increases injection risk.
3. Output truncation (`MAX_OUTPUT_CHARS`) prevents context window blowup.
4. Non-zero exits return as strings — the model reads `[exit N]` for self-correction (used in Episode 11's recovery loop).
5. **NEVER run unsandboxed on a real repo** — this is a teaching tool only.