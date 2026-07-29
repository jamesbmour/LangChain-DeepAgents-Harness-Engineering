# Episode 7 — The System Prompt: Engineering Personality & Rules (Tutorial Video Script)

## Overview
**Length:** ~12 minutes  
**Goal:** Show viewers how to write a system prompt that transforms a generic AI assistant into CodeIt — a specialized coding agent with clear personality, tool-use policies, safety rules, and editing conventions. Also covers loading project-specific context from `AGENTS.md` or `CODEIT.md`.

---

## Scene 1: Hook & The Personality Problem (0:00–1:30)

**On-screen:** Terminal showing a generic AI assistant trying to code without guidance — it's inconsistent, doesn't use tools properly, and makes unsafe decisions. Then the same task with CodeIt's system prompt works perfectly.

> **Host:** "A language model by itself is like a brilliant intern who's never been told what company they work for. They might be smart, but they don't know your standards, your conventions, or even what tools are available to them."
>
> *(Show before/after comparison)*

**Key points:**
- Without a system prompt, the model behaves inconsistently and may make unsafe decisions.
- A well-crafted system prompt turns a generic assistant into a specialized agent.
- This is the "same task, dramatically better behavior" episode — show clear before/after on camera.

---

## Scene 2: The SYSTEM_PROMPT Structure (1:30–4:00)

**On-screen:** Code editor showing the full `SYSTEM_PROMPT` constant with each section highlighted as it's discussed.

> **Host:** "Let's look at our system prompt — written for the model, not for viewers. It's imperative and concise."
>
> ```python
> SYSTEM_PROMPT = """You are CodeIt, a terminal coding agent.
> 
> # Role
> You help the user write, edit, and run code in their workspace. You read files,
> write files, run shell commands, and plan multi-step tasks.
> 
> # Tool-use policy
> - Prefer the built-in filesystem tools (ls, read_file, grep, glob) for exploration.
> - Use write_file only for new files. Use edit_file for changes to existing files.
> - Use run_shell for tests, installs, and git operations. The user will be asked
>   to approve mutating or destructive commands — prefer the least destructive option.
> - Use write_todos to plan any task that needs more than one step.
> 
> # Safety
> - Never run destructive commands (rm -rf, git push -f, dd, mkfs) without explaining why.
> - If a command might modify files outside the workspace, say so and stop.
> - The user can reject any action. Accept rejection gracefully and try a safer approach.
> 
> # Editing rules
> - For small targeted changes, use edit_file (search-replace). Never rewrite a whole file
>   to change a few lines.
> - After editing code that has tests, run the tests with run_shell('pytest -q') and fix failures.
> 
> # Communication
> - Be concise. Say what you're about to do, do it, then summarize the result in one line.
> - When a tool call fails, read the error, explain what went wrong in one sentence, and retry.
> - Don't apologize; fix.
> """
> ```

**Key points:**
- Written in imperative voice — direct instructions to the model.
- Organized into clear sections: Role, Tool-use policy, Safety, Editing rules, Communication.
- Each section covers a different aspect of agent behavior.

---

## Scene 3: The Safety Section (4:00–5:30)

**On-screen:** Code highlighting the Safety section with examples of destructive commands being blocked.

> **Host:** "The Safety section is critical — it prevents the agent from doing dangerous things."
>
> ```python
> # Safety
> - Never run destructive commands (rm -rf, git push -f, dd, mkfs) without explaining why.
> - If a command might modify files outside the workspace, say so and stop.
> - The user can reject any action. Accept rejection gracefully and try a safer approach.
> ```

**Key points:**
- Explicitly lists destructive commands: `rm -rf`, `git push -f`, `dd`, `mkfs`.
- Requires explanation before running dangerous operations.
- Enforces workspace boundaries — never modify files outside the sandbox.
- Teaches graceful handling of user rejection (from Episode 6's approval gate).

---

## Scene 4: Project Context Loading (5:30–7:30)

**On-screen:** Code showing `load_project_context()` and a demo with an AGENTS.md file in the workspace.

> **Host:** "Beyond our base system prompt, we also load project-specific context from `AGENTS.md` — Deep Agents' own convention for project instructions."
>
> ```python
> def load_project_context(root: str | Path | None = None) -> str:
>     """Read AGENTS.md (or CODEIT.md fallback) from the workspace. Return '' if absent."""
>     r = Path(root or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
>     for name in ("AGENTS.md", "CODEIT.md"):  # AGENTS.md wins if both exist
>         candidate = r / name
>         if candidate.is_file():
>             try:
>                 return f"\n\n# Project context ({name})\n\n" + candidate.read_text(encoding="utf-8")
>             except Exception:
>                 return ""  # degrade gracefully
>     return ""
> ```

**Live demo:** Show creating an AGENTS.md file and running the agent with it.

```bash
echo '# My Project\nUse FastAPI. The bug is in main.py.' > workspace/AGENTS.md
CODEIT_WORKDIR=./workspace python 07-system-prompt.py "What is this project?" --yolo
```

**Key points:**
- `AGENTS.md` wins if both AGENTS.md and CODEIT.md exist.
- Graceful degradation — missing file returns empty string, not an error.
- The loaded context is appended to the system prompt via `build_system_prompt()`.

---

## Scene 5: Composing the Full Prompt (7:30–9:00)

**On-screen:** Code showing `build_system_prompt()` and how it's used in `build_agent()`.

> **Host:** "The composition is simple — our base prompt plus any project context."
>
> ```python
> def build_system_prompt(root: str | Path | None = None) -> str:
>     """Compose the harness system prompt with any project context."""
>     return SYSTEM_PROMPT + load_project_context(root)
> 
> # In build_agent():
> return create_deep_agent(
>     model=get_model(),
>     tools=[run_shell],
>     system_prompt=build_system_prompt(root),  # ← composed prompt
>     backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
>     interrupt_on=INTERRUPT_ON,
>     checkpointer=MemorySaver(),
> )
> ```

**Key points:**
- The harness's middleware injects its own tool descriptions on top of our system prompt.
- We ADD policy — we don't try to replace the framework's built-in instructions.
- This is important: don't redefine tools like `write_todos` in your prompt; just reference them as policy.

---

## Scene 6: Live Demo — Before & After (9:00–11:00)

**On-screen:** Terminal session comparing agent behavior with and without the system prompt, then a demo with AGENTS.md context.

> **Host:** "Let's see the difference! First, let's run WITHOUT our system prompt — just a bare model:"
>
> *(Show generic assistant behavior — inconsistent tool use, no safety awareness)*

**Then WITH the system prompt:**
```bash
CODEIT_WORKDIR=./workspace python 07-system-prompt.py \
    "What is this project? Fix any issues." --yolo
```

*(Show CodeIt following its policies: reads AGENTS.md context, uses tools properly, respects safety rules)*

**Key points:**
- The system prompt makes behavior consistent and predictable.
- Safety rules prevent dangerous actions automatically.
- Project context from AGENTS.md informs the agent about project-specific conventions.

---

## Scene 7: Wrap-up & What's Next (11:00–12:00)

**On-screen:** Summary showing all system prompt capabilities and preview of Episode 8.

> **Host:** "In this episode, we've given CodeIt its personality — a comprehensive system prompt that covers role definition, tool-use policy, safety rules, editing conventions, and communication style."
>
> - **`SYSTEM_PROMPT`** — structured constant with Role, Tool-use policy, Safety, Editing rules, Communication.
> - **`load_project_context()`** — reads AGENTS.md or CODEIT.md from workspace for project-specific instructions.
> - **`build_system_prompt()`** — composes base prompt + project context.

**Key takeaways:**
1. System prompts should be imperative and concise — written for the model, not viewers.
2. Organize into clear sections: Role, Tool-use policy, Safety, Editing rules, Communication.
3. Load project-specific context from AGENTS.md (Deep Agents convention) or CODEIT.md fallback.
4. The harness injects its own tool descriptions on top of your prompt — ADD policy, don't replace it.
5. Graceful degradation is key: missing files should return empty strings, not errors.