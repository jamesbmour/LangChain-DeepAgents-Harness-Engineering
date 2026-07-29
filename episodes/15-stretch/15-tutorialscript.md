# Episode 15 — LangSmith Observability Appendix (Tutorial Video Script)

## Overview
**Length:** ~8 minutes  
**Goal:** Show viewers how to see what the agent is *doing* under the hood using LangSmith tracing. Every model call, tool call, subagent delegation, and message history becomes visible in a trace URL they can open in the browser. This episode adds minimal code — mostly docs + a small helper.

---

## Scene 1: Hook & Context (0:00–0:45)

**On-screen:** Terminal showing the CodeIt agent running with output, then a LangSmith dashboard view.

> **Host:** "You've built an agent that can reason, call tools, ask for approval, and even delegate to subagents. But when something goes wrong — or you just want to understand why it made a particular decision — how do you see what's actually happening under the hood?"
>
> "In this episode, we'll add LangSmith observability so every step of your agent's reasoning becomes visible in an interactive trace."

**Key points:**
- This is the buffer/appendix episode — minimal new code.
- LangSmith is already a dependency from Phase 0 setup.
- We're adding visibility, not changing behavior.

---

## Scene 2: What Is LangSmith Tracing? (0:45–1:30)

**On-screen:** Diagram showing the flow: User prompt → Agent graph → LangSmith trace → Browser dashboard.

> **Host:** "LangSmith is LangChain's built-in observability platform. When tracing is enabled via environment variables, every model invocation, tool call, and subagent delegation gets recorded as a structured trace."
>
> "The key insight: you don't need to change your agent code at all. Tracing is activated purely through configuration — the `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` environment variables."

**Code snippet shown:**
```bash
export LANGSMITH_API_KEY=your-key-here
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=codeit-demo
```

---

## Scene 3: The Observability Helper (1:30–3:30)

**On-screen:** Code editor showing `trace_url_for_run()` and `maybe_print_trace_url()`.

> **Host:** "Let's look at the two helper functions we've added. First, `trace_url_for_run` — it checks if tracing is enabled and builds a URL to the trace in LangSmith."
>
> ```python
> def trace_url_for_run(run_id: str | None) -> str | None:
>     """Build a LangSmith trace URL if tracing is on and a run_id is available."""
>     if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
>         return None
>     if not run_id:
>         return None
>     project = os.getenv("LANGSMITH_PROJECT", "default")
>     return f"https://smith.langchain.com/projects/p/{project}/r/{run_id}"
> ```
>
> "And `maybe_print_trace_url` tries to extract the run ID from the agent's final state and prints either the trace URL or a hint to check the dashboard."

**Key points:**
- The helper is best-effort — LangSmith UI URLs change over time.
- If no run_id is found, it tells viewers where to look in the dashboard.
- This is purely informational output; it doesn't affect agent behavior.

---

## Scene 4: Running with Tracing (3:30–5:30)

**On-screen:** Terminal session running episode 15 with tracing enabled.

> **Host:** "Now let's run the demo with tracing on. We'll use a simple `echo hello` command — something safe that won't trigger our approval gate."
>
> ```bash
> export LANGSMITH_API_KEY=your-key-here
> export LANGSMITH_TRACING=true
> export LANGSMITH_PROJECT=codeit-demo
> CODEIT_WORKDIR=./workspace python 15-stretch.py "Run: echo hello"
> ```

**Expected output shown:**
```
Tip: set LANGSMITH_TRACING=true to see a trace URL at the end.
tool call: run_shell({'command': 'echo hello'})
--yolo: auto-approving
assistant: That ran the `echo` command...
--- final answer ---
That ran the `echo` command in the shell and returned its output...
Trace: https://smith.langchain.com/projects/p/codeit-demo/r/<run-id>
```

**Key points:**
- The trace URL appears at the end of execution.
- Opening it shows every step: model calls, tool invocations, message history.
- Viewers can see exactly why the agent chose to call `run_shell` and how it processed the result.

---

## Scene 5: Reading a Trace (5:30–6:45)

**On-screen:** Screen recording of opening a LangSmith trace in the browser, navigating through the steps.

> **Host:** "When you open the trace URL, you see an interactive timeline. Each node represents either a model call or a tool invocation."
>
> "Clicking on a model call shows you exactly what prompt was sent — including all prior messages in context. Click on a tool call to see its arguments and return value."
>
> "This is invaluable for debugging: if your agent made a bad decision, you can trace back through the conversation history to find where things went wrong."

**What viewers will learn:**
- How to navigate the LangSmith trace UI.
- Where to find model inputs/outputs.
- How tool calls and their results are displayed.

---

## Scene 6: When Tracing Isn't Available (6:45–7:30)

**On-screen:** Code showing the fallback message in `maybe_print_trace_url`.

> **Host:** "What if you don't have a LangSmith API key, or tracing isn't enabled? The helper gracefully degrades."
>
> ```python
> elif os.getenv("LANGSMITH_TRACING", "").lower() == "true":
>     project = os.getenv("LANGSMITH_PROJECT", "default")
>     console.print(f"[dim]Tracing is on. Open the LangSmith dashboard → project '{project}' "
>                   f"to find this run by timestamp.[/dim]")
> ```
>
> "If tracing is enabled but we can't extract a URL, it tells you to check the dashboard directly and find your run by project name and timestamp."

---

## Scene 7: Wrap-up & What's Next (7:30–8:00)

**On-screen:** Summary of all 15 episodes with key capabilities.

> **Host:** "And that wraps up our 15-episode series! We've built CodeIt — a full-featured coding agent from scratch using Deep Agents."
>
> "From the model factory in Episode 1, through streaming, filesystem sandboxing, shell execution, approval gates, system prompts, surgical edits, repo mapping, todo planning, error recovery, subagents, MCP integration, and CLI streaming — to this final observability appendix."
>
> "The complete agent is under 250 lines of core code, yet it has all the capabilities you'd expect from a production coding assistant."

**Key takeaways:**
1. LangSmith tracing gives visibility into every step of your agent's reasoning.
2. No code changes needed — just environment variables.
3. The trace URL helper is best-effort but useful for quick debugging.
4. This episode adds zero new dependencies beyond what was already installed.

---

## Code Walkthrough Points (for host reference)

### Key Functions:
1. **`trace_url_for_run(run_id)`** — Builds a LangSmith dashboard URL from the run ID and project name. Returns `None` if tracing is off or no run ID available.

2. **`maybe_print_trace_url(state)`** — Tries multiple attribute paths to extract `run_id` from the agent state, then either prints the trace URL or a fallback hint.

3. **`build_agent()`** — Same as Episode 14 but with observability helpers integrated into the main flow.

### Environment Variables:
- `LANGSMITH_API_KEY` — Your LangSmith API key (required for tracing)
- `LANGSMITH_TRACING=true` — Enables tracing
- `LANGSMITH_PROJECT=codeit-demo` — Project name in LangSmith dashboard

### What Makes This Episode Special:
- **Minimal code:** The episode is mostly documentation and a small helper.
- **No behavior change:** Tracing doesn't alter how the agent works.
- **Debugging superpower:** Viewers get visibility into every model call, tool invocation, and message exchange.