# Episode 14 — Shipping CodeIt: A Real CLI with Event Streaming (Tutorial Video Script)

## Overview
**Length:** ~12 minutes  
**Goal:** Ship everything from Episodes 1-13 as a single `codeit` CLI command with a live Rich streaming view. Viewers run `codeit "add a health endpoint and test it"` and watch the agent work end-to-end: tokens stream, tool calls render in their own panel, subagent delegations appear, the todo list updates, approval prompts fire when needed.

---

## Scene 1: Hook & The Big Picture (0:00–1:00)

**On-screen:** Terminal showing `codeit --help` output with all flags listed.

> **Host:** "We've spent 13 episodes building individual pieces of a coding agent — model factory, streaming, filesystem sandboxing, shell execution, approval gates, system prompts, surgical edits, repo mapping, todo planning, error recovery, subagents, and MCP integration."
>
> "In this episode, we ship it all as a single `codeit` CLI command with a beautiful Rich-powered live streaming view. The viewer runs one command and watches the agent work in real time — just like Claude Code or Cursor."

**Key points:**
- This is the finale of the core series (Episode 15 is an optional appendix).
- Everything from Episodes 1-13 ships as a cohesive CLI tool.
- Rich panels give us that professional terminal UI look.

---

## Scene 2: The Typer App Structure (1:00–3:00)

**On-screen:** Code editor showing the `run` command definition with all flags.

> **Host:** "Let's start by looking at our CLI structure. We use Typer — a library that builds CLIs from Python type hints, giving us automatic help text and argument parsing."
>
> ```python
> @app.command()
> def run(
>     prompt: str = typer.Argument(..., help="The task for the agent."),
>     provider: str = typer.Option(None, "--provider", "-p", help="ollama | openai"),
>     model: str = typer.Option(None, "--model", "-m", help="model name"),
>     yolo: bool = typer.Option(False, "--yolo", "-y", help="DANGEROUS: auto-approve all actions."),
>     workdir: str = typer.Option(None, "--workdir", "-w", help="workspace directory"),
>     mcp: bool = typer.Option(False, "--mcp", help="connect to MCP server from env"),
>     skills: bool = typer.Option(False, "--skills", help="load skills from ./skills"),
>     approve: bool = typer.Option(False, "--approve", help="enable HITL approval gate"),
> ):
> ```

**Key points:**
- `typer.Argument` for positional arguments (the prompt).
- `typer.Option` with short flags (`-p`, `-m`, etc.) for optional parameters.
- `--yolo` flag auto-approves all actions — dangerous but useful for demos.
- `--approve` enables the human-in-the-loop approval gate from Episode 6.

---

## Scene 3: Flag Application Pattern (3:00–4:15)

**On-screen:** Code showing `_apply_flags()` function and how it mutates environment variables.

> **Host:** "Here's a clever pattern — instead of passing flags through every function call, we mutate the environment variables that our existing code already reads."
>
> ```python
> def _apply_flags(provider: str | None, model: str | None, yolo: bool, workdir: str | None):
>     """Mutate env so the rest of the code reads settings as usual (Ep 1 pattern)."""
>     if provider: os.environ["LLM_PROVIDER"] = provider
>     if model: os.environ["LLM_MODEL"] = model
>     if yolo: os.environ["CODEIT_AUTO_APPROVE"] = "true"
>     if workdir: os.environ["CODEIT_WORKDIR"] = str(Path(workdir).resolve())
> ```

**Key points:**
- This keeps the env-as-config pattern from Episode 1 consistent.
- No need to refactor existing code that reads `get_settings()` — it just works with CLI flags.
- `--yolo` sets `CODEIT_AUTO_APPROVE=true`, which our approval gate checks.

---

## Scene 4: The Streaming View (4:15–7:00)

**On-screen:** Code showing `_run_streaming()` and the Rich panel rendering, then a live demo of it running.

> **Host:** "Now for the fun part — the streaming view. We try Deep Agents' v3 event-streaming API first, which gives us typed projections like `messages` and `tool_calls`. If that's not available in the installed version, we gracefully fall back to v2 updates."
>
> ```python
> async def _run_streaming(agent, prompt: str, thread_id: str):
>     config = _config(thread_id)
>     input_msg = {"messages": [{"role": "user", "content": prompt}]}
>     try:
>         stream = agent.stream_events(input_msg, version="v3", config=config)
>         for name, item in stream.interleave("messages", "tool_calls"):
>             if name == "messages":
>                 console.print(Panel(Text(item.text, style="cyan"), 
>                                    title="assistant", border_style="blue"))
>             elif name == "tool_calls":
>                 color = "green" if item.error is None else "red"
>                 console.print(Panel(Text(f"{item.tool_name}({item.input})", style=color),
>                                    title="tool", border_style="magenta"))
>     except (TypeError, AttributeError):
>         # Fallback: v2 updates stream
>         for chunk in agent.stream(input_msg, config=config, 
>                                  stream_mode="updates", version="v2"):
>             _render_v2_chunk(chunk)
> ```

**Key points:**
- `stream.interleave("messages", "tool_calls")` gives us ordered output — messages and tool calls appear in execution order.
- Rich's `Panel` with different border colors makes the UI visually distinct: blue for assistant, magenta for tools.
- The fallback to v2 ensures compatibility across Deep Agents versions.

---

## Scene 5: Approval Driver (7:00–8:30)

**On-screen:** Code showing `run_with_approval()` function and a demo with `--approve` flag.

> **Host:** "When the user passes `--approve`, we use our approval-aware driver from Episode 6. This pauses on dangerous tool calls — like writing files or running shell commands — and asks for confirmation."
>
> ```python
> def run_with_approval(agent, prompt: str, thread_id: str = "cli") -> dict:
>     config = _config(thread_id)
>     auto = os.getenv("CODEIT_AUTO_APPROVE", "false").lower() == "true"
>     for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]},
>                           config=config, stream_mode="updates", version="v2"):
>         _render_v2_chunk(chunk)
>     state = agent.get_state(config)
>     while state.next:  # paused on an interrupt
>         pending = _pending_tool_call(state)
>         if not pending: break
>         name, args = pending
>         if auto:
>             cmd = Command(resume={"decisions": [{"type": "approve"}]})
>         else:
>             console.print(f"tool: {name}  args: {args}")
>             answer = console.input("[bold]Approve? (y/n): [/bold]").strip().lower()
>             cmd = Command(resume={"decisions": [{"type": "approve"}]}) if answer == "y" \
>                    else Command(resume={"decisions": [{"type": "reject", "message": "No."}]})
> ```

**Key points:**
- `state.next` indicates the agent is paused waiting for approval.
- `_pending_tool_call()` extracts which tool call needs approval.
- `--yolo` auto-approves everything; without it, interactive prompts appear.

---

## Scene 6: Live Demo (8:30–10:30)

**On-screen:** Terminal session running the CLI with different flag combinations.

> **Host:** "Let's see this in action! First, a simple task with auto-approval:"
>
> ```bash
> CODEIT_WORKDIR=./workspace python 14-cli-streaming.py "Run: echo hello" --yolo
> ```
>
> *(Show the Rich panels rendering live — assistant message in blue panel, tool call in magenta)*

**Then show with approval gate:**
```bash
CODEIT_WORKDIR=./workspace python 14-cli-streaming.py "Run: ls -la" --approve
# Approves when prompted
```

**Key points to highlight on screen:**
- The Rich panels give a professional, Claude Code-like feel.
- Tool calls are clearly separated from assistant responses.
- Approval prompts appear interactively when `--approve` is used without `--yolo`.

---

## Scene 7: MCP Integration (10:30–11:30)

**On-screen:** Code showing the async MCP loader and how it integrates with the CLI.

> **Host:** "The `--mcp` flag connects to an MCP server configured via environment variables. The loader is async — we use `asyncio.run()` at our entry point to handle this."
>
> ```python
> async def load_mcp_tools() -> list:
>     name = os.getenv("MCP_SERVER_NAME", "")
>     url = os.getenv("MCP_SERVER_URL", "")
>     if not name or not url:
>         return []
>     try:
>         from langchain_mcp_adapters.client import MultiServerMCPClient
>         client = MultiServerMCPClient({name: {"transport": "http", "url": url}})
>         return await client.get_tools()
>     except Exception as e:
>         print(f"MCP tools unavailable: {type(e).__name__}: {e}", file=sys.stderr)
>         return []
> ```

**Key points:**
- Graceful degradation — if MCP isn't configured or the package isn't installed, it returns an empty list.
- The full async flow is handled by `asyncio.run()` in our entry point.
- Full MCP integration details are covered in Episode 13.

---

## Scene 8: Wrap-up & What's Next (11:30–12:00)

**On-screen:** Summary showing all episodes and their capabilities, then a preview of Episode 15.

> **Host:** "And that's our CLI! We've taken everything from Episodes 1-13 — model factory, streaming, filesystem sandboxing, shell execution, approval gates, system prompts, surgical edits, repo mapping, todo planning, error recovery, subagents, and MCP integration — and shipped it as a single `codeit` command with a beautiful Rich-powered live view."
>
> "In Episode 15, we'll add LangSmith observability so you can see exactly what your agent is doing under the hood. But for now, you have a fully functional coding agent!"

**Key takeaways:**
1. Typer gives us professional CLI flags with zero boilerplate.
2. Rich panels create a polished terminal UI that rivals commercial agents.
3. The v3 event-streaming API provides typed projections; we fall back to v2 for compatibility.
4. Flag application via environment variables keeps the codebase consistent.
5. MCP integration is async and gracefully degrades when unavailable.