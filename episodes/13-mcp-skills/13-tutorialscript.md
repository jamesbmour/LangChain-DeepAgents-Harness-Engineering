# Episode 13 — MCP + Skills: Speaking the Standard Protocol (Tutorial Video Script)

## Overview
**Length:** ~12 minutes  
**Goal:** Show viewers how to extend their agent with two powerful ecosystem capabilities from Deep Agents: MCP tools that connect to external servers via a standard protocol, and skills that provide on-demand instructions loaded from a directory. Both are optional with graceful degradation — the agent works without them.

---

## Scene 1: Hook & The Power of Standards (0:00–1:00)

**On-screen:** Terminal showing an MCP server being started via npx, then the CodeIt agent connecting to it.

> **Host:** "So far, our agent has been self-contained — it uses its own tools and skills we've built in code. But what if you want your agent to talk to external systems? What if you want to add new capabilities without changing any code?"
>
> "In this episode, we'll connect our agent to an MCP server — the Model Context Protocol standard that lets any tool expose its functionality as callable functions. And we'll add skills — a directory of markdown instructions that load on demand when relevant."

**Key points:**
- MCP is an open standard for connecting AI agents to external tools and data sources.
- Skills are progressive-disclosure instruction files loaded from disk.
- Both integrate seamlessly with Deep Agents' middleware stack.

---

## Scene 2: The MCP Config Builder (1:00–3:00)

**On-screen:** Code editor showing `build_mcp_config()` and `load_mcp_tools()`.

> **Host:** "Let's start by looking at how we configure the MCP connection. We read three environment variables — server name, transport type, and URL."
>
> ```python
> def build_mcp_config() -> dict | None:
>     """Build the MultiServerMCPClient config from env. None if no server configured."""
>     name = os.getenv("MCP_SERVER_NAME", "")
>     url = os.getenv("MCP_SERVER_URL", "")
>     transport = os.getenv("MCP_TRANSPORT", "http")
>     if not name or not url:
>         return None
>     return {name: {"transport": transport, "url": url}}
> ```

**Key points:**
- Returns `None` when no MCP server is configured — the agent works without it.
- Supports HTTP and stdio transports (configurable via env).
- The config dict maps server names to their connection details.

---

## Scene 3: Async Tool Loading with Graceful Degradation (3:00–5:00)

**On-screen:** Code showing `load_mcp_tools()` with the try/except block, then a demo of running without MCP configured.

> **Host:** "MCP tool loading is async — `get_tools()` is a coroutine. Our loader catches any errors and returns an empty list if the server is unreachable."
>
> ```python
> async def load_mcp_tools(config: dict | None = None) -> list:
>     """Connect to the MCP server(s) and return their tools. Empty list if no config."""
>     if not config:
>         return []
>     from langchain_mcp_adapters.client import MultiServerMCPClient
>     try:
>         client = MultiServerMCPClient(config)
>         return await client.get_tools()
>     except Exception as e:
>         print(f"MCP tools unavailable: {type(e).__name__}: {e}", file=sys.stderr)
>         return []
> ```

**Key points:**
- The import is inside the function — if `langchain-mcp-adapters` isn't installed, it fails gracefully.
- Any connection errors are caught and logged to stderr.
- Returns an empty list so the agent continues working with just its built-in tools.

---

## Scene 4: Extended build_agent() (5:00–6:30)

**On-screen:** Code showing `build_agent()` accepting `mcp_tools` and `skills` parameters.

> **Host:** "Our `build_agent()` function now accepts two optional lists — MCP tools from an external server, and skill directory paths."
>
> ```python
> def build_agent(mcp_tools: list | None = None, skills: list | None = None, workdir: str | None = None):
>     root = Path(workdir or os.getenv("CODEIT_WORKDIR", "./workspace")).resolve()
>     root.mkdir(parents=True, exist_ok=True)
>     return create_deep_agent(
>         model=get_model(),
>         tools=[run_shell] + (mcp_tools or []),
>         system_prompt=("You are CodeIt, a coding agent. MCP tools (if present) come from an external "
>                        "server. Skills load on demand when relevant."),
>         backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
>         skills=skills,                  # list of directory paths; None = no skills
>         interrupt_on=INTERRUPT_ON, checkpointer=MemorySaver(),
>     )
> ```

**Key points:**
- MCP tools are appended to our custom `run_shell` tool.
- The `skills` parameter accepts a list of directory paths — Deep Agents' SkillsMiddleware handles loading them on demand.
- Both parameters default to `None`, so the agent works without either.

---

## Scene 5: Async Entry Point (6:30–8:00)

**On-screen:** Code showing `_async_main()` and how it orchestrates MCP loading + skills + approval.

> **Host:** "Since MCP tool loading is async, our main entry point uses `asyncio.run()`. The flow is: load MCP tools if requested → build the agent with all capabilities → run with approval gate."
>
> ```python
> async def _async_main(prompt: str, use_mcp: bool, skills_dir: str | None):
>     mcp_tools = await load_mcp_tools(build_mcp_config()) if use_mcp else []
>     if use_mcp:
>         console.print(f"[dim]Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}[/dim]")
>     skills = [skills_dir] if skills_dir else None
>     agent = build_agent(mcp_tools=mcp_tools, skills=skills)
>     state = run_with_approval(agent, prompt)
> ```

**Key points:**
- MCP loading only happens when `--mcp` flag is passed.
- Skills are loaded from `./skills/` directory if it exists and `--skills` is set.
- The approval driver from Episode 6 handles the interactive flow.

---

## Scene 6: Live Demo — Without MCP (8:00–9:30)

**On-screen:** Terminal session running episode 13 without MCP configured, showing graceful degradation.

> **Host:** "Let's see this in action! First, let's run without any MCP server configured:"
>
> ```bash
> CODEIT_WORKDIR=./workspace python 13-mcp-skills.py "Run: echo hello" --yolo
> ```
>
> *(Show the agent running successfully — it works with just its built-in tools)*

**Key points:**
- The agent runs perfectly fine without MCP.
- No errors or crashes when `langchain-mcp-adapters` isn't installed.
- Graceful degradation is a core design principle.

---

## Scene 7: Live Demo — With MCP (9:30–11:00)

**On-screen:** Terminal session starting an MCP filesystem server via npx, then running the agent with `--mcp`.

> **Host:** "Now let's connect to a real MCP server. We'll start the filesystem MCP server in one terminal:"
>
> ```bash
> # Terminal 1: Start MCP server
> npx -y @modelcontextprotocol/server-filesystem ./workspace &
> export MCP_SERVER_NAME=filesystem
> export MCP_TRANSPORT=http
> export MCP_SERVER_URL=http://localhost:8000/mcp
> ```
>
> "Then run the agent with `--mcp`:"
>
> ```bash
> python 13-mcp-skills.py "List files in workspace using MCP tools" --mcp --yolo
> ```

**Key points:**
- The filesystem MCP server exposes read/write/list operations as callable tools.
- These tools appear alongside our `run_shell` tool in the agent's toolkit.
- Viewers can see exactly how many MCP tools were loaded and their names.

---

## Scene 8: Skills — Progressive Disclosure (11:00–12:00)

**On-screen:** Code showing a sample SKILL.md file structure, then explaining progressive disclosure.

> **Host:** "Skills are markdown files with YAML frontmatter that provide instructions on demand. Instead of loading all possible knowledge into context at once, the agent only reads a skill's full instructions when it becomes relevant."
>
> ```markdown
> ---
> name: python-testing
> description: Best practices for writing Python tests with pytest
> triggers: [test, pytest, unittest]
> ---
> # Python Testing Skill
> When writing tests...
> ```

**Key points:**
- Skills keep context small — only loaded when relevant.
- The `triggers` field tells the agent when to consider loading a skill.
- This is progressive disclosure: knowledge stays on disk until needed.

---

## Scene 9: Wrap-up & What's Next (12:00–13:00)

**On-screen:** Summary showing MCP and skills integration, then preview of Episode 14.

> **Host:** "In this episode, we've given our agent two powerful new capabilities:"
>
> 1. **MCP tools** — connect to any MCP-compatible server for external system access
> 2. **Skills** — on-demand instruction loading from a directory with progressive disclosure
>
> "Both are optional and degrade gracefully when unavailable. In Episode 14, we'll ship everything as a polished CLI command with Rich-powered live streaming."

**Key takeaways:**
1. MCP is an open standard for tool interoperability — any server can expose tools to your agent.
2. Skills provide progressive disclosure of knowledge without bloating context.
3. Graceful degradation ensures the agent always works, even when optional components are missing.
4. Async loading with `asyncio.run()` handles the async nature of MCP connections.