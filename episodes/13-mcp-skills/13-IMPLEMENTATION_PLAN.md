# Episode 13 — MCP + Skills: Speaking the Standard Protocol

**Tag:** `ep-13` · **Shape:** use the battery — MCP + SkillsMiddleware · **Budget:** ~110 SLOC
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 13

---

## 13.1 · What this episode delivers

The agent gains two new capabilities from the Deep Agents ecosystem:

1. **MCP tools** — connect to an MCP server (e.g., a filesystem or git MCP server) via `langchain-mcp-adapters`. The server's tools register alongside our custom tools, exposed to the agent automatically. The agent can now drive external systems through a standard protocol.
2. **Skills** — a `skills/` directory with one example `SKILL.md` (YAML frontmatter + markdown instructions). `SkillsMiddleware` (built-in, #2 in the default stack) loads skill summaries at startup and the agent reads full instructions on demand. Progressive disclosure: context stays small until a skill is needed.

New pieces:
1. `codeit/mcp_client.py` — `build_mcp_config(settings)` and `load_mcp_tools(config)` (async) helpers.
2. `skills/python-testing/SKILL.md` — one example skill with proper frontmatter.
3. Extended `build_agent()` — accepts `mcp_tools` (a list of pre-loaded MCP tools) and `skills` (a list of skill directory paths).

---

## 13.2 · Pre-flight

1. Install `langchain-mcp-adapters`: `pip install langchain-mcp-adapters` (or `uv add`).
2. Confirm the MCP client API (per the docs):
   ```python
   from langchain_mcp_adapters.client import MultiServerMCPClient
   client = MultiServerMCPClient({"my_server": {"transport": "http", "url": "..."}})
   tools = await client.get_tools()
   # then: create_deep_agent(tools=[...our tools..., ...mcp tools...], ...)
   ```
   MCP tool loading is **async** — `get_tools()` is a coroutine. This means `build_agent` needs an async path, or we load MCP tools in an async helper before calling `build_agent`.
3. Pick a demo MCP server. Options:
   - **Filesystem MCP server** (`@modelcontextprotocol/server-filesystem` via npx) — exposes read/write tools. But we already have filesystem tools; less interesting.
   - **Git MCP server** — exposes git operations as tools. Good for the demo ("commit the change").
   - **A simple custom MCP server** — too much for one episode.
   - **Decision: use the filesystem MCP server** for the demo (easiest to run locally via `npx`), and mention git/Slack/GitHub MCP servers as next steps. The point is the *protocol*, not the specific server.
4. Confirm `skills` arg to `create_deep_agent` accepts a list of directory paths (per the skills doc: "Pass the path to your top-level skills directory in the `skills` argument").
5. Confirm `SkillsMiddleware` requires a backend (per the deep-agents-memory skill fix: "Skills require a proper backend to load from the filesystem"). We already have `FilesystemBackend` from Ep 3 — skills load from there. Confirm the skills directory must be *inside* the backend's root or can be a separate path.

---

## 13.3 · File specs

### `codeit/mcp_client.py` (~50 SLOC)

```python
from typing import Any
from codeit.settings import Settings, get_settings

def build_mcp_config(settings: Settings | None = None) -> dict | None:
    """Build the MultiServerMCPClient config from env. Returns None if no MCP server configured.

    Reads MCP_SERVER_NAME, MCP_TRANSPORT, MCP_URL from env (or a single MCP_SERVER_URL).
    Example .env addition:
      MCP_SERVER_NAME=filesystem
      MCP_TRANSPORT=http
      MCP_SERVER_URL=http://localhost:8000/mcp
    """
    import os
    s = settings or get_settings()
    name = os.getenv("MCP_SERVER_NAME", "")
    url = os.getenv("MCP_SERVER_URL", "")
    transport = os.getenv("MCP_TRANSPORT", "http")
    if not name or not url:
        return None
    return {
        name: {
            "transport": transport,
            "url": url,
        }
    }

async def load_mcp_tools(config: dict | None = None) -> list:
    """Connect to the MCP server(s) and return their tools. Empty list if no config.

    Usage (async):
        config = build_mcp_config()
        mcp_tools = await load_mcp_tools(config)
        agent = build_agent(mcp_tools=mcp_tools)
    """
    if not config:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient(config)
    try:
        tools = await client.get_tools()
        return tools
    except Exception as e:
        # Server unreachable — degrade gracefully. The agent still works without MCP tools.
        import sys
        print(f"MCP tools unavailable: {type(e).__name__}: {e}", file=sys.stderr)
        return []
```

**Teaching notes:**
- MCP is **optional** — if no server is configured (env not set) or the server is down, `load_mcp_tools` returns `[]` and the agent works normally. Graceful degradation is the contract.
- `load_mcp_tools` is **async** because `MultiServerMCPClient.get_tools()` is a coroutine. The demo script and the CLI (Ep 14) need an async entry point.
- We read MCP config from env (`MCP_SERVER_NAME`, `MCP_SERVER_URL`, `MCP_TRANSPORT`) — same provider-agnostic pattern as the LLM. No hardcoding.
- The `print(..., file=sys.stderr)` on failure avoids crashing the agent. The viewer sees the warning; the agent continues without MCP tools.

### `skills/python-testing/SKILL.md` (~30 SLOC, markdown)

```markdown
---
name: python-testing
description: Use this skill for Python testing tasks with pytest. Covers fixtures, mocking, async test patterns, and common pytest idioms. Load this when the user asks to write, run, or fix Python tests.
---

# python-testing

## Overview
This skill provides conventions and patterns for writing and running Python tests with pytest.

## When to use
- The user asks to write a new test
- The user asks to fix a failing test
- The user asks to run tests or interpret test output
- You're about to call run_tests and want to write idiomatic test code first

## Instructions

### Test file naming
- Name test files `test_<module>.py` (e.g. `test_main.py` for `main.py`)
- Name test functions `test_<behavior>` (e.g. `test_hello_returns_greeting`)

### Fixtures
- Use `@pytest.fixture` for shared setup
- Use `tmp_path` for filesystem tests (built-in fixture)
- Prefer fixtures over setUp/tearDown

### Running tests
- `pytest -q` for quiet output
- `pytest -x` to stop on first failure
- `pytest --tb=short` for concise tracebacks (this is what run_tests uses)

### Common patterns
- Use `pytest.raises(Exception)` to test expected errors
- Use `monkeypatch.setenv(...)` for env var tests
- Use `mocker.patch(...)` (from pytest-mock) for mocking — install if needed

### When a test fails
1. Read the full traceback (run_tests returns it)
2. Identify the line that failed and the assertion that didn't hold
3. Fix the code (not the test) unless the test itself is wrong
4. Re-run with run_tests to confirm the fix
```

**Teaching notes:**
- The YAML frontmatter (`name`, `description`) is **required** — per the deep-agents-core skill fix. Without it, `SkillsMiddleware` won't load the skill.
- The `description` is specific (per the fix: "Use specific descriptions to help agents decide when to use a skill"). "Use this skill for Python testing tasks..." tells the agent exactly when to load it.
- The skill body is instructions the agent reads *on demand* — only when it decides the skill is relevant. This is progressive disclosure: the summary loads at startup, the full body loads when needed.
- This is a *teaching* skill — small and illustrative. Real skills (per the skills doc) can include scripts, reference docs, templates in subdirectories.

### Extend `codeit/agent.py` (~30 SLOC)

```python
def build_agent(
    model: BaseChatModel | None = None,
    tools: list | None = None,
    system_prompt: str | None = None,
    backend=None,
    workdir: str | None = None,
    interrupt_on: dict | None = None,
    checkpointer=None,
    subagents: list | None = None,
    mcp_tools: list | None = None,   # NEW
    skills: list | None = None,       # NEW
):
    m = model or get_model()
    s = get_settings()
    wd = workdir or s.workdir
    if backend is None:
        root = Path(wd).resolve()
        root.mkdir(parents=True, exist_ok=True)
        backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    # Combine our custom tools with MCP tools (if any)
    all_tools = register_custom_tools(tools) + (mcp_tools or [])
    kwargs = dict(
        model=m,
        tools=all_tools,
        system_prompt=system_prompt or build_system_prompt(wd),
        backend=backend,
    )
    subagent_specs = subagents if subagents is not None else [_resolve_subagent_tools(s) for s in SUBAGENTS]
    if subagent_specs:
        kwargs["subagents"] = subagent_specs
    if skills:
        kwargs["skills"] = skills  # list of directory paths
    if interrupt_on:
        kwargs["interrupt_on"] = interrupt_on
        kwargs["checkpointer"] = checkpointer or MemorySaver()
    return create_deep_agent(**kwargs), s
```

**Teaching notes:**
- `mcp_tools` is a list of pre-loaded MCP tool callables (loaded by `load_mcp_tools` before `build_agent` is called). We just concatenate them with our custom tools.
- `skills` is a list of directory paths (e.g. `["./skills"]`). `create_deep_agent` passes them to `SkillsMiddleware`, which loads summaries at startup.
- Both are optional — the agent works without them. This is the graceful-degradation contract.

---

## 13.4 · Tests

### `tests/test_ep13_mcp_skills.py`

```python
import pytest
from pathlib import Path

def test_build_mcp_config_no_env(monkeypatch):
    monkeypatch.delenv("MCP_SERVER_NAME", raising=False)
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    from codeit.mcp_client import build_mcp_config
    assert build_mcp_config() is None

def test_build_mcp_config_with_env(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_NAME", "filesystem")
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
    monkeypatch.setenv("MCP_TRANSPORT", "http")
    from codeit.mcp_client import build_mcp_config
    config = build_mcp_config()
    assert config is not None
    assert "filesystem" in config
    assert config["filesystem"]["url"] == "http://localhost:8000/mcp"
    assert config["filesystem"]["transport"] == "http"

async def test_load_mcp_tools_no_config():
    from codeit.mcp_client import load_mcp_tools
    tools = await load_mcp_tools(None)
    assert tools == []

async def test_load_mcp_tools_unreachable_server(monkeypatch):
    """If the server is down, load_mcp_tools returns [] gracefully (no crash)."""
    from codeit.mcp_client import load_mcp_tools
    config = {"bogus": {"transport": "http", "url": "http://127.0.0.1:1/mcp"}}
    tools = await load_mcp_tools(config)
    assert tools == []  # degraded gracefully

def test_skill_md_has_frontmatter():
    skill_file = Path(__file__).parent.parent / "skills" / "python-testing" / "SKILL.md"
    assert skill_file.is_file()
    content = skill_file.read_text()
    assert content.startswith("---")
    # Frontmatter must have name and description
    assert "name: python-testing" in content
    assert "description:" in content

def test_build_agent_with_skills(tmp_path, monkeypatch):
    """build_agent should accept a skills list and pass it to create_deep_agent."""
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from codeit.agent import build_agent
    skills_dir = Path(__file__).parent.parent / "skills"
    agent, _ = build_agent(
        model=GenericFakeChatModel(messages=iter(["hi"])),
        workdir=str(tmp_path),
        skills=[str(skills_dir)],
    )
    assert agent is not None  # weak assertion; the strong one is that create_deep_agent didn't reject skills=

def test_build_agent_with_mcp_tools(tmp_path, monkeypatch):
    """build_agent should accept mcp_tools and include them in the tool list."""
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain.tools import tool
    from codeit.agent import build_agent

    @tool
    def fake_mcp_tool(query: str) -> str:
        """A fake MCP tool for testing."""
        return f"mcp result for {query}"

    agent, _ = build_agent(
        model=GenericFakeChatModel(messages=iter(["hi"])),
        workdir=str(tmp_path),
        mcp_tools=[fake_mcp_tool],
    )
    assert agent is not None
```

**Test notes:**
- The async tests (`test_load_mcp_tools_no_config`, `test_load_mcp_tools_unreachable_server`) require `pytest-asyncio`. Add it to dev deps if not already present, or use `asyncio.run(...)` in a sync test wrapper.
- `test_load_mcp_tools_unreachable_server` is the graceful-degradation test. Connect to a port nothing's listening on; assert `[]` is returned, not an exception. This is the contract: MCP is optional, never required.
- `test_skill_md_has_frontmatter` enforces the required frontmatter (per the deep-agents-core skill fix). Without `name` and `description`, the skill won't load.
- `test_build_agent_with_mcp_tools` uses a fake `@tool` to simulate an MCP tool — verifies the plumbing without a real MCP server.

---

## 13.5 · Demo script `scripts/demo_ep13.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Start a filesystem MCP server in the background (requires Node/npx)
echo "=== Starting filesystem MCP server (npx) ==="
npx -y @modelcontextprotocol/server-filesystem ./workspace &
MCP_PID=$!
sleep 3  # give it time to start

trap "kill $MCP_PID 2>/dev/null || true" EXIT

echo "=== Demo: agent uses an MCP-provided tool + a skill ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:32b \
CODEIT_WORKDIR=./workspace \
MCP_SERVER_NAME=filesystem \
MCP_SERVER_URL=http://localhost:8000/mcp \
MCP_TRANSPORT=http \
python -c "
import asyncio
from codeit.mcp_client import build_mcp_config, load_mcp_tools
from codeit.agent import build_agent, run

async def main():
    config = build_mcp_config()
    mcp_tools = await load_mcp_tools(config)
    print(f'Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}')
    agent, _ = build_agent(mcp_tools=mcp_tools, skills=['./skills'])
    state = run(agent, 'Add a health endpoint to main.py and write a test for it. Use the python-testing skill if helpful.')
    print('--- final ---')
    print(state['messages'][-1].content if state else 'no state')

asyncio.run(main())
"
echo
echo "Expected: MCP filesystem tools register alongside built-ins; the python-testing skill loads when the agent writes the test."
```

On camera:
1. Start the MCP server in a separate terminal (or background it).
2. Show the agent's tool list now includes MCP-provided tools (different names from our built-ins).
3. Ask the agent to add a health endpoint + test. Watch it load the `python-testing` skill on demand (the skill summary appears in context, then the full body when it decides to use it).
4. Show the MCP tool being called (e.g. a filesystem operation via the MCP server, distinct from our built-in `write_file`).

**Note:** if `npx @modelcontextprotocol/server-filesystem` isn't available or the URL/port differs, adapt. The point is to show *any* MCP server working. A simpler fallback: write a tiny custom MCP server in Python with `fastmcp` — but that's more code. Prefer the npx approach for the demo.

---

## 13.6 · README section to add

```
## Ep 13 — MCP + Skills
Adds: codeit/mcp_client.py (build_mcp_config, load_mcp_tools); skills/python-testing/SKILL.md; mcp_tools + skills args in build_agent
Run: see scripts/demo_ep13.sh (starts an MCP server via npx)
Key idea: langchain-mcp-adapters loads tools from any MCP server; SkillsMiddleware loads skill summaries at startup, full body on demand. Both optional — agent works without them.
New dep: langchain-mcp-adapters
Lines added: 110 SLOC
```

---

## 13.7 · Definition of Done

- [ ] `codeit/mcp_client.py` with `build_mcp_config`, `load_mcp_tools`.
- [ ] `skills/python-testing/SKILL.md` with valid frontmatter.
- [ ] `build_agent` accepts `mcp_tools` and `skills`.
- [ ] `langchain-mcp-adapters` added to `pyproject.toml`.
- [ ] `pytest -m "not live" tests/test_ep13_mcp_skills.py` green (including graceful-degradation tests).
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-12 HEAD -- '*.py'` adds < 220 lines.
- [ ] Ep 1-12 demos still pass.
- [ ] README "Ep 13" section written; new dependency noted.
- [ ] Commit + annotated tag `ep-13`.

---

## 13.8 · What NOT to do this episode

- Do not require an MCP server for the agent to run. MCP is optional; `load_mcp_tools` returns `[]` on no config or unreachable server.
- Do not crash the agent if the MCP server is down. Graceful degradation is the contract.
- Do not forget the YAML frontmatter in `SKILL.md`. Without `name` and `description`, `SkillsMiddleware` ignores the skill (per the deep-agents-core fix).
- Do not write a vague skill description. "Helpful skill" won't trigger; "Use this skill for Python testing tasks with pytest..." will (per the fix).
- Do not load skill *bodies* at startup — only summaries. The full body loads on demand (progressive disclosure). The framework handles this; we just write the `SKILL.md`.
- Do not put skills outside the backend's reach. `SkillsMiddleware` loads skills from the filesystem via the backend. If `skills/` isn't under the backend root or a configured path, it won't load. Verify the path setup during pre-flight.
- Do not try to make MCP tool loading sync. `get_tools()` is async. The CLI (Ep 14) needs an async entry point.

---

## 13.9 · Common gotchas

- **`pytest-asyncio` for async tests:** if not installed, async tests are skipped or error. Add `pytest-asyncio` to dev deps and configure `asyncio_mode = "auto"` in `pyproject.toml`'s `[tool.pytest.ini_options]`, or mark tests with `@pytest.mark.asyncio`.
- **MCP server startup time:** `npx` downloads the package on first run, which can take 10+ seconds. The demo script `sleep 3` may be too short — increase to `sleep 10` or poll the URL until it responds.
- **MCP transport types:** `http` is the easiest for a local server. `stdio` (spawn a subprocess) is also supported by `MultiServerMCPClient` but requires a `command` + `args` config instead of `url`. Pick `http` for the demo; mention `stdio` as the alternative for production.
- **Skills directory location:** if `skills/` is at the repo root but `CODEIT_WORKDIR` is `./workspace`, the backend (rooted at `./workspace`) can't see `skills/`. Either (a) put `skills/` inside the workspace, (b) pass `skills=["./skills"]` as an absolute path the middleware reads directly (verify this is supported), or (c) use a `CompositeBackend` with a route for skills. Simplest for the demo: put skills inside the workspace or pass an absolute path. Verify during pre-flight.
- **`SkillsMiddleware` requires a backend:** per the deep-agents-memory fix, skills need a backend to load from. We have `FilesystemBackend` from Ep 3 — confirm it's the one the middleware uses. If skills are in a separate location, you may need a `CompositeBackend` route.
- **MCP tool name collisions:** if an MCP server exposes a tool named `read_file`, it collides with our built-in. Deep Agents should handle this (the MCP tool wins or the built-in wins, depending on registration order). Verify during pre-flight; if it's a problem, rename via the MCP client's tool-mapping options or exclude the built-in.