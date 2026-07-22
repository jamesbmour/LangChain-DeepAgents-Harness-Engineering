# Episode 14 — Shipping CodeIt: a Real CLI with Event Streaming

**Tag:** `ep-14` · **Shape:** customize — Typer + rich · **Budget:** ~160 SLOC (may split into `ep-14` + `ep-15`)
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 14

---

## 14.1 · What this episode delivers

The finale. Everything from Eps 1-13 ships as a single `codeit` CLI command with a live `rich` streaming view. The viewer runs `codeit "add a health endpoint and test it"` and watches the agent work end-to-end: tokens stream, tool calls render in their own panel, subagent delegations appear, the todo list updates, approval prompts fire when needed.

New pieces:
1. `codeit/cli.py` — a `typer` app with flags (`--provider`, `--model`, `--yolo`, `--workdir`, `--mcp`, `--skills`) and a `rich` live-streaming view built on Deep Agents' **event-streaming API** (`agent.stream_events(..., version="v3")` with typed projections: `.messages`, `.tool_calls`, `.subagents`).
2. Async entry point (MCP tool loading from Ep 13 is async; the CLI must be async too).
3. A capstone demo: the agent builds a small complete app hands-free.

**Pre-split flag:** if CLI + event-streaming view push past 220 SLOC, split into `ep-14` (CLI scaffolding + sync `run` view) and `ep-15` (event-streaming view + capstone). The parent plan allows this.

---

## 14.2 · Pre-flight

1. Confirm the event-streaming API (per the Deep Agents event-streaming doc):
   ```python
   stream = agent.stream_events(input, version="v3")
   for message in stream.messages: ...
   for call in stream.tool_calls: ...
   for subagent in stream.subagents: ...
   # Sync interleave for ordered output:
   for name, item in stream.interleave("messages", "subagents"): ...
   ```
   This is the v0.6+ typed-projection API. Verify the installed `deepagents` version supports `stream_events` with `version="v3"`. If not, fall back to `agent.stream(..., stream_mode="updates", version="v2")` from Ep 2 — less rich, but works.
2. Confirm `typer` is installed and the `typer.testing.CliRunner` API for tests.
3. Confirm `rich`'s `Live` rendering for the streaming view: `from rich.live import Live; with Live(renderable): ...`.
4. Decide: sync or async CLI? MCP tool loading is async (Ep 13), so the CLI's `main` must be async (or wrap the async loading in `asyncio.run`). `typer` supports async commands via `asyncio.run` in the callback. Go with: async `main` that does MCP loading + agent build + streaming, wrapped in `asyncio.run` at the entry point.

---

## 14.3 · File specs

### `codeit/cli.py` (~140 SLOC)

```python
import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from codeit.settings import get_settings
from codeit.model import get_model
from codeit.agent import build_agent
from codeit.approval import run_with_approval, classify, _auto_approve
from codeit.tools import render_todos
from codeit.mcp_client import build_mcp_config, load_mcp_tools

console = Console()
app = typer.Typer(help="CodeIt — your terminal coding agent.")

def _apply_flags(provider: str | None, model: str | None, yolo: bool, workdir: str | None):
    """Mutate env based on CLI flags so the rest of the code reads settings as usual."""
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if model:
        os.environ["LLM_MODEL"] = model
    if yolo:
        os.environ["CODEIT_AUTO_APPROVE"] = "true"
    if workdir:
        os.environ["CODEIT_WORKDIR"] = str(Path(workdir).resolve())

async def _run_streaming(agent, prompt: str, thread_id: str):
    """Drive the agent with event streaming; render each projection in a rich panel."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": get_settings().max_iters * 2}
    input_msg = {"messages": [{"role": "user", "content": prompt}]}
    # Try the v3 event-streaming API; fall back to v2 updates if unavailable
    try:
        stream = agent.stream_events(input_msg, version="v3", config=config)
    except (TypeError, AttributeError):
        # Fallback: v2 updates stream (from Ep 2)
        for chunk in agent.stream(input_msg, config=config, stream_mode="updates", version="v2"):
            _render_v2_chunk(chunk)
        return
    # v3: interleave messages and tool_calls for ordered output
    for name, item in stream.interleave("messages", "tool_calls"):
        if name == "messages":
            console.print(Panel(Text(item.text, style="cyan"), title="assistant", border_style="blue"))
        elif name == "tool_calls":
            color = "green" if item.error is None else "red"
            console.print(Panel(Text(f"{item.tool_name}({item.input})", style=color), title="tool", border_style="magenta"))
    # Render final todos if present
    final_state = agent.get_state(config).values
    todos_str = render_todos(final_state)
    if todos_str and todos_str != "(no todos)":
        console.print(Panel(Text(todos_str), title="todos", border_style="yellow"))

def _render_v2_chunk(chunk):
    """Fallback renderer for v2 updates stream (used if v3 unavailable)."""
    from langchain_core.messages import AIMessage
    kind = chunk.get("type")
    if kind == "updates":
        for node_name, state in chunk.get("data", {}).items():
            msgs = state.get("messages", []) if isinstance(state, dict) else []
            for msg in msgs:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        console.print(Panel(Text(f"{tc['name']}({tc['args']})", style="magenta"), title="tool", border_style="magenta"))
                elif isinstance(msg, AIMessage) and msg.content:
                    console.print(Panel(Text(msg.content, style="cyan"), title="assistant", border_style="blue"))

@app.command()
def run(
    prompt: str = typer.Argument(..., help="The task for the agent."),
    provider: str = typer.Option(None, "--provider", "-p", help="ollama | openai"),
    model: str = typer.Option(None, "--model", "-m", help="model name, e.g. qwen2.5-coder:7b"),
    yolo: bool = typer.Option(False, "--yolo", "-y", help="DANGEROUS: auto-approve all actions."),
    workdir: str = typer.Option(None, "--workdir", "-w", help="workspace directory"),
    mcp: bool = typer.Option(False, "--mcp", help="connect to MCP server from env"),
    skills: bool = typer.Option(False, "--skills", help="load skills from ./skills"),
    approve: bool = typer.Option(False, "--approve", help="enable HITL approval gate"),
):
    """Run CodeIt on a task."""
    _apply_flags(provider, model, yolo, workdir)
    asyncio.run(_async_main(prompt, mcp, skills, approve))

async def _async_main(prompt: str, mcp: bool, skills: bool, approve: bool):
    mcp_tools = await load_mcp_tools(build_mcp_config()) if mcp else []
    skills_paths = ["./skills"] if skills else None
    interrupt_on = {"run_shell": True, "write_file": True, "edit_file": True, "edit_file_safe": True, "delete": True} if approve else None
    agent, _ = build_agent(mcp_tools=mcp_tools, skills=skills_paths, interrupt_on=interrupt_on)
    if approve:
        # Use the approval-aware driver from Ep 6
        run_with_approval(agent, prompt, thread_id="cli")
    else:
        await _run_streaming(agent, prompt, thread_id="cli")

def main():
    app()

if __name__ == "__main__":
    main()
```

**Teaching notes:**
- `typer` gives us flags + help for free. `rich`'s `Panel` gives us the boxed output that looks like Claude Code / `dcode`.
- `_apply_flags` mutates env so the rest of the code (which reads `get_settings()`) sees the flags. This keeps the env-as-config pattern from Ep 1 consistent.
- `_run_streaming` tries the v3 event-streaming API first, falls back to v2 updates if unavailable. The fallback path reuses Ep 2's chunk renderer. Resilience across Deep Agents versions.
- `stream.interleave("messages", "tool_calls")` gives ordered output — messages and tool calls appear in execution order, not in separate iterators. This is the sync interleave from the event-streaming doc.
- `--yolo` sets `CODEIT_AUTO_APPROVE=true` (Ep 6 reads this). `--approve` turns on the HITL gate (uses `run_with_approval`); without `--approve`, the agent runs ungated via `_run_streaming`. The viewer picks.
- `--mcp` and `--skills` are opt-in. Without them, no MCP, no skills — the agent works with just our custom tools.
- The capstone demo uses all flags: `codeit --mcp --skills --approve "add a health endpoint and test it"`.

### `pyproject.toml` — add the console script

```toml
[project.scripts]
codeit = "codeit.cli:main"
```

So `pip install -e .` (or `uv run`) makes `codeit` available on PATH.

---

## 14.4 · Tests

### `tests/test_ep14_cli.py`

```python
import pytest
from typer.testing import CliRunner
from codeit.cli import app

runner = CliRunner()

def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "CodeIt" in result.stdout

def test_run_help():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--provider" in result.stdout
    assert "--yolo" in result.stdout
    assert "--mcp" in result.stdout

def test_apply_flags(monkeypatch):
    from codeit.cli import _apply_flags
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    _apply_flags(provider="openai", model="gpt-4o-mini", yolo=True, workdir="/tmp/ws")
    import os
    assert os.environ["LLM_PROVIDER"] == "openai"
    assert os.environ["LLM_MODEL"] == "gpt-4o-mini"
    assert os.environ["CODEIT_AUTO_APPROVE"] == "true"
    assert os.environ["CODEIT_WORKDIR"] == "/tmp/ws"

def test_run_with_fake_model(tmp_path, monkeypatch):
    """End-to-end CLI run with a fake model injected via env. Asserts exit 0 + output."""
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    # We need to inject a fake model. The CLI reads get_model() which reads env.
    # For a deterministic test, monkeypatch codeit.model.get_model to return a fake.
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    import codeit.cli as cli_mod
    import codeit.model as model_mod
    monkeypatch.setattr(model_mod, "get_model", lambda *a, **kw: GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "ls", "args": {"path": "."}, "id": "c1", "type": "tool_call"}]),
        "Done.",
    ])))
    result = runner.invoke(app, ["run", "list files", "--workdir", str(tmp_path)])
    assert result.exit_code == 0
    # Output should mention the assistant's final message or the tool call
    assert "Done" in result.stdout or "ls" in result.stdout

def test_run_yolo_flag_sets_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    import os
    monkeypatch.delenv("CODEIT_AUTO_APPROVE", raising=False)
    # We can't easily run the full agent in a test, but we can assert --yolo sets env
    # via _apply_flags (tested above). This test is a placeholder for a fuller integration.
    pass
```

**Test notes:**
- `test_run_with_fake_model` is the integration test. It monkeypatches `codeit.model.get_model` to return a `GenericFakeChatModel` so the CLI doesn't need a real LLM. The `CliRunner` captures stdout; assert the assistant's final message or the tool call appears.
- `CliRunner.invoke` runs the command synchronously. Our CLI wraps async in `asyncio.run`, so this works.
- If the v3 event-streaming API isn't available in the installed version, `_run_streaming` falls back to v2 — the test should pass either way. Verify by running the test against the installed version.
- The `--approve` path uses `run_with_approval` which prompts for input — hard to test with `CliRunner` (it doesn't handle interactive stdin well). Test `run_with_approval` separately (Ep 6 tests) and leave the CLI `--approve` path as a manual demo.

---

## 14.5 · Demo script `scripts/demo_ep14.sh` (the capstone)

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p workspace

# Start an MCP server in the background if available
if command -v npx &>/dev/null; then
    npx -y @modelcontextprotocol/server-filesystem ./workspace &
    MCP_PID=$!
    sleep 5
    export MCP_SERVER_NAME=filesystem
    export MCP_SERVER_URL=http://localhost:8000/mcp
    export MCP_TRANSPORT=http
    trap "kill $MCP_PID 2>/dev/null || true" EXIT
    MCP_FLAG="--mcp"
else
    MCP_FLAG=""
fi

echo "=== Capstone: CodeIt builds a small app hands-free ==="
echo "This uses MCP tools, skills, subagents, planning, and the approval gate."
echo "Watch the rich panels: assistant (blue), tool calls (magenta), subagents (green), todos (yellow)."
echo
codeit run "Build a small FastAPI app: create main.py with a /hello endpoint and a /health endpoint, write a test for each, run the tests, and fix any failures. Plan it first, delegate exploration to the explorer subagent, and use the python-testing skill when writing tests." \
    --provider ollama --model qwen2.5-coder:32b \
    --workdir ./workspace \
    --skills \
    $MCP_FLAG \
    --approve
echo
echo "=== Result ==="
ls -la workspace/
echo
echo "If the agent asked for approval on writes/shell, you approved them. This is the HITL gate from Ep 6."
echo "Use a larger model (qwen2.5-coder:32b) or OpenAI for the capstone — small models can't hold the whole plan."
```

On camera (the finale):
1. Run the capstone command.
2. Watch the rich panels fill in real time: assistant message, tool call to `plan`, todo list appearing, delegation to `explorer` subagent (in its own panel), `write_file` calls with approval prompts, `run_tests`, recovery if tests fail, final summary.
3. Open `workspace/main.py` — real code, written by the agent.
4. Run the tests yourself — they pass.
5. Close: "That's CodeIt. Every episode built one piece of this. The framework did the heavy lifting; we configured, customized, and extended."

---

## 14.6 · README section to add

```
## Ep 14 — Shipping CodeIt (CLI + event streaming)
Adds: codeit/cli.py (typer app, rich event-streaming view, all flags); pyproject.toml console_script
Run: codeit run "your task" --provider ollama --model qwen2.5-coder:7b --workdir ./workspace
Capstone: codeit run "build a small FastAPI app..." --mcp --skills --approve
Key idea: agent.stream_events(version='v3') gives typed projections (messages, tool_calls, subagents); rich Panels render each. Falls back to v2 updates if v3 unavailable.
Lines added: 160 SLOC (split into ep-14 + ep-15 if over budget)
Install: pip install -e . (makes codeit available on PATH)
```

---

## 14.7 · Definition of Done

- [ ] `codeit/cli.py` with `typer` app, `rich` panels, all flags (`--provider`, `--model`, `--yolo`, `--workdir`, `--mcp`, `--skills`, `--approve`).
- [ ] `pyproject.toml` has `[project.scripts] codeit = "codeit.cli:main"`.
- [ ] `pip install -e .` makes `codeit` available on PATH.
- [ ] `pytest -m "not live" tests/test_ep14_cli.py` green (including `test_run_with_fake_model`).
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-13 HEAD -- '*.py'` adds < 220 lines. **If over, split into ep-14 + ep-15.**
- [ ] Ep 1-13 demos still pass (the underlying `codeit.*` modules haven't changed).
- [ ] README "Ep 14" section written; install instructions included.
- [ ] Capstone demo runs end-to-end on camera.
- [ ] Commit + annotated tag `ep-14` (or `ep-14` + `ep-15`).

---

## 14.8 · What NOT to do this episode

- Do not require `--mcp` or `--skills` for the CLI to work. Both are opt-in; the agent runs fine without them.
- Do not require `--approve`. The default is ungated streaming (matches Ep 2's behavior). `--approve` turns on the HITL gate.
- Do not crash if the v3 event-streaming API is unavailable. Fall back to v2 updates (the `_render_v2_chunk` path).
- Do not make the CLI only async. `typer` commands are sync; we wrap the async work in `asyncio.run` inside the command callback.
- Do not put interactive `input()` prompts in `_run_streaming` (the ungated path). Only `run_with_approval` (the `--approve` path) prompts. Mixing them confuses viewers.
- Do not forget the `[project.scripts]` entry in `pyproject.toml`. Without it, `codeit` isn't on PATH and the demo fails.
- Do not weaken the sandbox for the capstone. `--workdir ./workspace` keeps it confined; `virtual_mode=True` (Ep 4) still applies.

---

## 14.9 · Common gotchas

- **`stream_events` version:** the doc shows `version="v3"`. If the installed LangGraph/Deep Agents only supports v2, `stream_events` may not exist or may reject v3. The fallback to `agent.stream(stream_mode="updates", version="v2")` (Ep 2's path) covers this. Test both paths.
- **`stream.interleave(...)` sync vs async:** the event-streaming doc shows `stream.interleave("messages", "subagents")` for sync and `asyncio.gather` for async. We use sync interleave in `_run_streaming` (simpler). If the installed version only exposes async interleave, switch to `async def _run_streaming` and `async for`.
- **`rich.Live` vs sequential `console.print`:** `Live` redraws a single renderable (good for a progress bar); sequential `print` appends (good for a log). We use sequential `console.print(Panel(...))` — the panels stack up like a transcript. If you want a single updating view, use `Live`. The transcript style is more teachable.
- **`CliRunner` and interactive approval:** `CliRunner` doesn't handle interactive `input()` well. The `--approve` path (which uses `run_with_approval` and prompts y/n) isn't easily testable via `CliRunner`. Test `run_with_approval` in Ep 6's tests; leave the CLI `--approve` path as a manual demo.
- **`asyncio.run` inside typer:** `typer` commands are sync, so we call `asyncio.run(_async_main(...))` inside the command callback. Don't make the command itself async (typer doesn't support that directly without extra setup).
- **`--yolo` + `--approve` together:** `--yolo` sets `CODEIT_AUTO_APPROVE=true`, which `run_with_approval` reads to auto-approve. So `--yolo --approve` runs gated but auto-approves everything. Document this as "dangerous mode" on camera.
- **Capstone model size:** the capstone uses planning, subagents, recovery — all unreliable on small models. Mandate `qwen2.5-coder:32b` or OpenAI for the capstone demo. State this on camera and in README.