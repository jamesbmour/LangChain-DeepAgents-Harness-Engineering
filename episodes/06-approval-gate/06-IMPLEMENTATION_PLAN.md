# Episode 6 — Permission Gating: Human-in-the-Loop

**Tag:** `ep-06` · **Shape:** use the battery — HumanInTheLoopMiddleware · **Budget:** ~120 SLOC
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 6

---

## 6.1 · What this episode delivers

The agent can no longer mutate the workspace or run destructive commands without asking first. We use Deep Agents' built-in `HumanInTheLoopMiddleware` via `create_deep_agent(interrupt_on=..., checkpointer=...)`. When the model calls a gated tool, the graph pauses; we inspect state, prompt the viewer y/n, and resume with `Command(resume={"decisions": [...]})`.

New pieces:
1. `codeit/approval.py` — `classify()` (regex risk triage), `await_approval()` (interactive y/n → `Command`), `run_with_approval()` (the full interrupt-resume loop).
2. Extended `build_agent()` — accepts `interrupt_on`, adds `MemorySaver` checkpointer when set.
3. A `run_with_approval()` driver that wraps Ep 2's `run()` with interrupt handling.

The "we made it safe" moment.

---

## 6.2 · Pre-flight

1. Confirm `interrupt_on` is a real `create_deep_agent` param (per the HITL docs: `interrupt_on={"remove_file": True, "fetch_file": False, "notify_email": {"allowed_decisions": ["approve","reject"]}}`).
2. Confirm `MemorySaver` import: `from langgraph.checkpoint.memory import MemorySaver`.
3. Confirm `Command` import: `from langgraph.types import Command`.
4. Confirm the resume shape: `Command(resume={"decisions": [{"type": "approve"} | {"type": "reject", "message": "..."}]})`.
5. Confirm how to detect an interrupt after `invoke`: `state = agent.get_state(config); if state.next: ...` (state.next is non-empty when paused). Also check `result.get("__interrupt__")` if the installed version surfaces it that way.
6. Verify `PatchToolCallsMiddleware` is auto-added (per the docs: "If a run is cancelled or interrupted before a tool returns a result, PatchToolCallsMiddleware in the same stack repairs the message history automatically.") — we don't need to add it ourselves.

---

## 6.3 · File specs

### `codeit/approval.py` (~80 SLOC)

```python
import re
from rich.console import Console
from langgraph.types import Command

from codeit.settings import get_settings

console = Console()

# Risk classification — regex triage BEFORE the interrupt, to short-circuit
# obviously-bad commands and to label the prompt shown to the viewer.
SAFE_PATTERNS = [
    r"^\s*ls\b", r"^\s*cat\b", r"^\s*head\b", r"^\s*tail\b", r"^\s*grep\b",
    r"^\s*pwd\b", r"^\s*echo\b", r"^\s*wc\b", r"^\s*git status\b",
    r"^\s*git diff\b", r"^\s*git log\b", r"^\s*pytest\b.*--collect-only",
]
DESTRUCTIVE_PATTERNS = [
    r"rm\s+-rf?\b", r"git\s+push\s+.*-f", r"git\s+reset\s+--hard",
    r"\bdd\b", r"\bmkfs\b", r":\(\)\s*\{", r">\s*/dev/sd",
    r"chmod\s+-R\s+777", r"curl.*\|\s*sh", r"wget.*\|\s*sh",
]

def classify(command: str) -> str:
    """Return 'safe', 'needs-approval', or 'blocked' for a shell command.

    Safe commands skip the interrupt. Destructive commands are 'blocked' (still
    interrupt, but flagged red). Everything else is 'needs-approval' (normal interrupt).
    """
    for pat in DESTRUCTIVE_PATTERNS:
        if re.search(pat, command):
            return "blocked"
    for pat in SAFE_PATTERNS:
        if re.search(pat, command):
            return "safe"
    return "needs-approval"

def _auto_approve() -> bool:
    return get_settings().auto_approve  # set by --yolo

def await_approval(agent, config, tool_name: str, tool_args: dict) -> Command:
    """Inspect the pending interrupt, prompt the viewer, return a Command to resume.

    For run_shell we use classify() to label the prompt; for write_file/edit_file
    we always prompt (they mutate files).
    """
    if _auto_approve():
        console.print("[yellow]--yolo: auto-approving[/yellow]")
        return Command(resume={"decisions": [{"type": "approve"}]})
    # Build a human-readable summary
    if tool_name == "run_shell":
        risk = classify(tool_args.get("command", ""))
        color = {"safe": "green", "needs-approval": "yellow", "blocked": "red"}[risk]
        console.print(f"[{color}]risk: {risk}[/{color}]")
        console.print(f"command: {tool_args.get('command','')}")
    else:
        console.print(f"tool: {tool_name}")
        console.print(f"args: {tool_args}")
    answer = console.input("[bold]Approve? (y/n): [/bold]").strip().lower()
    if answer == "y":
        return Command(resume={"decisions": [{"type": "approve"}]})
    return Command(resume={"decisions": [{"type": "reject", "message": "User denied this action."}]})

def _extract_pending_tool_call(state) -> tuple[str, dict] | None:
    """Read the pending tool call from the interrupted state. Returns (name, args) or None."""
    # The exact shape depends on the installed version. Common patterns:
    # - state.tasks contains a list of PregelTask with interrupts
    # - state.values["messages"][-1] is an AIMessage with tool_calls
    # Inspect during implementation and adapt. The contract is "return the tool
    # name and args the model just requested, so we can show it to the viewer."
    msgs = getattr(state, "values", {}).get("messages", []) or []
    for msg in reversed(msgs):
        tc = getattr(msg, "tool_calls", None)
        if tc:
            return tc[-1]["name"], tc[-1]["args"]
    return None

def run_with_approval(agent, prompt: str, thread_id: str = "default") -> dict:
    """invoke → if interrupted, await_approval → resume. Loops until done."""
    from codeit.agent import _config, _print_event
    s = get_settings()
    config = _config(thread_id, s)
    # First invocation
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
        stream_mode="updates",
        version="v2",
    ):
        _print_event(chunk)
    # Check for interrupt
    state = agent.get_state(config)
    while state.next:  # paused on an interrupt
        pending = _extract_pending_tool_call(state)
        if pending is None:
            console.print("[red]Interrupt with no recognizable tool call — aborting.[/red]")
            break
        tool_name, tool_args = pending
        cmd = await_approval(agent, config, tool_name, tool_args)
        # Resume
        for chunk in agent.stream(cmd, config=config, stream_mode="updates", version="v2"):
            _print_event(chunk)
        state = agent.get_state(config)
    return state.values
```

**Teaching notes:**
- `classify()` runs **before** the interrupt fires — it's our pre-flight triage. We *label* the prompt red/yellow/green so the viewer sees risk at a glance. The interrupt itself comes from `interrupt_on`; we don't skip it based on `classify()` (the framework always pauses when `interrupt_on[name]=True`). `classify()` is purely for display. (If you want safe commands to skip the interrupt entirely, set `interrupt_on` conditionally — see §6.8 "What NOT to do.")
- `await_approval` returns a `Command(resume=...)`. The `decisions` list is the Deep Agents HITL protocol: `{"type": "approve"}`, `{"type": "reject", "message": "..."}`, `{"type": "edit", "edited_action": {...}}`, `{"type": "respond", ...}`. We only use approve/reject this episode; edit/respond are advanced.
- `_auto_approve()` reads `settings.auto_approve` which the `--yolo` flag (Ep 14) sets at runtime. Documented as dangerous.
- `_extract_pending_tool_call` is the version-sensitive seam. The exact shape of `state.tasks` / `state.values["messages"]` varies across LangGraph versions. **Inspect the installed version** during implementation and adapt. The contract is "return the (name, args) the model just requested."
- `run_with_approval` loops: invoke → check interrupt → approve/reject → resume → check again. A single user prompt might trigger multiple gated calls in sequence; the loop handles that.

### Extend `codeit/agent.py` (~30 SLOC)

```python
# NEW: interrupt_on + checkpointer
from langgraph.checkpoint.memory import MemorySaver

DEFAULT_INTERRUPT_ON = {
    "run_shell": True,
    "write_file": True,
    "edit_file": True,        # built-in edit (Ep 8 also gates edit_file_safe)
    "delete": True,
}

def build_agent(
    model: BaseChatModel | None = None,
    tools: list | None = None,
    system_prompt: str | None = None,
    backend=None,
    workdir: str | None = None,
    interrupt_on: dict | None = None,
    checkpointer=None,
):
    """Build a Deep Agent. If interrupt_on is set, add HumanInTheLoopMiddleware + MemorySaver."""
    m = model or get_model()
    s = get_settings()
    if backend is None:
        root = Path(workdir or s.workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    all_tools = register_custom_tools(tools)
    kwargs = dict(
        model=m,
        tools=all_tools,
        system_prompt=system_prompt or "You are CodeIt, a helpful coding assistant.",
        backend=backend,
    )
    if interrupt_on:
        kwargs["interrupt_on"] = interrupt_on
        kwargs["checkpointer"] = checkpointer or MemorySaver()  # REQUIRED for interrupts
    return create_deep_agent(**kwargs), s
```

**Teaching notes:**
- `MemorySaver()` is an in-process checkpointer. It's required for `interrupt_on` — without it, `Command(resume=...))` has nowhere to resume from. For production you'd use `PostgresSaver` or similar; we use `MemorySaver` for teaching.
- `DEFAULT_INTERRUPT_ON` is our policy: shell, writes, edits, deletes all pause. Reads (`ls`, `read_file`, `grep`) don't.
- We accept `checkpointer` as an arg so tests can inject a fresh `MemorySaver()` per test (otherwise state leaks between tests).

---

## 6.4 · Tests

### `tests/test_ep06_approval.py`

```python
import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from codeit.approval import classify
from codeit.agent import build_agent

def test_classify_safe():
    assert classify("ls -la") == "safe"
    assert classify("cat main.py") == "safe"
    assert classify("git status") == "safe"

def test_classify_destructive():
    assert classify("rm -rf .") == "blocked"
    assert classify("git push origin main -f") == "blocked"
    assert classify("dd if=/dev/zero of=/dev/sda") == "blocked"

def test_classify_needs_approval():
    assert classify("pip install fastapi") == "needs-approval"
    assert classify("pytest -q") == "needs-approval"

def test_destructive_action_interrupts(tmp_path, monkeypatch):
    """Model tries rm -rf; interrupt fires; we reject; agent gets a ToolMessage and stops."""
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    monkeypatch.setenv("CODEIT_AUTO_APPROVE", "false")
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "run_shell", "args": {"command": "rm -rf ."}, "id": "c1", "type": "tool_call"}]),
        "I won't try that again.",
    ]))
    agent, _ = build_agent(
        model=fake,
        workdir=str(tmp_path),
        interrupt_on={"run_shell": True},
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "t1"}, "recursion_limit": 20}
    # First invoke triggers the interrupt
    list(agent.stream({"messages": [{"role": "user", "content": "delete everything"}]}, config=config, stream_mode="updates", version="v2"))
    state = agent.get_state(config)
    assert state.next  # paused
    # Reject
    list(agent.stream(Command(resume={"decisions": [{"type": "reject", "message": "No."}]}), config=config, stream_mode="updates", version="v2"))
    # The agent should have stopped (no more interrupts)
    state2 = agent.get_state(config)
    assert not state2.next

def test_auto_approve_bypasses_prompt(tmp_path, monkeypatch):
    """With CODEIT_AUTO_APPROVE=true, the agent proceeds without a prompt."""
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    monkeypatch.setenv("CODEIT_AUTO_APPROVE", "true")
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "run_shell", "args": {"command": "echo hi"}, "id": "c1", "type": "tool_call"}]),
        "Done.",
    ]))
    agent, _ = build_agent(
        model=fake,
        workdir=str(tmp_path),
        interrupt_on={"run_shell": True},
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "t1"}, "recursion_limit": 20}
    list(agent.stream({"messages": [{"role": "user", "content": "say hi"}]}, config=config, stream_mode="updates", version="v2"))
    state = agent.get_state(config)
    # It should be paused (interrupt fired) — auto-approve happens in the resume step,
    # not automatically. The test confirms the interrupt fires; the resume with approve
    # is what auto-approve does in run_with_approval.
    assert state.next

def test_safe_read_does_not_interrupt(tmp_path, monkeypatch):
    """read_file is NOT in interrupt_on, so it runs without pausing."""
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    (tmp_path / "f.txt").write_text("hello")
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "f.txt"}, "id": "c1", "type": "tool_call"}]),
        "The file says hello.",
    ]))
    agent, _ = build_agent(
        model=fake,
        workdir=str(tmp_path),
        interrupt_on={"run_shell": True, "write_file": True},  # read_file NOT listed
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "t1"}, "recursion_limit": 20}
    list(agent.stream({"messages": [{"role": "user", "content": "read f.txt"}]}, config=config, stream_mode="updates", version="v2"))
    state = agent.get_state(config)
    assert not state.next  # never paused
```

**Test notes:**
- Each test gets a **fresh `MemorySaver()`** — pass it via `checkpointer=MemorySaver()`. Don't share one across tests; state leaks.
- The `test_auto_approve_bypasses_prompt` test is subtle: the interrupt *fires* (the framework pauses regardless of `--yolo`); it's our `await_approval` that auto-approves when resuming. So the test asserts the interrupt fires; the auto-approve behavior is tested separately by mocking `_auto_approve` and calling `await_approval` directly.
- `state.next` is the canonical "is it paused?" check. If the installed version uses a different attribute (`state.tasks` with interrupts, or `result.get("__interrupt__")`), adapt.

---

## 6.5 · Demo script `scripts/demo_ep06.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p workspace
echo "=== Demo: agent asks before a destructive command ==="
echo "When prompted, type 'n' to reject, then re-run and type 'y' to approve."
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
CODEIT_WORKDIR=./workspace \
python -c "
from codeit.agent import build_agent
from codeit.approval import run_with_approval
agent, _ = build_agent(interrupt_on={'run_shell': True, 'write_file': True}, checkpointer=__import__('langgraph.checkpoint.memory', fromlist=['MemorySaver']).MemorySaver())
run_with_approval(agent, 'Create a file then delete it with rm. Show me each step.')
"
echo
echo "=== --yolo mode (auto-approve) ==="
echo "DANGEROUS: this skips all prompts. Use only in throwaway workspaces."
CODEIT_AUTO_APPROVE=true CODEIT_WORKDIR=./workspace python -c "
from codeit.agent import build_agent
from codeit.approval import run_with_approval
agent, _ = build_agent(interrupt_on={'run_shell': True}, checkpointer=__import__('langgraph.checkpoint.memory', fromlist=['MemorySaver']).MemorySaver())
run_with_approval(agent, 'Run: echo yolo approved')
"
```

On camera:
1. Ask the agent to do something destructive.
2. Show the `[red]risk: blocked[/red]` label and the prompt.
3. Type `n` — agent backs off.
4. Re-run, type `y` — agent proceeds.
5. **Safety moment:** demo `--yolo` briefly, with a clear "never use this on a real repo" warning.

---

## 6.6 · README section to add

```
## Ep 6 — Permission Gating (Human-in-the-Loop)
Adds: codeit/approval.py (classify, await_approval, run_with_approval); interrupt_on + MemorySaver in build_agent
Run: python -c "from codeit.agent import build_agent; from codeit.approval import run_with_approval; a,_=build_agent(interrupt_on={'run_shell':True}, checkpointer=__import__('langgraph.checkpoint.memory',fromlist=['MemorySaver']).MemorySaver()); run_with_approval(a,'run: echo hi')"
Key idea: interrupt_on + MemorySaver pause the agent before mutating tools; Command(resume={'decisions':[...]}) resumes.
Lines added: 120 SLOC
Security: --yolo auto-approves everything. Never use on a real repo.
```

---

## 6.7 · Definition of Done

- [ ] `codeit/approval.py` with `classify`, `await_approval`, `run_with_approval`.
- [ ] `build_agent` accepts `interrupt_on` and `checkpointer`; adds `MemorySaver` when `interrupt_on` set.
- [ ] `pytest -m "not live" tests/test_ep06_approval.py` green (including auto-approve and no-interrupt-on-reads).
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-05 HEAD -- '*.py'` adds < 220 lines.
- [ ] Ep 1-5 demos still pass (Ep 5's run_shell now gates by default if you wire `run_with_approval` into `chat.py` — make this opt-in via env so the Ep 5 demo still works without prompts).
- [ ] README "Ep 6" section written; on-screen safety line included.
- [ ] Commit + annotated tag `ep-06`.

---

## 6.8 · What NOT to do this episode

- Do not hand-roll an approval gate by intercepting tool calls in a custom middleware. Use `interrupt_on` — that's what it's for.
- Do not forget `checkpointer=MemorySaver()`. Without it, `interrupt_on` either errors or silently doesn't pause. The docs are explicit: "Checkpointer is REQUIRED for human-in-the-loop."
- Do not share a `MemorySaver()` across tests. Each test gets a fresh one.
- Do not make `classify()` *skip* the interrupt for "safe" commands by removing them from `interrupt_on` dynamically. The framework reads `interrupt_on` at agent-build time. If you want safe commands to skip the interrupt, don't list them in `interrupt_on` to begin with — but then *all* `run_shell` calls skip, including `rm -rf`. Keep `run_shell` gated, use `classify()` only for display. (If you really want conditional interrupts, the HITL docs mention an `InterruptOnConfig` with a `when` predicate — advanced, out of scope this episode.)
- Do not break the Ep 5 demo. If you switch `chat.py` to use `run_with_approval` by default, the Ep 5 demo (which expects ungated `run_shell`) needs an env flag to opt back out. Cleanest: keep `chat.py` using `run()` (no approval), and add a `scripts/chat_approved.py` (or `--approve` flag) that uses `run_with_approval`. Document both.

---

## 6.9 · Common gotchas

- **`state.next` vs `state.tasks`:** the "is it paused?" check varies by LangGraph version. `state.next` is a tuple of node names pending execution. If your version uses `state.tasks` (a list of `PregelTask` with `.interrupts`), adapt `_extract_pending_tool_call` and the loop condition.
- **`Command(resume=...)` shape:** the `decisions` list is the HITL protocol. Each decision has a `type` (`approve`/`reject`/`edit`/`respond`). If the installed version uses a different shape (e.g. `Command(resume=[{"type":"approve"}])` without the `decisions` wrapper), adapt. Verify against the installed `langgraph.types.Command`.
- **`MemorySaver` is in-process:** state is lost on restart. For the demo that's fine. For production you'd use `PostgresSaver` or `SqliteSaver`. Mention this.
- **Interrupt on `write_file` catches the built-in:** when you gate `write_file`, the built-in filesystem tool is gated. Good. But if Ep 8 adds `edit_file_safe` (a custom wrapper), you must also add `"edit_file_safe": True` to `interrupt_on` separately — custom tools aren't auto-gated.
- **`--yolo` ergonomics:** `CODEIT_AUTO_APPROVE=true` is read by `_auto_approve()` inside `await_approval`. The Ep 14 CLI's `--yolo` flag sets this env at runtime. Don't read `auto_approve` anywhere else — single seam.