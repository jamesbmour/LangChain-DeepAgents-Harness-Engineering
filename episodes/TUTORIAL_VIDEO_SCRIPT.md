# CodeIt Tutorial Video Script
## Build Your Own AI Coding Agent with Deep Agents

**Series:** 15 episodes | **Total runtime:** ~6-8 hours of content
**Prerequisites:** Python 3.11+, Ollama (local) or OpenAI API key

---

## Episode 1: Your Agent's Brain — One File, Two Providers

**Runtime:** ~8 minutes

### Hook (0:00-0:30)
> "What if I told you we could build a coding agent — the kind that writes code, runs tests, and fixes its own bugs — in just 15 small Python files? Today we start with the brain: a model factory that works with both Ollama and OpenAI, switched by environment variables."

### Segment 1: The Settings Dataclass (0:30-2:00)
**On-screen code:** `codeit/settings.py`
- Show the `Settings` dataclass with frozen=True
- Explain each field: `llm_provider`, `llm_model`, `ollama_base_url`, `openai_api_key`, `openai_base_url`
- Show `get_settings()` reading from environment with `load_dotenv()`
- **Key point:** "This is a function, not a singleton — tests can monkeypatch env vars and call it fresh."

**Demo:** Show `.env.example` with all 7 variables

### Segment 2: The Model Factory (2:00-4:30)
**On-screen code:** `codeit/model.py`
- Show `get_model()` using `init_chat_model`
- Walk through the Ollama branch: `model_provider="ollama"`, `base_url=OLLAMA_BASE_URL`
- Walk through the OpenAI branch: validate `OPENAI_API_KEY`, honor `OPENAI_BASE_URL`
- **Key point:** "We pre-build the model ourselves so WE own the error messages — a clear 'OPENAI_API_KEY is required' instead of a cryptic stack trace."

**Demo:** 
```bash
LLM_PROVIDER=ollama LLM_MODEL=qwen3.5:2b python 01-model-factory.py "Say hello in one sentence."
```

### Segment 3: The Agent Builder (4:30-6:00)
**On-screen code:** `codeit/agent.py`
- Show `build_agent()` wrapping `create_deep_agent()`
- Explain: `tools=[]` for now — the harness still ships built-in planning/filesystem middleware, but with no custom tools the agent just talks
- Show `main()` with `agent.invoke()` and printing the last message

**Demo:** Show the agent responding to a simple prompt

### Segment 4: Testing (6:00-7:30)
**On-screen:** `tests/test_ep01_providers.py`
- Show `GenericFakeChatModel` from `langchain_core.language_models.fake_chat_models`
- Show the fake model fixture — no network, deterministic
- Show error path tests: missing OpenAI key, unknown provider

### Wrap-up (7:30-8:00)
> "Today we built the brain. Next episode: we make the loop visible — you'll see every step the agent takes, live on screen."

---

## Episode 2: The Agentic Loop, Made Visible

**Runtime:** ~12 minutes

### Hook (0:00-0:30)
> "A chatbot responds. An agent acts. The difference is the loop: model decides → tool runs → model sees result → model decides again. Today we make that loop visible with streaming."

### Segment 1: The Demo Tool (0:30-2:00)
**On-screen code:** `get_time` @tool
- Show the `@tool` decorator
- **Key point:** "The docstring is FOR THE MODEL. It tells the model WHEN to call this tool and what it returns. Tool docstrings are prompts."
- Show the docstring: "Return the current time. Use this when the user asks for the time."

### Segment 2: The Streaming Driver (2:00-6:00)
**On-screen code:** `run()` function
- Show `agent.stream()` with `stream_mode="updates"` and `version="v2"`
- Explain v2 chunks: dicts with `type`, `ns`, `data`
- Show `_print_event()` rendering each node event with Rich
- **Key point:** `recursion_limit` is a TOP-LEVEL config key, NOT inside `configurable`. It counts super-steps (model call + tool exec ≈ 2).
- Show `MemorySaver()` checkpointer — required for state persistence

**Demo:** Run with "What time is it? Use your tool." — watch the tool call and response stream live

### Segment 3: Error Handling (6:00-7:30)
- Show the broad `Exception` catch in `run()`
- Explain: handles `GraphRecursionError` and model errors gracefully
- Show red error output on failure

### Segment 4: Testing (7:30-10:00)
**On-screen:** `tests/test_ep02_loop.py`
- Show `FakeToolModel` — subclass of `GenericFakeChatModel` that overrides `bind_tools()`
- Show `_model_calling_time_once()` — first message has tool_calls, second is plain text
- Show `_model_looping_forever()` — uses `itertools.cycle` to test recursion limit
- Show `_model_no_tool_call()` — direct answer without tools

### Segment 5: Live Demo (10:00-11:30)
Run the full episode with Ollama:
```bash
LLM_PROVIDER=ollama LLM_MODEL=qwen3.5:2b python 02-agentic-loop.py "What time is it? Use your tool."
```

### Wrap-up (11:30-12:00)
> "The loop is visible. Next episode: we give the agent hands — filesystem tools to read and explore your codebase."

---

## Episode 3: Giving Your Agent Hands — Filesystem Tools

**Runtime:** ~10 minutes

### Hook (0:00-0:30)
> "An agent that can't touch files is just a very expensive chatbot. Today we attach a FilesystemBackend and the agent instantly gets ls, read_file, glob, and grep — zero hand-written tool code."

### Segment 1: The FilesystemBackend (0:30-3:00)
**On-screen code:** `FilesystemBackend(root_dir=..., virtual_mode=True)`
- Show the backend providing built-in tools automatically
- **Key point:** "SECURITY: `virtual_mode=True` is the sandbox. The default `virtual_mode=False` provides NO security even with `root_dir` set. Always pass `virtual_mode=True`."
- Explain: `virtual_mode=True` blocks `../`, `~/`, and absolute paths outside root

**Demo:** Show the agent listing files in the workspace

### Segment 2: Custom Tool — read_summary (3:00-5:00)
**On-screen code:** `read_summary` @tool
- Show how to register a custom tool alongside built-ins
- Explain the truncation policy: first 50 lines + note
- **Key point:** "Tool docstrings are prompts — 'Use this when you want a quick overview of a file without reading the whole thing.'"

### Segment 3: The Checkpointer (5:00-6:00)
- Show `checkpointer=MemorySaver()` added to `create_deep_agent`
- Explain: required for `agent.get_state(config).values` to work
- **Key point:** "Without a checkpointer, you can't retrieve the final state after streaming."

### Segment 4: Live Demo (6:00-8:30)
```bash
LLM_PROVIDER=ollama LLM_MODEL=gemma4:12b CODEIT_WORKDIR=./workspace python 03-filesystem-tools.py "What files are in this project?"
```

### Segment 5: Testing (8:30-9:30)
- Show acceptance test: `ls` lists files, `read_file` returns contents, traversal outside workspace raises

### Wrap-up (9:30-10:00)
> "The agent can now explore your codebase. Next episode: writing files — and why the sandbox matters."

---

## Episode 4: Writing Code — `write_file` + the Workspace Sandbox

**Runtime:** ~8 minutes

### Hook (0:00-0:30)
> "Reading is safe. Writing is where things get interesting. Today the agent creates a real FastAPI app — and we prove the sandbox works."

### Segment 1: The Sandbox Story (0:30-2:00)
- Reinforce `virtual_mode=True`
- Show `resolve_in_workspace(path)` helper — shares sandbox discipline across custom tools
- Explain `PathEscapeError` — raised when a resolved path leaves the workspace

**On-screen code:** `resolve_in_workspace()` function
- Show path resolution: `(root / path).resolve()` then `target.relative_to(root)`
- Explain: blocks `../`, `~/`, and absolute paths outside workspace

### Segment 2: write_file is Built-In (2:00-3:30)
- Show that `write_file` comes from the backend — no custom tool code needed
- **Key point:** "This is the shortest episode by design — the framework does the work. Spend screen time on what the sandbox protects against."

### Segment 3: Live Demo (3:30-6:00)
```bash
CODEIT_WORKDIR=./workspace python 04-write-sandbox.py "Create main.py with a FastAPI app: GET /hello returns {'msg':'hello'}"
```
- Show the file being written
- Verify on disk: `cat workspace/main.py`
- **Key point:** "Real code, written by the agent, in a sandboxed directory."

### Segment 4: Testing (6:00-7:00)
- Show acceptance test: write inside workspace succeeds; write outside raises; round-trip read matches

### Wrap-up (7:00-8:00)
> "The agent can now write code. Next episode: running shell commands — and why that's dangerous without a gate."

---

## Episode 5: Running Commands — The Shell Tool (and Why It's Dangerous)

**Runtime:** ~10 minutes

### Hook (0:00-0:30)
> "An agent that can write files but can't run them is like a chef who can't taste. Today we add a shell tool — and I'll show you why it needs a gate."

### Segment 1: The run_shell Tool (0:30-4:00)
**On-screen code:** `run_shell` @tool
- Show `subprocess.run(command, shell=True, cwd=cwd)`
- Explain: `shell=True` allows pipes/redirects — more dangerous but simpler
- Show output capture: stdout + stderr + exit code
- Show `MAX_OUTPUT_CHARS` truncation (20k chars ≈ 5k tokens)
- **Key point:** "We return non-zero exits as a STRING, not an exception — the model reads '[exit 1]' and can self-correct."

### Segment 2: The Danger (4:00-5:30)
- Show the warning at the end of `main()`:
  > "⚠️ The agent just ran real shell commands. It COULD have run `rm -rf .` and deleted the workspace."
- **Key point:** "run_shell confines CWD, not the process. `rm -rf /` would still try to delete the system if the user has permission. NEVER run unsandboxed on a real repo."

### Segment 3: Live Demo (5:30-7:30)
```bash
CODEIT_WORKDIR=./workspace python 05-shell-tool.py "Run: echo hello from the shell"
```

### Segment 4: Testing (7:30-9:00)
- Show acceptance test: safe command returns expected output and exit code; cwd is the workspace

### Wrap-up (9:00-10:00)
> "The agent can now run commands — but it could also delete everything. Next episode: the approval gate that makes this safe."

---

## Episode 6: Permission Gating — Human-in-the-Loop

**Runtime:** ~14 minutes

### Hook (0:00-0:30)
> "Last episode ended on a cliffhanger: the agent could run `rm -rf`. Today we add the approval gate — the agent asks before it acts."

### Segment 1: interrupt_on + MemorySaver (0:30-3:00)
**On-screen code:** `INTERRUPT_ON` dict and `create_deep_agent` call
- Show `interrupt_on={"run_shell": True, "write_file": True, "edit_file": True, "delete": True}`
- Show `checkpointer=MemorySaver()` — REQUIRED for interrupts
- **Key point:** "Without a checkpointer, Command(resume=...) has nowhere to resume from."

### Segment 2: The classify() Risk Triage (3:00-6:00)
**On-screen code:** `classify()` function
- Show `SAFE_PATTERNS` and `DESTRUCTIVE_PATTERNS` regex lists
- Walk through the logic: destructive → "blocked", safe → "safe", everything else → "needs-approval"
- **Key point:** "classify() is purely for display — the interrupt itself comes from interrupt_on. It labels the prompt red/yellow/green so the viewer sees risk at a glance."

### Segment 3: The Approval Loop (6:00-10:00)
**On-screen code:** `run_with_approval()` function
- Show the flow: `agent.stream()` → `agent.get_state()` → check `state.next` → prompt user → `Command(resume=...)`
- Show the `Command` resume primitive from `langgraph.types`
- Show auto-approve via `CODEIT_AUTO_APPROVE=true` or `--yolo`
- **Key point:** "The resume decisions are `[{'type': 'approve'}]` or `[{'type': 'reject', 'message': '...'}]`"

### Segment 4: Live Demo (10:00-12:30)
```bash
CODEIT_WORKDIR=./workspace CODEIT_AUTO_APPROVE=true python 06-approval-gate.py "Run: echo hello"
```
- Show the approval prompt appearing
- Show auto-approve bypassing the prompt

### Segment 5: Testing (12:30-13:30)
- Show acceptance test: dangerous action triggers interrupt and is rejected; auto-approve lets action through

### Wrap-up (13:30-14:00)
> "The agent now asks permission. Next episode: the system prompt that gives it personality and rules."

---

## Episode 7: The System Prompt — Engineering Personality & Rules

**Runtime:** ~10 minutes

### Hook (0:00-0:30)
> "A good agent isn't just smart — it has personality, rules, and knows the project it's working on. Today we write the system prompt that shapes everything."

### Segment 1: SYSTEM_PROMPT (0:30-4:00)
**On-screen code:** `SYSTEM_PROMPT` constant
- Show the structured prompt: Role, Tool-use policy, Safety, Editing rules, Communication
- **Key point:** "Written for the model, not for the viewer. Imperative, short."
- Show key rules:
  - "Use write_file only for new files. Use edit_file for changes to existing files."
  - "Never run destructive commands without explaining why."
  - "Be concise. Say what you're about to do, do it, then summarize."

### Segment 2: Project Context — AGENTS.md (4:00-6:30)
**On-screen code:** `load_project_context()` function
- Show reading `AGENTS.md` (Deep Agents convention) or `CODEIT.md` fallback
- Explain: graceful degradation — missing file returns empty string
- **Demo:** Create an AGENTS.md with project-specific instructions

```bash
echo '# My project\nUse FastAPI. The bug is in main.py.' > workspace/AGENTS.md
```

### Segment 3: Composing the Prompt (6:30-7:30)
**On-screen code:** `build_system_prompt()` function
- Show `SYSTEM_PROMPT + load_project_context(root)`
- Explain: the harness's own middleware injects tool descriptions on top; we ADD policy, not replace it

### Segment 4: Live Demo (7:30-9:00)
```bash
CODEIT_WORKDIR=./workspace python 07-system-prompt.py "What is this project?"
```

### Wrap-up (9:00-10:00)
> "The agent now has personality and knows the project. Next episode: surgical edits — the search-replace format used by Aider, Codex, and Cline."

---

## Episode 8: Surgical Edits — `edit_file` + the Fuzzy Fallback

**Runtime:** ~14 minutes

### Hook (0:00-0:30)
> "Rewriting a whole file to change one line is wasteful and dangerous. Today we teach the agent the search-replace edit format — and add a fuzzy fallback for when the model's memory of the code is slightly off."

### Segment 1: The Search-Replace Format (0:30-3:00)
**On-screen code:** `edit_file_safe()` function
- Show the three-argument contract: `path`, `search`, `replace`
- **Key point:** "For targeted changes to existing files — never rewrite a whole file to change a few lines."
- Show the docstring explaining three outcomes: exact / fuzzy / fail-with-diff

### Segment 2: The Fuzzy Fallback (3:00-7:00)
**On-screen code:** `_fuzzy_find()` function
- Show `difflib.SequenceMatcher` ratio-based matching
- Explain: O(n*m) scan, capped at 50k chars
- Show threshold: ≥90% similarity triggers fuzzy apply with a note
- Show the failure path: returns a unified diff so the model self-corrects

**Demo:** Show a near-miss scenario where the model's memory of the code is slightly off

### Segment 3: Path Resolution Fix (7:00-8:00)
- Show the `_resolve_in_workspace()` helper
- **Key point:** "We strip leading slashes so `/big.py` is treated as `big.py` relative to root — matching how the backend's write_file works."

### Segment 4: Live Demo (8:00-11:00)
```bash
CODEIT_WORKDIR=./workspace CODEIT_AUTO_APPROVE=true python 08-surgical-edits.py "Create big.py with content 'line 1\nline 2\nline 3\nline 4\nline 5', then change 'line 3' to 'EDITED'"
```
- Show the file being created and edited
- Verify on disk

### Segment 5: Testing (11:00-13:00)
- Show acceptance test: exact match edits correctly; near-miss triggers fuzzy apply; unresolvable search returns a diff

### Wrap-up (13:00-14:00)
> "The agent can now make surgical edits. Next episode: context management — a repo map so the agent doesn't read every file."

---

## Episode 9: Context Management — Repo Map & Token Trimming

**Runtime:** ~12 minutes

### Hook (0:00-0:30)
> "Reading every file in a large codebase burns tokens and time. Today we build a repo map — a compact overview of file paths and function signatures — so the agent can find what it needs without reading everything."

### Segment 1: The Repo Map Tool (0:30-5:00)
**On-screen code:** `build_repo_map()` function
- Show `os.walk` with skip list: `__pycache__`, `.git`, `.venv`, `node_modules`
- Show `ast.parse` extracting top-level `def`/`class` signatures
- Show `MAX_FILES` and `MAX_MAP_CHARS` limits
- **Key point:** "Aider ranks symbols by PageRank — we explicitly DON'T. This is the teaching version."

### Segment 2: Token Estimation & History Trimming (5:00-7:00)
**On-screen code:** `estimate_tokens()` and `trim_history()` functions
- Show the ~4 chars/token heuristic
- Show `trim_history()` dropping oldest messages until under cap
- **Key point:** "Deep Agents' FilesystemMiddleware already does context engineering internally. This is a teaching layer for custom control."

### Segment 3: Live Demo (7:00-9:30)
```bash
CODEIT_WORKDIR=./workspace python 09-repo-map.py "Use build_repo_map to list the files"
```

### Segment 4: Testing (9:30-11:00)
- Show acceptance test: map lists known symbols from sample_app; trim keeps newest turns under token cap

### Wrap-up (11:00-12:00)
> "The agent now has a bird's-eye view. Next episode: planning — teaching the agent to break big tasks into steps."

---

## Episode 10: Planning — TodoList & Task Decomposition

**Runtime:** ~12 minutes

### Hook (0:00-0:30)
> "A smart agent that doesn't plan is just lucky. Today we teach the agent to plan — and show you the built-in TodoList middleware that tracks progress."

### Segment 1: TodoListMiddleware is Built-In (0:30-2:00)
- Show that `TodoListMiddleware` is #1 in the default middleware stack
- Show `write_todos` is automatically available — we don't add it
- **Key point:** "Built-in tool names are reserved. We don't redefine `write_todos`."

### Segment 2: The plan() Sugar Tool (2:00-5:00)
**On-screen code:** `plan()` and `complete_todo()` tools
- Show `plan(steps)` returning a string instructing the model to call `write_todos`
- **Key point:** "Tools can't call other tools directly. We return a STRING instructing the model to call write_todos next; the model then does so."
- Show `complete_todo(id)` — instructs the model to mark a todo as completed
- **Key point:** "write_todos REPLACES the list, doesn't append. The model must pass the full list back with item `id` set to 'completed'."

### Segment 3: Rendering Todos (5:00-6:30)
**On-screen code:** `render_todos()` helper
- Show reading `state["todos"]` or `state["todo_list"]`
- Show pretty-printing with Rich: `[x]` completed, `[>]` in progress, `[ ]` pending
- **Key point:** "NOT a tool — a helper for the CLI to display state."

### Segment 4: Live Demo (6:30-9:00)
```bash
LLM_MODEL=qwen3.5:2b CODEIT_WORKDIR=./workspace python 10-todo-planning.py "Plan a 3-step task: read main.py, find a bug, run tests. Plan it first."
```
- Show the todo list being created and rendered

### Segment 5: Testing (9:00-11:00)
- Show acceptance test: `plan` seeds todos; `complete_todo(id)` flips status

### Wrap-up (11:00-12:00)
> "The agent now plans. Next episode: error recovery — the wow episode where the agent fixes its own bugs."

---

## Episode 11: Error Recovery — Self-Healing Loops

**Runtime:** ~16 minutes

### Hook (0:00-0:30)
> "This is the 'wow' episode. The agent writes code, runs it, reads the failure, and fixes it — autonomously. No human intervention."

### Segment 1: The run_tests Tool (0:30-4:00)
**On-screen code:** `run_tests()` function
- Show `subprocess.run(["pytest", path, "-q", "--tb=short"])`
- Show output capture with `MAX_OUTPUT_CHARS` truncation
- **Key point:** "Non-zero exit returns a STRING, not an exception — the model reads '[exit 1]' and self-corrects."
- Show the docstring: "Use this after editing code with tests, to check your work. If tests fail, read the failure, fix the code, then run_tests again."

### Segment 2: Failure Detection (4:00-7:00)
**On-screen code:** `_detect_test_failure()` function
- Show the heuristic: scan tool messages for `[exit 1]` + `FAILED` or `Error`
- **Key point:** "We only check the LAST run_tests tool result, not all tool messages — to avoid false positives from earlier failures in the conversation history."
- Show the fix: stop scanning when we hit `[exit 0]` or any non-failure tool result

### Segment 3: The Recovery Loop (7:00-11:00)
**On-screen code:** `run_with_recovery()` function
- Show the plain Python loop AROUND the graph (not a new graph node)
- Explain: "We don't fight the framework — we wrap it."
- Show the flow: `run_with_approval()` → `_detect_test_failure()` → re-invoke with failure output appended as a user message
- **Key point:** "The follow-up is a USER message so the model treats it as new input."
- Show `max_retries` cap (default 3)

### Segment 4: Live Demo (11:00-13:30)
```bash
# Create a failing test
echo 'def test_fail(): assert 1 + 1 == 3' > workspace/test_example.py
CODEIT_WORKDIR=./workspace CODEIT_AUTO_APPROVE=true python 11-error-recovery.py "Run the tests. If they fail, fix the code. Then run tests again."
```
- Show the agent detecting the failure, fixing the code, and re-running tests

### Segment 5: Testing (13:30-15:00)
- Show acceptance test: failure detection works; recovery loop re-invokes correctly

### Wrap-up (15:00-16:00)
> "The agent can now self-heal. Next episode: subagents — specialists in isolated context."

---

## Episode 12: Sub-Agents — Specialists in Isolated Context

**Runtime:** ~14 minutes

### Hook (0:00-0:30)
> "A generalist agent is good at everything and great at nothing. Today we create specialists — subagents that run in isolated context and return only a summary."

### Segment 1: SubAgentMiddleware is Built-In (0:30-2:00)
- Show that `SubAgentMiddleware` is #2 in the default middleware stack
- Show `task` tool is automatically available when subagents are configured
- **Key point:** "Built-in tool names are reserved. We don't redefine `task`."

### Segment 2: Subagent Specs (2:00-5:00)
**On-screen code:** `SUBAGENTS` list
- Show "explorer" — read-only subagent: tools = `[build_repo_map]`, no shell, no writes
- Show "tester" — test-runner: tools = `[run_tests, edit_file_safe]`, no shell, no write_file
- **Key point:** "Subagents are STATELESS — give complete instructions in one task call."
- Show system prompts for each subagent

### Segment 3: The spawn_subagent Sugar Tool (5:00-7:00)
**On-screen code:** `spawn_subagent()` function
- Show the docstring: "Use this for large or isolatable subtasks to keep the main context clean."
- **Key point:** "Returns a STRING instructing the model to call task(agent=..., instruction=...). Tools can't call other tools directly — the model orchestrates."
- Show argument: `agent` = 'explorer', 'tester', or 'general-purpose'

### Segment 4: Live Demo (7:00-10:00)
```bash
echo "print('hello')" > workspace/main.py
LLM_MODEL=qwen3.5:2b CODEIT_WORKDIR=./workspace python 12-subagents.py "Use the explorer subagent to map the codebase."
```
- Show the subagent being spawned and returning a summary

### Segment 5: Testing (10:00-12:00)
- Show acceptance test: subagent delegation works; isolated context returns summary

### Wrap-up (12:00-14:00)
> "The agent can now delegate. Next episode: MCP — speaking the standard protocol for tools."

---

## Episode 13: MCP + Skills — Speaking the Standard Protocol

**Runtime:** ~14 minutes

### Hook (0:00-0:30)
> "The agent ecosystem is fragmented — every tool has its own API. MCP, the Model Context Protocol, is the standard that fixes this. Today we connect to an MCP server and load skills."

### Segment 1: MCP Config (0:30-4:00)
**On-screen code:** `build_mcp_config()` function
- Show reading `MCP_SERVER_NAME`, `MCP_TRANSPORT`, `MCP_SERVER_URL` from env
- Show the config dict format: `{name: {"transport": "http", "url": url}}`
- **Key point:** "MCP is OPTIONAL — graceful degradation. If no server is configured, load_mcp_tools returns []."

### Segment 2: Async MCP Loading (4:00-7:00)
**On-screen code:** `load_mcp_tools()` async function
- Show `MultiServerMCPClient` from `langchain_mcp_adapters.client`
- Show `await client.get_tools()` — it's a coroutine
- Show the try/except: server unreachable → degrade gracefully
- **Key point:** "MCP tool loading is ASYNC — `get_tools()` is a coroutine. The demo uses `asyncio.run` to wrap the async loading."

### Segment 3: Skills (7:00-9:00)
- Show `skills` parameter to `create_deep_agent`
- Show `SkillsMiddleware` (built-in, #2 in default stack) loading skill summaries at startup
- Show `SKILL.md` format: YAML frontmatter + markdown instructions
- **Key point:** "Progressive disclosure: context stays small until a skill is needed."

### Segment 4: Live Demo (9:00-11:30)
```bash
python 13-mcp-skills.py --mcp "What MCP tools do you have?"
```
- Show graceful degradation when no MCP server is configured

### Segment 5: Testing (11:30-13:00)
- Show acceptance test: MCP tools load when configured; empty list when not

### Wrap-up (13:00-14:00)
> "The agent now speaks MCP. Next episode: the CLI — shipping CodeIt as a real command."

---

## Episode 14: Shipping CodeIt — A Real CLI with Event Streaming

**Runtime:** ~16 minutes

### Hook (0:00-0:30)
> "All those episodes were building toward this moment. Today we ship CodeIt as a single `codeit` CLI command with a live rich streaming view. The viewer runs `codeit run 'add a health endpoint'` and watches the agent work end-to-end."

### Segment 1: The Typer App (0:30-4:00)
**On-screen code:** `app = typer.Typer(...)` and `run()` command
- Show the CLI flags: `--provider`, `--model`, `--yolo`, `--workdir`, `--mcp`, `--skills`, `--approve`
- Show `_apply_flags()` mutating env so the rest of the code reads settings as usual
- **Key point:** "This is the Ep 1 pattern — env vars are the contract."

### Segment 2: v3 Event Streaming (4:00-8:00)
**On-screen code:** `_run_streaming()` async function
- Show `agent.stream_events(input_msg, version="v3", config=config)`
- Show the v3 interleave API: `stream.interleave("messages", "tool_calls")`
- Show rich Panel rendering for each event
- **Key point:** "We try v3 first and fall back to v2 — resilience across versions. The v3 event-streaming API may not exist in all installed versions."

### Segment 3: The Approval Driver (8:00-11:00)
**On-screen code:** `run_with_approval()` function
- Show the full interrupt/resume loop with rich panels
- Show auto-approve via `--yolo` flag
- **Key point:** "When --approve is set, we use the approval driver. Otherwise, we use the streaming view."

### Segment 4: Live Demo (11:00-13:30)
```bash
python 14-cli-streaming.py "Hello" --provider ollama --model gemma4:12b --workdir ./workspace
```
- Show the rich panel output
- Show the v3 streaming in action

### Segment 5: Testing (13:30-15:00)
- Show acceptance test: CLI parses flags correctly; streaming works

### Wrap-up (15:00-16:00)
> "CodeIt is shipped. Next episode: the stretch — LangSmith observability."

---

## Episode 15: LangSmith Observability Appendix

**Runtime:** ~10 minutes

### Hook (0:00-0:30)
> "You've built a powerful agent. But how do you debug it when things go wrong? Today we add LangSmith tracing — every model call, tool call, and subagent delegation becomes visible in a trace you can explore in the browser."

### Segment 1: LangSmith Tracing (0:30-3:00)
- Show the env vars: `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=codeit-demo`
- **Key point:** "LangSmith tracing is enabled purely by env vars — no code change to the agent itself is required."
- Show `.env.example` entries

### Segment 2: The Trace URL Helper (3:00-5:30)
**On-screen code:** `trace_url_for_run()` and `maybe_print_trace_url()` functions
- Show building the URL: `https://smith.langchain.com/projects/p/{project}/r/{run_id}`
- Show the best-effort nature: "The trace URL shape changes over time. The helper is best-effort."
- Show the fallback: "if the URL doesn't open, direct viewers to the LangSmith dashboard"
- Show the attribute paths tried: `state.run_id`, `state.get("run_id")`

### Segment 3: Live Demo (5:30-8:00)
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=codeit-demo
CODEIT_WORKDIR=./workspace python 15-stretch.py "Run: echo hello"
```
- Show the trace URL (or dashboard hint) at the end

### Segment 4: Testing (8:00-9:00)
- Show acceptance test: trace URL is built when tracing is on; dashboard hint when run_id is missing

### Wrap-up (9:00-10:00)
> "You now have a complete, observable coding agent. From a single file to a full CLI with streaming, planning, recovery, subagents, MCP, and observability — all in 15 episodes."

---

## Series Outro

**Runtime:** ~2 minutes

### What We Built (0:00-0:30)
> "We built CodeIt — a terminal coding agent that can read/write files, run commands, plan tasks, fix its own bugs, delegate to subagents, speak MCP, and ship as a CLI with full observability."

### Key Takeaways (0:30-1:30)
1. **Start small** — Episode 1 is just a model factory. Build incrementally.
2. **Safety first** — virtual_mode=True, interrupt_on, classify() — never skip the gate.
3. **Tool docstrings are prompts** — write them for the model, not for yourself.
4. **Errors return strings** — the model reads "[exit 1]" and self-corrects.
5. **Graceful degradation** — MCP down? Skills missing? The agent still works.

### Next Steps (1:30-2:00)
- Clone the repo: `git clone ...`
- Install: `uv sync`
- Try it: `LLM_PROVIDER=ollama LLM_MODEL=qwen3.5:2b python scripts/chat.py "Hello"`
- Extend it: add your own tools, subagents, and skills

---

## Fixes Applied During Testing

### Episode 3 — Missing Checkpointer
- **Issue:** `build_agent()` didn't include `checkpointer=MemorySaver()`, causing `ValueError: No checkpointer set` when `run()` called `agent.get_state()`.
- **Fix:** Added `from langgraph.checkpoint.memory import MemorySaver` import and `checkpointer=MemorySaver()` to `create_deep_agent()` call.

### Episode 8 — Path Resolution Bug
- **Issue:** `_resolve_in_workspace()` didn't handle absolute paths like `/big.py` — `(root / "/big.py").resolve()` resolved to filesystem root instead of `root/big.py`.
- **Fix:** Added `clean = path.lstrip("/")` to strip leading slashes before resolution.

### Episode 11 — False Positive Test Failure Detection
- **Issue:** `_detect_test_failure()` scanned ALL tool messages, causing false positives when earlier failures were in conversation history but the latest run succeeded.
- **Fix:** Modified to only check the LAST tool result, stopping when `[exit 0]` is found.
