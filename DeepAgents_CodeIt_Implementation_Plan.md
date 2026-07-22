# CodeIt-LangChain — Build Your Own AI Coding Agent with Deep Agents

## Implementation Plan for a 14-Episode YouTube Tutorial Series

**Audience:** the coding agent (Claude Code / opencode) that will generate the entire codebase for the JUST CODE IT tutorial series.
**Companion source:** `CodeIt_Implementation_Plan.md` — the Pydantic AI version. This plan re-platforms the same mission onto **LangChain Deep Agents**, taking the "Deep Agents throughout" pedagogical route the user chose.

---

## 0 · How to use this plan

You (the coding agent) will build one Python project, **CodeIt**, a terminal coding agent, incrementally across 14 episodes (plus an optional 15th). Work **strictly in order**, `ep-01` → `ep-14`. After each episode: run its acceptance test, run its demo command, commit, and create an annotated git tag. **Do not start an episode until the previous one's tag exists and its tests pass.**

This is a _teaching_ codebase. Optimize for clarity and small, legible diffs over cleverness. The viewer types this code on screen — every line must be explainable in one sentence.

Because Deep Agents is a batteries-included harness, episodes take one of three shapes:

- **"Use the battery"** — configure built-in middleware (Filesystem, TodoList, SubAgent, HITL) and demo what it does.
- **"Customize the battery"** — swap a backend, add a custom subagent, change approval rules.
- **"Add a new tool"** — write a `@tool` and register it (shell, tests, repo-map).

Every episode states which shape it is.

---

## 1 · Mission & final deliverables

Build a working, provider-agnostic coding agent that reads/writes/edits files via a sandboxed backend, runs shell commands under a permission gate, plans with TodoList, recovers from its own errors, spawns sub-agents, speaks MCP, loads skills, and ships as a streaming CLI — the same architecture as Claude Code / Codex / Cline / **Deep Agents Code (`dcode`)**, stripped to its teachable essence.

Deliverables at the end of the series:

1. A `codeit/` Python package (the harness) plus `tests/`, `examples/`, and tooling.
2. **14 annotated git tags** `ep-01`…`ep-14` (+ optional `ep-15`), each checkoutable and runnable.
3. A per-episode `README` section and a reproducible demo script per episode.
4. Every runnable entrypoint works against **Ollama (local)** and **OpenAI** by changing env vars only, via `init_chat_model()`.

---

## 2 · Operating rules (the contract — do not violate)

1. **Line budget:** each episode adds **< 220 lines of net-new Python** (SLOC = non-blank, non-comment `.py` lines, measured as `git diff <prev-tag> HEAD -- '*.py'` additions). Docstrings count; blank lines and comments do not. If an episode would exceed 220, **stop and flag for a split** — never compress to squeak under. Pre-flagged tight episodes: Ep 8 (edit customization), Ep 11 (recovery loop), Ep 14 (CLI + streaming).
2. **Always runnable:** at every tag, that episode's demo command must run end-to-end, and **every prior episode's demo must still run** — except where an episode explicitly supersedes earlier behavior.
3. **Provider-agnostic:** never hardcode a provider. All model access goes through `codeit/model.py` using `init_chat_model()` from `langchain.chat_models`. Switching Ollama↔OpenAI is env-only.
4. **Deterministic tests, live demos:** unit/acceptance tests must not call a real LLM — use `GenericFakeChatModel` from `langchain_core.language_models.fake_chat_models` with scripted `AIMessage` + `ToolCall` sequences. Tools are tested by direct calls against fixtures. The on-camera _demo_ uses a real model.
5. **Safety first:** filesystem access is confined via `FilesystemBackend(root_dir=..., virtual_mode=True)`. From Ep 6 on, mutating/dangerous actions pass through `HumanInTheLoopMiddleware` (`interrupt_on`). Every episode that touches shell/write repeats the on-screen warning: never run unsandboxed on a real repo.
6. **Pin, then verify:** pin exact dependency versions in `pyproject.toml`. The LangChain ecosystem iterates fast — **before writing agent code, verify the exact import paths and class names against the installed version** (`python -c "import deepagents; print(deepagents.__version__)"` and check the installed package's modules). Prefer the version's documented API over the names sketched in this plan if they differ. Consult the `langchain-docs` MCP server or `https://docs.langchain.com/llms.txt` for current signatures.
7. **Definition of done per episode:** budget respected → acceptance test green → demo command succeeds on both providers (or on Ollama with an OpenAI note) → README section written → demo script added → commit → annotated tag.

---

## 3 · Architectural decisions (read before coding)

- **The agentic loop is LangGraph, made visible via `agent.stream()`.** Deep Agents' `create_deep_agent()` returns a compiled LangGraph graph that already runs the request→tool→result cycle. To _teach_ the loop without abandoning the framework, Ep 2 drives the agent with `agent.stream(..., stream_mode="updates", subgraphs=True, version="v2")` so each node (model request, tool call, tool result) is observable and printed on screen. This is the honest "transparency" choice: idiomatic Deep Agents, but the viewer sees every turn. Do **not** hand-roll a raw completions loop or bypass the graph.
- **Deep Agents throughout.** Every episode uses `create_deep_agent()`. We do _not_ teach LangChain's `create_agent` or raw LangGraph first. The pedagogy is "configure, then customize, then extend" — not "build from scratch, then adopt the harness."
- **Workspace sandbox via `FilesystemBackend`.** `FilesystemBackend(root_dir=CODEIT_WORKDIR, virtual_mode=True)` confines the agent to the workspace. `virtual_mode=True` blocks `../` and `~/` escapes. The built-in filesystem tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) run through this backend. Path traversal outside the workspace raises.
- **Tool docstrings are prompts.** A `@tool`'s docstring and typed args become the schema/description the model sees. Write docstrings _for the model_ (what it does, when to use it, argument meaning), kept to 1–3 lines so they don't eat the budget.
- **Config is a singleton.** `get_settings()` reads `.env` once via `python-dotenv`; everything imports from there.
- **Built-in tool names are reserved.** Deep Agents owns `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `write_todos`, `task`. Do not redefine these. New tools get new names (`run_shell`, `run_tests`, `build_repo_map`).

---

## 4 · Target repo layout (end state after Ep 14)

Build toward this; each episode adds the files noted in its spec.

```
codeit/
  __init__.py
  settings.py        # Ep1  env/config singleton
  model.py           # Ep1  provider factory (Ollama | OpenAI) via init_chat_model
  agent.py           # Ep1  build_agent(); Ep2 stream driver; Ep11 recovery loop
  prompts.py         # Ep7  system prompt + AGENTS.md project-context loader
  approval.py        # Ep6  classify() + custom interrupt helpers (HITL sugar)
  repomap.py         # Ep9  repo map + token estimate + history trim
  testing.py         # Ep11 run_tests tool
  mcp_client.py      # Ep13 MCP server connection config
  cli.py             # Ep14 typer app + rich event-streaming view
  tools/
    __init__.py      # register_custom_tools(agent_config) helper
    shell.py         # Ep5  run_shell
tests/
  test_ep01_providers.py ... test_ep14_cli.py
  conftest.py         # GenericFakeChatModel fixture, tmp workspace fixture
examples/
  sample_app/        # a small target project so demos are reproducible on camera
scripts/
  chat.py            # Ep1 one-shot demo
  demo_ep02.sh ...   # reproducible per-episode demos
.env.example
pyproject.toml
README.md
DeepAgents_CodeIt_Implementation_Plan.md   # this file
```

Note: no `codeit/tools/files.py` and no `codeit/paths.py` — the filesystem and the sandbox live inside `FilesystemBackend`, provided by the framework. This is the single biggest structural change vs. the Pydantic AI plan.

---

## 5 · Tech stack, pinned deps & env contract

**Runtime:** Python 3.11+. Use `uv` (or venv+pip). **Pin exact versions** in `pyproject.toml`:

- `deepagents` (pin the latest stable at build time; record the version in README)
- `langchain>=1.0,<2.0`, `langchain-core>=1.0,<2.0`, `langgraph>=1.0,<2.0`, `langsmith>=0.3.0`
- `langchain-ollama` (Ollama provider)
- `langchain-openai` (OpenAI provider)
- `python-dotenv`, `rich`, `typer`
- dev: `pytest`, `ruff`

(The `deepagents` package installs `langgraph` transitively, but pin it explicitly for clarity and reproducibility.)

**`.env.example` (the full contract — `model.py` reads exactly these):**

```
LLM_PROVIDER=ollama              # ollama | openai
LLM_MODEL=qwen2.5-coder:7b       # e.g. gpt-4o-mini for openai
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=                  # required only when LLM_PROVIDER=openai
OPENAI_BASE_URL=                 # optional override
CODEIT_WORKDIR=./workspace
CODEIT_AUTO_APPROVE=false        # --yolo flag sets true at runtime
CODEIT_MAX_ITERS=25              # agent recursion_limit guard
```

**Recommended follow-along models:** default `qwen2.5-coder:7b` (runs ~8 GB, supports tool calls — verify it). For the heavier planning/sub-agent/recovery episodes, tell viewers to use `qwen2.5-coder:32b` / `qwen3-coder:30b` / `devstral`, or OpenAI. State on camera that small local models lose the thread on multi-file/planning work.

---

## 6 · Global coding standards

- Full type hints on every public function; `ruff format` + `ruff check` clean.
- Short docstrings on public functions; **tool functions get model-facing docstrings** (see §3).
- No dead code, no TODOs left in shipped tags, no commented-out blocks.
- Prefer pure functions for tools (easy to unit-test by direct call).
- Errors return a clear string to the model (so it can self-correct) rather than crashing the loop — but never silently swallow: log to stderr via `rich`.
- Keep prompt _text_ in `prompts.py` as module constants (text is cheap against the line budget; logic is not).
- Do not import directly from `langchain_community` for anything covered by a dedicated package; prefer `langchain-ollama`, `langchain-openai`.

---

## 7 · Phase 0 — Repo bootstrap (tag `ep-00-setup`, not a video)

Prepare the ground so Ep 1 stays under budget:

1. `git init`; create `pyproject.toml` with pinned deps, `ruff`/`pytest` config, and a `codeit` package skeleton (`__init__.py` only).
2. Add `.env.example`, `.gitignore` (ignore `.env`, `workspace/`, `__pycache__`), `README.md` skeleton with a "Pinned versions" section.
3. Create `examples/sample_app/` — a tiny, buggy-on-purpose Python app (e.g., a 3-file FastAPI "todo" service with one failing test) used as the demo target throughout. Keep it out of the SLOC budget (it's fixture, not taught code).
4. Create empty `tests/` with a `conftest.py` that provides a `fake_model` fixture (wraps `GenericFakeChatModel` with scripted `AIMessage`+`ToolCall` sequences) and a `tmp_workspace` fixture (creates a temp dir, points `CODEIT_WORKDIR` at it).
5. Commit, tag `ep-00-setup`. **This is a config/scaffold commit — it does not count against any episode's 220-line code budget.**

---

## 8 · Episode specifications

Each episode follows the same structure. Signatures are the contract; **adapt names only to match the pinned Deep Agents API** (§2.6). Verify against `/oss/python/deepagents/*.mdx` via the `langchain-docs` MCP server before writing each episode.

---

### Ep 1 — Your Agent's Brain: One File, Two Providers _(shape: add a tool-free agent)_

- **Tag:** `ep-01` · **Budget:** ~90 SLOC (settings ~35, model factory ~25, agent build ~15, demo ~15)
- **New:** `codeit/settings.py`, `codeit/model.py`, `codeit/agent.py`, `scripts/chat.py`
- **API:**
  - `get_settings() -> Settings` — loads `.env` once; fields per §5.
  - `get_model(settings: Settings | None = None) -> BaseChatModel` — uses `init_chat_model(model=..., model_provider=..., base_url=...)`. `openai` → `ChatOpenAI` (honor `OPENAI_BASE_URL` if set); `ollama` → `ChatOllama` pointed at `OLLAMA_BASE_URL`. (Confirm exact `init_chat_model` signature against installed `langchain.chat_models`.)
  - `build_agent(settings=None, tools=None, system_prompt: str | None=None) -> CompiledGraph` — wraps `create_deep_agent(model=..., tools=tools or [], system_prompt=...)`. Returns the compiled graph.
  - `scripts/chat.py` — build model, build agent, `agent.invoke({"messages": [{"role":"user","content":prompt}]}, config={"configurable":{"thread_id":"demo"}})`, print last message.
- **Behavior:** unknown provider raises a clear error; missing OpenAI key when provider=openai raises with a fix hint.
- **Out of scope:** tools, streaming (Ep 2+).
- **Acceptance test** (`test_ep01_providers.py`): `get_model()` builds without network for both providers (assert type/config); a live smoke test guarded by a `@pytest.mark.live` marker (skipped in CI). `build_agent()` returns a compiled graph with an `invoke` method.
- **Demo:** same prompt answered by `qwen2.5-coder:7b` and `gpt-4o-mini`, switched by env var. The "hello world" moment.

---

### Ep 2 — The Agentic Loop, Made Visible _(shape: use the battery — streaming)_

- **Tag:** `ep-02` · **Budget:** ~110 SLOC (stream driver ~70, history/thread_id ~20, dummy tool ~20)
- **New:** extend `codeit/agent.py`
- **API:**
  - `async def run(agent, prompt, thread_id="default") -> dict` — drives `agent.astream(..., stream_mode="updates", subgraphs=True, version="v2")`, printing each node event (model request / tool call / tool result) so the loop is visible via `rich`. Enforces `CODEIT_MAX_ITERS` by passing `recursion_limit` to the invoke/stream config.
  - `run_sync(agent, prompt, thread_id="default") -> dict` — sync wrapper.
  - A throwaway `get_time` `@tool` proves the loop fires.
- **Behavior:** `thread_id` is threaded across turns (state persists via the built-in checkpointer-free default; Ep 6 will add a real `MemorySaver`); loop terminates when the model stops requesting tools or the recursion cap trips.
- **Acceptance test:** with `GenericFakeChatModel` scripted to call `get_time` once then answer, assert the tool ran and the loop terminated; assert the recursion cap is enforced when the model loops forever (scripted to always call `get_time`).
- **Demo:** agent calls the dummy tool, sees the result, responds — the "this is an agent" moment, with each step streamed live on screen.
- **Builds on:** Ep 1 model factory + agent builder.

---

### Ep 3 — Giving Your Agent Hands: Filesystem Tools _(shape: use the battery — FilesystemMiddleware)_

- **Tag:** `ep-03` · **Budget:** ~80 SLOC (backend wiring ~30, custom read tool ~30, registry ~20)
- **New:** `codeit/tools/__init__.py`; add backend config to `codeit/agent.py`
- **API:**
  - `build_agent(..., backend=None)` — when `backend` is None and settings says so, defaults to `FilesystemBackend(root_dir=CODEIT_WORKDIR, virtual_mode=True)`. This automatically exposes the built-in `ls`, `read_file`, `glob`, `grep` tools.
  - _(teaching tool)_ `read_summary(path: str) -> str` — a thin custom `@tool` that calls `read_file` and truncates to first 50 lines + note. Exists to show viewers how to register a custom tool alongside built-ins.
  - `register_custom_tools(agent_config, tools: list) -> None` — central helper used by every later episode to add custom tools.
- **Behavior:** read-only this episode; large files truncated with a note; missing path returns a model-readable error string from the backend.
- **Out of scope:** any mutation (Ep 4).
- **Acceptance test:** agent with `FilesystemBackend` against `examples/sample_app` fixtures — `ls` lists files, `read_file` returns contents, `grep` finds a known symbol, traversal outside workspace raises. Direct-call test of `read_summary` against fixtures.
- **Demo:** "What's in this project?" → agent uses `ls` + `read_file` to answer. Viewers see the built-in tools need zero code.

---

### Ep 4 — Writing Code: `write_file` + the Workspace Sandbox _(shape: use the battery — FilesystemBackend write)_

- **Tag:** `ep-04` · **Budget:** ~70 SLOC (backend config solidified ~25, write demo ~20, sandbox test ~25)
- **New:** nothing new — `FilesystemBackend` already provides `write_file`. Reinforce the sandbox.
- **API:** `write_file(path, content)` is built-in. This episode _adds_:
  - A `resolve_in_workspace(path) -> Path` helper in `codeit/agent.py` (or a tiny `codeit/paths.py` if cleaner) used by _custom_ tools (Ep 5+) so they share the backend's sandbox discipline.
  - A README callout: whole-file write is simplest; Ep 8 makes `edit_file` the default editing path but `write_file` stays for new files.
- **Behavior:** `write_file` creates parent dirs; confined to `CODEIT_WORKDIR` by `virtual_mode=True`.
- **Acceptance test:** write inside workspace succeeds; write outside raises; round-trip `read_file` matches.
- **Demo:** "Create a FastAPI hello-world" → agent writes `main.py`; viewer runs it.
- **Teaching note:** this is the shortest episode by design — the framework does the work. Spend the screen time on _what the sandbox protects against_ and why `virtual_mode=True` matters.

---

### Ep 5 — Running Commands: the Shell Tool (and why it's dangerous) _(shape: add a new tool)_

- **Tag:** `ep-05` · **Budget:** ~100 SLOC (exec ~50, output capture/truncate ~30, wiring ~20)
- **New:** `codeit/tools/shell.py`
- **API:** `run_shell(command: str) -> str` — a custom `@tool` that runs the command in `CODEIT_WORKDIR`, captures stdout+stderr+exit code, truncates long output with a note. Resolves cwd through `resolve_in_workspace`. Registered via `register_custom_tools`.
- **Behavior:** **no gating yet** — this is the cliffhanger. End the episode by pointing at Ep 6. Repeat the sandbox warning: even confined to the workspace, an agent can `rm -rf` the workspace itself.
- **Acceptance test:** a safe command (`echo`/`python -c`) returns expected output and exit code; cwd is the workspace.
- **Demo:** agent runs `pip install`, executes the app, reads output. Tease: "next episode we make this safe."

---

### Ep 6 — Permission Gating: Human-in-the-Loop _(shape: use the battery — HumanInTheLoopMiddleware)_

- **Tag:** `ep-06` · **Budget:** ~120 SLOC (gate config ~40, classify ~40, sugar ~20, checkpointer ~20)
- **New:** `codeit/approval.py`; extend `codeit/agent.py` to add `interrupt_on` + `checkpointer`
- **API:**
  - `build_agent(..., interrupt_on=None)` — when set, passes `interrupt_on={"run_shell": True, "write_file": True, "edit_file": True}` to `create_deep_agent`; adds `MemorySaver()` checkpointer (required for interrupts).
  - `classify(command: str) -> str` — returns `"safe" | "needs-approval" | "blocked"` via a regex allow-list (read-only vs mutating vs destructive patterns). Used inside a pre-check wrapper around `run_shell` to short-circuit obviously-bad commands before they ever hit the interrupt.
  - `await_approval(agent, config) -> Command` — helper that calls `agent.get_state(config)`, prints the pending action via `rich`, prompts y/n (auto-approve when `CODEIT_AUTO_APPROVE`/`--yolo`), returns `Command(resume={"decisions":[{"type":"approve"} | {"type":"reject","message":...}]})`.
  - `run_with_approval(agent, prompt, thread_id) -> dict` — the full loop: `invoke` → if `state.next`, `await_approval` → `invoke(Command(...))`.
- **Behavior:** destructive commands (`rm -rf`, `git push -f`, etc.) trigger an interrupt; viewer approves/rejects; `--yolo` auto-approves everything (documented as dangerous).
- **Acceptance test:** dangerous action triggers an interrupt and is rejected without approval; auto-approve lets a mutating action through; safe read passes untouched (no interrupt). Uses `GenericFakeChatModel` scripted to call `run_shell("rm -rf .")`.
- **Demo:** agent asks before a destructive command; viewer approves/denies. The "we made it safe" moment.

---

### Ep 7 — The System Prompt: engineering personality & rules _(shape: customize the battery)_

- **Tag:** `ep-07` · **Budget:** ~80 SLOC of _logic_ (loader/injection) + prompt text as constants
- **New:** `codeit/prompts.py`
- **API:**
  - `SYSTEM_PROMPT: str` — structured harness prompt (role, tool-use policy, safety, editing rules). Passed as `system_prompt=` to `create_deep_agent`.
  - `load_project_context(root: str = ".") -> str` — reads `AGENTS.md` (Deep Agents convention) or `CODEIT.md` fallback from the workspace if present.
  - `build_system_prompt(root) -> str` — composes `SYSTEM_PROMPT + "\n\n" + project_context`; passed to `build_agent`.
- **Acceptance test:** project file is discovered and injected; absent file degrades gracefully (returns just `SYSTEM_PROMPT`).
- **Demo:** same task, dramatically better behavior after prompt tuning — a before/after. Show the `AGENTS.md` file going in.

---

### Ep 8 — Surgical Edits: `edit_file` + tuning the edit format _(shape: use + customize the battery)_

- **Tag:** `ep-08` · **Budget:** ~140 SLOC (format explainer ~30, fuzzy fallback wrapper ~70, tests ~40). Tightest episode — if it crosses 220, split fuzzy fallback into `ep-08b`.
- **New:** extend `codeit/tools/__init__.py`
- **API:** `edit_file(path, search, replace)` is built-in (Aider/Codex/Cline-style search-replace). This episode _adds_:
  - `edit_file_safe(path: str, search: str, replace: str) -> str` — a custom `@tool` that wraps the built-in `edit_file` with a `difflib` fuzzy fallback (≥90% similarity) when the exact search misses, applied with a note; on total failure, returns a unified diff so the model self-corrects. This is the _teaching_ layer over the built-in: we explain the search-replace format, then harden it.
  - Goes through the Ep 6 gate (`interrupt_on={"edit_file_safe": True}`).
- **Behavior:** teaches the industry-standard reliable edit format (Aider/Codex/Cline lineage). Built-in `edit_file` is the default; `edit_file_safe` is the upgrade for tricky near-misses.
- **Acceptance test:** exact match edits correctly; near-miss triggers fuzzy apply; unresolvable search returns a diff and leaves the file unchanged.
- **Demo:** change 3 lines in a 200-line file without rewriting it; deliberately trigger a near-miss to show self-correction.

---

### Ep 9 — Context Management: Repo Map & not blowing the token window _(shape: add a new tool + customize)_

- **Tag:** `ep-09` · **Budget:** ~120 SLOC (repo map ~80, token estimate/trim ~30, wiring ~10). If over, split map (`ep-09`) from trimming (`ep-09b`).
- **New:** `codeit/repomap.py`
- **API:**
  - `build_repo_map(root: str = ".") -> str` — custom `@tool` that walks the repo (respecting `.gitignore`), uses `ast` to extract file → top-level `def`/`class` signatures (simple version; note Aider's PageRank ranking as out-of-scope advanced work). Registered via `register_custom_tools`.
  - `estimate_tokens(text: str) -> int` — ~4-chars/token heuristic.
  - `trim_history(messages, max_tokens: int) -> list` — drop/compact oldest turns to fit. _Note:_ Deep Agents' `FilesystemMiddleware` already does context engineering internally; this episode teaches the _concept_ and gives the viewer a visible trim function for custom control, complementing the built-in behavior.
- **Acceptance test:** map lists known symbols from `sample_app`; trim keeps newest turns under a token cap.
- **Demo:** "where is X defined?" answered across multiple files without reading them all, using `build_repo_map`.

---

### Ep 10 — Planning: TodoList & task decomposition _(shape: use + customize the battery — TodoListMiddleware)_

- **Tag:** `ep-10` · **Budget:** ~90 SLOC (TodoList config ~30, custom plan tool ~40, render ~20)
- **New:** nothing new — `TodoListMiddleware` is built-in via `write_todos`. This episode _adds_:
  - A custom `plan(steps: list[str]) -> str` `@tool` that calls `write_todos` under the hood to seed the todo list from an explicit plan (teaches the viewer the bridge between "plan" and the built-in state).
  - `render_todos(state) -> str` — reads `state["todos"]` and pretty-prints via `rich` for the CLI (used by Ep 14).
- **Behavior:** `write_todos` is the built-in tool; current todo state is injected into context each turn by the middleware. `plan` is a sugar tool that makes the planning step explicit and legible.
- **Acceptance test:** `plan` seeds todos; `complete_todo(id)` (a tiny custom tool) flips status; `list_todos()` renders injected state deterministically. Use `GenericFakeChatModel` scripted to call `plan` then `write_todos` updates.
- **Demo:** "Add auth to this app" → agent writes a 5-step plan via `plan`, executes step by step, checks items off via `write_todos`.
- **Teaching note:** this is shorter than CodeIt's Ep 10 because the middleware does the state injection. Spend the screen time on _reading the todo state back out of `result["todos"]`_ and rendering it.

---

### Ep 11 — Error Recovery: Self-Healing Loops _(shape: add a new tool + custom LangGraph node)_

- **Tag:** `ep-11` · **Budget:** ~140 SLOC (test/lint tool ~50, recovery loop ~70, wiring ~20). Tight — may split into `ep-11` (tool) + `ep-11b` (loop).
- **New:** `codeit/testing.py`; recovery logic in `codeit/agent.py`
- **API:**
  - `run_tests(path: str = ".") -> str` — custom `@tool` that runs pytest/ruff in the workspace, captures failures. Registered via `register_custom_tools`.
  - `run_with_recovery(agent, prompt, thread_id, max_retries=3) -> dict` — wraps `run` (Ep 2): after the agent finishes, if `run_tests` was called and reported failures, re-invoke the agent with the failure output appended as a user message; cap retries (reuse `CODEIT_MAX_ITERS`). Implemented as a plain Python loop _around_ the Deep Agents graph (not a new graph node), so we don't fight the framework.
- **Behavior:** on failure, re-invoke with the traceback; the agent reads it and fixes. Cap prevents infinite loops.
- **Acceptance test:** with `GenericFakeChatModel` scripted to "write buggy code" → `run_tests` fails → "fix" on the 2nd pass, assert the loop converges and stops at the retry cap otherwise.
- **Demo:** agent writes buggy code, runs it, reads the traceback, fixes it autonomously — the big "wow."
- **Builds on:** Ep 5 shell + Ep 8 edit + Ep 10 plan.

---

### Ep 12 — Sub-Agents: specialists in isolated context _(shape: use + customize the battery — SubAgentMiddleware)_

- **Tag:** `ep-12` · **Budget:** ~110 SLOC (custom subagent factory ~60, delegation demo ~30, summary tool ~20)
- **New:** extend `codeit/agent.py` with a subagent factory
- **API:**
  - `build_agent(..., subagents=None)` — accepts a list of subagent specs. Default subagent `"general-purpose"` is built-in (same tools as main). We add custom subagents:
    - `"explorer"`: read-only tools (`ls`, `read_file`, `grep`, `build_repo_map`), no shell/write. For "map the codebase."
    - `"tester"`: `run_tests` + `edit_file_safe`, no shell. For "make the tests pass."
  - The `task` built-in tool is how the main agent delegates: `task(agent="explorer", instruction="...")`.
  - `spawn_subagent(task: str, agent: str = "general-purpose") -> str` — a thin custom `@tool` wrapper around the built-in `task` for ergonomics and to show viewers how delegation looks in code.
- **Behavior:** subagents run with isolated context; only a summary returns to the parent; subagent messages don't leak into parent history (framework behavior).
- **Acceptance test:** with `GenericFakeChatModel`, parent delegates via `task` and receives a summary string; sub-agent's messages don't appear in parent `state["messages"]`. Custom `"explorer"` subagent has only read-only tools (assert via its config).
- **Demo:** main agent delegates "map the codebase" to the `explorer` sub-agent; only the summary returns, saving context. Then delegates "make tests pass" to `tester`.
- **Teaching note:** subagents are _stateless_ — emphasize complete instructions in a single `task` call (from the Deep Agents docs).

---

### Ep 13 — MCP + Skills: speaking the standard protocol _(shape: use the battery — MCP + SkillsMiddleware)_

- **Tag:** `ep-13` · **Budget:** ~110 SLOC (MCP config ~50, skills dir ~30, wiring/demo ~30)
- **New:** `codeit/mcp_client.py`; a `skills/` directory with one example `SKILL.md`
- **API:**
  - `build_agent(..., mcp_servers=None, skills=None)` — passes MCP server config (e.g., filesystem or git MCP server via Pydantic AI's MCP support / `langchain-mcp-adapters`) and a skills directory to `create_deep_agent`. Verify the exact param name (`mcp_servers` / `mcp_tools` / `mcp_config`) against the installed Deep Agents version.
  - `codeit/mcp_client.py` — `build_mcp_config(settings) -> MCPServerConfig` — assembles the MCP server connection from env (e.g., `MCP_FILESYSTEM_ROOT`).
  - `skills/python-testing/SKILL.md` — one example skill with proper YAML frontmatter (`name`, `description`) so viewers see the format. The agent loads it on demand.
- **Behavior:** MCP-provided tools register alongside built-ins; skills load on demand based on the agent's judgment.
- **Acceptance test:** when an MCP server is reachable, its tools appear in the agent's tool list (skipped if no server); the example skill is discoverable via the skills middleware (assert the skill file parses with frontmatter).
- **Demo:** `codeit "add a health endpoint and test it"` uses an MCP-provided tool (e.g., a git MCP tool to commit the change) and the `python-testing` skill for test conventions.

---

### Ep 14 — Shipping CodeIt: a Real CLI with Event Streaming _(shape: customize — Typer + rich)_

- **Tag:** `ep-14` · **Budget:** ~160 SLOC (CLI ~70, event-streaming view ~70, wiring ~20). **If this crosses 220, split:** `ep-14` (CLI) and `ep-15` (event-streaming + capstone) — within the 14–15 window the user approved.
- **New:** `codeit/cli.py`
- **API:**
  - `typer` app `codeit` with a `rich` live-streaming view built on Deep Agents' **event-streaming API** (v0.6+, `agent.astream_events` / typed projections) — separate iterators for subagents, messages, tool calls, values, so each is rendered in its own panel.
  - Flags: `--provider`, `--model`, `--yolo`, `--workdir`, `--mcp`, `--skills`.
  - Single command: `codeit "your task here"`.
- **Behavior:** streams tokens, tool calls, subagent updates, todo changes live; honors `--yolo` (sets `CODEIT_AUTO_APPROVE`), `--workdir` (sets `CODEIT_WORKDIR`), provider/model overrides.
- **Acceptance test:** `typer.testing.CliRunner` drives `codeit "say hi"` with a `GenericFakeChatModel` injected via env/test fixture; asserts exit 0 + streamed output contains the model's reply and the tool-call panel.
- **Demo:** `codeit "add a health endpoint and test it"` runs end-to-end with live streaming, an MCP-provided tool, a subagent delegation, and the todo list updating. **Finale:** agent builds a small complete app hands-free on camera.

---

### Ep 15 (optional buffer / stretch)

Choose one:

- **A. Codebase RAG/memory over a large repo** — embeddings via Ollama `nomic-embed-text` + `langchain-ollama`'s `OllamaEmbeddings` + a vector store (`langchain-chroma`), exposed as a `recall` tool.
- **B. LangSmith observability appendix** — `LANGSMITH_TRACING=true` from the start (already set in `.env.example`); walk through traces, datasets, and evals. (Per user choice: optional appendix only.)
- **C. Raw-SDK "no-framework" appendix** — re-implement Ep 1-2 with `langchain-core` primitives to show what Deep Agents does for you. Doubles SEO surface.

Same DoD and tagging rules.

---

## 9 · Testing & verification strategy

- **CI (no network):** `pytest -m "not live"` must pass at every tag. All agent-behavior tests use `GenericFakeChatModel` with scripted `AIMessage` + `ToolCall` sequences; all tool tests call tools directly against `examples/sample_app` and `tmp_path` workspaces.
- **Live smoke (manual):** `pytest -m live` runs one real call per provider — used to record demos, skipped in CI.
- **Regression:** keep every prior `test_epNN_*.py` in the suite; later tags must keep them green (honoring the Ep 4→8 supersession note: `edit_file` becomes default but `write_file` tests must still pass).
- **Lint/format gate:** `ruff check` + `ruff format --check` in the same CI step.

---

## 10 · Companion assets (per episode)

1. **README section** — 4–6 lines: what this episode adds, the new file(s), the run command, and the sub-220-line note. Include the pinned Deep Agents / LangChain versions once, up top.
2. **`scripts/demo_epNN.sh`** — the exact commands used on camera, runnable against `examples/sample_app`, provider-selectable by env.
3. **On-screen safety line** for any episode touching shell/write: never run unsandboxed on a real repo.

---

## 11 · Guardrails to bake in (from the research caveats)

- **Version pinning + import verification** (§2.6): The LangChain ecosystem evolves fast and `deepagents` is pre-1.0 in places; verify against the installed version before writing `model.py`/`agent.py`. Record the exact version in README so viewers reproduce it. Prefer the `langchain-docs` MCP server for current signatures.
- **Small-model honesty:** planning, sub-agent, and recovery episodes are unreliable on ≤8B local models — code must degrade gracefully (clear errors, retry caps) and README must recommend a larger model or OpenAI for those episodes.
- **Tool-parser quirk:** some Ollama builds mishandle tool calls for `qwen2.5-coder`; README tells viewers to confirm the model's tool-call support and update Ollama. Prefer `qwen3-coder` if available.
- **Security posture:** the `FilesystemBackend(virtual_mode=True)` sandbox (Ep 4) and `interrupt_on` approval gate (Ep 6) are load-bearing, not decoration. Never weaken them for demo convenience; `--yolo` is documented as dangerous.
- **Don't fight the framework:** Deep Agents owns filesystem, todos, subagents, HITL. Do not re-implement these. When you need custom behavior, _configure_ the middleware or _wrap_ the built-in tool — don't replace it. The only from-scratch pieces are `run_shell`, `run_tests`, `build_repo_map`, the fuzzy `edit_file_safe` wrapper, and the recovery loop (which lives _around_ the graph, not inside it).
- **Split over compress:** if any episode's diff crosses 220 SLOC, split it and add the next tag rather than shrinking the teaching code. Pre-flagged: Ep 8, Ep 11, Ep 14.

---

## 12 · Definition of done — whole series

- [ ] Tags `ep-00-setup`, `ep-01`…`ep-14` (+ optional `ep-15`) all exist, checkoutable, runnable.
- [ ] Each episode's net-new Python < 220 SLOC (verified via tag-to-tag diff).
- [ ] `pytest -m "not live"` green and `ruff` clean at every tag.
- [ ] Every episode runs on Ollama and OpenAI by env var only (or Ollama + documented OpenAI note).
- [ ] README has pinned versions, per-episode notes, and model recommendations; each episode has a demo script.
- [ ] Final `codeit "..."` capstone builds a small app hands-free with event streaming, MCP, a subagent, and the todo list updating.

**Begin with Phase 0, then Ep 1. Do not advance past an episode until its Definition of Done is met.**

---

## 13 · Mapping back to the original CodeIt plan (for reviewers)

| CodeIt (Pydantic AI) episode                | This plan (Deep Agents) episode | What changed                                                                           |
| ------------------------------------------- | ------------------------------- | -------------------------------------------------------------------------------------- |
| Ep 1 model factory                          | Ep 1                            | `init_chat_model` replaces Pydantic AI factory                                         |
| Ep 2 agentic loop via `agent.iter()`        | Ep 2                            | `agent.astream(stream_mode="updates", subgraphs=True, version="v2")` replaces `iter()` |
| Ep 3 read_file/list_dir/grep (hand-rolled)  | Ep 3                            | `FilesystemBackend` provides them; we add one teaching tool                            |
| Ep 4 write_file + sandbox                   | Ep 4                            | `FilesystemBackend(virtual_mode=True)` is the sandbox; `write_file` built-in           |
| Ep 5 run_shell                              | Ep 5                            | Same — custom `@tool` (no built-in equivalent)                                         |
| Ep 6 approval gate (hand-rolled)            | Ep 6                            | `HumanInTheLoopMiddleware` via `interrupt_on` + `MemorySaver`                          |
| Ep 7 system prompt + AGENTS.md              | Ep 7                            | Same — `AGENTS.md` is Deep Agents' own convention                                      |
| Ep 8 edit_file (hand-rolled search-replace) | Ep 8                            | `edit_file` is built-in; we add a fuzzy-fallback wrapper as the teaching layer         |
| Ep 9 repo map + token trim                  | Ep 9                            | Same — custom tool; note FilesystemMiddleware already does context engineering         |
| Ep 10 todo (hand-rolled)                    | Ep 10                           | `TodoListMiddleware` + `write_todos` built-in; we add a `plan` sugar tool              |
| Ep 11 run_tests + recovery                  | Ep 11                           | Same — custom tool; recovery loop lives _around_ the graph                             |
| Ep 12 subagent (hand-rolled)                | Ep 12                           | `SubAgentMiddleware` + `task` built-in; we add custom subagent specs                   |
| Ep 13 MCP + CLI                             | Ep 13 + Ep 14                   | Split: Ep 13 = MCP + skills; Ep 14 = CLI + event streaming                             |
| Ep 14 stretch                               | Ep 15 stretch                   | RAG, LangSmith appendix, or raw-SDK appendix                                           |

---

This plan keeps CodeIt's mission, pacing, line budgets, safety posture, and per-episode tag/demo/test discipline, while re-platforming onto Deep Agents so the viewer learns the framework they'd actually ship.
