# Episode 15 — Stretch / Buffer (optional)

**Tag:** `ep-15` (optional) · **Budget:** sub-220 SLOC, same DoD as other episodes
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 15

---

## 15.1 · Purpose

This is the buffer episode. Use it for **one** of the three options below, or skip it entirely if Eps 1-14 landed within budget and the series feels complete. Same Definition of Done and tagging rules apply if used.

The user chose "Optional appendix only" for LangSmith, so Option B (LangSmith) is the lowest-friction choice. But Options A (RAG) and C (raw-SDK) are also valid. Pick based on what the series most needs.

---

## 15.2 · Option A — Codebase RAG / Memory over a large repo

**When to pick:** the series has focused on a small `examples/sample_app` and viewers want to see CodeIt scale to a real codebase. RAG lets the agent search a large repo semantically instead of `grep`-ing line by line.

**Adds:**
1. `codeit/rag.py` — an indexing helper (`index_repo(root) -> None`) that walks the workspace, splits files into chunks, embeds them via `langchain-ollama`'s `OllamaEmbeddings` (model `nomic-embed-text`), and stores them in `langchain-chroma`'s `Chroma` vector store at `.codeit/chroma/`.
2. A `recall(query: str) -> str` `@tool` that embeds the query, retrieves the top-k chunks, and returns them with file paths so the model can `read_file` for full context.
3. A `--index` CLI flag (or a `codeit index` subcommand) to build the index before a run.

**Pre-flight:**
- `pip install langchain-ollama langchain-chroma chromadb` (or `uv add`).
- `ollama pull nomic-embed-text` (the embedding model).
- Confirm `OllamaEmbeddings` import: `from langchain_ollama import OllamaEmbeddings`. Verify against the installed `langchain-ollama` version — the community-package import (`from langchain_community.embeddings import OllamaEmbeddings`) is deprecated per the langchain-dependencies skill fix.
- Confirm `Chroma` import: `from langchain_chroma import Chroma` (dedicated package, not `langchain_community.vectorstores`).

**Sketch (~150 SLOC):**
```python
# codeit/rag.py
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from codeit.paths import workspace_root

def _embeddings():
    return OllamaEmbeddings(model="nomic-embed-text", base_url=get_settings().ollama_base_url)

def index_repo(root: str = ".") -> str:
    """Walk the workspace, split files, embed, store in Chroma at .codeit/chroma/."""
    # ... walk, read, split with RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
    # embed, add to Chroma(persist_directory=str(workspace_root()/".codeit/chroma"), embedding_function=_embeddings())
    # return a summary like "Indexed N files, M chunks."

@tool
def recall(query: str, k: int = 4) -> str:
    """Semantic search over the indexed codebase. Returns top-k chunks with file paths.

    Use this when grep isn't enough — e.g. 'where do we handle authentication?' across a large repo.
    Requires the repo to be indexed first (run: codeit index). Argument k: number of chunks (default 4).
    """
    # load Chroma, similarity_search(query, k=k), format results with file paths
```

**Demo:**
```bash
codeit index --workdir ./big-repo
codeit run "where is authentication handled?" --workdir ./big-repo
# agent calls recall(), gets semantic hits, then read_file for the right file.
```

**DoD:** `index_repo` runs; `recall` returns relevant chunks; test with a tiny fixture repo asserts the right file surfaces for a semantic query.

---

## 15.3 · Option B — LangSmith observability appendix

**When to pick:** the user chose "Optional appendix only" for LangSmith, so this is the natural fit. Show viewers how to see what the agent is *doing* under the hood.

**Adds:**
1. `.env.example` already has `LANGSMITH_API_KEY=` and `LANGSMITH_TRACING=true` (set these in Phase 0).
2. A short `scripts/demo_ep15_langsmith.sh` that runs a task with tracing on, then opens the trace URL.
3. A README section explaining how to read traces: each model call, tool call, subagent delegation, and the full message history at each step.
4. (Optional) a `codeit/observability.py` helper that wraps `run()` to log trace URLs to stderr via `rich` at the end of a run.

**Pre-flight:**
- `pip install langsmith` (already a dep from Phase 0).
- Set `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` in `.env`.
- Confirm the env var names per the ecosystem-primer skill: `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=<name>`. Older names no longer work.

**Sketch (~60 SLOC):**
```python
# codeit/observability.py
import os
from rich.console import Console
console = Console()

def trace_url_for_run(run_id: str | None) -> str | None:
    """Build a LangSmith trace URL if tracing is on and a run_id is available."""
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        return None
    if not run_id:
        return None
    project = os.getenv("LANGSMITH_PROJECT", "default")
    # URL shape: https://smith.langchain.com/o/<org>/projects/p/<project>/r/<run_id>
    # The exact shape depends on the LangSmith UI; this is a best-effort helper.
    return f"https://smith.langchain.com/projects/p/{project}/r/{run_id}"

def maybe_print_trace_url(state):
    """After a run, print the trace URL to stderr if tracing is on."""
    url = trace_url_for_run(getattr(state, "run_id", None) if state else None)
    if url:
        console.print(f"[dim]Trace: {url}[/dim]")
```

**Demo:**
```bash
export LANGSMITH_API_KEY=...
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=codeit-demo
codeit run "fix the failing test" --workdir ./examples/sample_app --approve
# At the end, a trace URL prints. Open it to see every model call, tool call, and the full message history.
```

**DoD:** with tracing on, a trace URL prints after the run; the README explains how to read traces. No code change required if the `.env.example` already has the vars — this is mostly a documentation + demo episode.

---

## 15.4 · Option C — Raw-SDK "no-framework" appendix

**When to pick:** viewers want to understand what Deep Agents does *for* them by seeing the same demo built without it. Doubles SEO surface ("LangChain from scratch" + "Deep Agents").

**Adds:**
1. `examples/raw_sdk/` — a parallel mini-implementation of Eps 1-2 using only `langchain-core` primitives: `ChatOllama` / `ChatOpenAI` directly, a hand-rolled tool loop with `model.invoke()` + manual tool dispatch, no `create_deep_agent`, no middleware.
2. A `scripts/demo_ep15_raw.sh` that runs the same "what time is it?" task through both the raw-SDK version and the Deep Agents version, side by side.
3. A README section explaining the trade-off: the raw-SDK version is ~3x more code for the same behavior, and it doesn't get planning, filesystem, subagents, or HITL for free.

**Pre-flight:**
- Confirm `ChatOllama` / `ChatOpenAI` import paths from `langchain_ollama` / `langchain_openai` (not `langchain_community`).
- Confirm `ToolCall` parsing: the raw loop needs to inspect `AIMessage.tool_calls`, dispatch each to the right function, append `ToolMessage` results, and re-invoke. This is exactly what Deep Agents does internally — the point of the demo.

**Sketch (~120 SLOC):**
```python
# examples/raw_sdk/agent.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool

@tool
def get_time() -> str:
    """Return the current time."""
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")

TOOLS = {"get_time": get_time}

def run_raw(prompt: str, model) -> str:
    messages = [HumanMessage(content=prompt)]
    for _ in range(10):  # cap
        resp = model.invoke(messages)
        messages.append(resp)
        if not resp.tool_calls:
            return resp.content
        for tc in resp.tool_calls:
            result = TOOLS[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"]))
    return "cap reached"

if __name__ == "__main__":
    model = ChatOllama(model="qwen2.5-coder:7b")
    print(run_raw("what time is it? use the tool", model))
```

**Demo:**
```bash
python examples/raw_sdk/agent.py  # ~40 lines of loop code
# vs
codeit run "what time is it? use the tool"  # Deep Agents does it in 1 line
```

**DoD:** the raw-SDK version runs the same task; the README side-by-side shows the line-count difference and lists what the raw version *doesn't* have (planning, filesystem, subagents, HITL, persistence, skills, MCP).

---

## 15.5 · Definition of Done (whichever option is picked)

- [ ] The chosen option's files are written and under 220 SLOC.
- [ ] `pytest -m "not live" tests/test_ep15_*.py` green (if the option adds testable code; Option B may have no new tests).
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-14 HEAD -- '*.py'` adds < 220 lines.
- [ ] Ep 1-14 demos still pass.
- [ ] README "Ep 15" section written.
- [ ] Commit + annotated tag `ep-15` (optional — skip if the episode isn't used).

---

## 15.6 · What NOT to do this episode

- Do not pick more than one option. This is a buffer, not a marathon. Pick the one the series most needs (probably B, given the user's "optional appendix" choice for LangSmith).
- Do not let the option break Eps 1-14. All prior demos must still pass at `ep-15`.
- Do not add new hard dependencies for the whole series. If Option A adds `langchain-chroma` + `chromadb`, it's an *optional* extra — the base agent works without RAG. Document the extra install in the README.
- Do not make Option C (raw-SDK) the *primary* path. It's a teaching contrast, not a replacement. The `codeit` CLI from Ep 14 remains the real deliverable.

---

## 15.7 · Common gotchas

- **Option A — `chromadb` is heavy:** the `chromadb` package pulls in a lot. For a teaching codebase, consider an in-memory `Chroma` (no persist) for the demo, or `FAISS` via `langchain-community` (lighter but the deprecated import path). Prefer `langchain-chroma` if you go Chroma.
- **Option A — `OllamaEmbeddings` import:** must be `from langchain_ollama import OllamaEmbeddings` (dedicated package), not `from langchain_community.embeddings import OllamaEmbeddings` (deprecated, per the langchain-dependencies fix).
- **Option B — env var names:** `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT`. Older names (`LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING`) no longer work per the ecosystem-primer skill. Verify against the installed `langsmith` version.
- **Option B — trace URL shape:** the LangSmith UI URL structure changes over time. The `trace_url_for_run` helper is best-effort; if the URL doesn't open, direct viewers to the LangSmith dashboard and have them find the run by project + timestamp.
- **Option C — tool dispatch loop:** the raw-SDK loop must handle `ToolCall` dicts with `name`, `args`, `id`, `type` keys. If the installed `langchain-core` uses a different shape (e.g. `ToolCall` as a Pydantic model vs dict), adapt. The point is to show the manual dispatch — the exact shape is secondary.
- **Option C — no persistence:** the raw-SDK version has no `MemorySaver`, no thread_id, no resume. That's the contrast — Deep Agents gives you all of that. Emphasize on camera.

---

## 15.8 · Recommendation

Given the user's choices (LangSmith = optional appendix only, 14-15 episodes), **Option B (LangSmith observability appendix)** is the recommended pick. It:
- Honors the "optional appendix" preference.
- Adds the least new code (mostly docs + a small helper).
- Gives viewers a debugging superpower they'll use in every later project.
- Doesn't require a heavy new dep (langsmith is already pinned from Phase 0).

If the series feels incomplete after Ep 14 (e.g. viewers want to see scaling), pick Option A. If it feels too framework-magic and viewers want to peek under the hood, pick Option C. Default to B.