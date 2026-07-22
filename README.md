# CodeIt

A small coding-agent harness built on [Deep Agents](https://github.com/langchain-ai/deepagents)
(LangChain / LangGraph). Follow-along build, one episode at a time.

## Pinned versions

| Package | Version |
| --- | --- |
| deepagents | 0.6.12 |
| langchain | 1.2.9 |
| langchain-core | 1.2.9 |
| langchain-ollama | 1.0.1 |
| langchain-openai | (installed, latest 1.x) |
| python | 3.11+ |

Install: `pip install -e ".[dev]"` (or `uv sync`).

## Episodes

### Ep 1 — Your Agent's Brain
Adds: `codeit/settings.py`, `codeit/model.py`, `codeit/agent.py`, `scripts/chat.py`
Run: `LLM_PROVIDER=ollama python scripts/chat.py "hello"`
Pinned: `deepagents==0.6.12`, `langchain==1.2.9`, `langchain-ollama==1.0.1`
Lines added: ~90 SLOC

### Ep 2 — The Agentic Loop, Made Visible
Adds: `run()` + `get_time` demo tool in `codeit/agent.py`; updates `scripts/chat.py`; `tests/test_ep02_loop.py`
Run: `LLM_PROVIDER=ollama python scripts/chat.py "What time is it? Use your tool."`
Key idea: `agent.stream(stream_mode=["updates","values"], version="v2")` makes the LangGraph loop visible and captures final state without a checkpointer.
Lines added: ~110 SLOC
