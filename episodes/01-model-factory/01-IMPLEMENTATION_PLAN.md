# Episode 1 — Your Agent's Brain: One File, Two Providers

**Tag:** `ep-01` · **Shape:** add a tool-free agent · **Budget:** ~90 SLOC
**Parent plan:** `DeepAgents_CodeIt_Implementation_Plan.md` §8 Ep 1

---

## 1.1 · What this episode delivers

The viewer builds the smallest possible Deep Agent that can answer a prompt. Two pieces of infrastructure land here and are reused by every later episode:

1. **`codeit/settings.py`** — a typed config singleton loaded from `.env` once.
2. **`codeit/model.py`** — a provider factory that returns a `BaseChatModel` for either Ollama or OpenAI, switched by env var only.
3. **`codeit/agent.py`** — a thin `build_agent()` wrapper around `create_deep_agent()` with no tools yet.
4. **`scripts/chat.py`** — a one-shot CLI demo: send a prompt, print the reply.

This is the "hello world" of Deep Agents. No tools, no loop visualization, no streaming — just prove the model factory and the agent builder work on both providers.

---

## 1.2 · Pre-flight (do this before writing code)

1. `pip install deepagents langchain-ollama langchain-openai python-dotenv` (or `uv add ...`).
2. `python -c "import deepagents; print(deepagents.__version__)"` and record the exact version in `README.md` under "Pinned versions".
3. Verify `init_chat_model` signature in the installed `langchain.chat_models`:
   ```bash
   python -c "from langchain.chat_models import init_chat_model; import inspect; print(inspect.signature(init_chat_model))"
   ```
   The factory in §1.3 must match this signature. Prefer the installed API over this plan if they differ.
4. Confirm Ollama is running: `curl http://localhost:11434/api/tags` (or skip live demo).
5. Pull the demo model: `ollama pull qwen2.5-coder:7b`.

---

## 1.3 · File specs

### `codeit/settings.py` (~35 SLOC)

```python
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env once at import time

@dataclass(frozen=True)
class Settings:
    llm_provider: str          # "ollama" | "openai"
    llm_model: str             # e.g. "qwen2.5-coder:7b" or "gpt-4o-mini"
    ollama_base_url: str
    openai_api_key: str
    openai_base_url: str
    workdir: str
    auto_approve: bool
    max_iters: int

def get_settings() -> Settings:
    """Load settings from environment; called once, cached by frozen dataclass."""
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5-coder:7b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        workdir=os.getenv("CODEIT_WORKDIR", "./workspace"),
        auto_approve=os.getenv("CODEIT_AUTO_APPROVE", "false").lower() == "true",
        max_iters=int(os.getenv("CODEIT_MAX_ITERS", "25")),
    )
```

**Teaching notes (say these on camera):**
- Frozen dataclass = immutable singleton; safe to pass anywhere.
- `load_dotenv()` at import means `.env` is read once; no need to call it in every module.
- `get_settings()` is a function, not a module-level constant, so tests can monkeypatch env and call it fresh.

### `codeit/model.py` (~25 SLOC)

```python
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from codeit.settings import Settings, get_settings

def get_model(settings: Settings | None = None) -> BaseChatModel:
    """Return a chat model for the configured provider. Switching providers is env-only."""
    s = settings or get_settings()
    if s.llm_provider == "ollama":
        return init_chat_model(
            model=s.llm_model,
            model_provider="ollama",
            base_url=s.ollama_base_url,
        )
    elif s.llm_provider == "openai":
        if not s.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                "Set it in .env or `export OPENAI_API_KEY=...`."
            )
        kwargs = {"model": s.llm_model, "model_provider": "openai"}
        if s.openai_base_url:
            kwargs["base_url"] = s.openai_base_url
        return init_chat_model(**kwargs)
    raise ValueError(
        f"Unknown LLM_PROVIDER={s.llm_provider!r}. Use 'ollama' or 'openai'."
    )
```

**Teaching notes:**
- `init_chat_model` is LangChain's provider-agnostic factory; Deep Agents uses it internally too. We use the same primitive so the viewer sees one consistent API.
- The `provider:model` string format (e.g. `openai:gpt-4o-mini`) is what `create_deep_agent` accepts directly — but we pre-build the model so we can hold `base_url` and key validation here, in *our* code, where we control the error messages.
- Never log the API key. The error message tells the viewer exactly what to fix.

### `codeit/agent.py` (~15 SLOC)

```python
from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from codeit.model import get_model
from codeit.settings import get_settings

def build_agent(
    model: BaseChatModel | None = None,
    tools: list | None = None,
    system_prompt: str | None = None,
):
    """Build a Deep Agent. No tools, no backend, no interrupts yet — just the brain."""
    m = model or get_model()
    _ = get_settings()  # ensure .env loaded; settings used by later episodes
    return create_deep_agent(
        model=m,
        tools=tools or [],
        system_prompt=system_prompt or "You are CodeIt, a helpful coding assistant.",
    )
```

**Teaching notes:**
- `create_deep_agent` returns a compiled LangGraph graph. It has `.invoke()`, `.stream()`, `.astream()`, `.get_state()`.
- We accept `model` as an arg so tests can inject `GenericFakeChatModel` (Ep 2+) without touching `.env`.
- `tools=[]` for now — built-in planning/filesystem/subagent middleware still ships, but with no custom tools the agent just talks.

### `scripts/chat.py` (~15 SLOC)

```python
import sys
from codeit.agent import build_agent

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello in one sentence."
    agent = build_agent()
    config = {"configurable": {"thread_id": "demo"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
```

---

## 1.4 · Tests

### `tests/test_ep01_providers.py`

- `test_get_model_ollama_no_network`: call `get_model(Settings(llm_provider="ollama", ...))` with a fake base_url; assert the returned object is a `ChatOllama` (or whatever `init_chat_model` returns for provider="ollama"). Do NOT call `.invoke()` — no network.
- `test_get_model_openai_missing_key_raises`: `Settings(llm_provider="openai", openai_api_key="")` → `get_model()` raises `ValueError` with the fix hint.
- `test_get_model_unknown_provider_raises`: `Settings(llm_provider="bedrock")` → raises.
- `test_build_agent_returns_graph`: `build_agent(model=fake_model)` returns an object with `.invoke` and `.stream` attributes (compiled graph).
- `test_live_ollama` (`@pytest.mark.live`): real `qwen2.5-coder:7b` round-trip; skipped in CI.
- `test_live_openai` (`@pytest.mark.live`): real OpenAI round-trip; skipped in CI.

### `tests/conftest.py` (additions, if not already from Phase 0)

- `fake_model` fixture: returns `GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))` by default; tests can override `messages` per-test.
- `tmp_workspace` fixture: `tmp_path` dir, monkeypatch `CODEIT_WORKDIR` to it. (Used from Ep 4 on; harmless to add now.)

---

## 1.5 · Demo script `scripts/demo_ep01.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "=== Ollama ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b python scripts/chat.py "What is 2+2? Answer in one word."
echo
echo "=== OpenAI ==="
LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini python scripts/chat.py "What is 2+2? Answer in one word."
```

On camera: run with Ollama first (free, local), then flip to OpenAI with one env var. The whole point: **provider switching is env-only**.

---

## 1.6 · README section to add

```
## Ep 1 — Your Agent's Brain
Adds: codeit/settings.py, codeit/model.py, codeit/agent.py, scripts/chat.py
Run: LLM_PROVIDER=ollama python scripts/chat.py "hello"
Pinned: deepagents==<X.Y.Z>, langchain==<1.0.*>, langchain-ollama==<X.Y.Z>
Lines added: 90 SLOC
```

---

## 1.7 · Definition of Done (per-episode checklist)

- [ ] `codeit/settings.py`, `codeit/model.py`, `codeit/agent.py`, `scripts/chat.py` written.
- [ ] `pytest -m "not live" tests/test_ep01_providers.py` green.
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `git diff ep-00-setup HEAD -- '*.py'` adds < 220 non-blank non-comment lines.
- [ ] `scripts/demo_ep01.sh` runs end-to-end on Ollama (and OpenAI if key available).
- [ ] README "Ep 1" section written with pinned versions.
- [ ] Commit + annotated tag `ep-01` (`git tag -a ep-01 -m "Ep 1: model factory + agent build"`).

---

## 1.8 · What NOT to do this episode

- Do not add tools (Ep 3+).
- Do not add streaming (Ep 2).
- Do not add a backend / filesystem (Ep 3-4).
- Do not add a checkpointer (Ep 6 needs it for HITL; not yet).
- Do not call `create_deep_agent(model="ollama:...")` with a string — we pre-build the model in `model.py` so we own error messages. (You *can* pass a string directly, but then we lose the missing-key hint. Keep the factory.)
- Do not log `openai_api_key` anywhere, ever.

---

## 1.9 · Common gotchas

- **`init_chat_model` signature drift:** if the installed version uses `provider=` instead of `model_provider=`, adapt. The contract is "provider-agnostic factory in `model.py`" — the exact kwarg name is implementation detail.
- **Ollama model not pulled:** `init_chat_model` will succeed but `.invoke()` fails. The demo script should `ollama pull` first or the README should tell viewers to.
- **`create_deep_agent` imports:** confirm the import path `from deepagents import create_deep_agent` against the installed version. If it moved to `deepagents.graph` or similar, update.