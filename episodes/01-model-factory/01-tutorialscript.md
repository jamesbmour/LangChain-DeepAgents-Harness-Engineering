# Episode 1 — Your Agent's Brain: One File, Two Providers (Tutorial Video Script)

## Overview
**Length:** ~12 minutes  
**Goal:** Show viewers how to build the foundation of a coding agent — a provider-agnostic model factory that can switch between local Ollama and cloud OpenAI with just environment variables. This is Episode 1: everything builds on top of this.

---

## Scene 1: Hook & The Provider Dilemma (0:00–2:00)

**On-screen:** Terminal showing two different model providers — Ollama running locally, then switching to OpenAI with a single env var change.

> **Host:** "Every AI agent needs a brain — the language model that powers its reasoning. But which one should you use? Local models like Ollama give you privacy and no API costs, but cloud models like GPT-4 offer better performance."
>
> *(Show switching between providers)*

**Key points:**
- The choice of LLM provider is fundamental — it affects cost, speed, privacy, and capability.
- We want to switch providers without changing code — just environment variables.
- This episode builds the model factory that makes this possible.

---

## Scene 2: Settings — Typed Configuration from Environment (2:00–4:30)

**On-screen:** Code editor showing the `Settings` dataclass and `get_settings()` function, then explanation of each field.

> **Host:** "We start with a typed settings object loaded from environment variables. This is our configuration layer — immutable, type-safe, and easy to test."
>
> ```python
> @dataclass(frozen=True)  # frozen = immutable; safe to pass anywhere
> class Settings:
>     llm_provider: str    # "ollama" | "openai"
>     llm_model: str       # e.g. "qwen2.5-coder:7b" or "gpt-4o-mini"
>     ollama_base_url: str
>     openai_api_key: str
>     openai_base_url: str
> 
> def get_settings() -> Settings:
>     """Read settings from env, with sensible defaults for local Ollama."""
>     return Settings(
>         llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
>         llm_model=os.getenv("LLM_MODEL", "qwen3.5:2b"),
>         ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
>         openai_api_key=os.getenv("OPENAI_API_KEY", ""),
>         openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
>     )
> ```

**Key points:**
- `frozen=True` makes the dataclass immutable — safe to pass around without accidental mutation.
- Defaults favor local Ollama: `LLM_PROVIDER=ollama`, model defaults to a small local model.
- `.env` file is loaded at import time via `load_dotenv()` from python-dotenv.

---

## Scene 3: The Model Factory (4:30–7:30)

**On-screen:** Code showing the `get_model()` function with detailed walkthrough of each provider branch, then a live demo switching between providers.

> **Host:** "Now for the core — our model factory that creates the right chat model based on configuration."
>
> ```python
> def get_model(settings: Settings | None = None) -> BaseChatModel:
>     """Return a chat model for the configured provider.
>     
>     We pre-build the model ourselves (instead of passing a "provider:model"
>     string to create_deep_agent) so WE own the error messages — e.g. a clear
>     'OPENAI_API_KEY is required' hint instead of a cryptic stack trace.
>     """
>     s = settings or get_settings()
>     
>     if s.llm_provider == "ollama":
>         # Ollama runs locally; no API key needed. Make sure `ollama serve` is up.
>         return init_chat_model(
>             model=s.llm_model,
>             model_provider="ollama",
>             base_url=s.ollama_base_url,
>         )
>     
>     if s.llm_provider == "openai":
>         if not s.openai_api_key:
>             raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai. Set it in .env or `export OPENAI_API_KEY=...`.")
>         kwargs = {"model": s.llm_model, "model_provider": "openai"}
>         if s.openai_base_url:  # optional — for Azure/OpenAI-compatible endpoints
>             kwargs["base_url"] = s.openai_base_url
>         return init_chat_model(**kwargs)
>     
>     raise ValueError(f"Unknown LLM_PROVIDER={s.llm_provider!r}. Use 'ollama' or 'openai'.")
> ```

**Live demo:** Show switching providers:
```bash
# Local Ollama (default):
LLM_PROVIDER=ollama python 01-model-factory.py "Say hello in one sentence."

# Cloud OpenAI:
LLM_PROVIDER=openai OPENAI_API_KEY=your-key LLM_MODEL=gpt-4o-mini \
    python 01-model-factory.py "Say hello in one sentence."
```

**Key points:**
- `init_chat_model` is LangChain's provider-agnostic factory — same function for both providers.
- We pre-build the model so we own error messages (clear hints vs cryptic stack traces).
- OpenAI API key validation happens BEFORE any network call — fail fast with a helpful message.

---

## Scene 4: The Agent Builder (7:30–9:00)

**On-screen:** Code showing `build_agent()` and how it wraps `create_deep_agent`, then explanation of the Deep Agents architecture.

> **Host:** "With our model factory ready, we build the agent — a thin wrapper around Deep Agents' `create_deep_agent`."
>
> ```python
> def build_agent(model: BaseChatModel | None = None, system_prompt: str | None = None):
>     """Build a tool-free Deep Agent. Just the brain."""
>     m = model or get_model()
>     # tools=[] for now — the harness still ships built-in planning/filesystem
>     # middleware, but with no custom tools the agent just talks.
>     return create_deep_agent(
>         model=m,
>         tools=[],
>         system_prompt=system_prompt or "You are CodeIt, a helpful coding assistant.",
>     )
> ```

**Key points:**
- `create_deep_agent` returns a compiled LangGraph graph with `.invoke()`, `.stream()`, `.get_state()`.
- We pass `tools=[]` for now — the agent is just a chatbot in this episode.
- The system prompt gives it personality: "You are CodeIt, a helpful coding assistant."

---

## Scene 5: Running the Agent (9:00–10:30)

**On-screen:** Terminal session showing the full run with `agent.invoke()`, then explanation of each step.

> **Host:** "Let's see it running! We use `invoke()` — which runs the full agent loop to completion."
>
> ```python
> def main() -> None:
>     prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello in one sentence."
>     
>     agent = build_agent()
>     # thread_id identifies this conversation. Later episodes reuse it for
>     # memory and human-in-the-loop interrupts.
>     config = {"configurable": {"thread_id": "demo"}}
>     
>     result = agent.invoke(
>         {"messages": [{"role": "user", "content": prompt}]},
>         config=config,
>     )
>     
>     # The final state holds every message; the last one is the agent's reply.
>     print(result["messages"][-1].content)
> ```

**Live demo:** Show running with different prompts:
```bash
python 01-model-factory.py "Say hello in one sentence."
# Output: Hello! I'm CodeIt, a helpful coding assistant ready to help you with your projects.

python 01-model-factory.py "What is 2+2?"
# Output: 2 + 2 = 4
```

**Key points:**
- `thread_id` identifies the conversation — later episodes use it for memory and interrupts.
- `invoke()` runs synchronously to completion (vs `stream()` which we'll see in Episode 2).
- The final state contains all messages; we extract the last one as the reply.

---

## Scene 6: Environment Variables & .env Files (10:30–11:30)

**On-screen:** Code showing `.env.example` and how `load_dotenv()` works, then a demo of creating a `.env` file.

> **Host:** "Configuration is managed through environment variables — we load them from a `.env` file at import time."
>
> ```python
> # .env.example
> LLM_PROVIDER=ollama          # or: openai
> LLM_MODEL=qwen2.5-coder:7b   # or: gpt-4o-mini
> OLLAMA_BASE_URL=http://localhost:11434
> OPENAI_API_KEY=your-key-here  # required when LLM_PROVIDER=openai
> ```

**Key points:**
- `load_dotenv()` reads `.env` once at import — optional but convenient for local development.
- Environment variables on the command line also work (no `.env` needed).
- The `.env.example` file documents all available configuration options.

---

## Scene 7: Wrap-up & What's Next (11:30–12:00)

**On-screen:** Summary showing all model factory capabilities and preview of Episode 2.

> **Host:** "In this episode, we've built the foundation — a provider-agnostic model factory that can switch between local Ollama and cloud OpenAI with just environment variables."
>
> - **`Settings` dataclass** — typed configuration loaded from env vars (immutable via `frozen=True`).
> - **`get_model()`** — one function, two providers, switched by `LLM_PROVIDER` env var.
> - **`build_agent()`** — thin wrapper around `create_deep_agent`, returns a compiled LangGraph graph.

**Key takeaways:**
1. Provider-agnostic model factory: switch between Ollama and OpenAI with just env vars.
2. Pre-build the model so you own error messages (clear hints vs cryptic stack traces).
3. Validate API keys BEFORE any network call — fail fast with helpful messages.
4. `create_deep_agent` returns a compiled LangGraph graph with `.invoke()`, `.stream()`, and `.get_state()`.
5. In Episode 2, we'll add streaming to make the agentic loop visible live.