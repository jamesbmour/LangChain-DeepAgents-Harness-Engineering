# Episode 2 — The Agentic Loop, Made Visible (Tutorial Video Script)

## Overview
**Length:** ~13 minutes  
**Goal:** Show viewers the fundamental difference between a chatbot and an agent: the model DECIDES to call tools autonomously. We drive the Deep Agents graph with `agent.stream(stream_mode="updates", version="v2")` so each step of the loop prints live — model request → tool call → tool result → final answer.

---

## Scene 1: Hook & Chatbot vs Agent (0:00–2:00)

**On-screen:** Side-by-side comparison showing a chatbot responding to "What time is it?" with training data, versus an agent that calls a `get_time` tool and returns the actual current time.

> **Host:** "Here's the fundamental question: what makes an AI agent different from a chatbot? A chatbot responds from its training data — it knows facts up to its cutoff date but can't interact with the world."
>
> *(Show chatbot saying 'I don't have access to real-time information')*

**Key points:**
- Chatbots are stateless responders — they answer from training data.
- Agents use tools to interact with the world and make decisions autonomously.
- The key difference: **the model decides when to call a tool**.

---

## Scene 2: The get_time Tool (2:00–4:30)

**On-screen:** Code editor showing the `get_time` tool implementation, then explanation of how `@tool` decorator works.

> **Host:** "Let's look at our demo tool — `get_time`. It's a simple function decorated with LangChain's `@tool`, which turns it into an agent-callable tool."
>
> ```python
> from langchain.tools import tool
> 
> @tool
> def get_time() -> str:
>     """Return the current time. Use this when the user asks for the time."""
>     import datetime
>     return datetime.datetime.now().isoformat(timespec="seconds")
> ```

**Key points:**
- The `@tool` decorator turns a plain function into a LangChain tool.
- **The docstring is FOR THE MODEL** — it tells the model WHEN to call this tool and what it returns. Tool docstrings are prompts.
- This is the simplest possible demonstration of agentic behavior: the model decides to call `get_time`, gets the result, and incorporates it into its answer.

---

## Scene 3: The Streaming Driver (4:30–7:30)

**On-screen:** Code showing `_config()`, `_print_event()`, and `run()` functions with detailed walkthrough of each part.

> **Host:** "Now for the core — our streaming driver that makes every step visible."
>
> ```python
> def _config(thread_id: str) -> dict:
>     # NOTE: recursion_limit is a TOP-LEVEL config key, NOT inside configurable.
>     max_iters = int(os.getenv("CODEIT_MAX_ITERS", "25"))
>     return {
>         "configurable": {"thread_id": thread_id},
>         "recursion_limit": max_iters * 2,
>     }
> ```

**Then the event printer:**
```python
def _print_event(chunk) -> None:
    """Pretty-print one v2 stream chunk. v2 chunks are dicts with type/ns/data."""
    kind = chunk.get("type")
    if kind != "updates":
        return
    for node_name, state in chunk.get("data", {}).items():
        if not isinstance(state, dict):
            continue
        for msg in state.get("messages", []):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    console.print(f"[cyan]tool call:[/cyan] {tc['name']}({tc['args']})")
            elif isinstance(msg, AIMessage) and msg.content:
                console.print(f"[green]assistant:[/green] {msg.content}")
```

**Key points:**
- `recursion_limit` is a top-level config key — it caps super-steps to prevent infinite loops.
- v2 stream chunks are dicts with `type`, `ns`, and `data` fields.
- We filter for `"updates"` type events only (not all event types).
- Tool calls print in cyan, assistant messages in green — visual distinction helps viewers follow along.

---

## Scene 4: The run() Function (7:30–9:00)

**On-screen:** Code showing the `run()` function with error handling and final state retrieval.

> **Host:** "The `run` function ties it all together — streaming, rendering, and error handling."
>
> ```python
> def run(agent, prompt: str, thread_id: str = "default") -> dict:
>     """Drive the agent with streaming so each node event prints live."""
>     config = _config(thread_id)
>     try:
>         for chunk in agent.stream(
>             {"messages": [{"role": "user", "content": prompt}]},
>             config=config, stream_mode="updates", version="v2"):
>             _print_event(chunk)
>     except Exception as e:
>         # GraphRecursionError or model errors — print clearly, don't crash.
>         console.print(f"[red]error:[/red] {type(e).__name__}: {e}")
>     return agent.get_state(config).values
> ```

**Key points:**
- `stream_mode="updates"` gives us per-node state changes — perfect for live rendering.
- `version="v2"` uses the newer typed dict format (vs v1's bare `{node: state}`).
- Broad exception handling catches `GraphRecursionError` and model errors gracefully.
- Final state is retrieved via `agent.get_state(config).values`.

---

## Scene 5: Live Demo — The Agentic Loop in Action (9:00–11:30)

**On-screen:** Terminal session showing the agent receiving a prompt, deciding to call `get_time`, executing it, and returning the answer.

> **Host:** "Let's see this working! We'll ask our agent what time it is."
>
> ```bash
> LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
>     python 02-agentic-loop.py "What time is it? Use your tool."
> ```

**What viewers will see:**
1. **Tool call appears in cyan**: `tool call: get_time({})` — the model DECIDED to use the tool!
2. **Assistant response in green**: The actual current time, incorporating the tool's output.
3. **Final answer section**: Clean summary of what happened.

**Key points:**
- This is the "aha!" moment — viewers see the agent making autonomous decisions.
- The model didn't just say "I don't know" — it called a tool and got real data.
- Each step prints live, so viewers can follow the reasoning process.

---

## Scene 6: Recursion Limit & Safety (11:30–12:30)

**On-screen:** Text explaining recursion_limit and how it prevents infinite loops.

> **Host:** "One critical detail — without a recursion limit, a looping model could hang forever."
>
> ```python
> # recursion_limit counts super-steps (model call + tool exec ≈ 2)
> # We cap it so a looping model can't hang the viewer's machine.
> max_iters = int(os.getenv("CODEIT_MAX_ITERS", "25"))
> return {"configurable": {"thread_id": thread_id}, "recursion_limit": max_iters * 2}
> ```

**Key points:**
- `recursion_limit` is a safety mechanism — it caps the number of super-steps.
- Default is 25 iterations × 2 = 50 super-steps, configurable via env var.
- Without this, an agent could loop forever calling tools without converging.

---

## Scene 7: Wrap-up & What's Next (12:30–13:00)

**On-screen:** Summary showing the agentic loop capabilities and preview of Episode 3.

> **Host:** "In this episode, we've crossed the threshold from chatbot to agent."
>
> - **`get_time` tool** — proves the model can decide to call tools autonomously.
> - **Streaming driver** — `agent.stream(stream_mode="updates", version="v2")` makes every step visible live.
> - **Recursion limit** — safety mechanism preventing infinite loops.

**Key takeaways:**
1. The fundamental difference between chatbots and agents: the model DECIDES to call tools autonomously.
2. Tool docstrings are prompts — they tell the model WHEN and HOW to use each tool.
3. `stream_mode="updates"` with `version="v2"` gives per-node state changes for live rendering.
4. Always set a recursion limit to prevent infinite loops from misbehaving models.