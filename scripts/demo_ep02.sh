#!/usr/bin/env bash
set -euo pipefail

echo "=== Ep 2: the agent calls a tool, live ==="
echo "Expected output: a [tool call: get_time({})] line, then [tool result: get_time -> ...],"
echo "then [assistant: ...] with the final answer."
echo
LLM_PROVIDER=ollama LLM_MODEL="${LLM_MODEL:-qwen2.5-coder:7b}" \
  python scripts/chat.py "What time is it? Use your tool."