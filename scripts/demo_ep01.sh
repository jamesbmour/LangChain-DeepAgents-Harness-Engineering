#!/usr/bin/env bash
set -euo pipefail

echo "=== Ollama ==="
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b \
  python scripts/chat.py "What is 2+2? Answer in one word."
echo

echo "=== OpenAI ==="
LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini \
  python scripts/chat.py "What is 2+2? Answer in one word."