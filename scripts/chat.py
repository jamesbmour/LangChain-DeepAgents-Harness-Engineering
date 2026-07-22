#!/usr/bin/env python3
"""One-shot CLI demo: stream the agent loop, then print the final answer."""

import sys

from codeit.agent import build_agent, run


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What time is it? Use your tool."
    agent = build_agent()
    state = run(agent, prompt)
    last = state.get("messages", [])[-1] if state else None
    if last is not None:
        print("\n--- final answer ---")
        print(getattr(last, "content", last))


if __name__ == "__main__":
    main()
