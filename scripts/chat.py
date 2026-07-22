#!/usr/bin/env python3
"""One-shot CLI demo: send a prompt, print the reply."""

import sys

from codeit.agent import build_agent


def main() -> None:
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
