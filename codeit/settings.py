"""Typed config singleton loaded from ``.env`` once at import time."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # reads .env once at import time


@dataclass(frozen=True)
class Settings:
    """Immutable settings; safe to pass anywhere. Built by :func:`get_settings`."""

    llm_provider: str  # "ollama" | "openai"
    llm_model: str  # e.g. "qwen2.5-coder:7b" or "gpt-4o-mini"
    ollama_base_url: str
    openai_api_key: str
    openai_base_url: str
    workdir: str
    auto_approve: bool
    max_iters: int


def get_settings() -> Settings:
    """Load settings from the environment.

    A function (not a module-level constant) so tests can monkeypatch env
    vars and call it fresh.
    """
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
