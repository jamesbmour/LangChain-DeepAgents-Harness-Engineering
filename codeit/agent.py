"""Thin ``build_agent()`` wrapper around ``create_deep_agent`` — the brain."""

from collections.abc import Callable

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from codeit.model import get_model
from codeit.settings import get_settings


def build_agent(
    model: BaseChatModel | None = None,
    tools: list[BaseTool | Callable] | None = None,
    system_prompt: str | None = None,
):
    """Build a Deep Agent. No tools, no backend, no interrupts yet — just the brain.

    ``model`` is accepted as an arg so tests can inject a fake model without
    touching ``.env``. Returns a compiled LangGraph graph (``.invoke``,
    ``.stream``, ``.astream``, ``.get_state``).
    """
    m = model or get_model()
    _ = get_settings()  # ensure .env loaded; settings used by later episodes
    return create_deep_agent(
        model=m,
        tools=tools or [],
        system_prompt=system_prompt or "You are CodeIt, a helpful coding assistant.",
    )
