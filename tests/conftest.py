"""Shared pytest fixtures for the CodeIt test suite."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


@pytest.fixture
def fake_model() -> GenericFakeChatModel:
    """A scripted chat model that never touches the network.

    Tests that need a different reply sequence can construct their own
    ``GenericFakeChatModel``; this fixture gives a sane default ("hi").
    """
    return GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))


@pytest.fixture
def tmp_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A temp directory pointed at by ``CODEIT_WORKDIR``.

    Used from Ep 4 on; harmless to register now so later episodes inherit it.
    """
    monkeypatch.setenv("CODEIT_WORKDIR", str(tmp_path))
    yield tmp_path
