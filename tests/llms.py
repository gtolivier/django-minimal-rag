"""The test bench's LLM backends, and helpers configuring them for a test."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_django import Settings


class CannedLLM:
    """An LLM backend answering every prompt with the ``reply`` it was built with.

    The reply is given through the setting's OPTIONS, so a test decides what
    the LLM answers.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, prompt: str, *, system: str) -> str:
        """Return the canned reply, whatever the prompt."""
        return self.reply


class RecordingLLM:
    """An LLM backend appending each ``(prompt, system)`` it is given to ``calls``.

    The list is given through the setting's OPTIONS, so it outlives the
    backend instances built on each call. Every prompt is answered with
    ``reply``.
    """

    def __init__(self, calls: list[tuple[str, str]], reply: str = "") -> None:
        self.calls = calls
        self.reply = reply

    def complete(self, prompt: str, *, system: str) -> str:
        """Record the prompt and the system message, and return ``reply``."""
        self.calls.append((prompt, system))
        return self.reply


def use_canned_llm(settings: "Settings", reply: str) -> None:
    """Configure ``CannedLLM``, answering every prompt with ``reply``."""
    settings.MINIMAL_RAG_LLM = {
        "BACKEND": "tests.llms.CannedLLM",
        "OPTIONS": {"reply": reply},
    }


def remove_llm_setting(settings: "Settings") -> None:
    """Leave the LLM setting missing, whether or not the test settings define it.

    It is set first so the deletion works either way.
    """
    settings.MINIMAL_RAG_LLM = {"BACKEND": "tests.llms.CannedLLM"}
    del settings.MINIMAL_RAG_LLM


@pytest.fixture
def llm_calls(settings: "Settings") -> list[tuple[str, str]]:
    """Configure ``RecordingLLM`` and return the ``(prompt, system)`` it records."""
    calls: list[tuple[str, str]] = []
    settings.MINIMAL_RAG_LLM = {
        "BACKEND": "tests.llms.RecordingLLM",
        "OPTIONS": {"calls": calls},
    }
    return calls
