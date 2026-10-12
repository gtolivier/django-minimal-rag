"""LLM backends."""

from typing import Protocol

from django_minimal_rag.backends import load_backend

LLM_SETTING = "MINIMAL_RAG_LLM"


class LLM(Protocol):
    """What the package needs from an LLM backend."""

    def complete(self, prompt: str, *, system: str) -> str:
        """Return the LLM's answer to the prompt."""
        ...


def get_llm() -> LLM:
    """Return the backend configured by the MINIMAL_RAG_LLM setting."""
    llm: LLM = load_backend(LLM_SETTING)
    return llm


class FakeLLM:
    """An LLM for tests: it answers with the prompt it was given."""

    def complete(self, prompt: str, *, system: str) -> str:  # noqa: ARG002 - part of the LLM interface, unused by the fake
        """Return the prompt itself."""
        return prompt
