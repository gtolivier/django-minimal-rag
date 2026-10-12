"""LLM backends."""

from collections.abc import Mapping
from typing import Any, Protocol

from django.conf import settings
from django.utils.module_loading import import_string


class LLM(Protocol):
    """What the package needs from an LLM backend."""

    def complete(self, prompt: str, *, system: str) -> str:
        """Return the LLM's answer to the prompt."""
        ...


def get_llm() -> LLM:
    """Return the backend configured by the MINIMAL_RAG_LLM setting."""
    config: Mapping[str, Any] = getattr(settings, "MINIMAL_RAG_LLM")  # noqa: B009 - the setting is a host project's, unknown to the django-stubs plugin
    backend: type[LLM] = import_string(config["BACKEND"])
    return backend(**config.get("OPTIONS", {}))


class FakeLLM:
    """An LLM for tests: it answers with the prompt it was given."""

    def complete(self, prompt: str, *, system: str) -> str:  # noqa: ARG002 - part of the LLM interface, unused by the fake
        """Return the prompt itself."""
        return prompt
