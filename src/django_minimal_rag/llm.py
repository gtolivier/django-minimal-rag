"""LLM backends."""

from collections.abc import Mapping
from typing import Any, Protocol

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from django_minimal_rag.embeddings import BACKEND_KEY, OPTIONS_KEY

LLM_SETTING = "MINIMAL_RAG_LLM"


class LLM(Protocol):
    """What the package needs from an LLM backend."""

    def complete(self, prompt: str, *, system: str) -> str:
        """Return the LLM's answer to the prompt."""
        ...


def get_llm() -> LLM:
    """Return the backend configured by the MINIMAL_RAG_LLM setting."""
    config = _llm_config()
    backend: type[LLM] = import_string(config[BACKEND_KEY])
    return backend(**config.get(OPTIONS_KEY, {}))


def _llm_config() -> Mapping[str, Any]:
    try:
        config: Mapping[str, Any] = getattr(settings, LLM_SETTING)
    except AttributeError as error:
        msg = f"The {LLM_SETTING} setting is required."
        raise ImproperlyConfigured(msg) from error
    return config


class FakeLLM:
    """An LLM for tests: it answers with the prompt it was given."""

    def complete(self, prompt: str, *, system: str) -> str:  # noqa: ARG002 - part of the LLM interface, unused by the fake
        """Return the prompt itself."""
        return prompt
