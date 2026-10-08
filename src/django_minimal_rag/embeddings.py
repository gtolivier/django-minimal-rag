"""Embedding backends."""

from collections.abc import Sequence
from typing import Any, Protocol

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

FAKE_EMBEDDINGS_DEFAULT_DIMENSION = 8
EMBEDDINGS_SETTING = "MINIMAL_RAG_EMBEDDINGS"


class Embeddings(Protocol):
    """An embedding backend."""

    @property
    def model(self) -> str:
        """Name of the embedding model."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text."""
        ...


class FakeEmbeddings:
    """Deterministic embeddings for tests."""

    def __init__(self, dimensions: int = FAKE_EMBEDDINGS_DEFAULT_DIMENSION) -> None:
        if dimensions <= 0:
            msg = "dimensions must be positive"
            raise ValueError(msg)
        self.dimensions = dimensions
        self.model = f"fake-{dimensions}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text."""
        return [[float(sum(map(ord, text)))] * self.dimensions for text in texts]


def get_embeddings() -> Embeddings:
    """Return the backend configured by the MINIMAL_RAG_EMBEDDINGS setting."""
    try:
        config: dict[str, Any] = getattr(settings, EMBEDDINGS_SETTING)
    except AttributeError as error:
        msg = f"The {EMBEDDINGS_SETTING} setting is not set."
        raise ImproperlyConfigured(msg) from error
    try:
        path: str = config["BACKEND"]
    except KeyError as error:
        msg = f"The {EMBEDDINGS_SETTING} setting has no BACKEND."
        raise ImproperlyConfigured(msg) from error
    backend: type[Embeddings] = import_string(path)
    return backend(**config.get("OPTIONS", {}))
