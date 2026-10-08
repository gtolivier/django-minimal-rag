"""Embedding backends."""

from collections.abc import Sequence
from typing import Protocol

from django.conf import settings
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
    config: dict[str, str] = getattr(settings, EMBEDDINGS_SETTING)
    backend: type[Embeddings] = import_string(config["BACKEND"])
    return backend()
