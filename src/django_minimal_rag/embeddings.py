"""Embedding backends."""

from collections.abc import Sequence
from typing import Protocol

FAKE_EMBEDDINGS_DEFAULT_DIMENSION = 8


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
