"""Embedding backends."""

import zlib
from collections.abc import Sequence
from typing import Protocol

from django_minimal_rag.backends import load_backend

FAKE_EMBEDDINGS_DEFAULT_DIMENSION = 8
FAKE_EMBEDDINGS_MIN_DIMENSION = 2
# Keeps 24 bits: every such integer is exactly representable in float32.
FLOAT32_EXACT_MASK = 0xFFFFFF
EMBEDDINGS_SETTING = "MINIMAL_RAG_EMBEDDINGS"


class Embeddings(Protocol):
    """An embedding backend."""

    @property
    def model(self) -> str:
        """Name of the embedding model."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, in the order of ``texts``.

        ``texts`` may hold any number of texts: a backend whose API limits the
        size of a request splits ``texts`` into batches itself.
        """
        ...


class FakeEmbeddings:
    """Deterministic embeddings for tests."""

    def __init__(self, dimensions: int = FAKE_EMBEDDINGS_DEFAULT_DIMENSION) -> None:
        if dimensions < FAKE_EMBEDDINGS_MIN_DIMENSION:
            msg = f"dimensions must be at least {FAKE_EMBEDDINGS_MIN_DIMENSION}"
            raise ValueError(msg)
        self.dimensions = dimensions
        self.model = f"fake-{dimensions}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text."""
        return [self._embed_one(text.encode()) for text in texts]

    def _embed_one(self, encoded_text: bytes) -> list[float]:
        return [
            float(zlib.crc32(encoded_text, index) & FLOAT32_EXACT_MASK)
            for index in range(self.dimensions)
        ]


def get_embeddings() -> Embeddings:
    """Return the backend configured by the MINIMAL_RAG_EMBEDDINGS setting."""
    embeddings: Embeddings = load_backend(EMBEDDINGS_SETTING)
    return embeddings
