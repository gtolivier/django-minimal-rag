"""Embedding backends."""

FAKE_EMBEDDINGS_DEFAULT_DIMENSION = 8


class FakeEmbeddings:
    """Deterministic embeddings for tests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text."""
        return [[0.0] * FAKE_EMBEDDINGS_DEFAULT_DIMENSION for _ in texts]
