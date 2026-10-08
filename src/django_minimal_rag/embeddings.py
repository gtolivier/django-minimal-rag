"""Embedding backends."""


class FakeEmbeddings:
    """Deterministic embeddings for tests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text."""
        return [[0.0] * 8 for _ in texts]
