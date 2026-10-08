"""Embedding backends."""


class FakeEmbeddings:
    """Deterministic embeddings for tests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text."""
        return [[] for _ in texts]
