"""Embedding backends."""

FAKE_EMBEDDINGS_DEFAULT_DIMENSION = 8


class FakeEmbeddings:
    """Deterministic embeddings for tests."""

    def __init__(self, dimensions: int = FAKE_EMBEDDINGS_DEFAULT_DIMENSION) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text."""
        return [[0.0] * self.dimensions for _ in texts]
