"""The test bench's embedding backend: one a host project would configure."""

from collections.abc import Sequence


class SampleEmbeddings:
    """An embedding backend defined by the test bench and shipped by no package."""

    model = "sample"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text."""
        return [[float(len(text))] for text in texts]
