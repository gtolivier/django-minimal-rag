"""The test bench's embedding backend: one a host project would configure."""

from collections.abc import Sequence


class SampleEmbeddings:
    """An embedding backend defined by the test bench and shipped by no package."""

    model = "sample"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text."""
        return [[float(len(text))] for text in texts]


class RecordingEmbeddings:
    """A backend appending every text it embeds to the ``embedded`` list.

    The list is given through the setting's OPTIONS, so it outlives the
    backend instances built on each call. A text's vector is its position in
    that list: embedding the same text again gives it a new vector.
    """

    model = "recording"

    def __init__(self, embedded: list[str]) -> None:
        self.embedded = embedded

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Record ``texts`` and return one vector per text."""
        vectors = []
        for text in texts:
            vectors.append([float(len(self.embedded)), 1.0])
            self.embedded.append(text)
        return vectors
