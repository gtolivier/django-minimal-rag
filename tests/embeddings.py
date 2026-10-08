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

    When a ``calls`` list is given too, each ``embed()`` call also appends the
    list of the texts it was given to it.
    """

    model = "recording"

    def __init__(
        self, embedded: list[str], calls: list[list[str]] | None = None
    ) -> None:
        self.embedded = embedded
        self.calls = calls

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Record ``texts`` and return one vector per text."""
        if self.calls is not None:
            self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vectors.append([float(len(self.embedded)), 1.0])
            self.embedded.append(text)
        return vectors


class EmbeddingFailedError(Exception):
    """The error raised by ``FailingEmbeddings``."""


class FailingEmbeddings:
    """A backend raising ``EmbeddingFailedError`` when asked to embed ``failing_text``.

    Any other text gets a vector, so the texts embedded before it succeed.
    """

    model = "failing"

    def __init__(self, failing_text: str) -> None:
        self.failing_text = failing_text

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, or raise if ``failing_text`` is among them."""
        if self.failing_text in texts:
            msg = f"Cannot embed {self.failing_text!r}."
            raise EmbeddingFailedError(msg)
        return [[float(len(text)), 1.0] for text in texts]


class MiscountingEmbeddings:
    """A backend returning ``vector_count`` vectors, however many texts it is given.

    Set ``vector_count`` below or above the number of texts embedded to get
    fewer or more vectors than asked.
    """

    model = "miscounting"

    def __init__(self, vector_count: int) -> None:
        self.vector_count = vector_count

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return ``vector_count`` vectors, ignoring ``texts``."""
        return [[float(position), 1.0] for position in range(self.vector_count)]
