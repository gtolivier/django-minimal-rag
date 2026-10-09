"""Retrieval of indexed chunks."""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from django.db.models import Case, F, When
from pgvector.django import CosineDistance

from django_minimal_rag.embeddings import Embeddings, get_embeddings
from django_minimal_rag.models import Chunk

DEFAULT_LIMIT = 5
MIN_LIMIT = 1


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by ``retrieve()``, with its cosine ``distance``.

    ``title``, ``url`` and ``source_key`` are those of the chunk's document.
    """

    text: str
    title: str
    url: str
    source_key: str
    distance: float


def retrieve(
    question: str,
    *,
    permissions: AbstractSet[str],
    max_distance: float,
    limit: int = DEFAULT_LIMIT,
) -> list[RetrievedChunk]:
    """Return the at most ``limit`` chunks nearest to ``question``, nearest first.

    The question is embedded by the configured backend, and only chunks
    embedded by the same model are compared to it. ``max_distance`` is the
    relevance threshold: a chunk whose cosine distance to the question is
    above it is left out. ``permissions`` are the names of the permissions
    the reader holds: a chunk is returned only if the reader holds every
    permission its document requires.

    Raises ``ValueError`` if ``limit`` is below ``MIN_LIMIT``, or if the
    backend does not return exactly one vector for the question.
    """
    _require_positive_limit(limit)
    embeddings = get_embeddings()
    question_embedding = _embed_question(question, embeddings)
    rows = (
        Chunk.objects.filter(
            embedding_model=embeddings.model,
            document__permissions__contained_by=list(permissions),
        )
        .annotate(distance=_distance_to(question_embedding, embeddings.model))
        .order_by("distance")
        .values(
            "text",
            "distance",
            title=F("document__title"),
            url=F("document__url"),
            source_key=F("document__source__source_key"),
        )[:limit]
    )
    # The threshold is applied to the nearest rows, not in the query: a WHERE
    # on the distance would compute it a second time per row. Rows come
    # nearest first, so those within the threshold are the same either way.
    return [RetrievedChunk(**row) for row in rows if row["distance"] <= max_distance]


def _distance_to(question_embedding: list[float], model: str) -> Case:
    """The cosine distance of a chunk to the question, NULL if not of ``model``.

    Vectors of another model may have another dimension, on which the distance
    would fail. The conditions of a WHERE clause have no guaranteed evaluation
    order, whatever the order of the ``filter()`` calls; a CASE does.
    """
    return Case(
        When(
            embedding_model=model,
            then=CosineDistance("embedding", question_embedding),
        )
    )


def _require_positive_limit(limit: int) -> None:
    """Raise ``ValueError`` if ``limit`` is below ``MIN_LIMIT``."""
    if limit < MIN_LIMIT:
        msg = f"limit must be at least {MIN_LIMIT}, got {limit}"
        raise ValueError(msg)


def _embed_question(question: str, embeddings: Embeddings) -> list[float]:
    """Return the one vector ``embeddings`` gives ``question``.

    Raises ``ValueError`` when the backend returns another number of vectors.
    """
    vectors = embeddings.embed([question])
    if len(vectors) != 1:
        msg = f"expected 1 vector for the question, got {len(vectors)}"
        raise ValueError(msg)
    (question_embedding,) = vectors
    return question_embedding
