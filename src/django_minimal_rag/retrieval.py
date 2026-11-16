"""Retrieval of indexed chunks."""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from django.db.models import Case, F, Q, QuerySet, When
from pgvector.django import CosineDistance

from django_minimal_rag.embeddings import Embeddings, get_embeddings
from django_minimal_rag.models import Chunk

DEFAULT_LIMIT = 5
MIN_LIMIT = 1
# The wildcard of RFC 4647 language ranges, which is not accepted.
_WILDCARD = "*"
# Separates the subtags of a language tag (BCP 47), as in "fr-CA".
_SUBTAG_SEPARATOR = "-"


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
    language: str | None = None,
) -> list[RetrievedChunk]:
    """Return the at most ``limit`` chunks nearest to ``question``, nearest first.

    The question is embedded by the configured backend, and only chunks
    embedded by the same model are compared to it. ``max_distance`` is the
    relevance threshold: a chunk whose cosine distance to the question is
    above it is left out. ``permissions`` are the names of the permissions
    the reader holds: a chunk is returned only if the reader holds every
    permission its document requires. If ``language`` is given, only chunks
    of documents in that language, one of its subtags (``"fr"`` covers
    ``"fr-CA"``, not ``"frr"``), or of unknown language, are compared.
    Chunks at the same distance are ordered by their document's id, then
    by their rank in it.

    Raises ``ValueError`` if ``limit`` is below ``MIN_LIMIT``, if ``language``
    is blank, holds whitespace or is the wildcard ``"*"`` (pass ``None`` to
    search every language), if the backend does not return exactly one
    vector for the question, or if that vector is all zeros.
    """
    _require_positive_limit(limit)
    _require_well_formed_language(language)
    embeddings = get_embeddings()
    question_embedding = _embed_question(question, embeddings)
    chunks = _candidate_chunks(embeddings.model, permissions, language)
    rows = (
        chunks.annotate(distance=_distance_to(question_embedding, embeddings.model))
        .order_by("distance", "document_id", "rank")
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


def _candidate_chunks(
    model: str, permissions: AbstractSet[str], language: str | None
) -> QuerySet[Chunk]:
    """The chunks of ``model`` the reader may see.

    If ``language`` is given, only those of documents in that language, one
    of its subtags, or of unknown language.
    """
    chunks = Chunk.objects.filter(
        embedding_model=model,
        document__permissions__contained_by=list(permissions),
    )
    if language is None:
        return chunks
    return chunks.filter(
        Q(document__language__iexact=language)
        | Q(document__language__istartswith=f"{language}{_SUBTAG_SEPARATOR}")
        | Q(document__language__isnull=True)
    )


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


def _require_well_formed_language(language: str | None) -> None:
    """Raise ``ValueError`` if ``language`` is blank, holds whitespace or is ``"*"``."""
    if language is None:
        return
    if not language.strip():
        msg = "language must not be blank"
        raise ValueError(msg)
    if any(char.isspace() for char in language):
        msg = "language must not contain whitespace"
        raise ValueError(msg)
    if language == _WILDCARD:
        msg = f'language must not be "{_WILDCARD}"; pass None to search every language'
        raise ValueError(msg)


def _embed_question(question: str, embeddings: Embeddings) -> list[float]:
    """Return the one vector ``embeddings`` gives ``question``.

    Raises ``ValueError`` when the backend returns another number of vectors,
    or a vector of all zeros, which has no direction to compare chunks to.
    """
    vectors = embeddings.embed([question])
    if len(vectors) != 1:
        msg = f"expected 1 vector for the question, got {len(vectors)}"
        raise ValueError(msg)
    (question_embedding,) = vectors
    if not any(question_embedding):
        msg = "the question's vector is all zeros, it has no direction"
        raise ValueError(msg)
    return question_embedding
