"""Retrieval of indexed chunks."""

from collections.abc import Collection

from pgvector.django import CosineDistance

from django_minimal_rag.embeddings import get_embeddings
from django_minimal_rag.models import Chunk

DEFAULT_LIMIT = 5
MIN_LIMIT = 1


def retrieve(
    query: str,
    *,
    permissions: Collection[str],
    max_distance: float,
    limit: int = DEFAULT_LIMIT,
) -> list[Chunk]:
    """Return the chunks relevant to ``query``, each with its ``distance``."""
    _require_positive_limit(limit)
    embeddings = get_embeddings()
    (query_embedding,) = embeddings.embed([query])
    return list(
        # The django-stubs plugin objects to annotating the name Chunk declares
        # for type checkers (no-redef), and cannot resolve it as a field when
        # ordering (misc); at runtime the annotation is what sets it.
        # Filtering on the model first keeps vectors of other dimensions out of
        # the distance computation, which would fail on them.
        Chunk.objects.filter(
            embedding_model=embeddings.model,
            document__permissions__contained_by=list(permissions),
        )
        .annotate(  # type: ignore[no-redef,misc]
            distance=CosineDistance("embedding", query_embedding)
        )
        .filter(distance__lte=max_distance)
        .order_by("distance")[:limit]
    )


def _require_positive_limit(limit: int) -> None:
    """Raise ``ValueError`` if ``limit`` is below ``MIN_LIMIT``."""
    if limit < MIN_LIMIT:
        msg = f"limit must be at least {MIN_LIMIT}, got {limit}"
        raise ValueError(msg)
