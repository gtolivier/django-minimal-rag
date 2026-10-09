"""Retrieval of indexed chunks."""

from collections.abc import Collection

from pgvector.django import CosineDistance

from django_minimal_rag.embeddings import get_embeddings
from django_minimal_rag.models import Chunk

DEFAULT_LIMIT = 5


def retrieve(
    query: str,
    *,
    permissions: Collection[str],  # noqa: ARG001 - not used yet, the tests so far need no permissions
    max_distance: float,  # noqa: ARG001 - idem
    limit: int = DEFAULT_LIMIT,
) -> list[Chunk]:
    """Return the chunks relevant to ``query``, each with its ``distance``."""
    (query_embedding,) = get_embeddings().embed([query])
    return list(
        # The django-stubs plugin objects to annotating the name Chunk declares
        # for type checkers (no-redef), and cannot resolve it as a field when
        # ordering (misc); at runtime the annotation is what sets it.
        Chunk.objects.annotate(  # type: ignore[no-redef,misc]
            distance=CosineDistance("embedding", query_embedding)
        ).order_by("distance")[:limit]
    )
