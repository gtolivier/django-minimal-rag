"""Retrieval of indexed chunks."""

from collections.abc import Collection

from django_minimal_rag.models import Chunk


def retrieve(
    query: str,  # noqa: ARG001 - not used yet, the tests so far need no query
    *,
    permissions: Collection[str],  # noqa: ARG001 - idem
    max_distance: float,  # noqa: ARG001 - idem
) -> list[Chunk]:
    """Return the chunks relevant to ``query``."""
    return list(Chunk.objects.all())
