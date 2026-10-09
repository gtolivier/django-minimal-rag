"""Retrieval of indexed chunks."""

from collections.abc import Collection
from typing import Any


def retrieve(
    query: str,  # noqa: ARG001 - not used yet, the first test needs no query
    *,
    permissions: Collection[str],  # noqa: ARG001 - idem
    max_distance: float,  # noqa: ARG001 - idem
) -> list[Any]:
    """Return the chunks relevant to ``query``."""
    return []
