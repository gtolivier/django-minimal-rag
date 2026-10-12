"""Answering questions from indexed chunks."""

from collections.abc import Set as AbstractSet


def answer(
    question: str,  # noqa: ARG001 - part of the public signature, not used yet
    *,
    permissions: AbstractSet[str],  # noqa: ARG001 - part of the public signature, not used yet
    max_distance: float,  # noqa: ARG001 - part of the public signature, not used yet
) -> str | None:
    """Return None: nothing is indexed to answer from."""
    return None
