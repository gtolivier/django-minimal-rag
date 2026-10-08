"""Indexing of documents into the storage models."""

from collections.abc import Mapping


class Indexer:
    """Stores documents, chunks and embeddings."""

    def replace(self, groups: Mapping[str, object]) -> None:
        """Replace the indexed content with the given groups."""
        del groups
