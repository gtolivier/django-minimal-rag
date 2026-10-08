"""Indexing of documents into the storage models."""

from collections.abc import Mapping

from django_minimal_rag.models import Source


class Indexer:
    """Stores documents, chunks and embeddings."""

    def replace(self, groups: Mapping[str, object]) -> None:
        """Replace the indexed content with the given groups."""
        for source_key in groups:
            Source.objects.create(source_key=source_key)
