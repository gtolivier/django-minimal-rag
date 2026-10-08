"""Indexing of documents into the storage models."""

from collections.abc import Mapping, Sequence

from django_minimal_rag.documents import Document as DocumentProtocol
from django_minimal_rag.models import Document, Source


class Indexer:
    """Stores documents, chunks and embeddings."""

    def replace(self, groups: Mapping[str, Sequence[DocumentProtocol]]) -> None:
        """Replace the indexed content with the given groups."""
        for source_key, documents in groups.items():
            source = Source.objects.create(source_key=source_key)
            for document in documents:
                Document.objects.create(
                    source=source,
                    title=document.title,
                    url=document.url,
                    language=document.language,
                    permissions=sorted(document.permissions),
                )
