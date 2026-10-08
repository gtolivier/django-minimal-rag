"""Indexing of documents into the storage models."""

from collections.abc import Mapping, Sequence

from django_minimal_rag.chunking import chunk_group
from django_minimal_rag.documents import Document as DocumentProtocol
from django_minimal_rag.embeddings import get_embeddings
from django_minimal_rag.models import Chunk, Document, Source


class Indexer:
    """Stores documents, chunks and embeddings."""

    def replace(self, groups: Mapping[str, Sequence[DocumentProtocol]]) -> None:
        """Replace the indexed content with the given groups."""
        embeddings = get_embeddings()
        for source_key, documents in groups.items():
            source = Source.objects.create(source_key=source_key)
            stored = {
                id(document): Document.objects.create(
                    source=source,
                    title=document.title,
                    url=document.url,
                    language=document.language,
                    permissions=sorted(document.permissions),
                )
                for document in documents
            }
            chunks = chunk_group(documents)
            vectors = embeddings.embed([chunk.text for chunk in chunks])
            for chunk, vector in zip(chunks, vectors, strict=True):
                Chunk.objects.create(
                    document=stored[id(chunk.document)],
                    rank=chunk.rank,
                    text=chunk.text,
                    embedding_model=embeddings.model,
                    embedding=vector,
                )
