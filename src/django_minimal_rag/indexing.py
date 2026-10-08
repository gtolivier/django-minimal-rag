"""Indexing of documents into the storage models."""

from collections.abc import Mapping, Sequence

from django_minimal_rag.chunking import Chunk as TextChunk
from django_minimal_rag.chunking import chunk_group
from django_minimal_rag.documents import Document as DocumentProtocol
from django_minimal_rag.embeddings import Embeddings, get_embeddings
from django_minimal_rag.models import Chunk, Document, Source


class Indexer:
    """Stores documents, chunks and embeddings."""

    def replace(self, groups: Mapping[str, Sequence[DocumentProtocol]]) -> None:
        """Replace the indexed content with the given groups."""
        embeddings = get_embeddings()
        for source_key, documents in groups.items():
            if not documents:
                Source.objects.filter(source_key=source_key).delete()
                continue
            source, _ = Source.objects.get_or_create(source_key=source_key)
            source.document_set.all().delete()
            stored_documents = _store_documents(source, documents)
            _store_chunks(chunk_group(documents), stored_documents, embeddings)


def _store_documents(
    source: Source, documents: Sequence[DocumentProtocol]
) -> dict[int, Document]:
    """Store the ``documents`` of ``source``, keyed by the ``id`` of each.

    They are keyed by identity because a document may be unhashable.
    """
    return {
        id(document): Document.objects.create(
            source=source,
            title=document.title,
            url=document.url,
            language=document.language,
            permissions=sorted(document.permissions),
        )
        for document in documents
    }


def _store_chunks(
    chunks: Sequence[TextChunk],
    stored_documents: Mapping[int, Document],
    embeddings: Embeddings,
) -> None:
    """Store ``chunks`` with their embeddings, each under its stored document."""
    vectors = embeddings.embed([chunk.text for chunk in chunks])
    for chunk, vector in zip(chunks, vectors, strict=True):
        Chunk.objects.create(
            document=stored_documents[id(chunk.document)],
            rank=chunk.rank,
            text=chunk.text,
            embedding_model=embeddings.model,
            embedding=vector,
        )
