"""Indexing of documents into the storage models."""

from collections.abc import Mapping, Sequence
from typing import Any

from django.db import transaction

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
        with transaction.atomic():
            for source_key, documents in groups.items():
                if documents:
                    _replace_source(source_key, documents, embeddings)
                else:
                    _remove_source(source_key)


def _replace_source(
    source_key: str, documents: Sequence[DocumentProtocol], embeddings: Embeddings
) -> None:
    """Store ``documents`` as the only content of the source ``source_key``."""
    source, _ = Source.objects.get_or_create(source_key=source_key)
    stored_vectors = _stored_vectors(source, embeddings.model)
    source.document_set.all().delete()
    stored_documents = _store_documents(source, documents)
    _store_chunks(chunk_group(documents), stored_documents, embeddings, stored_vectors)


def _remove_source(source_key: str) -> None:
    """Remove the source stored under ``source_key``, with its content."""
    Source.objects.filter(source_key=source_key).delete()


def _stored_vectors(source: Source, embedding_model: str) -> dict[str, Any]:
    """Return the vectors ``embedding_model`` gave the chunks of ``source``, by text."""
    return {
        chunk.text: chunk.embedding
        for chunk in Chunk.objects.filter(
            document__source=source, embedding_model=embedding_model
        )
    }


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
    stored_vectors: Mapping[str, Any],
) -> None:
    """Store ``chunks`` with their embeddings, each under its stored document."""
    vectors = _vectors(chunks, embeddings, stored_vectors)
    for chunk, vector in zip(chunks, vectors, strict=True):
        Chunk.objects.create(
            document=stored_documents[id(chunk.document)],
            rank=chunk.rank,
            text=chunk.text,
            embedding_model=embeddings.model,
            embedding=vector,
        )


def _vectors(
    chunks: Sequence[TextChunk],
    embeddings: Embeddings,
    stored_vectors: Mapping[str, Any],
) -> list[Any]:
    """Return one vector per chunk, embedding only the texts not yet stored.

    A chunk whose text was already embedded by the same model keeps its vector.
    """
    new_texts = [chunk.text for chunk in chunks if chunk.text not in stored_vectors]
    new_vectors = iter(embeddings.embed(new_texts) if new_texts else [])
    return [
        stored_vectors[chunk.text]
        if chunk.text in stored_vectors
        else next(new_vectors)
        for chunk in chunks
    ]
