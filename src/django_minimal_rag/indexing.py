"""Indexing of documents into the storage models."""

from collections.abc import Mapping, Sequence
from typing import Any

from django.db import transaction

from django_minimal_rag.chunking import split_text
from django_minimal_rag.documents import Document as DocumentProtocol
from django_minimal_rag.embeddings import Embeddings, get_embeddings
from django_minimal_rag.models import Chunk, Document, Source


class Indexer:
    """Stores documents, chunks and embeddings."""

    @transaction.atomic
    def replace(self, groups: Mapping[str, Sequence[DocumentProtocol]]) -> None:
        """Replace the indexed content with the given groups."""
        embeddings = get_embeddings() if any(groups.values()) else None
        for source_key, documents in groups.items():
            if documents and embeddings:
                _replace_source(source_key, documents, embeddings)
            else:
                _remove_source(source_key)


def _replace_source(
    source_key: str, documents: Sequence[DocumentProtocol], embeddings: Embeddings
) -> None:
    """Store ``documents`` as the only content of the source ``source_key``."""
    documents = list(documents)  # a group may be iterable only once
    source, _ = Source.objects.select_for_update().get_or_create(source_key=source_key)
    stored_vectors = _stored_vectors(source, embeddings.model)
    source.document_set.all().delete()
    stored_documents = _store_documents(source, documents)
    pieces = _split_documents(documents, stored_documents)
    _store_chunks(pieces, embeddings, stored_vectors)


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
) -> list[Document]:
    """Store the ``documents`` of ``source``, one stored document per occurrence."""
    return Document.objects.bulk_create(
        Document(
            source=source,
            title=document.title,
            url=document.url,
            language=document.language,
            permissions=sorted(document.permissions),
        )
        for document in documents
    )


def _split_documents(
    documents: Sequence[DocumentProtocol], stored_documents: Sequence[Document]
) -> list[tuple[Document, str]]:
    """Split each of ``documents`` into texts, each paired with its stored document.

    ``stored_documents`` holds the stored document of each of ``documents``, in
    order.
    """
    return [
        (stored_document, text)
        for document, stored_document in zip(documents, stored_documents, strict=True)
        for text in split_text(document.text)
    ]


def _store_chunks(
    pieces: Sequence[tuple[Document, str]],
    embeddings: Embeddings,
    stored_vectors: Mapping[str, Any],
) -> None:
    """Store the ``pieces`` of text with their embeddings, ranked in order.

    Each piece is a stored document and a text of it.
    """
    vectors = _vectors([text for _, text in pieces], embeddings, stored_vectors)
    Chunk.objects.bulk_create(
        Chunk(
            document=stored_document,
            rank=rank,
            text=text,
            embedding_model=embeddings.model,
            embedding=vector,
        )
        for rank, ((stored_document, text), vector) in enumerate(
            zip(pieces, vectors, strict=True)
        )
    )


def _vectors(
    texts: Sequence[str],
    embeddings: Embeddings,
    stored_vectors: Mapping[str, Any],
) -> list[Any]:
    """Return one vector per text of ``texts``, embedding only those not yet stored.

    A text already embedded by the same model keeps its vector.
    """
    new_texts = [text for text in texts if text not in stored_vectors]
    new_vectors = iter(embeddings.embed(new_texts) if new_texts else [])
    return [
        stored_vectors[text] if text in stored_vectors else next(new_vectors)
        for text in texts
    ]
