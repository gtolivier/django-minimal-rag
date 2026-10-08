"""Indexing of documents into the storage models."""

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

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
        if not any(groups.values()):
            for source_key in sorted(groups):
                _remove_source(source_key)
            return
        embeddings = get_embeddings()
        plans = []
        for source_key, documents in sorted(groups.items()):
            if documents:
                plans.append(_prepare_source(source_key, documents, embeddings))
            else:
                _remove_source(source_key)
        _store_all_chunks(plans, embeddings)


class _Plan(NamedTuple):
    """The chunks a source still needs, once its documents are stored."""

    pieces: list[tuple[Document, str]]
    """Each text to store as a chunk, paired with its stored document, in order."""
    stored_vectors: dict[str, Any]
    """The vectors the source's previous chunks had, by text."""


def _prepare_source(
    source_key: str, documents: Sequence[DocumentProtocol], embeddings: Embeddings
) -> _Plan:
    """Store ``documents`` as the only documents of ``source_key``, without chunks.

    Return the pieces to chunk and the vectors already stored for the source.
    """
    documents = list(documents)  # a group may be iterable only once
    _check_source_keys(source_key, documents)
    source, _ = Source.objects.select_for_update().get_or_create(source_key=source_key)
    stored_vectors = _stored_vectors(source, embeddings.model)
    source.document_set.all().delete()
    stored_documents = _store_documents(source, documents)
    return _Plan(_split_documents(documents, stored_documents), stored_vectors)


def _check_source_keys(source_key: str, documents: Sequence[DocumentProtocol]) -> None:
    """Raise ``ValueError`` if one of ``documents`` is not of ``source_key``."""
    for document in documents:
        if document.source_key != source_key:
            message = (
                f"Document of source key {document.source_key!r} "
                f"found in the group {source_key!r}"
            )
            raise ValueError(message)


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


def _store_all_chunks(plans: Sequence[_Plan], embeddings: Embeddings) -> None:
    """Store the chunks of all ``plans``, each source's pieces ranked in order."""
    ranked_pieces = (
        (rank, stored_document, text)
        for plan in plans
        for rank, (stored_document, text) in enumerate(plan.pieces)
    )
    Chunk.objects.bulk_create(
        Chunk(
            document=stored_document,
            rank=rank,
            text=text,
            embedding_model=embeddings.model,
            embedding=vector,
        )
        for (rank, stored_document, text), vector in zip(
            ranked_pieces, _vectors(plans, embeddings), strict=True
        )
    )


def _vectors(plans: Sequence[_Plan], embeddings: Embeddings) -> list[Any]:
    """Return one vector per piece of ``plans``, embedding new texts in one call.

    A text already embedded by the same model keeps its vector.
    """
    new_texts = list(
        dict.fromkeys(
            text
            for plan in plans
            for _, text in plan.pieces
            if text not in plan.stored_vectors
        )
    )
    new_vectors = dict(zip(new_texts, _embed(new_texts, embeddings), strict=True))
    return [
        plan.stored_vectors[text] if text in plan.stored_vectors else new_vectors[text]
        for plan in plans
        for _, text in plan.pieces
    ]


def _embed(texts: Sequence[str], embeddings: Embeddings) -> list[Any]:
    """Return the vectors ``embeddings`` gives ``texts``, one per text.

    Raises ``ValueError`` when the backend returns another number of vectors.
    """
    if not texts:
        return []
    vectors = embeddings.embed(texts)
    if len(vectors) != len(texts):
        message = f"The backend returned {len(vectors)} vectors for {len(texts)} texts"
        raise ValueError(message)
    return vectors
