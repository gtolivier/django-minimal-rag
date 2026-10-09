"""Indexing of documents into the storage models."""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any, NamedTuple

from django.db import transaction
from django.db.models import QuerySet
from django.db.models.functions import Collate

from django_minimal_rag.chunking import split_text
from django_minimal_rag.documents import Document as DocumentProtocol
from django_minimal_rag.embeddings import Embeddings, get_embeddings
from django_minimal_rag.models import Chunk, Document, Source

MODEL_LABEL_SEPARATOR = ":"
"""What ends the model label at the start of a source key."""


class Indexer:
    """Stores documents, chunks and embeddings."""

    @transaction.atomic
    def prune(self, model_label: str, kept_keys: AbstractSet[str]) -> None:
        """Remove the sources of ``model_label`` whose key is not in ``kept_keys``.

        ``model_label`` is the part of a source key before the separator.
        Raise ``ValueError`` if it is empty.
        """
        _check_model_label(model_label)
        # kept_keys may hold every instance of a model: sent to SQL, it could go
        # over PostgreSQL's parameter limit. The removed keys are usually few.
        stored_keys = Source.objects.filter(
            source_key__startswith=f"{model_label}{MODEL_LABEL_SEPARATOR}"
        ).values_list("source_key", flat=True)
        removed_keys = set(stored_keys).difference(kept_keys)
        _delete_sources(Source.objects.filter(source_key__in=removed_keys))

    @transaction.atomic
    def replace(self, groups: Mapping[str, Sequence[DocumentProtocol]]) -> None:
        """Replace the indexed content with the given groups."""
        embeddings: Embeddings | None = None  # built only if a group has documents
        plans = []
        for source_key, documents in _checked_groups(groups).items():
            if not documents:
                _remove_source(source_key)
                continue
            if embeddings is None:
                embeddings = get_embeddings()
            plans.append(_prepare_source(source_key, documents, embeddings))
        if embeddings is not None:
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
    source, _ = Source.objects.select_for_update().get_or_create(source_key=source_key)
    stored_vectors = _stored_vectors(source, embeddings.model)
    source.document_set.all().delete()
    stored_documents = _store_documents(source, documents)
    return _Plan(_split_documents(documents, stored_documents), stored_vectors)


def _checked_groups(
    groups: Mapping[str, Sequence[DocumentProtocol]],
) -> dict[str, list[DocumentProtocol]]:
    """Return ``groups`` sorted by source key, each as a list of its documents.

    Raise ``ValueError`` if a document is not of its group's source key.
    """
    sorted_groups = {
        source_key: list(documents)  # a group may be iterable only once
        for source_key, documents in sorted(groups.items())
    }
    for source_key, documents in sorted_groups.items():
        _check_source_keys(source_key, documents)
    return sorted_groups


def _check_source_keys(source_key: str, documents: Sequence[DocumentProtocol]) -> None:
    """Raise ``ValueError`` if one of ``documents`` is not of ``source_key``."""
    for document in documents:
        if document.source_key != source_key:
            message = (
                f"Document of source key {document.source_key!r} "
                f"found in the group {source_key!r}"
            )
            raise ValueError(message)


def _check_model_label(model_label: str) -> None:
    """Raise ``ValueError`` if ``model_label`` is empty."""
    if not model_label:
        message = "The model label must not be empty"
        raise ValueError(message)


def _remove_source(source_key: str) -> None:
    """Remove the source stored under ``source_key``, with its content."""
    _delete_sources(Source.objects.filter(source_key=source_key))


def _delete_sources(sources: QuerySet[Source]) -> None:
    """Delete ``sources`` with their content, once their rows are locked.

    Must run in a transaction. A replacement locks a source row before changing
    its documents; the rows are locked first, in the same source key order, so
    that no document is deleted while a replacement inserts new ones under it.
    """
    locked_pks = _lock_sources(sources)
    # Not sources.delete(): under READ COMMITTED, a second query also sees rows
    # committed after the locks were taken, so it could delete an unlocked source.
    Source.objects.filter(pk__in=locked_pks).delete()


def _lock_sources(sources: QuerySet[Source]) -> list[int]:
    """Lock the rows of ``sources`` in one query, in the order replace() uses.

    Return the primary keys of the rows locked. Must run in a transaction.
    """
    # replace() sorts the keys as Python does, by code point. The "C" collation
    # compares bytes, which for UTF-8 is code-point order; the database's
    # default collation may order them otherwise.
    in_python_order = Collate("source_key", "C")
    # Evaluating the queryset is what takes the locks.
    return list(
        sources.select_for_update()
        .order_by(in_python_order)
        .values_list("pk", flat=True)
    )


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
