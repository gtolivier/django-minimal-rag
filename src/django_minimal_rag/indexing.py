"""Indexing of documents into the storage models."""

import datetime
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any, NamedTuple

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import transaction
from django.db.models import Q, QuerySet
from django.db.models.functions import Collate
from django.utils import timezone

from django_minimal_rag.chunking import split_text
from django_minimal_rag.documents import Document as DocumentProtocol
from django_minimal_rag.embeddings import Embeddings, get_embeddings
from django_minimal_rag.models import Chunk, Document, Source

MODEL_LABEL_SEPARATOR = ":"
"""What ends the model label at the start of a source key."""
DEFAULT_BATCH_SIZE = 100
"""Most chunks per embedding call when no batch size is given."""
DEFAULT_CLAIM_TIMEOUT = datetime.timedelta(minutes=10)
"""How long a claim keeps other runs off a chunk when no claim timeout is given."""
DEFER_EMBEDDING_SETTING = "MINIMAL_RAG_DEFER_EMBEDDING"
"""When true, replace() does not run embed_pending() once it commits."""


class Indexer:
    """Stores documents, chunks and embeddings."""

    @transaction.atomic
    def prune(self, model_label: str, kept_keys: AbstractSet[str]) -> None:
        """Remove the sources of ``model_label`` whose key is not in ``kept_keys``.

        ``model_label`` is the part of a source key before the separator.
        Raise ``ValueError`` if it is empty or holds the separator.
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


def embed_pending(
    batch_size: int | None = None, claim_timeout: datetime.timedelta | None = None
) -> int:
    """Embed the chunks still waiting for a vector; return how many were.

    Embed them in calls of at most ``batch_size`` chunks
    (``DEFAULT_BATCH_SIZE`` by default).
    Skip the chunks claimed more recently than ``claim_timeout``
    (``DEFAULT_CLAIM_TIMEOUT`` by default).
    """
    _check_batch_size(batch_size)
    _check_claim_timeout(claim_timeout)
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if claim_timeout is None:
        claim_timeout = DEFAULT_CLAIM_TIMEOUT
    embedded = 0
    embeddings: Embeddings | None = None  # built only if a chunk is pending
    while batch := _claim_pending_chunks(claim_timeout, batch_size):
        if embeddings is None:
            embeddings = get_embeddings()
        embedded += _embed_chunks(batch, embeddings)
    return embedded


def _claim_pending_chunks(claim_timeout: datetime.timedelta, limit: int) -> list[Chunk]:
    """Claim at most ``limit`` chunks of ``_pending_chunks(claim_timeout)``."""
    claimed_at = timezone.now()
    ids = list(
        _pending_chunks(claim_timeout)
        .order_by("pk")
        .values_list("pk", flat=True)[:limit]
    )
    # One UPDATE takes the chunks, so that no other run takes them too.
    _pending_chunks(claim_timeout).filter(pk__in=ids).update(
        embedding_claimed_at=claimed_at
    )
    return list(
        Chunk.objects.filter(
            pk__in=ids, embedding__isnull=True, embedding_claimed_at=claimed_at
        ).order_by("pk")
    )


def _pending_chunks(claim_timeout: datetime.timedelta) -> QuerySet[Chunk]:
    """Return the chunks waiting for a vector, without those claimed recently.

    A chunk is claimed recently when it was claimed less than ``claim_timeout``
    ago.
    """
    return Chunk.objects.filter(embedding__isnull=True).filter(
        Q(embedding_claimed_at__isnull=True)
        | Q(embedding_claimed_at__lte=timezone.now() - claim_timeout)
    )


def _check_batch_size(batch_size: int | None) -> None:
    """Raise ``ValueError`` if ``batch_size`` is given and below 1."""
    if batch_size is not None and batch_size < 1:
        message = f"batch_size must be at least 1, got {batch_size}"
        raise ValueError(message)


def _check_claim_timeout(claim_timeout: datetime.timedelta | None) -> None:
    """Raise ``ValueError`` if ``claim_timeout`` is given and not positive."""
    if claim_timeout is not None and claim_timeout <= datetime.timedelta(0):
        message = f"claim_timeout must be positive, got {claim_timeout!r}"
        raise ValueError(message)


def _embed_chunks(chunks: Sequence[Chunk], embeddings: Embeddings) -> int:
    """Give ``chunks`` the vectors of ``embeddings``, in one call, and save them.

    Return how many of them were still stored.
    """
    vectors = _embed_distinct((chunk.text for chunk in chunks), embeddings)
    for chunk in chunks:
        chunk.embedding = vectors[chunk.text]
        chunk.embedding_model = embeddings.model
    return Chunk.objects.bulk_update(chunks, ["embedding", "embedding_model"])


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
    """Raise ``ValueError`` if ``model_label`` is empty or holds the separator."""
    if not model_label:
        message = "The model label must not be empty"
        raise ValueError(message)
    if MODEL_LABEL_SEPARATOR in model_label:
        message = f"The model label must not contain {MODEL_LABEL_SEPARATOR!r}"
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
    """Store the chunks of all ``plans``, each source's pieces ranked in order.

    A text without a stored vector is stored without one. Unless the
    ``DEFER_EMBEDDING_SETTING`` setting is true, embed_pending() embeds it once
    the transaction commits.
    """
    _store_chunks_without_embedding(plans, embeddings)
    if not _is_embedding_deferred():
        transaction.on_commit(embed_pending)


def _is_embedding_deferred() -> bool:
    """Return whether the ``DEFER_EMBEDDING_SETTING`` setting is true."""
    defer_embedding: object = getattr(settings, DEFER_EMBEDDING_SETTING, False)
    if not isinstance(defer_embedding, bool):
        message = f"The {DEFER_EMBEDDING_SETTING} setting must be a bool."
        raise ImproperlyConfigured(message)
    return defer_embedding


def _store_chunks_without_embedding(
    plans: Sequence[_Plan], embeddings: Embeddings
) -> None:
    """Store the pieces of ``plans`` as chunks, without embedding any text.

    Each source's pieces are ranked in order. A text with a vector stored in one
    of ``plans`` keeps it; any other text is stored without a vector.
    """
    ranked_pieces = (
        (rank, stored_document, text)
        for plan in plans
        for rank, (stored_document, text) in enumerate(plan.pieces)
    )
    stored_vectors = {
        text: vector for plan in plans for text, vector in plan.stored_vectors.items()
    }
    Chunk.objects.bulk_create(
        Chunk(
            document=stored_document,
            rank=rank,
            text=text,
            embedding_model=embeddings.model,
            embedding=stored_vectors[text],
        )
        if text in stored_vectors
        else Chunk(document=stored_document, rank=rank, text=text)
        for rank, stored_document, text in ranked_pieces
    )


def _embed_distinct(texts: Iterable[str], embeddings: Embeddings) -> dict[str, Any]:
    """Return the vector ``embeddings`` gives each of ``texts``, by text.

    A text repeated in ``texts`` is embedded once, in a single call for all.
    """
    distinct_texts = list(dict.fromkeys(texts))
    return dict(zip(distinct_texts, _embed(distinct_texts, embeddings), strict=True))


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
