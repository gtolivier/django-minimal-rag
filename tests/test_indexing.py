import dataclasses
import datetime
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.db import (
    DEFAULT_DB_ALIAS,
    connection,
    connections,
    transaction,
)
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from django_minimal_rag.chunking import chunk_group
from django_minimal_rag.embeddings import FakeEmbeddings
from django_minimal_rag.indexing import Indexer, embed_pending
from django_minimal_rag.models import Chunk, Document, Source
from tests.documents import MutableDocument, SampleDocument
from tests.embeddings import (
    EmbeddingFailedError,
    use_fake_embeddings,
    use_miscounting_embeddings,
)
from tests.indexing import replace_and_run_on_commit
from tests.sequences import SinglePassSequence

if TYPE_CHECKING:
    from django.db.backends.base.base import BaseDatabaseWrapper
    from pytest_django import Settings


pytestmark = pytest.mark.usefixtures("fake_embeddings")


def use_recording_embeddings(settings: "Settings") -> list[str]:
    """Configure ``RecordingEmbeddings``; return the list it records texts in."""
    embedded: list[str] = []
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.RecordingEmbeddings",
        "OPTIONS": {"embedded": embedded},
    }
    return embedded


def use_call_recording_embeddings(settings: "Settings") -> list[list[str]]:
    """Configure ``RecordingEmbeddings``; return the list of its ``embed()`` calls.

    Each call is recorded as the list of the texts it was given.
    """
    calls: list[list[str]] = []
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.RecordingEmbeddings",
        "OPTIONS": {"embedded": [], "calls": calls},
    }
    return calls


def faq_entry(number: int) -> SampleDocument:
    """FAQ entry ``number`` of the host project, in English and public."""
    return SampleDocument(
        text=f"Answer to question {number}.",
        source_key=f"faq:{number}",
        title=f"Question {number}",
        url=f"https://example.com/faq/{number}/",
        language="en",
        permissions=frozenset(),
    )


def faq_entry_under(source_key: str, number: int) -> SampleDocument:
    """FAQ entry ``number``, belonging to the source ``source_key``."""
    return dataclasses.replace(faq_entry(number), source_key=source_key)


def paragraphs_of_one_chunk_each(count: int) -> list[str]:
    """``count`` paragraphs, each too long to share a chunk with another."""
    return [f"Paragraph {number}: " + "word " * 120 for number in range(count)]


def store_pending_chunks(source_key: str, texts: Iterable[str]) -> None:
    """Store a source of one document whose chunks, of ``texts``, await a vector.

    Each chunk is stored as before its embedding: no vector, no model.
    """
    source = Source.objects.create(source_key=source_key)
    document = Document.objects.create(
        source=source,
        title=f"Document of {source_key}",
        url=f"https://example.com/{source_key}/",
        language="en",
    )
    Chunk.objects.bulk_create(
        Chunk(
            document=document,
            rank=rank,
            text=text,
            embedding=None,
            embedding_model="",
        )
        for rank, text in enumerate(texts)
    )


def stored_content(source: Source) -> tuple[list[object], list[object]]:
    """The documents and chunks stored under ``source``, ordered by primary key."""
    documents: list[object] = list(
        Document.objects.filter(source=source)
        .order_by("pk")
        .values_list("pk", "title", "url", "language", "permissions")
    )
    chunks: list[object] = [
        (
            chunk.pk,
            chunk.document_id,
            chunk.rank,
            chunk.text,
            chunk.embedding_model,
            list(chunk.embedding),
        )
        for chunk in Chunk.objects.filter(document__source=source).order_by("pk")
    ]
    return documents, chunks


def source_keys_locked_by(queries: Iterable[Mapping[str, str]]) -> list[str]:
    """The ``source_key`` of each source row ``queries`` lock, in order.

    Only queries reading the source table ``FOR UPDATE`` count; each query's
    ``sql`` has its parameters interpolated.
    """
    table = connection.ops.quote_name(Source._meta.db_table)
    column = connection.ops.quote_name("source_key")
    key_lookup = re.compile(
        rf"{re.escape(table)}\.{re.escape(column)} = '((?:[^']|'')*)'"
    )
    return [
        match.group(1).replace("''", "'")
        for query in queries
        if f"FROM {table}" in query["sql"] and "FOR UPDATE" in query["sql"]
        for match in key_lookup.finditer(query["sql"])
    ]


@pytest.fixture
def other_connection() -> "Iterator[BaseDatabaseWrapper]":
    """A database connection of its own, in autocommit mode, closed afterwards.

    Outside the test's transaction, it sees only what is committed, and
    contends for row locks like another process.
    """
    other = connections.create_connection(DEFAULT_DB_ALIAS)
    yield other
    other.close()


def source_keys_locked_against(other_connection: "BaseDatabaseWrapper") -> list[str]:
    """The sorted ``source_key`` of each source row ``other_connection`` cannot lock.

    ``other_connection`` must be in autocommit mode, so that its probing locks
    are released as soon as taken.
    """
    table = other_connection.ops.quote_name(Source._meta.db_table)
    with other_connection.cursor() as cursor:
        cursor.execute(f"SELECT source_key FROM {table}")
        stored = {source_key for (source_key,) in cursor.fetchall()}
        # FOR SHARE conflicts with the FOR UPDATE lock of a replacement.
        cursor.execute(f"SELECT source_key FROM {table} FOR SHARE SKIP LOCKED")
        unlocked = {source_key for (source_key,) in cursor.fetchall()}
    return sorted(stored - unlocked)


def wait_until_a_query_waits_for_a_lock(
    other_connection: "BaseDatabaseWrapper", timeout: float = 10.0
) -> None:
    """Return once a query of the database waits for a lock; fail after ``timeout``.

    ``other_connection`` must be in autocommit mode, so that each poll sees
    the current activity.
    """
    deadline = time.monotonic() + timeout
    with other_connection.cursor() as cursor:
        while time.monotonic() < deadline:
            cursor.execute(
                "SELECT EXISTS (SELECT FROM pg_stat_activity"
                " WHERE datname = current_database() AND wait_event_type = 'Lock')"
            )
            if cursor.fetchone()[0]:
                return
            time.sleep(0.01)
    pytest.fail(f"No query waited for a lock within {timeout} s")


@pytest.mark.django_db
def test_replace_with_no_groups_stores_nothing() -> None:
    replace_and_run_on_commit({})

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_replace_with_no_groups_succeeds_without_an_embedding_backend_configured(
    settings: "Settings",
) -> None:
    del settings.MINIMAL_RAG_EMBEDDINGS

    replace_and_run_on_commit({})


@pytest.mark.django_db
def test_replace_group_of_one_document_creates_source_with_group_key() -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})

    assert list(Source.objects.values_list("source_key", flat=True)) == ["faq:1"]


@pytest.mark.django_db
def test_replace_group_of_one_document_stores_it_under_the_group_source() -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})

    assert list(
        Document.objects.values_list("source__source_key", "title", "url", "language")
    ) == [("faq:1", "Question 1", "https://example.com/faq/1/", "en")]


@pytest.mark.django_db
def test_replace_stores_each_group_under_its_own_source() -> None:
    replace_and_run_on_commit(
        {
            "faq:1": [faq_entry(1)],
            "faq:2": [faq_entry(2), faq_entry_under("faq:2", 3)],
        }
    )

    assert sorted(Source.objects.values_list("source_key", flat=True)) == [
        "faq:1",
        "faq:2",
    ]
    assert sorted(Document.objects.values_list("source__source_key", "title")) == [
        ("faq:1", "Question 1"),
        ("faq:2", "Question 2"),
        ("faq:2", "Question 3"),
    ]


@pytest.mark.django_db
def test_replace_of_a_stored_source_keeps_it_with_only_the_new_content() -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})
    source_pk = Source.objects.get().pk

    replace_and_run_on_commit(
        {"faq:1": [faq_entry_under("faq:1", 2), faq_entry_under("faq:1", 3)]}
    )

    assert list(Source.objects.values_list("pk", "source_key")) == [
        (source_pk, "faq:1")
    ]
    assert sorted(Document.objects.values_list("source", "title")) == [
        (source_pk, "Question 2"),
        (source_pk, "Question 3"),
    ]
    assert sorted(Chunk.objects.values_list("document__title", "text")) == [
        ("Question 2", "Answer to question 2."),
        ("Question 3", "Answer to question 3."),
    ]


@pytest.mark.django_db
def test_replace_with_an_empty_group_removes_its_stored_source() -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1), faq_entry_under("faq:1", 2)]})

    replace_and_run_on_commit({"faq:1": []})

    assert not Source.objects.filter(source_key="faq:1").exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_replace_with_an_empty_group_for_an_unknown_key_stores_nothing() -> None:
    replace_and_run_on_commit({"faq:1": []})

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_replace_with_only_empty_groups_removes_sources_without_a_backend_configured(
    settings: "Settings",
) -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})
    del settings.MINIMAL_RAG_EMBEDDINGS

    replace_and_run_on_commit({"faq:1": [], "faq:2": []})

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_replace_leaves_sources_absent_from_the_call_as_they_are() -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})
    source = Source.objects.get()
    document = Document.objects.get()
    chunk = Chunk.objects.get()

    replace_and_run_on_commit({"faq:2": [faq_entry(2)]})

    assert Source.objects.filter(source_key="faq:1").get().pk == source.pk
    assert list(
        Document.objects.filter(source=source).values_list(
            "pk", "title", "url", "language", "permissions"
        )
    ) == [(document.pk, "Question 1", "https://example.com/faq/1/", "en", [])]
    kept_chunk = Chunk.objects.filter(document__source=source).get()
    assert (kept_chunk.pk, kept_chunk.rank, kept_chunk.text) == (
        chunk.pk,
        chunk.rank,
        chunk.text,
    )
    assert kept_chunk.embedding_model == chunk.embedding_model
    assert list(kept_chunk.embedding) == list(chunk.embedding)


@pytest.mark.django_db
def test_replace_stores_a_frozenset_of_permissions_as_their_sorted_list() -> None:
    document = dataclasses.replace(
        faq_entry(1), permissions=frozenset({"faq.view_faq", "app.change_note"})
    )

    replace_and_run_on_commit({"faq:1": [document]})

    assert list(Document.objects.values_list("permissions", flat=True)) == [
        ["app.change_note", "faq.view_faq"]
    ]


@pytest.mark.django_db
def test_replace_stores_a_short_document_text_as_one_chunk_of_rank_0() -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})

    document = Document.objects.get()
    assert list(Chunk.objects.values_list("document", "rank", "text")) == [
        (document.pk, 0, "Answer to question 1.")
    ]


@pytest.mark.django_db
def test_replace_stores_a_long_document_text_as_its_chunks_in_rank_order() -> None:
    paragraphs = paragraphs_of_one_chunk_each(5)
    long_document = SampleDocument(
        text="\n\n".join(paragraphs),
        source_key="guide:1",
        title="Guide",
        url="https://example.com/guide/",
        language="en",
        permissions=frozenset(),
    )
    expected = [(chunk.rank, chunk.text) for chunk in chunk_group([long_document])]

    replace_and_run_on_commit({"guide:1": [long_document]})

    document = Document.objects.get()
    stored = Chunk.objects.filter(document=document).order_by("rank")
    assert len(expected) > 1
    assert list(stored.values_list("rank", "text")) == expected


@pytest.mark.django_db
def test_replace_stores_each_document_of_a_group_with_its_chunks_ranked_across() -> (
    None
):
    paragraphs = paragraphs_of_one_chunk_each(2)
    two_chunk_document = SampleDocument(
        text="\n\n".join(paragraphs),
        source_key="guide:1",
        title="Guide, part 1",
        url="https://example.com/guide/1/",
        language="en",
        permissions=frozenset(),
    )
    one_chunk_document = SampleDocument(
        text="Short closing part.",
        source_key="guide:1",
        title="Guide, part 2",
        url="https://example.com/guide/2/",
        language="en",
        permissions=frozenset(),
    )

    replace_and_run_on_commit({"guide:1": [two_chunk_document, one_chunk_document]})

    stored = Chunk.objects.order_by("rank")
    assert list(stored.values_list("rank", "document__title", "text")) == [
        (0, "Guide, part 1", paragraphs[0].rstrip()),
        (1, "Guide, part 1", paragraphs[1].rstrip()),
        (2, "Guide, part 2", "Short closing part."),
    ]


@pytest.mark.django_db
def test_replace_stores_a_document_given_twice_in_a_group_twice_with_its_chunks() -> (
    None
):
    entry = faq_entry(1)

    replace_and_run_on_commit({"faq:1": [entry, entry]})

    first, second = Document.objects.order_by("pk")
    stored = Chunk.objects.order_by("rank")
    assert list(stored.values_list("document", "rank", "text")) == [
        (first.pk, 0, "Answer to question 1."),
        (second.pk, 1, "Answer to question 1."),
    ]


@pytest.mark.django_db
def test_replace_stores_all_documents_and_chunks_of_a_group_iterable_only_once() -> (
    None
):
    group = SinglePassSequence([faq_entry(1), faq_entry_under("faq:1", 2)])

    replace_and_run_on_commit({"faq:1": group})

    assert sorted(Document.objects.values_list("source__source_key", "title")) == [
        ("faq:1", "Question 1"),
        ("faq:1", "Question 2"),
    ]
    assert sorted(Chunk.objects.values_list("document__title", "text")) == [
        ("Question 1", "Answer to question 1."),
        ("Question 2", "Answer to question 2."),
    ]


@pytest.mark.django_db
def test_replace_stores_an_unhashable_document_with_its_chunks() -> None:
    document = MutableDocument(
        text="Answer to question 1.",
        source_key="faq:1",
        title="Question 1",
        url="https://example.com/faq/1/",
        language="en",
        permissions=frozenset(),
    )

    replace_and_run_on_commit({"faq:1": [document]})

    assert list(
        Document.objects.values_list("source__source_key", "title", "url", "language")
    ) == [("faq:1", "Question 1", "https://example.com/faq/1/", "en")]
    assert list(Chunk.objects.values_list("document__title", "rank", "text")) == [
        ("Question 1", 0, "Answer to question 1.")
    ]


@pytest.mark.django_db
def test_replace_embeds_each_chunk_with_the_configured_backend(
    settings: "Settings",
) -> None:
    use_fake_embeddings(settings, dimensions=3)
    backend = FakeEmbeddings(dimensions=3)
    [expected_vector] = backend.embed(["Answer to question 1."])

    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})

    chunk = Chunk.objects.get()
    assert chunk.embedding_model == "fake-3"
    assert list(chunk.embedding) == expected_vector


@pytest.mark.django_db
def test_replace_with_the_same_texts_embeds_nothing_and_keeps_each_vector(
    settings: "Settings",
) -> None:
    embedded = use_recording_embeddings(settings)
    replace_and_run_on_commit({"faq:1": [faq_entry(1), faq_entry_under("faq:1", 2)]})
    vectors_before = {
        chunk.text: list(chunk.embedding) for chunk in Chunk.objects.all()
    }
    embedded_before = len(embedded)

    replace_and_run_on_commit({"faq:1": [faq_entry(1), faq_entry_under("faq:1", 2)]})

    assert embedded[embedded_before:] == []
    assert {
        chunk.text: list(chunk.embedding) for chunk in Chunk.objects.all()
    } == vectors_before


@pytest.mark.django_db
def test_replace_changing_one_text_embeds_it_only_and_keeps_the_other_vectors(
    settings: "Settings",
) -> None:
    embedded = use_recording_embeddings(settings)
    replace_and_run_on_commit(
        {
            "faq:1": [
                faq_entry(1),
                faq_entry_under("faq:1", 2),
                faq_entry_under("faq:1", 3),
            ]
        }
    )
    vectors_before = {
        chunk.text: list(chunk.embedding) for chunk in Chunk.objects.all()
    }
    embedded_before = len(embedded)
    changed_entry = dataclasses.replace(
        faq_entry_under("faq:1", 2), text="New answer to question 2."
    )

    replace_and_run_on_commit(
        {"faq:1": [faq_entry(1), changed_entry, faq_entry_under("faq:1", 3)]}
    )

    assert embedded[embedded_before:] == ["New answer to question 2."]
    vectors_after = {chunk.text: list(chunk.embedding) for chunk in Chunk.objects.all()}
    unchanged = ["Answer to question 1.", "Answer to question 3."]
    assert [vectors_after[text] for text in unchanged] == [
        vectors_before[text] for text in unchanged
    ]
    assert sorted(vectors_after) == [
        "Answer to question 1.",
        "Answer to question 3.",
        "New answer to question 2.",
    ]


@pytest.mark.django_db
def test_replace_re_embeds_a_text_stored_under_another_model(
    settings: "Settings",
) -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1), faq_entry_under("faq:1", 2)]})
    embedded = use_recording_embeddings(settings)

    replace_and_run_on_commit({"faq:1": [faq_entry(1), faq_entry_under("faq:1", 2)]})

    assert sorted(embedded) == ["Answer to question 1.", "Answer to question 2."]
    assert {
        chunk.text: (chunk.embedding_model, list(chunk.embedding))
        for chunk in Chunk.objects.all()
    } == {
        text: ("recording", [float(position), 1.0])
        for position, text in enumerate(embedded)
    }


@pytest.mark.django_db
def test_replace_of_several_groups_embeds_the_new_texts_of_all_in_one_backend_call(
    settings: "Settings",
) -> None:
    calls = use_call_recording_embeddings(settings)

    replace_and_run_on_commit(
        {"faq:1": [faq_entry(1)], "faq:2": [faq_entry(2)], "faq:3": [faq_entry(3)]}
    )

    assert calls == [
        ["Answer to question 1.", "Answer to question 2.", "Answer to question 3."]
    ]


@pytest.mark.django_db
def test_replace_deferring_embedding_stores_new_texts_without_vector_or_backend_call(
    settings: "Settings",
) -> None:
    calls = use_call_recording_embeddings(settings)
    settings.MINIMAL_RAG_DEFER_EMBEDDING = True

    replace_and_run_on_commit({"faq:1": [faq_entry(1)], "faq:2": [faq_entry(2)]})

    assert calls == []
    assert list(
        Chunk.objects.order_by("text").values_list(
            "text", "embedding", "embedding_model"
        )
    ) == [
        ("Answer to question 1.", None, ""),
        ("Answer to question 2.", None, ""),
    ]


@pytest.mark.django_db
def test_replace_deferring_embedding_keeps_the_stored_vector_of_an_unchanged_text(
    settings: "Settings",
) -> None:
    use_recording_embeddings(settings)
    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})
    vector_before = list(Chunk.objects.get().embedding)
    settings.MINIMAL_RAG_DEFER_EMBEDDING = True

    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})

    chunk = Chunk.objects.get()
    vector_after = None if chunk.embedding is None else list(chunk.embedding)
    assert (chunk.text, chunk.embedding_model, vector_after) == (
        "Answer to question 1.",
        "recording",
        vector_before,
    )


@pytest.mark.django_db
def test_replace_deferring_embedding_stores_unchanged_text_of_other_model_unembedded(
    settings: "Settings",
) -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})
    assert Chunk.objects.get().embedding_model != "recording"
    calls = use_call_recording_embeddings(settings)
    settings.MINIMAL_RAG_DEFER_EMBEDDING = True

    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})

    assert calls == []
    assert list(Chunk.objects.values_list("text", "embedding", "embedding_model")) == [
        ("Answer to question 1.", None, "")
    ]


@pytest.mark.django_db
def test_replace_with_defer_embedding_false_embeds_the_new_texts_as_when_unset(
    settings: "Settings",
) -> None:
    use_fake_embeddings(settings, dimensions=3)
    backend = FakeEmbeddings(dimensions=3)
    [expected_vector] = backend.embed(["Answer to question 1."])
    settings.MINIMAL_RAG_DEFER_EMBEDDING = False

    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})

    chunk = Chunk.objects.get()
    assert chunk.embedding_model == "fake-3"
    assert list(chunk.embedding) == expected_vector


@pytest.mark.parametrize(
    "non_bool",
    [
        "yes",
        # 1 == True: a check by equality with the bools would accept it.
        1,
    ],
)
@pytest.mark.django_db
def test_replace_raises_and_stores_nothing_when_defer_embedding_is_not_a_bool(
    settings: "Settings", non_bool: object
) -> None:
    # Truthy, but not a bool: refused rather than read as true.
    settings.MINIMAL_RAG_DEFER_EMBEDDING = non_bool

    with pytest.raises(ImproperlyConfigured, match="MINIMAL_RAG_DEFER_EMBEDDING"):
        replace_and_run_on_commit({"faq:1": [faq_entry(1)]})

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_replace_embeds_a_new_text_of_two_groups_once_and_gives_both_its_vector(
    settings: "Settings",
) -> None:
    embedded = use_recording_embeddings(settings)
    shared_text = "Shared answer."
    entry_1 = dataclasses.replace(faq_entry(1), text=shared_text)
    entry_2 = dataclasses.replace(faq_entry(2), text=shared_text)

    replace_and_run_on_commit({"faq:1": [entry_1], "faq:2": [entry_2]})

    assert embedded == [shared_text]
    assert {
        chunk.document.source.source_key: list(chunk.embedding)
        for chunk in Chunk.objects.select_related("document__source")
    } == {"faq:1": [0.0, 1.0], "faq:2": [0.0, 1.0]}


@pytest.mark.django_db
def test_replace_stores_nothing_when_the_backend_raises_on_a_later_group(
    settings: "Settings",
) -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1)], "faq:2": [faq_entry(2)]})
    source = Source.objects.get(source_key="faq:1")
    content_before = stored_content(source)
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.FailingEmbeddings",
        "OPTIONS": {"failing_text": "New answer to question 2."},
    }
    new_entry_1 = dataclasses.replace(faq_entry(1), text="New answer to question 1.")
    new_entry_2 = dataclasses.replace(faq_entry(2), text="New answer to question 2.")

    with pytest.raises(EmbeddingFailedError):
        replace_and_run_on_commit({"faq:1": [new_entry_1], "faq:2": [new_entry_2]})

    assert stored_content(source) == content_before


@pytest.mark.django_db
def test_replace_raises_and_stores_nothing_when_the_backend_returns_too_few_vectors(
    settings: "Settings",
) -> None:
    use_miscounting_embeddings(settings, vector_count=1)
    two_chunk_document = SampleDocument(
        text="\n\n".join(paragraphs_of_one_chunk_each(2)),
        source_key="guide:1",
        title="Guide",
        url="https://example.com/guide/",
        language="en",
        permissions=frozenset(),
    )

    # Both counts, in any order and wording: 2 texts, 1 vector.
    with pytest.raises(ValueError, match=r"(?s)^(?=.*\b2\b)(?=.*\b1\b)"):
        replace_and_run_on_commit({"guide:1": [two_chunk_document]})

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_replace_raises_when_the_backend_returns_too_many_vectors(
    settings: "Settings",
) -> None:
    use_miscounting_embeddings(settings, vector_count=3)

    # Both counts, in any order and wording: 1 text, 3 vectors.
    with pytest.raises(ValueError, match=r"(?s)^(?=.*\b1\b)(?=.*\b3\b)"):
        replace_and_run_on_commit({"faq:1": [faq_entry(1)]})


@pytest.mark.django_db
def test_replace_raises_and_stores_nothing_for_a_document_of_another_source_key() -> (
    None
):
    # Both keys, in any order and wording: the group's and the document's.
    with pytest.raises(ValueError, match=r"(?s)^(?=.*\bfaq:1\b)(?=.*\bfaq:2\b)"):
        replace_and_run_on_commit({"faq:1": [faq_entry(2)]})

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_replace_raises_and_stores_nothing_for_the_last_document_of_a_later_group() -> (
    None
):
    # Both keys, in any order and wording: the second group's and its last
    # document's.
    with pytest.raises(ValueError, match=r"(?s)^(?=.*\bfaq:5\b)(?=.*\bfaq:7\b)"):
        replace_and_run_on_commit(
            {
                "faq:3": [faq_entry(3)],
                "faq:5": [faq_entry(5), faq_entry_under("faq:5", 6), faq_entry(7)],
            }
        )

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_replace_locks_the_row_of_each_replaced_source_until_its_transaction_ends(
    other_connection: "BaseDatabaseWrapper",
) -> None:
    Indexer().replace({"faq:1": [faq_entry(1)]})
    with transaction.atomic():
        Indexer().replace({"faq:1": [faq_entry_under("faq:1", 2)]})

        locked_inside_the_transaction = source_keys_locked_against(other_connection)

    assert locked_inside_the_transaction == ["faq:1"]


@pytest.mark.django_db
def test_replace_locks_the_replaced_sources_in_sorted_source_key_order() -> None:
    # Two calls replacing overlapping sources must lock their rows in the same
    # order, whatever the order of their groups, or they can deadlock.
    replace_and_run_on_commit(
        {"faq:1": [faq_entry(1)], "faq:2": [faq_entry(2)], "faq:3": [faq_entry(3)]}
    )

    with CaptureQueriesContext(connection) as captured:
        replace_and_run_on_commit(
            {"faq:2": [faq_entry(2)], "faq:3": [faq_entry(3)], "faq:1": [faq_entry(1)]}
        )

    assert source_keys_locked_by(captured.captured_queries) == [
        "faq:1",
        "faq:2",
        "faq:3",
    ]


@pytest.mark.django_db
def test_replace_with_an_empty_group_locks_its_source_before_deleting_anything() -> (
    None
):
    # A replacement locks the source row, then deletes its documents; a removal
    # must take the same lock first, or the two can deadlock.
    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})

    with CaptureQueriesContext(connection) as captured:
        replace_and_run_on_commit({"faq:1": []})

    queries = captured.captured_queries
    first_delete = next(
        index
        for index, query in enumerate(queries)
        if query["sql"].startswith("DELETE")
    )
    assert source_keys_locked_by(queries[:first_delete]) == ["faq:1"]


@pytest.mark.django_db
def test_replace_locks_removed_and_replaced_sources_in_sorted_source_key_order() -> (
    None
):
    # Two calls sharing sources must lock their rows in the same order, whether
    # each source is replaced or removed, or they can deadlock.
    replace_and_run_on_commit(
        {"faq:1": [faq_entry(1)], "faq:2": [faq_entry(2)], "faq:3": [faq_entry(3)]}
    )

    with CaptureQueriesContext(connection) as captured:
        replace_and_run_on_commit({"faq:3": [faq_entry(3)], "faq:1": [], "faq:2": []})

    assert source_keys_locked_by(captured.captured_queries) == [
        "faq:1",
        "faq:2",
        "faq:3",
    ]


@pytest.mark.django_db
def test_prune_with_no_source_stored_does_nothing() -> None:
    Indexer().prune("faq", frozenset({"faq:1"}))

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_prune_with_no_kept_key_removes_a_stored_source_with_its_content() -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})

    Indexer().prune("faq", set())

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_prune_keeps_the_kept_sources_with_their_content_and_removes_the_others() -> (
    None
):
    replace_and_run_on_commit(
        {
            "faq:1": [faq_entry(1)],
            "faq:2": [faq_entry(2)],
            "faq:3": [faq_entry(3)],
            "faq:4": [faq_entry(4)],
        }
    )
    kept_sources = list(
        Source.objects.filter(source_key__in=["faq:1", "faq:3"]).order_by("source_key")
    )
    content_before = [stored_content(source) for source in kept_sources]

    Indexer().prune("faq", {"faq:1", "faq:3"})

    assert list(
        Source.objects.order_by("source_key").values_list("pk", "source_key")
    ) == [(source.pk, source.source_key) for source in kept_sources]
    assert [stored_content(source) for source in kept_sources] == content_before
    assert sorted(Document.objects.values_list("source__source_key", flat=True)) == [
        "faq:1",
        "faq:3",
    ]
    assert sorted(
        Chunk.objects.values_list("document__source__source_key", flat=True)
    ) == ["faq:1", "faq:3"]


@pytest.mark.django_db
def test_prune_leaves_the_sources_of_another_model_label_with_their_content() -> None:
    replace_and_run_on_commit({"news:1": [faq_entry_under("news:1", 1)]})
    other_source = Source.objects.get()
    content_before = stored_content(other_source)

    Indexer().prune("faq", set())

    assert list(Source.objects.values_list("pk", "source_key")) == [
        (other_source.pk, "news:1")
    ]
    assert stored_content(other_source) == content_before


@pytest.mark.django_db
def test_prune_leaves_a_source_whose_model_label_only_starts_with_the_given_one() -> (
    None
):
    # The model label is matched up to the colon: "app.note" is not a model
    # label of "app.notebook:3".
    replace_and_run_on_commit(
        {"app.notebook:3": [faq_entry_under("app.notebook:3", 3)]}
    )
    other_source = Source.objects.get()
    content_before = stored_content(other_source)

    Indexer().prune("app.note", set())

    assert list(Source.objects.values_list("pk", "source_key")) == [
        (other_source.pk, "app.notebook:3")
    ]
    assert stored_content(other_source) == content_before


@pytest.mark.django_db
def test_prune_matches_the_model_label_literally_not_as_a_like_pattern() -> None:
    # In a SQL LIKE pattern, "_" matches any character: "my_app.page" would
    # then be a model label of "myXapp.page:1".
    replace_and_run_on_commit({"myXapp.page:1": [faq_entry_under("myXapp.page:1", 1)]})
    other_source = Source.objects.get()
    content_before = stored_content(other_source)

    Indexer().prune("my_app.page", set())

    assert list(Source.objects.values_list("pk", "source_key")) == [
        (other_source.pk, "myXapp.page:1")
    ]
    assert stored_content(other_source) == content_before


@pytest.mark.django_db
def test_prune_raises_and_removes_nothing_for_an_empty_model_label() -> None:
    # ":1" is the key an empty model label would match up to the colon.
    replace_and_run_on_commit({":1": [faq_entry_under(":1", 1)]})
    stored_source = Source.objects.get()
    content_before = stored_content(stored_source)

    # The message names what is wrong, in any case and wording around it.
    with pytest.raises(ValueError, match=r"(?i)\bmodel label\b"):
        Indexer().prune("", set())

    assert list(Source.objects.values_list("pk", "source_key")) == [
        (stored_source.pk, ":1")
    ]
    assert stored_content(stored_source) == content_before


@pytest.mark.django_db
def test_prune_raises_and_removes_nothing_for_a_model_label_holding_the_separator() -> (
    None
):
    # "faq::1" is a key the model label "faq:" would match up to the colon.
    replace_and_run_on_commit({"faq::1": [faq_entry_under("faq::1", 1)]})
    stored_source = Source.objects.get()
    content_before = stored_content(stored_source)

    # The message names what is wrong, in any case and wording around it.
    with pytest.raises(ValueError, match=r"(?i)\bmodel label\b"):
        Indexer().prune("faq:", set())

    assert list(Source.objects.values_list("pk", "source_key")) == [
        (stored_source.pk, "faq::1")
    ]
    assert stored_content(stored_source) == content_before


@pytest.mark.django_db(transaction=True)
def test_prune_outside_a_transaction_locks_the_sources_it_removes_before_deleting(
    other_connection: "BaseDatabaseWrapper",
) -> None:
    # A replacement locks a source row, then changes its documents; a prune
    # deleting the source without that lock could delete documents while a
    # replacement inserts new ones under it.
    Indexer().replace({"faq:1": [faq_entry(1)], "faq:2": [faq_entry(2)]})
    locked_at_first_delete: list[list[str]] = []

    def probe_locks_before_first_delete(
        execute: Callable[[str, Any, bool, dict[str, Any]], Any],
        sql: str,
        params: Any,
        many: bool,
        context: dict[str, Any],
    ) -> Any:
        if sql.startswith("DELETE") and not locked_at_first_delete:
            locked_at_first_delete.append(source_keys_locked_against(other_connection))
        return execute(sql, params, many, context)

    with connection.execute_wrapper(probe_locks_before_first_delete):
        Indexer().prune("faq", set())

    assert locked_at_first_delete == [["faq:1", "faq:2"]]


@pytest.mark.django_db(transaction=True)
def test_prune_locks_the_sources_it_removes_in_the_sorted_order_replace_uses(
    other_connection: "BaseDatabaseWrapper",
) -> None:
    # A prune and a replacement sharing sources must lock their rows in the
    # same order, or they can deadlock. replace() sorts the keys as Python
    # does: "faq:B" before "faq:a", where a database collation such as
    # en_US puts "faq:a" first.
    # Stored "faq:a" first, so that a sequential scan meets it first.
    Indexer().replace({"faq:a": [faq_entry_under("faq:a", 1)]})
    Indexer().replace({"faq:B": [faq_entry_under("faq:B", 2)]})
    # A thread does not propagate what it raises: kept here to re-raise.
    raised_by_the_prune: list[Exception] = []

    def prune_by_sequential_scan() -> None:
        # Runs in a thread, on a connection of its own. Without index scans,
        # the database meets the rows in storage order: only a sort locks them
        # in key order, whatever plan it would choose otherwise.
        try:
            with transaction.atomic(), connection.cursor() as cursor:
                cursor.execute("SET LOCAL enable_indexscan = off")
                cursor.execute("SET LOCAL enable_bitmapscan = off")
                Indexer().prune("faq", set())
        except Exception as error:
            raised_by_the_prune.append(error)
        finally:
            connection.close()

    pruning = threading.Thread(target=prune_by_sequential_scan)
    try:
        with transaction.atomic():
            # Holding the row "faq:a" makes the prune wait there, still
            # holding the rows it locked before it.
            Source.objects.select_for_update().get(source_key="faq:a")
            pruning.start()
            wait_until_a_query_waits_for_a_lock(other_connection)
            locked_while_the_prune_waits = source_keys_locked_against(other_connection)
    finally:
        if pruning.is_alive():
            pruning.join()
        # Raised here, it also surfaces when a prune failing early makes the
        # wait above time out.
        if raised_by_the_prune:
            raise raised_by_the_prune[0]

    assert locked_while_the_prune_waits == ["faq:B", "faq:a"]


@pytest.mark.django_db(transaction=True)
def test_prune_leaves_a_source_committed_after_it_locked_its_sources(
    other_connection: "BaseDatabaseWrapper",
) -> None:
    # Under READ COMMITTED, a query run after the locks sees the rows committed
    # since: deleting by the same filter would remove a source never locked.
    Indexer().replace({"faq:1": [faq_entry(1)]})
    table = other_connection.ops.quote_name(Source._meta.db_table)
    committed_after_the_locks: list[int] = []

    def commit_a_source_after_the_locks(
        execute: Callable[[str, Any, bool, dict[str, Any]], Any],
        sql: str,
        params: Any,
        many: bool,
        context: dict[str, Any],
    ) -> Any:
        result = execute(sql, params, many, context)
        if "FOR UPDATE" in sql and not committed_after_the_locks:
            with other_connection.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO {table} (source_key) VALUES ('faq:5') RETURNING id"
                )
                committed_after_the_locks.append(cursor.fetchone()[0])
        return result

    with connection.execute_wrapper(commit_a_source_after_the_locks):
        Indexer().prune("faq", set())

    assert list(Source.objects.values_list("pk", "source_key")) == [
        (committed_after_the_locks[0], "faq:5")
    ]


@pytest.mark.django_db
def test_embed_pending_with_every_chunk_embedded_returns_0_without_a_backend(
    settings: "Settings",
) -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1)], "faq:2": [faq_entry(2)]})
    del settings.MINIMAL_RAG_EMBEDDINGS

    assert embed_pending() == 0


@pytest.mark.django_db
def test_embed_pending_gives_a_chunk_without_a_vector_the_backend_vector_and_model(
    settings: "Settings",
) -> None:
    use_fake_embeddings(settings, dimensions=3)
    [expected_vector] = FakeEmbeddings(dimensions=3).embed(["Answer to question 1."])
    source = Source.objects.create(source_key="faq:1")
    document = Document.objects.create(
        source=source,
        title="Question 1",
        url="https://example.com/faq/1/",
        language="en",
    )
    # A chunk stored before its embedding: no vector, no model.
    Chunk.objects.create(
        document=document,
        rank=0,
        text="Answer to question 1.",
        embedding=None,
        embedding_model="",
    )

    embedded_count = embed_pending()

    chunk = Chunk.objects.get()
    assert embedded_count == 1
    assert chunk.embedding_model == "fake-3"
    assert list(chunk.embedding) == expected_vector


@pytest.mark.django_db
def test_embed_pending_embeds_the_pending_chunks_of_several_sources_in_one_call(
    settings: "Settings",
) -> None:
    calls = use_call_recording_embeddings(settings)
    store_pending_chunks("faq:1", ["Answer to question 1.", "More on question 1."])
    store_pending_chunks("faq:2", ["Answer to question 2."])

    embedded_count = embed_pending()

    assert embedded_count == 3
    assert len(calls) == 1
    [embedded_texts] = calls
    assert sorted(embedded_texts) == [
        "Answer to question 1.",
        "Answer to question 2.",
        "More on question 1.",
    ]
    # RecordingEmbeddings gives each text its position in the call.
    assert {
        chunk.text: (chunk.embedding_model, list(chunk.embedding))
        for chunk in Chunk.objects.all()
    } == {
        text: ("recording", [float(position), 1.0])
        for position, text in enumerate(embedded_texts)
    }


@pytest.mark.django_db
def test_embed_pending_leaves_a_chunk_embedded_by_another_model_as_it_is(
    settings: "Settings",
) -> None:
    replace_and_run_on_commit({"faq:1": [faq_entry(1)]})
    embedded_chunk = Chunk.objects.get()
    calls = use_call_recording_embeddings(settings)
    assert embedded_chunk.embedding_model != "recording"
    store_pending_chunks("faq:2", ["Answer to question 2."])

    embedded_count = embed_pending()

    assert embedded_count == 1
    assert calls == [["Answer to question 2."]]
    kept_chunk = Chunk.objects.get(pk=embedded_chunk.pk)
    assert kept_chunk.embedding_model == embedded_chunk.embedding_model
    assert list(kept_chunk.embedding) == list(embedded_chunk.embedding)
    pending_chunk = Chunk.objects.get(text="Answer to question 2.")
    assert pending_chunk.embedding_model == "recording"
    assert list(pending_chunk.embedding) == [0.0, 1.0]


@pytest.mark.django_db
def test_embed_pending_embeds_a_shared_text_once_and_gives_each_chunk_its_vector(
    settings: "Settings",
) -> None:
    embedded = use_recording_embeddings(settings)
    shared_text = "Shared answer."
    store_pending_chunks("faq:1", [shared_text])
    store_pending_chunks("faq:2", [shared_text])

    embedded_count = embed_pending()

    assert embedded_count == 2
    assert embedded == [shared_text]
    assert {
        chunk.document.source.source_key: (chunk.embedding_model, list(chunk.embedding))
        for chunk in Chunk.objects.select_related("document__source")
    } == {"faq:1": ("recording", [0.0, 1.0]), "faq:2": ("recording", [0.0, 1.0])}


@pytest.mark.django_db
def test_embed_pending_with_a_batch_size_embeds_the_chunks_in_calls_of_that_size(
    settings: "Settings",
) -> None:
    calls = use_call_recording_embeddings(settings)
    texts = [f"Answer to question {number}." for number in range(1, 6)]
    store_pending_chunks("faq:1", texts)

    embedded_count = embed_pending(batch_size=2)

    assert embedded_count == 5
    assert len(calls) == 3
    assert all(len(call) <= 2 for call in calls)
    embedded_texts = [text for call in calls for text in call]
    assert sorted(embedded_texts) == sorted(texts)
    # RecordingEmbeddings gives each text its position among all texts embedded.
    assert {
        chunk.text: (chunk.embedding_model, list(chunk.embedding))
        for chunk in Chunk.objects.all()
    } == {
        text: ("recording", [float(position), 1.0])
        for position, text in enumerate(embedded_texts)
    }


@pytest.mark.django_db(transaction=True)
def test_embed_pending_calls_the_backend_with_no_transaction_open(
    settings: "Settings",
) -> None:
    # A backend call can take long: a transaction open around it would hold
    # its locks, and a connection, for that long.
    transaction_open: list[bool] = []
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.TransactionProbingEmbeddings",
        "OPTIONS": {"transaction_open": transaction_open},
    }
    store_pending_chunks("faq:1", ["Answer to question 1.", "Answer to question 2."])

    embedded_count = embed_pending(batch_size=1)

    assert embedded_count == 2
    assert transaction_open == [False, False]


@pytest.mark.django_db
@pytest.mark.parametrize("batch_size", [0, -1])
def test_embed_pending_with_a_batch_size_below_1_raises_and_embeds_nothing(
    settings: "Settings", batch_size: int
) -> None:
    calls = use_call_recording_embeddings(settings)
    store_pending_chunks("faq:1", ["Answer to question 1.", "Answer to question 2."])

    with pytest.raises(ValueError, match="batch_size"):
        embed_pending(batch_size=batch_size)

    assert calls == []
    assert list(
        Chunk.objects.order_by("rank").values_list(
            "text", "embedding", "embedding_model"
        )
    ) == [
        ("Answer to question 1.", None, ""),
        ("Answer to question 2.", None, ""),
    ]


@pytest.mark.django_db
def test_embed_pending_raises_and_stores_no_vector_when_the_backend_miscounts(
    settings: "Settings",
) -> None:
    use_miscounting_embeddings(settings, vector_count=1)
    store_pending_chunks("faq:1", ["Answer to question 1.", "Answer to question 2."])

    # Both counts, in any order and wording: 2 texts, 1 vector.
    with pytest.raises(ValueError, match=r"(?s)^(?=.*\b2\b)(?=.*\b1\b)"):
        embed_pending()

    assert list(
        Chunk.objects.order_by("rank").values_list(
            "text", "embedding", "embedding_model"
        )
    ) == [
        ("Answer to question 1.", None, ""),
        ("Answer to question 2.", None, ""),
    ]


@pytest.mark.django_db
def test_embed_pending_raises_the_backend_error_and_leaves_the_batch_to_the_next_run(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.FailingEmbeddings",
        "OPTIONS": {"failing_text": "Answer to question 2."},
    }
    texts = ["Answer to question 1.", "Answer to question 2."]
    store_pending_chunks("faq:1", texts)

    with pytest.raises(EmbeddingFailedError):
        embed_pending()

    # Neither embedded nor claimed: no run is kept from taking them.
    assert list(
        Chunk.objects.order_by("rank").values_list(
            "text", "embedding", "embedding_model", "embedding_claimed_at"
        )
    ) == [
        ("Answer to question 1.", None, "", None),
        ("Answer to question 2.", None, "", None),
    ]
    calls = use_call_recording_embeddings(settings)

    embedded_count = embed_pending()

    assert embedded_count == 2
    [embedded_texts] = calls
    assert sorted(embedded_texts) == texts
    # RecordingEmbeddings gives each text its position in the call.
    assert {
        chunk.text: (chunk.embedding_model, list(chunk.embedding))
        for chunk in Chunk.objects.all()
    } == {
        text: ("recording", [float(position), 1.0])
        for position, text in enumerate(embedded_texts)
    }


@pytest.mark.django_db
def test_embed_pending_keeps_the_vectors_of_earlier_batches_when_a_later_one_fails(
    settings: "Settings",
) -> None:
    calls: list[list[str]] = []
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.FailingEmbeddings",
        "OPTIONS": {"failing_text": "Answer to question 4.", "calls": calls},
    }
    texts = [f"Answer to question {number}." for number in range(1, 5)]
    # Stored last, the failing text's chunk is taken in a later batch.
    store_pending_chunks("faq:1", texts)

    with pytest.raises(EmbeddingFailedError):
        embed_pending(batch_size=2)

    *earlier_calls, failed_call = calls
    assert "Answer to question 4." in failed_call
    embedded_texts = [text for call in earlier_calls for text in call]
    assert embedded_texts
    # The failed batch's chunks still await a vector, the others have theirs.
    assert sorted(
        Chunk.objects.filter(embedding__isnull=True, embedding_model="").values_list(
            "text", flat=True
        )
    ) == sorted(failed_call)
    # FailingEmbeddings gives each text its length.
    assert {
        chunk.text: (chunk.embedding_model, list(chunk.embedding))
        for chunk in Chunk.objects.filter(embedding__isnull=False)
    } == {text: ("failing", [float(len(text)), 1.0]) for text in embedded_texts}


@pytest.mark.django_db
def test_embed_pending_gives_no_vector_to_a_chunk_deleted_during_its_backend_call(
    settings: "Settings",
) -> None:
    store_pending_chunks("faq:1", ["Answer to question 1.", "More on question 1."])
    store_pending_chunks("faq:2", ["Answer to question 2."])

    def remove_faq_2() -> None:
        # The source is removed while its chunk is being embedded.
        Source.objects.filter(source_key="faq:2").delete()

    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.InterleavingEmbeddings",
        "OPTIONS": {"during_embed": remove_faq_2},
    }

    embed_pending()

    # The deleted chunk stays deleted; InterleavingEmbeddings gives each text
    # its length, and the rest of the batch gets its vectors.
    assert {
        chunk.text: (chunk.embedding_model, list(chunk.embedding))
        for chunk in Chunk.objects.all()
    } == {
        "Answer to question 1.": ("interleaving", [21.0, 1.0]),
        "More on question 1.": ("interleaving", [19.0, 1.0]),
    }


@pytest.mark.django_db
def test_embed_pending_run_during_another_runs_backend_call_skips_its_claimed_chunks(
    settings: "Settings",
) -> None:
    claimed_texts = ["Answer to question 1.", "More on question 1."]
    store_pending_chunks("faq:1", claimed_texts)
    calls: list[list[str]] = []
    second_run_counts: list[int] = []

    def run_a_second_embed_pending() -> None:
        # Only during the first run's backend call: the second run's own
        # backend call comes back here too.
        if len(calls) > 1:
            return
        # Stored once the first run has claimed its batch, so left to the second.
        store_pending_chunks("faq:2", ["Answer to question 2."])
        second_run_counts.append(embed_pending())

    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.InterleavingEmbeddings",
        "OPTIONS": {"during_embed": run_a_second_embed_pending, "calls": calls},
    }

    first_run_count = embed_pending()

    assert first_run_count == 2
    assert second_run_counts == [1]
    first_call, *second_run_calls = calls
    assert sorted(first_call) == claimed_texts
    assert second_run_calls == [["Answer to question 2."]]
    # InterleavingEmbeddings gives each text its length.
    assert {
        chunk.text: (chunk.embedding_model, list(chunk.embedding))
        for chunk in Chunk.objects.all()
    } == {
        "Answer to question 1.": ("interleaving", [21.0, 1.0]),
        "More on question 1.": ("interleaving", [19.0, 1.0]),
        "Answer to question 2.": ("interleaving", [21.0, 1.0]),
    }


@pytest.mark.django_db
def test_embed_pending_skips_a_chunk_claimed_within_the_claim_timeout(
    settings: "Settings",
) -> None:
    calls = use_call_recording_embeddings(settings)
    store_pending_chunks("faq:1", ["Claimed answer.", "Unclaimed answer."])
    # Another run claimed this chunk two minutes ago and may still embed it.
    claimed_at = timezone.now() - datetime.timedelta(minutes=2)
    Chunk.objects.filter(text="Claimed answer.").update(embedding_claimed_at=claimed_at)

    embedded_count = embed_pending(claim_timeout=datetime.timedelta(minutes=10))

    assert embedded_count == 1
    assert calls == [["Unclaimed answer."]]
    claimed_chunk = Chunk.objects.get(text="Claimed answer.")
    assert claimed_chunk.embedding is None
    assert claimed_chunk.embedding_model == ""
    unclaimed_chunk = Chunk.objects.get(text="Unclaimed answer.")
    assert unclaimed_chunk.embedding_model == "recording"
    assert list(unclaimed_chunk.embedding) == [0.0, 1.0]


@pytest.mark.django_db
def test_embed_pending_embeds_a_chunk_whose_claim_is_older_than_the_claim_timeout(
    settings: "Settings",
) -> None:
    calls = use_call_recording_embeddings(settings)
    store_pending_chunks("faq:1", ["Abandoned answer."])
    # Another run claimed this chunk twenty minutes ago and never embedded it.
    claimed_at = timezone.now() - datetime.timedelta(minutes=20)
    Chunk.objects.update(embedding_claimed_at=claimed_at)

    embedded_count = embed_pending(claim_timeout=datetime.timedelta(minutes=10))

    assert embedded_count == 1
    assert calls == [["Abandoned answer."]]
    chunk = Chunk.objects.get()
    assert chunk.embedding_model == "recording"
    assert list(chunk.embedding) == [0.0, 1.0]
