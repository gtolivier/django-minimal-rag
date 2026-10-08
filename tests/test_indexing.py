import dataclasses
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import pytest
from django.db import (
    DEFAULT_DB_ALIAS,
    OperationalError,
    connection,
    connections,
    transaction,
)
from django.test.utils import CaptureQueriesContext
from psycopg.errors import LockNotAvailable

from django_minimal_rag.chunking import chunk_group
from django_minimal_rag.embeddings import FakeEmbeddings
from django_minimal_rag.indexing import Indexer
from django_minimal_rag.models import Chunk, Document, Source
from tests.documents import MutableDocument, SampleDocument
from tests.embeddings import EmbeddingFailedError
from tests.sequences import SinglePassSequence

if TYPE_CHECKING:
    from pytest_django import Settings


@pytest.fixture(autouse=True)
def fake_embeddings(settings: "Settings") -> None:
    """Configure the embedding backend meant for tests."""
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
    }


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


def use_miscounting_embeddings(settings: "Settings", vector_count: int) -> None:
    """Configure ``MiscountingEmbeddings``, returning ``vector_count`` vectors."""
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.MiscountingEmbeddings",
        "OPTIONS": {"vector_count": vector_count},
    }


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


@pytest.mark.django_db
def test_replace_with_no_groups_stores_nothing() -> None:
    Indexer().replace({})

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_replace_with_no_groups_succeeds_without_an_embedding_backend_configured(
    settings: "Settings",
) -> None:
    del settings.MINIMAL_RAG_EMBEDDINGS

    Indexer().replace({})


@pytest.mark.django_db
def test_replace_group_of_one_document_creates_source_with_group_key() -> None:
    Indexer().replace({"faq:1": [faq_entry(1)]})

    assert list(Source.objects.values_list("source_key", flat=True)) == ["faq:1"]


@pytest.mark.django_db
def test_replace_group_of_one_document_stores_it_under_the_group_source() -> None:
    Indexer().replace({"faq:1": [faq_entry(1)]})

    assert list(
        Document.objects.values_list("source__source_key", "title", "url", "language")
    ) == [("faq:1", "Question 1", "https://example.com/faq/1/", "en")]


@pytest.mark.django_db
def test_replace_stores_each_group_under_its_own_source() -> None:
    Indexer().replace(
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
    Indexer().replace({"faq:1": [faq_entry(1)]})
    source_pk = Source.objects.get().pk

    Indexer().replace(
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
    Indexer().replace({"faq:1": [faq_entry(1), faq_entry_under("faq:1", 2)]})

    Indexer().replace({"faq:1": []})

    assert not Source.objects.filter(source_key="faq:1").exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_replace_with_an_empty_group_for_an_unknown_key_stores_nothing() -> None:
    Indexer().replace({"faq:1": []})

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_replace_with_only_empty_groups_removes_sources_without_a_backend_configured(
    settings: "Settings",
) -> None:
    Indexer().replace({"faq:1": [faq_entry(1)]})
    del settings.MINIMAL_RAG_EMBEDDINGS

    Indexer().replace({"faq:1": [], "faq:2": []})

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_replace_leaves_sources_absent_from_the_call_as_they_are() -> None:
    Indexer().replace({"faq:1": [faq_entry(1)]})
    source = Source.objects.get()
    document = Document.objects.get()
    chunk = Chunk.objects.get()

    Indexer().replace({"faq:2": [faq_entry(2)]})

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

    Indexer().replace({"faq:1": [document]})

    assert list(Document.objects.values_list("permissions", flat=True)) == [
        ["app.change_note", "faq.view_faq"]
    ]


@pytest.mark.django_db
def test_replace_stores_a_short_document_text_as_one_chunk_of_rank_0() -> None:
    Indexer().replace({"faq:1": [faq_entry(1)]})

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

    Indexer().replace({"guide:1": [long_document]})

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

    Indexer().replace({"guide:1": [two_chunk_document, one_chunk_document]})

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

    Indexer().replace({"faq:1": [entry, entry]})

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

    Indexer().replace({"faq:1": group})

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

    Indexer().replace({"faq:1": [document]})

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
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
        "OPTIONS": {"dimensions": 3},
    }
    backend = FakeEmbeddings(dimensions=3)
    [expected_vector] = backend.embed(["Answer to question 1."])

    Indexer().replace({"faq:1": [faq_entry(1)]})

    chunk = Chunk.objects.get()
    assert chunk.embedding_model == "fake-3"
    assert list(chunk.embedding) == expected_vector


@pytest.mark.django_db
def test_replace_with_the_same_texts_embeds_nothing_and_keeps_each_vector(
    settings: "Settings",
) -> None:
    embedded = use_recording_embeddings(settings)
    Indexer().replace({"faq:1": [faq_entry(1), faq_entry_under("faq:1", 2)]})
    vectors_before = {
        chunk.text: list(chunk.embedding) for chunk in Chunk.objects.all()
    }
    embedded_before = len(embedded)

    Indexer().replace({"faq:1": [faq_entry(1), faq_entry_under("faq:1", 2)]})

    assert embedded[embedded_before:] == []
    assert {
        chunk.text: list(chunk.embedding) for chunk in Chunk.objects.all()
    } == vectors_before


@pytest.mark.django_db
def test_replace_changing_one_text_embeds_it_only_and_keeps_the_other_vectors(
    settings: "Settings",
) -> None:
    embedded = use_recording_embeddings(settings)
    Indexer().replace(
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

    Indexer().replace(
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
    Indexer().replace({"faq:1": [faq_entry(1), faq_entry_under("faq:1", 2)]})
    embedded = use_recording_embeddings(settings)

    Indexer().replace({"faq:1": [faq_entry(1), faq_entry_under("faq:1", 2)]})

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

    Indexer().replace(
        {"faq:1": [faq_entry(1)], "faq:2": [faq_entry(2)], "faq:3": [faq_entry(3)]}
    )

    assert calls == [
        ["Answer to question 1.", "Answer to question 2.", "Answer to question 3."]
    ]


@pytest.mark.django_db
def test_replace_embeds_a_new_text_of_two_groups_once_and_gives_both_its_vector(
    settings: "Settings",
) -> None:
    embedded = use_recording_embeddings(settings)
    shared_text = "Shared answer."
    entry_1 = dataclasses.replace(faq_entry(1), text=shared_text)
    entry_2 = dataclasses.replace(faq_entry(2), text=shared_text)

    Indexer().replace({"faq:1": [entry_1], "faq:2": [entry_2]})

    assert embedded == [shared_text]
    assert {
        chunk.document.source.source_key: list(chunk.embedding)
        for chunk in Chunk.objects.select_related("document__source")
    } == {"faq:1": [0.0, 1.0], "faq:2": [0.0, 1.0]}


@pytest.mark.django_db
def test_replace_stores_nothing_when_the_backend_raises_on_a_later_group(
    settings: "Settings",
) -> None:
    Indexer().replace({"faq:1": [faq_entry(1)], "faq:2": [faq_entry(2)]})
    source = Source.objects.get(source_key="faq:1")
    content_before = stored_content(source)
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.FailingEmbeddings",
        "OPTIONS": {"failing_text": "New answer to question 2."},
    }
    new_entry_1 = dataclasses.replace(faq_entry(1), text="New answer to question 1.")
    new_entry_2 = dataclasses.replace(faq_entry(2), text="New answer to question 2.")

    with pytest.raises(EmbeddingFailedError):
        Indexer().replace({"faq:1": [new_entry_1], "faq:2": [new_entry_2]})

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
        Indexer().replace({"guide:1": [two_chunk_document]})

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
        Indexer().replace({"faq:1": [faq_entry(1)]})


@pytest.mark.django_db
def test_replace_raises_and_stores_nothing_for_a_document_of_another_source_key() -> (
    None
):
    # Both keys, in any order and wording: the group's and the document's.
    with pytest.raises(ValueError, match=r"(?s)^(?=.*\bfaq:1\b)(?=.*\bfaq:2\b)"):
        Indexer().replace({"faq:1": [faq_entry(2)]})

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
        Indexer().replace(
            {
                "faq:3": [faq_entry(3)],
                "faq:5": [faq_entry(5), faq_entry_under("faq:5", 6), faq_entry(7)],
            }
        )

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_replace_locks_the_row_of_each_replaced_source_until_its_transaction_ends() -> (
    None
):
    Indexer().replace({"faq:1": [faq_entry(1)]})
    source = Source.objects.get()
    # A connection of its own, outside the test's transaction: it sees only
    # what is committed, and contends for row locks like another process.
    other_connection = connections.create_connection(DEFAULT_DB_ALIAS)
    table = other_connection.ops.quote_name(Source._meta.db_table)
    # FOR SHARE, not FOR UPDATE: inserting documents already takes a FOR KEY
    # SHARE lock on the source row (foreign-key check), which FOR SHARE does
    # not conflict with, unlike FOR UPDATE / FOR NO KEY UPDATE.
    probe = f"SELECT 1 FROM {table} WHERE id = %s FOR SHARE NOWAIT"
    try:
        with transaction.atomic():
            Indexer().replace({"faq:1": [faq_entry_under("faq:1", 2)]})

            with (
                pytest.raises(OperationalError) as raised,
                other_connection.cursor() as cursor,
            ):
                cursor.execute(probe, [source.pk])
    finally:
        other_connection.close()

    assert isinstance(raised.value.__cause__, LockNotAvailable)


@pytest.mark.django_db
def test_replace_locks_the_replaced_sources_in_sorted_source_key_order() -> None:
    # Two calls replacing overlapping sources must lock their rows in the same
    # order, whatever the order of their groups, or they can deadlock.
    Indexer().replace(
        {"faq:1": [faq_entry(1)], "faq:2": [faq_entry(2)], "faq:3": [faq_entry(3)]}
    )

    with CaptureQueriesContext(connection) as captured:
        Indexer().replace(
            {"faq:2": [faq_entry(2)], "faq:3": [faq_entry(3)], "faq:1": [faq_entry(1)]}
        )

    assert source_keys_locked_by(captured.captured_queries) == [
        "faq:1",
        "faq:2",
        "faq:3",
    ]
