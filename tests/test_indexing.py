from typing import TYPE_CHECKING

import pytest

from django_minimal_rag.chunking import chunk_group
from django_minimal_rag.embeddings import FakeEmbeddings
from django_minimal_rag.indexing import Indexer
from django_minimal_rag.models import Chunk, Document, Source
from tests.documents import SampleDocument

if TYPE_CHECKING:
    from pytest_django import Settings


@pytest.fixture(autouse=True)
def fake_embeddings(settings: "Settings") -> None:
    """Configure the embedding backend meant for tests."""
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
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


@pytest.mark.django_db
def test_replace_with_no_groups_stores_nothing() -> None:
    Indexer().replace({})

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()


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
            "faq:2": [faq_entry(2), faq_entry(3)],
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

    Indexer().replace({"faq:1": [faq_entry(2), faq_entry(3)]})

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
    Indexer().replace({"faq:1": [faq_entry(1), faq_entry(2)]})

    Indexer().replace({"faq:1": []})

    assert not Source.objects.filter(source_key="faq:1").exists()
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
    document = SampleDocument(
        text="Answer to question 1.",
        source_key="faq:1",
        title="Question 1",
        url="https://example.com/faq/1/",
        language="en",
        permissions=frozenset({"faq.view_faq", "app.change_note"}),
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
    paragraphs = [f"Paragraph {number}: " + "word " * 120 for number in range(5)]
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
    paragraphs = [f"Paragraph {number}: " + "word " * 120 for number in range(2)]
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
