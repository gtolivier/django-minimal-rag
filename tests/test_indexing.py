import pytest

from django_minimal_rag.indexing import Indexer
from django_minimal_rag.models import Chunk, Document, Source
from tests.documents import SampleDocument


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
