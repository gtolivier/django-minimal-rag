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
