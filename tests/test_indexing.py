import pytest

from django_minimal_rag.indexing import Indexer
from django_minimal_rag.models import Chunk, Document, Source


@pytest.mark.django_db
def test_replace_with_no_groups_stores_nothing() -> None:
    Indexer().replace({})

    assert not Source.objects.exists()
    assert not Document.objects.exists()
    assert not Chunk.objects.exists()
