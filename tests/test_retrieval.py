from typing import TYPE_CHECKING

import pytest

from django_minimal_rag.retrieval import retrieve

if TYPE_CHECKING:
    from pytest_django import Settings


@pytest.fixture(autouse=True)
def fake_embeddings(settings: "Settings") -> None:
    """Configure the embedding backend meant for tests."""
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
    }


@pytest.mark.django_db
def test_retrieve_with_nothing_indexed_returns_an_empty_list() -> None:
    retrieved = retrieve(
        "What are the opening hours?", permissions=frozenset(), max_distance=2.0
    )

    assert retrieved == []
