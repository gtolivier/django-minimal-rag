from collections.abc import Mapping
from typing import TYPE_CHECKING

import pytest

from django_minimal_rag.indexing import Indexer
from django_minimal_rag.retrieval import retrieve
from tests.documents import SampleDocument

if TYPE_CHECKING:
    from pytest_django import Settings


@pytest.fixture(autouse=True)
def fake_embeddings(settings: "Settings") -> None:
    """Configure the embedding backend meant for tests."""
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
    }


def use_chosen_embeddings(
    settings: "Settings", vectors: Mapping[str, list[float]]
) -> None:
    """Configure ``ChosenEmbeddings``, giving each text of ``vectors`` its vector."""
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.ChosenEmbeddings",
        "OPTIONS": {"vectors": vectors},
    }


def public_page(text: str) -> SampleDocument:
    """A public page of the host project, in English, holding ``text``."""
    return SampleDocument(
        text=text,
        source_key="page:1",
        title="Opening hours",
        url="https://example.com/opening-hours/",
        language="en",
        permissions=frozenset(),
    )


@pytest.mark.django_db
def test_retrieve_with_nothing_indexed_returns_an_empty_list() -> None:
    retrieved = retrieve(
        "What are the opening hours?", permissions=frozenset(), max_distance=2.0
    )

    assert retrieved == []


@pytest.mark.django_db
def test_retrieve_returns_the_text_of_a_single_chunk_indexed_near_the_question(
    settings: "Settings",
) -> None:
    chunk_text = "The shop opens at nine."
    question = "When does the shop open?"
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [1.0, 0.1]})
    Indexer().replace({"page:1": [public_page(chunk_text)]})

    retrieved = retrieve(question, permissions=frozenset(), max_distance=0.1)

    assert [chunk.text for chunk in retrieved] == [chunk_text]
