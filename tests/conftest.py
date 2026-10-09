from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_django import Settings


@pytest.fixture
def fake_embeddings(settings: "Settings") -> None:
    """Configure the embedding backend meant for tests."""
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
    }
