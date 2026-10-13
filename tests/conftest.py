from typing import TYPE_CHECKING

import pytest

from tests.llms import llm_calls

if TYPE_CHECKING:
    from pytest_django import Settings

# Fixtures defined next to the backends they configure, made available to
# every test module by importing them here. Listed so the import reads as a
# re-export: the tests request a fixture by name, never use the import.
__all__ = ["fake_embeddings", "llm_calls"]


@pytest.fixture
def fake_embeddings(settings: "Settings") -> None:
    """Configure the embedding backend meant for tests."""
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
    }
