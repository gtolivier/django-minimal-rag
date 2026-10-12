from typing import TYPE_CHECKING

import pytest

from django_minimal_rag.answering import answer

if TYPE_CHECKING:
    from pytest_django import Settings


pytestmark = pytest.mark.usefixtures("fake_embeddings")


def use_recording_llm(settings: "Settings", calls: list[tuple[str, str]]) -> None:
    """Configure ``RecordingLLM``, appending each ``(prompt, system)`` to ``calls``."""
    settings.MINIMAL_RAG_LLM = {
        "BACKEND": "tests.llms.RecordingLLM",
        "OPTIONS": {"calls": calls},
    }


@pytest.mark.django_db
def test_answer_with_nothing_indexed_returns_none_without_calling_the_llm(
    settings: "Settings",
) -> None:
    llm_calls: list[tuple[str, str]] = []
    use_recording_llm(settings, llm_calls)

    answered = answer(
        "What are the opening hours?", permissions=frozenset(), max_distance=2.0
    )

    assert answered is None
    assert llm_calls == []
