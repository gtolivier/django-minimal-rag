from typing import TYPE_CHECKING

import pytest

from django_minimal_rag.answering import answer
from django_minimal_rag.indexing import Indexer
from tests.documents import public_page
from tests.embeddings import use_chosen_embeddings

if TYPE_CHECKING:
    from pytest_django import Settings


pytestmark = pytest.mark.usefixtures("fake_embeddings")


def use_recording_llm(settings: "Settings", calls: list[tuple[str, str]]) -> None:
    """Configure ``RecordingLLM``, appending each ``(prompt, system)`` to ``calls``."""
    settings.MINIMAL_RAG_LLM = {
        "BACKEND": "tests.llms.RecordingLLM",
        "OPTIONS": {"calls": calls},
    }


def use_canned_llm(settings: "Settings", reply: str) -> None:
    """Configure ``CannedLLM``, answering every prompt with ``reply``."""
    settings.MINIMAL_RAG_LLM = {
        "BACKEND": "tests.llms.CannedLLM",
        "OPTIONS": {"reply": reply},
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


@pytest.mark.django_db
def test_answer_with_only_a_chunk_beyond_max_distance_returns_none_without_llm(
    settings: "Settings",
) -> None:
    chunk_text = "The shop opens at nine on weekdays."
    question = "Where is the shop?"
    # Cosine distance of the chunk to the question: 1.0, the vectors are orthogonal.
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [0.0, 1.0]})
    Indexer().replace({"page:1": [public_page(chunk_text)]})
    llm_calls: list[tuple[str, str]] = []
    use_recording_llm(settings, llm_calls)

    answered = answer(question, permissions=frozenset(), max_distance=0.5)

    assert answered is None
    assert llm_calls == []


@pytest.mark.django_db
@pytest.mark.parametrize(
    "reply",
    ["The shop opens at nine.", "It opens at 9 a.m., every day but Sunday."],
)
def test_answer_with_a_chunk_near_the_question_returns_the_llm_reply_as_its_text(
    settings: "Settings", reply: str
) -> None:
    chunk_text = "The shop opens at nine on weekdays."
    question = "When does the shop open?"
    # Cosine distance of the chunk to the question: about 0.005.
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [1.0, 0.1]})
    Indexer().replace({"page:1": [public_page(chunk_text)]})
    use_canned_llm(settings, reply)

    answered = answer(question, permissions=frozenset(), max_distance=0.1)

    assert answered is not None
    assert answered.text == reply
