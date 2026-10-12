from typing import TYPE_CHECKING

from django_minimal_rag.llm import FakeLLM, get_llm
from tests.llms import CannedLLM

if TYPE_CHECKING:
    from pytest_django import Settings


def test_fake_llm_answers_with_the_prompt_it_was_given() -> None:
    llm = FakeLLM()

    first = llm.complete("What is a vector?", system="Answer briefly.")
    second = llm.complete("Who wrote the docs?", system="Cite your sources.")

    assert first == "What is a vector?"
    assert second == "Who wrote the docs?"


def test_get_llm_builds_the_backend_of_the_setting_with_its_options(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_LLM = {
        "BACKEND": "tests.llms.CannedLLM",
        "OPTIONS": {"reply": "A vector is a list of numbers."},
    }

    llm = get_llm()

    assert isinstance(llm, CannedLLM)
    assert (
        llm.complete("What is a vector?", system="Answer briefly.")
        == "A vector is a list of numbers."
    )
