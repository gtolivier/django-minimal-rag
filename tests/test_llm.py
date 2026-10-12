from typing import TYPE_CHECKING

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_minimal_rag.llm import FakeLLM, get_llm
from tests.llms import CannedLLM, remove_llm_setting, use_canned_llm

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
    use_canned_llm(settings, "A vector is a list of numbers.")

    llm = get_llm()

    assert isinstance(llm, CannedLLM)
    assert (
        llm.complete("What is a vector?", system="Answer briefly.")
        == "A vector is a list of numbers."
    )


def test_get_llm_without_the_setting_raises_improperly_configured(
    settings: "Settings",
) -> None:
    remove_llm_setting(settings)

    with pytest.raises(ImproperlyConfigured, match="MINIMAL_RAG_LLM"):
        get_llm()


@pytest.mark.parametrize(
    "malformed_setting",
    [
        # A common mistake: the backend's dotted path given directly, instead
        # of a mapping with that path under "BACKEND".
        pytest.param("tests.llms.CannedLLM", id="not-a-mapping"),
        pytest.param({"OPTIONS": {"reply": "Hello."}}, id="no-backend"),
        # A common mistake: the backend class itself instead of its dotted path.
        pytest.param({"BACKEND": CannedLLM}, id="non-string-backend"),
        pytest.param({"BACKEND": "tests.llms.NoSuchLLM"}, id="unimportable-backend"),
        pytest.param(
            {"BACKEND": "tests.llms.CannedLLM", "OPTIONS": [("reply", "Hello.")]},
            id="non-mapping-options",
        ),
    ],
)
def test_get_llm_with_a_malformed_setting_raises_improperly_configured(
    settings: "Settings",
    malformed_setting: object,
) -> None:
    settings.MINIMAL_RAG_LLM = malformed_setting

    with pytest.raises(ImproperlyConfigured, match="MINIMAL_RAG_LLM"):
        get_llm()
