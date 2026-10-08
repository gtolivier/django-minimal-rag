import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_minimal_rag.embeddings import FakeEmbeddings, get_embeddings
from tests.embeddings import SampleEmbeddings

if TYPE_CHECKING:
    from pytest_django import Settings

    # The Protocol is only an annotation here: satisfying it is checked by
    # mypy, so the import is not needed when the tests run.
    from django_minimal_rag.embeddings import Embeddings

EMBED_A_TEXT_IN_A_CHILD_PROCESS = (
    "import json\n"
    "from django_minimal_rag.embeddings import FakeEmbeddings\n"
    "print(json.dumps(FakeEmbeddings().embed(['a text'])[0]))\n"
)


def embed_a_text_in_a_child_process(hash_seed: str) -> list[float]:
    result = subprocess.run(
        [sys.executable, "-c", EMBED_A_TEXT_IN_A_CHILD_PROCESS],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    )
    vector: list[float] = json.loads(result.stdout)
    return vector


def test_fake_embeddings_of_no_texts_is_an_empty_list() -> None:
    assert FakeEmbeddings().embed([]) == []


def test_fake_embeddings_of_one_text_is_one_vector_of_eight_floats() -> None:
    vectors = FakeEmbeddings().embed(["a text"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 8
    assert all(isinstance(value, float) for value in vectors[0])


def test_fake_embeddings_with_three_dimensions_are_vectors_of_three_floats() -> None:
    vectors = FakeEmbeddings(dimensions=3).embed(["a text"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 3
    assert all(isinstance(value, float) for value in vectors[0])


def test_fake_embeddings_of_two_different_texts_are_different_vectors() -> None:
    first, second = FakeEmbeddings().embed(["a text", "another text"])

    assert first != second


def test_fake_embeddings_of_two_different_texts_of_the_same_length_differ() -> None:
    first, second = FakeEmbeddings().embed(["a text", "b text"])

    assert first != second


def test_fake_embeddings_returns_one_vector_per_text_in_the_texts_order() -> None:
    embeddings = FakeEmbeddings()
    [short_vector] = embeddings.embed(["a"])
    [long_vector] = embeddings.embed(["a longer text"])

    assert embeddings.embed(["a", "a longer text"]) == [short_vector, long_vector]
    assert embeddings.embed(["a longer text", "a"]) == [long_vector, short_vector]


def test_fake_embeddings_of_the_same_text_from_two_instances_are_equal() -> None:
    [first] = FakeEmbeddings().embed(["a text"])
    [second] = FakeEmbeddings().embed(["a text"])

    assert first == second


def test_fake_embeddings_of_the_same_text_are_equal_in_other_processes() -> None:
    [vector] = FakeEmbeddings().embed(["a text"])

    assert embed_a_text_in_a_child_process(hash_seed="1") == vector
    assert embed_a_text_in_a_child_process(hash_seed="2") == vector


def test_fake_embeddings_with_zero_dimensions_raise_value_error() -> None:
    with pytest.raises(ValueError):
        FakeEmbeddings(dimensions=0)


def test_fake_embeddings_model_is_fake_8() -> None:
    assert FakeEmbeddings().model == "fake-8"


def test_fake_embeddings_with_three_dimensions_model_is_fake_3() -> None:
    assert FakeEmbeddings(dimensions=3).model == "fake-3"


def test_fake_embeddings_are_accepted_where_embeddings_are_expected() -> None:
    fake = FakeEmbeddings()

    embeddings: Embeddings = fake

    assert embeddings is fake


def test_get_embeddings_returns_the_backend_of_the_setting(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
    }

    assert isinstance(get_embeddings(), FakeEmbeddings)


def test_get_embeddings_returns_a_backend_defined_outside_the_package(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.SampleEmbeddings",
    }

    assert isinstance(get_embeddings(), SampleEmbeddings)


def test_get_embeddings_builds_the_backend_with_the_options_of_the_setting(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
        "OPTIONS": {"dimensions": 3},
    }

    [vector] = get_embeddings().embed(["a text"])

    assert len(vector) == 3


def test_get_embeddings_builds_a_new_backend_from_the_setting_on_every_call(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
    }
    first = get_embeddings()
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
        "OPTIONS": {"dimensions": 3},
    }

    second = get_embeddings()

    assert second is not first
    assert second.model == "fake-3"


def test_get_embeddings_without_the_setting_raises_improperly_configured(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
    }
    del settings.MINIMAL_RAG_EMBEDDINGS

    with pytest.raises(ImproperlyConfigured):
        get_embeddings()
