import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
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

FAKE_EMBEDDINGS_BACKEND = "django_minimal_rag.embeddings.FakeEmbeddings"

EMBED_A_TEXT_IN_A_CHILD_PROCESS = (
    "import json\n"
    "from django_minimal_rag.embeddings import FakeEmbeddings\n"
    "print(json.dumps(FakeEmbeddings().embed(['a text'])[0]))\n"
)


class EmbeddingsWithoutModel:
    """Every member of an embedding backend except `model`."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class EmbeddingsWithoutEmbed:
    """Every member of an embedding backend except `embed`."""

    model: str = "without-embed"


class EmbeddingsWithListOnlyEmbed:
    """An embedding backend whose `embed` accepts a `list`, not any sequence."""

    model: str = "list-only-embed"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


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


def cosine_similarity(first: list[float], second: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(first, second, strict=True))
    return dot_product / (math.hypot(*first) * math.hypot(*second))


def test_fake_embeddings_of_no_texts_is_an_empty_list() -> None:
    assert FakeEmbeddings().embed([]) == []


def test_fake_embeddings_of_one_text_is_one_vector_of_the_default_dimension() -> None:
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


def test_fake_embeddings_of_two_different_texts_are_not_parallel() -> None:
    first, second = FakeEmbeddings().embed(["a text", "another text"])

    assert cosine_similarity(first, second) != pytest.approx(1)


def test_fake_embeddings_of_two_different_texts_of_the_same_length_differ() -> None:
    first, second = FakeEmbeddings().embed(["a text", "b text"])

    assert first != second


def test_fake_embeddings_of_two_anagrams_differ() -> None:
    first, second = FakeEmbeddings().embed(["a text", "t axte"])

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


def test_fake_embeddings_with_one_dimension_raise_value_error() -> None:
    # With one dimension every vector is parallel: a cosine lookup could not
    # tell texts apart.
    with pytest.raises(ValueError):
        FakeEmbeddings(dimensions=1)


def test_fake_embeddings_model_names_the_default_dimension() -> None:
    assert FakeEmbeddings().model == "fake-8"


def test_fake_embeddings_model_names_the_given_dimensions() -> None:
    assert FakeEmbeddings(dimensions=3).model == "fake-3"


def test_fake_embeddings_are_accepted_where_embeddings_are_expected() -> None:
    fake = FakeEmbeddings()

    embeddings: Embeddings = fake

    assert embeddings is fake


# A class missing any one member is not an embedding backend. That is a
# static behavior, so it is checked by mypy alone: the functions below are
# never called, and pytest does not collect them (their names do not start
# with `test_`). Each returns its argument where `Embeddings` are expected;
# mypy must refuse that return, and strict mypy reports the ignore as unused
# if it does not, which fails the type check.


def _rejects_without_model(embeddings: EmbeddingsWithoutModel) -> "Embeddings":
    return embeddings  # type: ignore[return-value]  # no `model`: not Embeddings


def _rejects_without_embed(embeddings: EmbeddingsWithoutEmbed) -> "Embeddings":
    return embeddings  # type: ignore[return-value]  # no `embed`: not Embeddings


def _rejects_list_only_embed(
    embeddings: EmbeddingsWithListOnlyEmbed,
) -> "Embeddings":
    return embeddings  # type: ignore[return-value]  # `embed` refuses a tuple


def test_get_embeddings_returns_the_backend_of_the_setting(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": FAKE_EMBEDDINGS_BACKEND,
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
        "BACKEND": FAKE_EMBEDDINGS_BACKEND,
        "OPTIONS": {"dimensions": 3},
    }

    [vector] = get_embeddings().embed(["a text"])

    assert len(vector) == 3


def test_get_embeddings_builds_a_new_backend_from_the_setting_on_every_call(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": FAKE_EMBEDDINGS_BACKEND,
    }
    first = get_embeddings()
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": FAKE_EMBEDDINGS_BACKEND,
        "OPTIONS": {"dimensions": 3},
    }

    second = get_embeddings()

    assert second is not first
    assert second.model == "fake-3"


def test_get_embeddings_with_an_unchanged_setting_builds_two_distinct_backends(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": FAKE_EMBEDDINGS_BACKEND,
    }
    first = get_embeddings()

    second = get_embeddings()

    assert second is not first


def test_get_embeddings_without_the_setting_raises_improperly_configured(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": FAKE_EMBEDDINGS_BACKEND,
    }
    del settings.MINIMAL_RAG_EMBEDDINGS

    with pytest.raises(ImproperlyConfigured):
        get_embeddings()


def test_get_embeddings_without_a_backend_raises_improperly_configured(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {"OPTIONS": {"dimensions": 3}}

    with pytest.raises(ImproperlyConfigured):
        get_embeddings()


def test_get_embeddings_with_a_non_string_backend_raises_improperly_configured(
    settings: "Settings",
) -> None:
    # A common mistake: the backend class itself instead of its dotted path.
    settings.MINIMAL_RAG_EMBEDDINGS = {"BACKEND": FakeEmbeddings}

    with pytest.raises(ImproperlyConfigured):
        get_embeddings()


def test_get_embeddings_with_a_non_mapping_setting_raises_improperly_configured(
    settings: "Settings",
) -> None:
    # A common mistake: the backend's dotted path given directly, instead of
    # a mapping with that path under "BACKEND".
    settings.MINIMAL_RAG_EMBEDDINGS = FAKE_EMBEDDINGS_BACKEND

    with pytest.raises(ImproperlyConfigured):
        get_embeddings()


def test_get_embeddings_with_non_mapping_options_raises_improperly_configured(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": FAKE_EMBEDDINGS_BACKEND,
        "OPTIONS": [("dimensions", 3)],
    }

    with pytest.raises(ImproperlyConfigured):
        get_embeddings()


def test_get_embeddings_with_an_unimportable_backend_raises_improperly_configured(
    settings: "Settings",
) -> None:
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.NoSuchEmbeddings",
    }

    with pytest.raises(ImproperlyConfigured):
        get_embeddings()
