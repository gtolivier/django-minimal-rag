from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from typing import TYPE_CHECKING

import pytest
from django.db.models import F

from django_minimal_rag.indexing import Indexer
from django_minimal_rag.models import Chunk
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


def use_fake_embeddings(settings: "Settings", dimensions: int) -> None:
    """Configure ``FakeEmbeddings`` of ``dimensions``, whose model is "fake-<n>"."""
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "django_minimal_rag.embeddings.FakeEmbeddings",
        "OPTIONS": {"dimensions": dimensions},
    }


def use_miscounting_embeddings(settings: "Settings", vector_count: int) -> None:
    """Configure ``MiscountingEmbeddings``, returning ``vector_count`` vectors."""
    settings.MINIMAL_RAG_EMBEDDINGS = {
        "BACKEND": "tests.embeddings.MiscountingEmbeddings",
        "OPTIONS": {"vector_count": vector_count},
    }


def public_page(text: str, source_key: str = "page:1") -> SampleDocument:
    """A public page of the host project, in English, holding ``text``."""
    return SampleDocument(
        text=text,
        source_key=source_key,
        title="Opening hours",
        url="https://example.com/opening-hours/",
        language="en",
        permissions=frozenset(),
    )


def index_public_pages(*texts: str) -> None:
    """Index a public page holding each of ``texts``, in order, under "page:1"."""
    Indexer().replace({"page:1": [public_page(text) for text in texts]})


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
    index_public_pages(chunk_text)

    retrieved = retrieve(question, permissions=frozenset(), max_distance=0.1)

    assert [chunk.text for chunk in retrieved] == [chunk_text]


@pytest.mark.django_db
def test_each_retrieved_chunk_carries_its_document_title_url_and_source_key(
    settings: "Settings",
) -> None:
    chunk_text = "The shop opens at nine."
    question = "When does the shop open?"
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [1.0, 0.1]})
    document = public_page(chunk_text)
    Indexer().replace({document.source_key: [document]})

    retrieved = retrieve(question, permissions=frozenset(), max_distance=0.1)

    assert [(chunk.title, chunk.url, chunk.source_key) for chunk in retrieved] == [
        (document.title, document.url, document.source_key)
    ]


@pytest.mark.django_db
def test_each_retrieved_chunk_carries_its_cosine_distance_to_the_question(
    settings: "Settings",
) -> None:
    nearer_text = "The shop opens at nine."
    farther_text = "The shop closes at six."
    question = "When does the shop open?"
    # Cosine similarities with the question: 4/5 and 3/5.
    use_chosen_embeddings(
        settings,
        {nearer_text: [4.0, 3.0], farther_text: [3.0, 4.0], question: [1.0, 0.0]},
    )
    index_public_pages(nearer_text, farther_text)

    retrieved = retrieve(question, permissions=frozenset(), max_distance=1.0)

    assert {chunk.text: chunk.distance for chunk in retrieved} == {
        nearer_text: pytest.approx(0.2),
        farther_text: pytest.approx(0.4),
    }


@pytest.mark.django_db
def test_retrieved_chunks_are_ordered_nearest_first_by_cosine_distance(
    settings: "Settings",
) -> None:
    farthest_text = "The shop is on Main Street."
    nearest_text = "The shop opens at nine."
    middle_text = "The shop opens at nine on Sundays too."
    question = "When does the shop open?"
    # Cosine distances to the question: nearest ~0.08, middle 0.2, farthest 0.4.
    # Euclidean distances order them otherwise: middle ~0.63, farthest ~4.47,
    # nearest ~12.08. The chunks are indexed in neither of those orders.
    use_chosen_embeddings(
        settings,
        {
            farthest_text: [3.0, 4.0],
            nearest_text: [12.0, 5.0],
            middle_text: [0.8, 0.6],
            question: [1.0, 0.0],
        },
    )
    index_public_pages(farthest_text, nearest_text, middle_text)

    retrieved = retrieve(question, permissions=frozenset(), max_distance=1.0)

    assert [chunk.text for chunk in retrieved] == [
        nearest_text,
        middle_text,
        farthest_text,
    ]


@pytest.mark.django_db
def test_retrieve_returns_at_most_limit_chunks_the_nearest_ones(
    settings: "Settings",
) -> None:
    farthest_text = "The shop has a car park."
    far_text = "The shop is on Main Street."
    middle_text = "The shop opens at nine on Sundays too."
    nearest_text = "The shop opens at nine."
    question = "When does the shop open?"
    # Cosine distances to the question: nearest ~0.08, middle 0.2, far 0.4,
    # farthest ~0.62. The two nearest are indexed last.
    use_chosen_embeddings(
        settings,
        {
            farthest_text: [5.0, 12.0],
            far_text: [3.0, 4.0],
            middle_text: [4.0, 3.0],
            nearest_text: [12.0, 5.0],
            question: [1.0, 0.0],
        },
    )
    index_public_pages(farthest_text, far_text, middle_text, nearest_text)

    retrieved = retrieve(question, permissions=frozenset(), max_distance=1.0, limit=2)

    assert [chunk.text for chunk in retrieved] == [nearest_text, middle_text]


@pytest.mark.django_db
def test_retrieve_without_limit_returns_at_most_the_five_nearest_chunks(
    settings: "Settings",
) -> None:
    sixth_text = "The shop sells bread."
    fifth_text = "The shop has a car park."
    fourth_text = "The shop is on Main Street."
    third_text = "The shop opens at nine on Sundays too."
    second_text = "The shop opens at nine on weekdays."
    first_text = "The shop opens at nine."
    question = "When does the shop open?"
    # Cosine distances to the question: first ~0.01, second ~0.08, third 0.2,
    # fourth 0.4, fifth ~0.62, sixth 1.0. The farthest is indexed first.
    use_chosen_embeddings(
        settings,
        {
            sixth_text: [0.0, 1.0],
            fifth_text: [5.0, 12.0],
            fourth_text: [3.0, 4.0],
            third_text: [4.0, 3.0],
            second_text: [12.0, 5.0],
            first_text: [7.0, 1.0],
            question: [1.0, 0.0],
        },
    )
    index_public_pages(
        sixth_text, fifth_text, fourth_text, third_text, second_text, first_text
    )

    retrieved = retrieve(question, permissions=frozenset(), max_distance=2.0)

    assert [chunk.text for chunk in retrieved] == [
        first_text,
        second_text,
        third_text,
        fourth_text,
        fifth_text,
    ]


@pytest.mark.django_db
def test_chunks_at_the_same_distance_are_ordered_by_document_id_then_rank(
    settings: "Settings",
) -> None:
    # Two paragraphs too long to share a chunk: the first document is split in
    # two chunks, of ranks 0 and 1.
    first_opening = " ".join(["The shop opens at nine."] * 30)
    first_closing = " ".join(["The shop closes at six."] * 30)
    second_text = "The shop is on Main Street."
    third_text = "The shop has a car park."
    question = "When does the shop open?"
    # Every chunk is at a cosine distance of 0.0 from the question.
    use_chosen_embeddings(
        settings,
        {
            text: [1.0, 0.0]
            for text in (
                first_opening,
                first_closing,
                second_text,
                third_text,
                question,
            )
        },
    )
    # The ranks are those of a source's chunks: the second document's chunk
    # has rank 2, the third's rank 0. Ordering by rank alone would put the
    # third document's chunk second.
    Indexer().replace(
        {
            "page:1": [
                public_page(f"{first_opening}\n\n{first_closing}"),
                public_page(second_text),
            ],
            "page:2": [public_page(third_text, "page:2")],
        }
    )
    # Rewriting the rows from the last written to the first stores a new
    # version of each after the others: PostgreSQL then reads them in the
    # reverse of the order they were indexed in.
    for chunk_pk in Chunk.objects.order_by("-pk").values_list("pk", flat=True):
        Chunk.objects.filter(pk=chunk_pk).update(rank=F("rank"))

    retrieved = retrieve(question, permissions=frozenset(), max_distance=1.0)

    assert [chunk.text for chunk in retrieved] == [
        first_opening,
        first_closing,
        second_text,
        third_text,
    ]


@pytest.mark.django_db
def test_a_retrieved_chunk_cannot_be_modified(settings: "Settings") -> None:
    chunk_text = "The shop opens at nine."
    question = "When does the shop open?"
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [1.0, 0.0]})
    index_public_pages(chunk_text)
    (chunk,) = retrieve(question, permissions=frozenset(), max_distance=1.0)

    with pytest.raises(FrozenInstanceError):
        # The assignment is the point of the test: mypy rightly rejects it.
        chunk.text = "The shop opens at ten."  # type: ignore[misc]


@pytest.mark.django_db
def test_retrieve_leaves_out_a_chunk_farther_from_the_question_than_max_distance(
    settings: "Settings",
) -> None:
    farther_text = "The shop is on Main Street."
    nearer_text = "The shop opens at nine."
    question = "When does the shop open?"
    # Cosine distances to the question: nearer 0.2, farther 0.4.
    use_chosen_embeddings(
        settings,
        {farther_text: [3.0, 4.0], nearer_text: [4.0, 3.0], question: [1.0, 0.0]},
    )
    index_public_pages(farther_text, nearer_text)

    retrieved = retrieve(question, permissions=frozenset(), max_distance=0.3)

    assert [chunk.text for chunk in retrieved] == [nearer_text]


@pytest.mark.django_db
def test_retrieve_returns_a_chunk_exactly_at_max_distance_from_the_question(
    settings: "Settings",
) -> None:
    chunk_text = "The shop is on Main Street."
    question = "When does the shop open?"
    # The chunk is orthogonal to the question: cosine distance exactly 1.0.
    use_chosen_embeddings(settings, {chunk_text: [0.0, 1.0], question: [1.0, 0.0]})
    index_public_pages(chunk_text)

    retrieved = retrieve(question, permissions=frozenset(), max_distance=1.0)

    assert [chunk.text for chunk in retrieved] == [chunk_text]


@pytest.mark.django_db
def test_retrieve_ignores_chunks_embedded_by_another_model_even_of_another_dimension(
    settings: "Settings",
) -> None:
    same_dimension_text = "The shop opened at eight last year."
    other_dimension_text = "The shop opened at seven long ago."
    current_text = "The shop opens at nine."
    question = "When does the shop open?"
    # Left by previous models: "fake-2" gives 2 dimensions, like the current
    # model, and "fake-3" gives 3. Fake vectors have no negative component, so
    # the 2-dimension one is within a cosine distance of 1.0 of the question.
    use_fake_embeddings(settings, dimensions=2)
    Indexer().replace({"page:1": [public_page(same_dimension_text, "page:1")]})
    use_fake_embeddings(settings, dimensions=3)
    Indexer().replace({"page:2": [public_page(other_dimension_text, "page:2")]})
    use_chosen_embeddings(settings, {current_text: [1.0, 0.0], question: [1.0, 0.0]})
    Indexer().replace({"page:3": [public_page(current_text, "page:3")]})

    retrieved = retrieve(question, permissions=frozenset(), max_distance=2.0)

    assert [chunk.text for chunk in retrieved] == [current_text]


@pytest.mark.django_db
def test_retrieve_leaves_out_a_chunk_requiring_a_permission_the_reader_lacks(
    settings: "Settings",
) -> None:
    secret_text = "The shop opens at eight for staff."
    public_text = "The shop opens at nine."
    question = "When does the shop open?"
    # Cosine distances to the question: secret 0.0, public 0.2. With a limit
    # of 1, the public chunk is only returned if the secret one is left out by
    # the query itself, before the limit applies.
    use_chosen_embeddings(
        settings,
        {secret_text: [1.0, 0.0], public_text: [4.0, 3.0], question: [1.0, 0.0]},
    )
    secret_page = replace(
        public_page(secret_text), permissions=frozenset({"app.view_secret"})
    )
    Indexer().replace({"page:1": [secret_page, public_page(public_text)]})

    retrieved = retrieve(
        question,
        permissions=frozenset({"app.view_other"}),
        max_distance=1.0,
        limit=1,
    )

    assert [chunk.text for chunk in retrieved] == [public_text]


@pytest.mark.django_db
def test_retrieve_returns_a_chunk_only_to_a_reader_holding_all_its_permissions(
    settings: "Settings",
) -> None:
    chunk_text = "The shop opens at eight for managers."
    question = "When does the shop open?"
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [1.0, 0.0]})
    page = replace(
        public_page(chunk_text),
        permissions=frozenset({"app.view_a", "app.view_b"}),
    )
    Indexer().replace({"page:1": [page]})

    retrieved_by_full_reader = retrieve(
        question,
        permissions=frozenset({"app.view_a", "app.view_b", "app.view_c"}),
        max_distance=1.0,
    )
    retrieved_by_partial_reader = retrieve(
        question,
        permissions=frozenset({"app.view_a", "app.view_c"}),
        max_distance=1.0,
    )

    assert [chunk.text for chunk in retrieved_by_full_reader] == [chunk_text]
    assert retrieved_by_partial_reader == []


@pytest.mark.django_db
def test_retrieve_with_a_limit_below_one_raises_value_error() -> None:
    with pytest.raises(ValueError, match="limit"):
        retrieve(
            "What are the opening hours?",
            permissions=frozenset(),
            max_distance=1.0,
            limit=0,
        )


@pytest.mark.django_db
def test_retrieve_raises_value_error_when_the_backend_gives_the_question_two_vectors(
    settings: "Settings",
) -> None:
    use_miscounting_embeddings(settings, vector_count=2)

    # The word "vector" and the count 2, in any order and wording: not the
    # incidental error of unpacking the backend's result.
    with pytest.raises(ValueError, match=r"(?is)^(?=.*\bvectors?\b)(?=.*\b2\b)"):
        retrieve(
            "What are the opening hours?", permissions=frozenset(), max_distance=1.0
        )


@pytest.mark.django_db
def test_retrieve_raises_value_error_when_the_backend_gives_the_question_a_zero_vector(
    settings: "Settings",
) -> None:
    chunk_text = "The shop opens at nine."
    question = "When does the shop open?"
    # A vector of norm 0 has no direction: its cosine distance to the chunk
    # cannot be computed, which must not pass for "no relevant chunk".
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [0.0, 0.0]})
    index_public_pages(chunk_text)

    # The word "zero", in any wording: the error is about the question's vector.
    with pytest.raises(ValueError, match=r"(?i)\bzeros?\b"):
        retrieve(question, permissions=frozenset(), max_distance=2.0)
