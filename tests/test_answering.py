from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_minimal_rag.answering import ANSWER_INSTRUCTIONS, Citation, answer
from django_minimal_rag.indexing import Indexer
from tests.documents import SampleDocument, public_page
from tests.embeddings import use_chosen_embeddings

if TYPE_CHECKING:
    from pytest_django import Settings


pytestmark = pytest.mark.usefixtures("fake_embeddings")


@pytest.fixture
def llm_calls(settings: "Settings") -> list[tuple[str, str]]:
    """Configure ``RecordingLLM`` and return the ``(prompt, system)`` it records."""
    calls: list[tuple[str, str]] = []
    settings.MINIMAL_RAG_LLM = {
        "BACKEND": "tests.llms.RecordingLLM",
        "OPTIONS": {"calls": calls},
    }
    return calls


def use_canned_llm(settings: "Settings", reply: str) -> None:
    """Configure ``CannedLLM``, answering every prompt with ``reply``."""
    settings.MINIMAL_RAG_LLM = {
        "BACKEND": "tests.llms.CannedLLM",
        "OPTIONS": {"reply": reply},
    }


def index_three_chunks_near(
    settings: "Settings", question: str
) -> tuple[str, str, str]:
    """Index three chunks near ``question``, each in its own public page.

    Return their texts nearest first. The cosine distances to the question
    are about 0.005, 0.106 and 0.293; the chunks are indexed in an order
    other than their distance order.
    """
    nearest = "The shop opens at nine on weekdays."
    middle = "On Saturdays the shop opens at ten."
    farthest = "The shop is closed on Sundays."
    use_chosen_embeddings(
        settings,
        {
            question: [1.0, 0.0],
            nearest: [1.0, 0.1],
            middle: [1.0, 0.5],
            farthest: [1.0, 1.0],
        },
    )
    Indexer().replace(
        {
            "page:1": [public_page(middle, source_key="page:1")],
            "page:2": [public_page(farthest, source_key="page:2")],
            "page:3": [public_page(nearest, source_key="page:3")],
        }
    )
    return nearest, middle, farthest


def page_cited_as(citation: Citation, text: str) -> SampleDocument:
    """A public page holding ``text``, from the document ``citation`` names."""
    return replace(
        public_page(text, citation.source_key), title=citation.title, url=citation.url
    )


@pytest.mark.django_db
def test_answer_with_nothing_indexed_returns_none_without_calling_the_llm(
    llm_calls: list[tuple[str, str]],
) -> None:
    answered = answer(
        "What are the opening hours?", permissions=frozenset(), max_distance=2.0
    )

    assert answered is None
    assert llm_calls == []


@pytest.mark.django_db
def test_answer_without_the_llm_setting_raises_even_with_nothing_indexed(
    settings: "Settings",
) -> None:
    # Set first so the deletion works whether or not the test settings
    # define it: the setting is then missing for answer.
    settings.MINIMAL_RAG_LLM = {"BACKEND": "tests.llms.CannedLLM"}
    del settings.MINIMAL_RAG_LLM

    with pytest.raises(ImproperlyConfigured, match="MINIMAL_RAG_LLM"):
        answer("What are the opening hours?", permissions=frozenset(), max_distance=2.0)


@pytest.mark.django_db
def test_answer_with_only_a_chunk_beyond_max_distance_returns_none_without_llm(
    settings: "Settings", llm_calls: list[tuple[str, str]]
) -> None:
    chunk_text = "The shop opens at nine on weekdays."
    question = "Where is the shop?"
    # Cosine distance of the chunk to the question: 1.0, the vectors are orthogonal.
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [0.0, 1.0]})
    Indexer().replace({"page:1": [public_page(chunk_text)]})

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


@pytest.mark.django_db
@pytest.mark.parametrize(
    "question", ["When does the shop open?", "Is the shop open on weekdays?"]
)
def test_answer_gives_the_llm_a_prompt_holding_the_question(
    settings: "Settings", llm_calls: list[tuple[str, str]], question: str
) -> None:
    chunk_text = "The shop opens at nine on weekdays."
    # Cosine distance of the chunk to the question: about 0.005.
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [1.0, 0.1]})
    Indexer().replace({"page:1": [public_page(chunk_text)]})

    answer(question, permissions=frozenset(), max_distance=0.1)

    [(prompt, _system)] = llm_calls
    assert question in prompt


@pytest.mark.django_db
@pytest.mark.parametrize(
    "question", ["When does the shop open?", "Is the shop open on weekdays?"]
)
def test_answer_gives_the_llm_a_prompt_ending_with_the_question_element(
    settings: "Settings", llm_calls: list[tuple[str, str]], question: str
) -> None:
    chunk_text = "The shop opens at nine on weekdays."
    # Cosine distance of the chunk to the question: about 0.005.
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [1.0, 0.1]})
    Indexer().replace({"page:1": [public_page(chunk_text)]})

    answer(question, permissions=frozenset(), max_distance=0.1)

    [(prompt, _system)] = llm_calls
    assert prompt.endswith(f"<question>{question}</question>")


@pytest.mark.django_db
def test_answer_gives_the_llm_every_retrieved_chunk_nearest_first(
    settings: "Settings", llm_calls: list[tuple[str, str]]
) -> None:
    question = "When does the shop open?"
    nearest, middle, farthest = index_three_chunks_near(settings, question)

    answer(question, permissions=frozenset(), max_distance=0.5)

    [(prompt, _system)] = llm_calls
    assert all(text in prompt for text in (nearest, middle, farthest))
    assert prompt.index(nearest) < prompt.index(middle) < prompt.index(farthest)


@pytest.mark.django_db
def test_answer_gives_the_llm_only_the_limit_nearest_chunks(
    settings: "Settings", llm_calls: list[tuple[str, str]]
) -> None:
    question = "When does the shop open?"
    # All three within max_distance.
    nearest, middle, farthest = index_three_chunks_near(settings, question)

    answer(question, permissions=frozenset(), max_distance=0.5, limit=2)

    [(prompt, _system)] = llm_calls
    assert nearest in prompt
    assert middle in prompt
    assert farthest not in prompt


@pytest.mark.django_db
def test_answer_gives_the_llm_each_chunk_under_its_document_title(
    settings: "Settings", llm_calls: list[tuple[str, str]]
) -> None:
    question = "When does the shop open?"
    weekday_text = "The shop opens at nine from Monday to Friday."
    weekend_text = "On Saturdays the shop opens at ten."
    weekday_title = "Weekday schedule"
    weekend_title = "Saturday schedule"
    # Cosine distances to the question: about 0.005 and 0.106.
    use_chosen_embeddings(
        settings,
        {question: [1.0, 0.0], weekday_text: [1.0, 0.1], weekend_text: [1.0, 0.5]},
    )
    Indexer().replace(
        {
            "page:1": [
                replace(public_page(weekday_text, "page:1"), title=weekday_title)
            ],
            "page:2": [
                replace(public_page(weekend_text, "page:2"), title=weekend_title)
            ],
        }
    )

    answer(question, permissions=frozenset(), max_distance=0.5)

    [(prompt, _system)] = llm_calls
    assert all(title in prompt for title in (weekday_title, weekend_title))
    # Each title comes before its own chunk, and after the previous chunk.
    assert (
        prompt.index(weekday_title)
        < prompt.index(weekday_text)
        < prompt.index(weekend_title)
        < prompt.index(weekend_text)
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("title", "chunk_text"),
    [
        ("Weekday schedule", "The shop opens at nine on weekdays."),
        ("Shop FAQ", "On Saturdays the shop opens at ten."),
    ],
)
def test_answer_gives_the_llm_the_document_title_and_extract_in_elements(
    settings: "Settings",
    llm_calls: list[tuple[str, str]],
    title: str,
    chunk_text: str,
) -> None:
    question = "When does the shop open?"
    # Cosine distance of the chunk to the question: about 0.005.
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [1.0, 0.1]})
    Indexer().replace({"page:1": [replace(public_page(chunk_text), title=title)]})

    answer(question, permissions=frozenset(), max_distance=0.1)

    [(prompt, _system)] = llm_calls
    assert prompt == "\n".join(
        [
            "<document>",
            f"<title>{title}</title>",
            f"<extract>{chunk_text}</extract>",
            "</document>",
            f"<question>{question}</question>",
        ]
    )


@pytest.mark.django_db
def test_answer_gives_the_llm_the_chunks_of_one_document_in_one_document_element(
    settings: "Settings", llm_calls: list[tuple[str, str]]
) -> None:
    question = "When does the shop open?"
    weekday_text = "The shop opens at nine from Monday to Friday."
    saturday_text = "On Saturdays the shop opens at ten."
    sunday_text = "The shop is closed on Sundays."
    # Cosine distances to the question: weekday 0.005, saturday 0.106 and
    # sunday 0.293. Weekday and sunday are chunks of the same document, so
    # its chunks come before and after the other document's.
    use_chosen_embeddings(
        settings,
        {
            question: [1.0, 0.0],
            weekday_text: [1.0, 0.1],
            saturday_text: [1.0, 0.5],
            sunday_text: [1.0, 1.0],
        },
    )
    week = Citation(
        title="Week schedule",
        url="https://example.com/hours/week/",
        source_key="page:1",
    )
    saturday = Citation(
        title="Saturday schedule",
        url="https://example.com/hours/saturday/",
        source_key="page:2",
    )
    week_pages = [page_cited_as(week, text) for text in (weekday_text, sunday_text)]
    saturday_page = page_cited_as(saturday, saturday_text)
    Indexer().replace(
        {week.source_key: week_pages, saturday.source_key: [saturday_page]}
    )

    answer(question, permissions=frozenset(), max_distance=0.5)

    [(prompt, _system)] = llm_calls
    assert prompt == "\n".join(
        [
            "<document>",
            f"<title>{week.title}</title>",
            f"<extract>{weekday_text}</extract>",
            f"<extract>{sunday_text}</extract>",
            "</document>",
            "<document>",
            f"<title>{saturday.title}</title>",
            f"<extract>{saturday_text}</extract>",
            "</document>",
            f"<question>{question}</question>",
        ]
    )


@pytest.mark.django_db
def test_answer_gives_the_llm_the_answer_instructions_as_system_prompt(
    settings: "Settings", llm_calls: list[tuple[str, str]]
) -> None:
    chunk_text = "The shop opens at nine on weekdays."
    question = "When does the shop open?"
    # Cosine distance of the chunk to the question: about 0.005.
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [1.0, 0.1]})
    Indexer().replace({"page:1": [public_page(chunk_text)]})

    answer(question, permissions=frozenset(), max_distance=0.1)

    [(_prompt, system)] = llm_calls
    assert system == ANSWER_INSTRUCTIONS


@pytest.mark.django_db
def test_answer_never_gives_the_llm_a_chunk_the_reader_may_not_read(
    settings: "Settings", llm_calls: list[tuple[str, str]]
) -> None:
    secret_text = "The shop opens at eight for staff."
    public_text = "The shop opens at nine."
    question = "When does the shop open?"
    # Cosine distances to the question: secret 0.0, public 0.2.
    use_chosen_embeddings(
        settings,
        {secret_text: [1.0, 0.0], public_text: [4.0, 3.0], question: [1.0, 0.0]},
    )
    secret_page = replace(
        public_page(secret_text), permissions=frozenset({"app.view_secret"})
    )
    Indexer().replace({"page:1": [secret_page, public_page(public_text)]})

    answer(question, permissions=frozenset({"app.view_other"}), max_distance=1.0)

    [(prompt, _system)] = llm_calls
    assert public_text in prompt
    assert secret_text not in prompt


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("title", "url", "source_key"),
    [
        ("Weekday schedule", "https://example.com/hours/weekdays/", "page:12"),
        ("Shop FAQ", "https://shop.example.org/faq/#hours", "faq:7"),
    ],
)
def test_answer_cites_the_document_of_its_retrieved_chunk(
    settings: "Settings", title: str, url: str, source_key: str
) -> None:
    chunk_text = "The shop opens at nine on weekdays."
    question = "When does the shop open?"
    # Cosine distance of the chunk to the question: about 0.005.
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [1.0, 0.1]})
    page = replace(public_page(chunk_text, source_key), title=title, url=url)
    Indexer().replace({source_key: [page]})
    use_canned_llm(settings, "The shop opens at nine.")

    answered = answer(question, permissions=frozenset(), max_distance=0.1)

    assert answered is not None
    assert answered.citations == (
        Citation(title=title, url=url, source_key=source_key),
    )


@pytest.mark.django_db
def test_answer_cites_each_document_once_in_the_order_of_its_nearest_chunk(
    settings: "Settings",
) -> None:
    question = "When does the shop open?"
    weekday_text = "The shop opens at nine from Monday to Friday."
    saturday_text = "On Saturdays the shop opens at ten."
    sunday_text = "The shop is closed on Sundays."
    # Cosine distances to the question: weekday 0.005, saturday 0.106 and
    # sunday 0.293. Weekday and sunday are chunks of the same document, so
    # its chunks come before and after the other document's.
    use_chosen_embeddings(
        settings,
        {
            question: [1.0, 0.0],
            weekday_text: [1.0, 0.1],
            saturday_text: [1.0, 0.5],
            sunday_text: [1.0, 1.0],
        },
    )
    week = Citation(
        title="Week schedule",
        url="https://example.com/hours/week/",
        source_key="page:1",
    )
    saturday = Citation(
        title="Saturday schedule",
        url="https://example.com/hours/saturday/",
        source_key="page:2",
    )
    week_pages = [page_cited_as(week, text) for text in (weekday_text, sunday_text)]
    saturday_page = page_cited_as(saturday, saturday_text)
    Indexer().replace(
        {week.source_key: week_pages, saturday.source_key: [saturday_page]}
    )
    use_canned_llm(settings, "The shop opens at nine, at ten on Saturdays.")

    answered = answer(question, permissions=frozenset(), max_distance=0.5)

    assert answered is not None
    assert answered.citations == (week, saturday)


@pytest.mark.django_db
def test_answer_cites_each_document_of_one_source_with_its_own_title_and_url(
    settings: "Settings",
) -> None:
    question = "When does the shop open?"
    english_text = "The shop opens at nine on weekdays."
    french_text = "La boutique ouvre à neuf heures en semaine."
    # Cosine distances to the question: french 0.005, english 0.106. The
    # french document is indexed second but cited first.
    use_chosen_embeddings(
        settings,
        {question: [1.0, 0.0], french_text: [1.0, 0.1], english_text: [1.0, 0.5]},
    )
    source_key = "app.note:3"
    english = Citation(
        title="Opening hours",
        url="https://example.com/en/opening-hours/",
        source_key=source_key,
    )
    french = Citation(
        title="Horaires d'ouverture",
        url="https://example.com/fr/horaires/",
        source_key=source_key,
    )
    english_page = page_cited_as(english, english_text)
    french_page = replace(page_cited_as(french, french_text), language="fr")
    Indexer().replace({source_key: [english_page, french_page]})
    use_canned_llm(settings, "The shop opens at nine.")

    answered = answer(question, permissions=frozenset(), max_distance=0.5)

    assert answered is not None
    assert answered.citations == (french, english)


@pytest.mark.django_db
def test_answer_gives_its_citations_as_a_tuple(settings: "Settings") -> None:
    chunk_text = "The shop opens at nine on weekdays."
    question = "When does the shop open?"
    # Cosine distance of the chunk to the question: about 0.005.
    use_chosen_embeddings(settings, {chunk_text: [1.0, 0.0], question: [1.0, 0.1]})
    Indexer().replace({"page:1": [public_page(chunk_text)]})
    use_canned_llm(settings, "The shop opens at nine.")

    answered = answer(question, permissions=frozenset(), max_distance=0.1)

    assert answered is not None
    assert isinstance(answered.citations, tuple)
