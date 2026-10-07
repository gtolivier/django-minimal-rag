from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from django_minimal_rag import Document
from tests.documents import SampleDocument


@dataclass(frozen=True)
class DocumentWithoutText:
    """Every member of a document except `text`."""

    source_key: str
    title: str
    url: str
    language: str | None
    permissions: AbstractSet[str]


def test_test_bench_document_is_accepted_where_a_document_is_expected() -> None:
    sample = SampleDocument(
        text="Opening hours are 9am to 5pm.",
        source_key="faq:opening-hours",
        title="Opening hours",
        url="https://example.test/faq/opening-hours",
        language="en",
        permissions=frozenset({"public"}),
    )

    document: Document = sample

    assert document is sample


def test_class_without_text_is_not_a_document() -> None:
    without_text = DocumentWithoutText(
        source_key="faq:opening-hours",
        title="Opening hours",
        url="https://example.test/faq/opening-hours",
        language="en",
        permissions=frozenset({"public"}),
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_text  # type: ignore[assignment]

    assert not hasattr(document, "text")
