from django_minimal_rag import Document
from tests.documents import SampleDocument


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
