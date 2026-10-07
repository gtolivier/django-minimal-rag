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


@dataclass(frozen=True)
class DocumentWithoutSourceKey:
    """Every member of a document except `source_key`."""

    text: str
    title: str
    url: str
    language: str | None
    permissions: AbstractSet[str]


@dataclass(frozen=True)
class DocumentWithoutTitle:
    """Every member of a document except `title`."""

    text: str
    source_key: str
    url: str
    language: str | None
    permissions: AbstractSet[str]


@dataclass(frozen=True)
class DocumentWithoutUrl:
    """Every member of a document except `url`."""

    text: str
    source_key: str
    title: str
    language: str | None
    permissions: AbstractSet[str]


@dataclass(frozen=True)
class DocumentWithoutLanguage:
    """Every member of a document except `language`."""

    text: str
    source_key: str
    title: str
    url: str
    permissions: AbstractSet[str]


@dataclass(frozen=True)
class DocumentWithoutPermissions:
    """Every member of a document except `permissions`."""

    text: str
    source_key: str
    title: str
    url: str
    language: str | None


class ReadOnlyDocument:
    """Every member of a document, as a property without a setter."""

    @property
    def text(self) -> str:
        return "Opening hours are 9am to 5pm."

    @property
    def source_key(self) -> str:
        return "faq:opening-hours"

    @property
    def title(self) -> str:
        return "Opening hours"

    @property
    def url(self) -> str:
        return "https://example.test/faq/opening-hours"

    @property
    def language(self) -> str | None:
        return "en"

    @property
    def permissions(self) -> AbstractSet[str]:
        return frozenset({"public"})


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


def test_class_with_read_only_members_is_a_document() -> None:
    read_only = ReadOnlyDocument()

    document: Document = read_only

    assert document is read_only


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


def test_class_without_source_key_is_not_a_document() -> None:
    without_source_key = DocumentWithoutSourceKey(
        text="Opening hours are 9am to 5pm.",
        title="Opening hours",
        url="https://example.test/faq/opening-hours",
        language="en",
        permissions=frozenset({"public"}),
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_source_key  # type: ignore[assignment]

    assert not hasattr(document, "source_key")


def test_class_without_title_is_not_a_document() -> None:
    without_title = DocumentWithoutTitle(
        text="Opening hours are 9am to 5pm.",
        source_key="faq:opening-hours",
        url="https://example.test/faq/opening-hours",
        language="en",
        permissions=frozenset({"public"}),
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_title  # type: ignore[assignment]

    assert not hasattr(document, "title")


def test_class_without_url_is_not_a_document() -> None:
    without_url = DocumentWithoutUrl(
        text="Opening hours are 9am to 5pm.",
        source_key="faq:opening-hours",
        title="Opening hours",
        language="en",
        permissions=frozenset({"public"}),
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_url  # type: ignore[assignment]

    assert not hasattr(document, "url")


def test_class_without_language_is_not_a_document() -> None:
    without_language = DocumentWithoutLanguage(
        text="Opening hours are 9am to 5pm.",
        source_key="faq:opening-hours",
        title="Opening hours",
        url="https://example.test/faq/opening-hours",
        permissions=frozenset({"public"}),
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_language  # type: ignore[assignment]

    assert not hasattr(document, "language")


def test_class_without_permissions_is_not_a_document() -> None:
    without_permissions = DocumentWithoutPermissions(
        text="Opening hours are 9am to 5pm.",
        source_key="faq:opening-hours",
        title="Opening hours",
        url="https://example.test/faq/opening-hours",
        language="en",
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_permissions  # type: ignore[assignment]

    assert not hasattr(document, "permissions")
