from collections.abc import Set as AbstractSet
from dataclasses import dataclass

import pytest

from django_minimal_rag import Document
from tests.documents import SampleDocument

# The members of one FAQ entry, shared by every document built in this module.
TEXT = "Opening hours are 9am to 5pm."
SOURCE_KEY = "faq:opening-hours"
TITLE = "Opening hours"
URL = "https://example.test/faq/opening-hours"
LANGUAGE = "en"
PERMISSIONS = frozenset({"public"})


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
        return TEXT

    @property
    def source_key(self) -> str:
        return SOURCE_KEY

    @property
    def title(self) -> str:
        return TITLE

    @property
    def url(self) -> str:
        return URL

    @property
    def language(self) -> str | None:
        return LANGUAGE

    @property
    def permissions(self) -> AbstractSet[str]:
        return PERMISSIONS


@pytest.fixture
def sample() -> SampleDocument:
    """The test bench's document, holding the FAQ entry of this module."""
    return SampleDocument(
        text=TEXT,
        source_key=SOURCE_KEY,
        title=TITLE,
        url=URL,
        language=LANGUAGE,
        permissions=PERMISSIONS,
    )


def test_test_bench_document_is_accepted_where_a_document_is_expected(
    sample: SampleDocument,
) -> None:
    document: Document = sample

    assert document is sample


def test_class_with_read_only_members_is_a_document() -> None:
    read_only = ReadOnlyDocument()

    document: Document = read_only

    assert document is read_only


def test_class_without_text_is_not_a_document() -> None:
    without_text = DocumentWithoutText(
        source_key=SOURCE_KEY,
        title=TITLE,
        url=URL,
        language=LANGUAGE,
        permissions=PERMISSIONS,
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_text  # type: ignore[assignment]

    assert not hasattr(document, "text")


def test_class_without_source_key_is_not_a_document() -> None:
    without_source_key = DocumentWithoutSourceKey(
        text=TEXT,
        title=TITLE,
        url=URL,
        language=LANGUAGE,
        permissions=PERMISSIONS,
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_source_key  # type: ignore[assignment]

    assert not hasattr(document, "source_key")


def test_class_without_title_is_not_a_document() -> None:
    without_title = DocumentWithoutTitle(
        text=TEXT,
        source_key=SOURCE_KEY,
        url=URL,
        language=LANGUAGE,
        permissions=PERMISSIONS,
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_title  # type: ignore[assignment]

    assert not hasattr(document, "title")


def test_class_without_url_is_not_a_document() -> None:
    without_url = DocumentWithoutUrl(
        text=TEXT,
        source_key=SOURCE_KEY,
        title=TITLE,
        language=LANGUAGE,
        permissions=PERMISSIONS,
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_url  # type: ignore[assignment]

    assert not hasattr(document, "url")


def test_class_without_language_is_not_a_document() -> None:
    without_language = DocumentWithoutLanguage(
        text=TEXT,
        source_key=SOURCE_KEY,
        title=TITLE,
        url=URL,
        permissions=PERMISSIONS,
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_language  # type: ignore[assignment]

    assert not hasattr(document, "language")


def test_class_without_permissions_is_not_a_document() -> None:
    without_permissions = DocumentWithoutPermissions(
        text=TEXT,
        source_key=SOURCE_KEY,
        title=TITLE,
        url=URL,
        language=LANGUAGE,
    )

    # The rejection is the behavior under test: mypy must refuse this
    # assignment, and strict mypy reports the ignore as unused if it does not.
    document: Document = without_permissions  # type: ignore[assignment]

    assert not hasattr(document, "permissions")


def test_members_read_through_a_document_have_their_exact_types(
    sample: SampleDocument,
) -> None:
    document: Document = sample

    text: str = document.text
    source_key: str = document.source_key
    title: str = document.title
    url: str = document.url
    language: str | None = document.language
    permissions: AbstractSet[str] = document.permissions

    assert (text, source_key, title, url, language, permissions) == (
        sample.text,
        sample.source_key,
        sample.title,
        sample.url,
        sample.language,
        sample.permissions,
    )
