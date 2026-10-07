from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import assert_type

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


# A class missing any one member is not a document. That is a static
# behavior, so it is checked by mypy alone: the functions below are never
# called, and pytest does not collect them (their names do not start with
# `test_`). Each returns its argument where a `Document` is expected; mypy
# must refuse that return, and strict mypy reports the ignore as unused if
# it does not, which fails the type check.


def _rejects_without_text(document: DocumentWithoutText) -> Document:
    return document  # type: ignore[return-value]  # no `text`: not a Document


def _rejects_without_source_key(document: DocumentWithoutSourceKey) -> Document:
    return document  # type: ignore[return-value]  # no `source_key`: not a Document


def _rejects_without_title(document: DocumentWithoutTitle) -> Document:
    return document  # type: ignore[return-value]  # no `title`: not a Document


def _rejects_without_url(document: DocumentWithoutUrl) -> Document:
    return document  # type: ignore[return-value]  # no `url`: not a Document


def _rejects_without_language(document: DocumentWithoutLanguage) -> Document:
    return document  # type: ignore[return-value]  # no `language`: not a Document


def _rejects_without_permissions(document: DocumentWithoutPermissions) -> Document:
    return document  # type: ignore[return-value]  # no `permissions`: not a Document


def test_members_read_through_a_document_have_their_exact_types(
    sample: SampleDocument,
) -> None:
    document: Document = sample

    # assert_type pins each type exactly: mypy fails if a member is loosened
    # (to Any or object, for instance), which an annotated assignment accepts.
    text = assert_type(document.text, str)
    source_key = assert_type(document.source_key, str)
    title = assert_type(document.title, str)
    url = assert_type(document.url, str)
    language = assert_type(document.language, str | None)
    permissions = assert_type(document.permissions, AbstractSet[str])

    assert (text, source_key, title, url, language, permissions) == (
        sample.text,
        sample.source_key,
        sample.title,
        sample.url,
        sample.language,
        sample.permissions,
    )
