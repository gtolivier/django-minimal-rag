"""The test bench's document: what a host project would hand to this package."""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass


@dataclass(frozen=True)
class SampleDocument:
    """A document to index, defined by the test bench and by no other package."""

    text: str
    source_key: str
    title: str
    url: str
    language: str | None
    permissions: AbstractSet[str]


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


@dataclass
class MutableDocument:
    """A document a host project may define as a plain dataclass: unhashable."""

    text: str
    source_key: str
    title: str
    url: str
    language: str | None
    permissions: AbstractSet[str]
