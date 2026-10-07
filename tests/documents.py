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
