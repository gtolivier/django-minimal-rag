"""The document protocol: what a host project hands to this package."""

from typing import Protocol


class Document(Protocol):
    """A document to index."""
