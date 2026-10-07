"""The document protocol: what a host project hands to this package."""

from typing import Protocol


class Document(Protocol):
    """A document to index."""

    @property
    def text(self) -> str: ...

    @property
    def source_key(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def url(self) -> str: ...

    @property
    def language(self) -> str | None: ...
