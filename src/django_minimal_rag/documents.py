"""The document protocol: what a host project hands to this package."""

from collections.abc import Set as AbstractSet
from typing import Protocol


class Document(Protocol):
    """A document to index, whatever produced it.

    Every member is required of every producer. Each is read-only, so a
    dataclass field and a property both satisfy it.
    """

    @property
    def text(self) -> str:
        """The text to index."""

    @property
    def source_key(self) -> str:
        """The group this document belongs to, not the document itself.

        `replace()` replaces a group's documents together. `prune()` reads
        the part before the colon as its `model_label`, matched whole:
        `app.note` never matches `app.notebook:3`.
        """

    @property
    def title(self) -> str:
        """The document's title."""

    @property
    def url(self) -> str:
        """The URL of the document's source page.

        Whether it may be empty is not settled yet.
        """

    @property
    def language(self) -> str | None:
        """The language of the text.

        What `None` means is not settled yet.
        """

    @property
    def permissions(self) -> AbstractSet[str]:
        """Permission names, as `app_label.codename`.

        What they require of a reader is not settled yet.
        """
