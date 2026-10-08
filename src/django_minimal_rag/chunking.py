"""Splitting text into chunks."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial

from django_minimal_rag.documents import Document

PARAGRAPH_SEPARATOR = "\n\n"
LINE_SEPARATOR = "\n"
WORD_SEPARATOR = " "
BLANK_LINES = re.compile(r"\n\s*\n")

DEFAULT_MAX_LENGTH = 1000

Length = Callable[[str], int]


def split_text(
    text: str, *, max_length: int = DEFAULT_MAX_LENGTH, length: Length = len
) -> list[str]:
    """Split ``text`` into chunks measuring at most ``max_length``.

    Sizes are measured by ``length``, which counts characters by default.
    ``max_length`` defaults to ``DEFAULT_MAX_LENGTH``.
    """
    _require_positive_max_length(max_length)
    content = _normalize_line_endings(text).strip()
    if not content:
        return []
    limit = _SizeLimit(max_length=max_length, length=length)
    return _pack_splitting_oversized(
        [_strip_line_ends(paragraph) for paragraph in BLANK_LINES.split(content)],
        separator=PARAGRAPH_SEPARATOR,
        limit=limit,
        split=partial(_split_between_lines, limit=limit),
    )


@dataclass(frozen=True)
class Chunk:
    """A piece of a document's text, with its rank in the group."""

    # The document may be unhashable; equal chunks still share text and rank.
    document: Document = field(hash=False)
    text: str
    rank: int


def chunk_group(
    documents: Sequence[Document],
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    length: Length = len,
) -> list[Chunk]:
    """Chunk the ``documents`` of a group, each split by ``split_text``."""
    _require_positive_max_length(max_length)
    pieces = [
        (document, text)
        for document in documents
        for text in split_text(document.text, max_length=max_length, length=length)
    ]
    return [
        Chunk(document=document, text=text, rank=rank)
        for rank, (document, text) in enumerate(pieces)
    ]


@dataclass(frozen=True)
class _SizeLimit:
    """The largest size a chunk may have, and how sizes are measured."""

    max_length: int
    length: Length

    def fits(self, text: str) -> bool:
        """Whether ``text`` measures at most ``max_length``."""
        return self.length(text) <= self.max_length


def _require_positive_max_length(max_length: int) -> None:
    """Raise ``ValueError`` if ``max_length`` is below 1."""
    if max_length < 1:
        msg = f"max_length must be at least 1, got {max_length}"
        raise ValueError(msg)


def _normalize_line_endings(text: str) -> str:
    """Replace every line boundary of ``text`` with ``LINE_SEPARATOR``."""
    return LINE_SEPARATOR.join(text.splitlines())


def _strip_line_ends(paragraph: str) -> str:
    """Strip the whitespace at the end of each line of ``paragraph``."""
    return LINE_SEPARATOR.join(
        line.rstrip() for line in paragraph.split(LINE_SEPARATOR)
    )


def _pack_splitting_oversized(
    pieces: list[str],
    *,
    separator: str,
    limit: _SizeLimit,
    split: Callable[[str], list[str]],
) -> list[str]:
    """Pack consecutive ``pieces`` that fit ``limit``; ``split`` the others.

    Packed pieces are joined with ``separator``, as by ``_pack``.
    """
    chunks: list[str] = []
    fitting_pieces: list[str] = []
    for piece in pieces:
        if limit.fits(piece):
            fitting_pieces.append(piece)
            continue
        chunks += _pack(fitting_pieces, separator=separator, limit=limit)
        fitting_pieces = []
        chunks += split(piece)
    return chunks + _pack(fitting_pieces, separator=separator, limit=limit)


def _split_between_lines(paragraph: str, *, limit: _SizeLimit) -> list[str]:
    """Split ``paragraph`` between lines, packed into pieces that fit ``limit``.

    A line that does not fit ``limit`` is split between words.
    """
    return _pack_splitting_oversized(
        paragraph.split(LINE_SEPARATOR),
        separator=LINE_SEPARATOR,
        limit=limit,
        split=partial(_split_between_words, limit=limit),
    )


def _split_between_words(text: str, *, limit: _SizeLimit) -> list[str]:
    """Split ``text`` between words into pieces that fit ``limit``.

    A word that does not fit ``limit`` is cut first, so it fits a piece.
    """
    words = [piece for word in text.split() for piece in _cut_word(word, limit=limit)]
    return _pack(words, separator=WORD_SEPARATOR, limit=limit)


def _cut_word(word: str, *, limit: _SizeLimit) -> list[str]:
    """Cut ``word`` into its longest prefixes that fit ``limit``, unless it fits."""
    if limit.fits(word):
        return [word]
    pieces: list[str] = []
    rest = word
    while rest:
        piece = _longest_fitting_prefix(rest, limit=limit)
        pieces.append(piece)
        rest = rest[len(piece) :]
    return pieces


def _longest_fitting_prefix(text: str, *, limit: _SizeLimit) -> str:
    """The longest prefix of ``text`` that fits ``limit``.

    The prefix is never empty, so cutting a word always moves forward: a first
    character of ``text`` that does not fit ``limit`` raises ``ValueError``.

    The search is bracketed first, so every measured prefix stays within twice
    the size of the result, however long ``text`` is.
    """
    fitting, too_long = _bracket_longest_fitting_prefix(text, limit=limit)
    while fitting + 1 < too_long:
        middle = (fitting + too_long) // 2
        if limit.fits(text[:middle]):
            fitting = middle
        else:
            too_long = middle
    return text[:fitting]


def _bracket_longest_fitting_prefix(text: str, *, limit: _SizeLimit) -> tuple[int, int]:
    """Two sizes around the longest prefix of ``text`` that fits ``limit``.

    The prefix of the first size fits; that of the second, larger one does
    not, or would go past the end of ``text``. They are found by doubling the
    size from 1, once ``_require_first_character_fits`` has checked it.
    """
    _require_first_character_fits(text, limit=limit)
    fitting = 1
    while fitting * 2 <= len(text) and limit.fits(text[: fitting * 2]):
        fitting *= 2
    return fitting, min(fitting * 2, len(text) + 1)


def _require_first_character_fits(text: str, *, limit: _SizeLimit) -> None:
    """Raise ``ValueError`` if the first character of ``text`` exceeds ``limit``."""
    character = text[:1]
    if not limit.fits(character):
        msg = (
            f"character {character!r} measures {limit.length(character)}, "
            f"more than max_length ({limit.max_length})"
        )
        raise ValueError(msg)


def _pack(pieces: list[str], *, separator: str, limit: _SizeLimit) -> list[str]:
    """Join consecutive ``pieces`` with ``separator`` while they fit ``limit``."""
    chunks = pieces[:1]
    for piece in pieces[1:]:
        candidate = chunks[-1] + separator + piece
        if limit.fits(candidate):
            chunks[-1] = candidate
        else:
            chunks.append(piece)
    return chunks
