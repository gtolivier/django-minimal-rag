"""Splitting text into chunks."""

import re
from collections.abc import Callable
from dataclasses import dataclass

PARAGRAPH_SEPARATOR = "\n\n"
WORD_SEPARATOR = " "
BLANK_LINES = re.compile(r"\n\s*\n")

Length = Callable[[str], int]


def split_text(text: str, *, max_length: int, length: Length = len) -> list[str]:
    """Split ``text`` into chunks measuring at most ``max_length``.

    Sizes are measured by ``length``, which counts characters by default.
    """
    content = text.strip()
    if not content:
        return []
    limit = _SizeLimit(max_length=max_length, length=length)
    chunks: list[str] = []
    short_paragraphs: list[str] = []
    for paragraph in BLANK_LINES.split(content):
        if limit.fits(paragraph):
            short_paragraphs.append(paragraph)
            continue
        chunks += _pack_paragraphs(short_paragraphs, limit=limit)
        short_paragraphs = []
        chunks += _split_between_words(paragraph, limit=limit)
    return chunks + _pack_paragraphs(short_paragraphs, limit=limit)


@dataclass(frozen=True)
class _SizeLimit:
    """The largest size a chunk may have, and how sizes are measured."""

    max_length: int
    length: Length

    def fits(self, text: str) -> bool:
        """Whether ``text`` measures at most ``max_length``."""
        return self.length(text) <= self.max_length


def _pack_paragraphs(paragraphs: list[str], *, limit: _SizeLimit) -> list[str]:
    """Join consecutive ``paragraphs`` by blank lines while they fit ``limit``."""
    return _pack(paragraphs, separator=PARAGRAPH_SEPARATOR, limit=limit)


def _split_between_words(paragraph: str, *, limit: _SizeLimit) -> list[str]:
    """Split ``paragraph`` between words into pieces that fit ``limit``.

    A word that does not fit ``limit`` is cut first, so it fits a piece.
    """
    words = [
        piece for word in paragraph.split() for piece in _cut_word(word, limit=limit)
    ]
    return _pack(words, separator=WORD_SEPARATOR, limit=limit)


def _cut_word(word: str, *, limit: _SizeLimit) -> list[str]:
    """Cut ``word`` into its longest prefixes that fit ``limit``, unless it fits."""
    pieces: list[str] = []
    rest = word
    while rest:
        end = len(rest)
        while end > 1 and not limit.fits(rest[:end]):
            end -= 1
        pieces.append(rest[:end])
        rest = rest[end:]
    return pieces


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
