"""Splitting text into chunks."""

import re

PARAGRAPH_SEPARATOR = "\n\n"
WORD_SEPARATOR = " "
BLANK_LINES = re.compile(r"\n\s*\n")


def split_text(text: str, *, max_length: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_length`` characters."""
    content = text.strip()
    if not content:
        return []
    chunks: list[str] = []
    short_paragraphs: list[str] = []
    for paragraph in BLANK_LINES.split(content):
        if len(paragraph) <= max_length:
            short_paragraphs.append(paragraph)
            continue
        chunks += _pack_paragraphs(short_paragraphs, max_length=max_length)
        short_paragraphs = []
        chunks += _split_between_words(paragraph, max_length=max_length)
    return chunks + _pack_paragraphs(short_paragraphs, max_length=max_length)


def _pack_paragraphs(paragraphs: list[str], *, max_length: int) -> list[str]:
    """Join consecutive ``paragraphs`` by blank lines while they fit ``max_length``."""
    return _pack(paragraphs, separator=PARAGRAPH_SEPARATOR, max_length=max_length)


def _split_between_words(paragraph: str, *, max_length: int) -> list[str]:
    """Split ``paragraph`` between words into pieces of at most ``max_length``."""
    words = [
        word[start : start + max_length]
        for word in paragraph.split()
        for start in range(0, len(word), max_length)
    ]
    return _pack(words, separator=WORD_SEPARATOR, max_length=max_length)


def _pack(pieces: list[str], *, separator: str, max_length: int) -> list[str]:
    """Join consecutive ``pieces`` with ``separator`` while they fit ``max_length``."""
    chunks = pieces[:1]
    for piece in pieces[1:]:
        candidate = chunks[-1] + separator + piece
        if len(candidate) <= max_length:
            chunks[-1] = candidate
        else:
            chunks.append(piece)
    return chunks
