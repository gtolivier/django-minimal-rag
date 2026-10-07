"""Splitting text into chunks."""

import re

PARAGRAPH_SEPARATOR = "\n\n"
BLANK_LINES = re.compile(r"\n\s*\n")


def split_text(text: str, *, max_length: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_length`` characters."""
    content = text.strip()
    if not content:
        return []
    chunks: list[str] = []
    for paragraph in BLANK_LINES.split(content):
        if len(paragraph) > max_length:
            chunks.extend(_pack_words(paragraph.split(), max_length=max_length))
        else:
            chunks.append(paragraph)
    return _pack_paragraphs(chunks, max_length=max_length)


def _pack_words(words: list[str], *, max_length: int) -> list[str]:
    """Join consecutive ``words`` with one space while they fit in ``max_length``."""
    chunks = words[:1]
    for word in words[1:]:
        candidate = chunks[-1] + " " + word
        if len(candidate) <= max_length:
            chunks[-1] = candidate
        else:
            chunks.append(word)
    return chunks


def _pack_paragraphs(paragraphs: list[str], *, max_length: int) -> list[str]:
    """Join consecutive ``paragraphs`` greedily while they fit in ``max_length``."""
    chunks = paragraphs[:1]
    for paragraph in paragraphs[1:]:
        candidate = chunks[-1] + PARAGRAPH_SEPARATOR + paragraph
        if len(candidate) <= max_length:
            chunks[-1] = candidate
        else:
            chunks.append(paragraph)
    return chunks
