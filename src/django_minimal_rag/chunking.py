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
    run: list[str] = []
    for paragraph in BLANK_LINES.split(content):
        if len(paragraph) <= max_length:
            run.append(paragraph)
            continue
        chunks += _pack(run, separator=PARAGRAPH_SEPARATOR, max_length=max_length)
        run = []
        chunks += _split_paragraph(paragraph, max_length=max_length)
    return chunks + _pack(run, separator=PARAGRAPH_SEPARATOR, max_length=max_length)


def _split_paragraph(paragraph: str, *, max_length: int) -> list[str]:
    """Split ``paragraph`` between words if it is longer than ``max_length``."""
    if len(paragraph) <= max_length:
        return [paragraph]
    return _pack(paragraph.split(), separator=WORD_SEPARATOR, max_length=max_length)


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
