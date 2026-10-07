"""Splitting text into chunks."""

PARAGRAPH_SEPARATOR = "\n\n"


def split_text(text: str, *, max_length: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_length`` characters."""
    content = text.strip()
    if not content:
        return []
    return _pack_paragraphs(content.split(PARAGRAPH_SEPARATOR), max_length=max_length)


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
