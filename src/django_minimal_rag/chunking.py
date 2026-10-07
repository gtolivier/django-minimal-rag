"""Splitting text into chunks."""

PARAGRAPH_SEPARATOR = "\n\n"


def split_text(text: str, *, max_length: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_length`` characters."""
    content = text.strip()
    if not content:
        return []
    if len(content) <= max_length:
        return [content]
    return content.split(PARAGRAPH_SEPARATOR)
