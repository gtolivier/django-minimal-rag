"""Splitting text into chunks."""


def split_text(text: str, *, max_length: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_length`` characters."""
    content = text.strip()
    if not content:
        return []
    if len(content) <= max_length:
        return [content]
    return content.split("\n\n")
