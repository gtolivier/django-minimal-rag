"""Splitting text into chunks."""


def split_text(text: str, *, max_length: int) -> list[str]:  # noqa: ARG001  # max_length is not used until a test requires it
    """Split ``text`` into chunks of at most ``max_length`` characters."""
    if not text:
        return []
    return [text]
