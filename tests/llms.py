"""The test bench's LLM backends."""


class CannedLLM:
    """An LLM backend answering every prompt with the ``reply`` it was built with.

    The reply is given through the setting's OPTIONS, so a test decides what
    the LLM answers.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, prompt: str, *, system: str) -> str:
        """Return the canned reply, whatever the prompt."""
        return self.reply


class RecordingLLM:
    """An LLM backend appending each ``(prompt, system)`` it is given to ``calls``.

    The list is given through the setting's OPTIONS, so it outlives the
    backend instances built on each call. Every prompt is answered with
    ``reply``.
    """

    def __init__(self, calls: list[tuple[str, str]], reply: str = "") -> None:
        self.calls = calls
        self.reply = reply

    def complete(self, prompt: str, *, system: str) -> str:
        """Record the prompt and the system message, and return ``reply``."""
        self.calls.append((prompt, system))
        return self.reply
