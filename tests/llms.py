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
