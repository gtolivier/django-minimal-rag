"""LLM backends."""


class FakeLLM:
    """An LLM for tests: it answers with the prompt it was given."""

    def complete(self, prompt: str, *, system: str) -> str:  # noqa: ARG002 - part of the LLM interface, unused by the fake
        """Return the prompt itself."""
        return prompt
