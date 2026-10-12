"""Answering questions from indexed chunks."""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from django_minimal_rag.llm import get_llm
from django_minimal_rag.retrieval import retrieve

CHUNK_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Answer:
    """An answer to a question, as written by the LLM."""

    text: str


def answer(
    question: str,
    *,
    permissions: AbstractSet[str],
    max_distance: float,
) -> Answer | None:
    """Return the LLM's answer from the chunks near ``question``.

    Return None, without calling the LLM, when no chunk is near enough.
    """
    chunks = retrieve(question, permissions=permissions, max_distance=max_distance)
    if not chunks:
        return None
    prompt = CHUNK_SEPARATOR.join([*(chunk.text for chunk in chunks), question])
    return Answer(text=get_llm().complete(prompt, system=""))
