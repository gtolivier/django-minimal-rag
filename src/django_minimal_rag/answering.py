"""Answering questions from indexed chunks."""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from django_minimal_rag.llm import get_llm
from django_minimal_rag.retrieval import RetrievedChunk, retrieve

PROMPT_PART_SEPARATOR = "\n\n"
ANSWER_INSTRUCTIONS = (
    "Answer the question using only the document extracts given. "
    "If they do not contain the answer, say so."
)


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
    return Answer(
        text=get_llm().complete(
            _build_prompt(question, chunks), system=ANSWER_INSTRUCTIONS
        )
    )


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Return each chunk under its document's title, then ``question``."""
    return PROMPT_PART_SEPARATOR.join(
        [*(f"{chunk.title}\n{chunk.text}" for chunk in chunks), question]
    )
