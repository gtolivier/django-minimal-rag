"""Answering questions from indexed chunks."""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from django_minimal_rag.llm import get_llm
from django_minimal_rag.retrieval import DEFAULT_LIMIT, RetrievedChunk, retrieve

PROMPT_PART_SEPARATOR = "\n\n"
ANSWER_INSTRUCTIONS = (
    "Answer the question using only the document extracts given. "
    "If they do not contain the answer, say so. "
    "Answer in the language of the question."
)


@dataclass(frozen=True)
class Citation:
    """The document a retrieved chunk comes from."""

    title: str
    url: str
    source_key: str


@dataclass(frozen=True)
class Answer:
    """An answer to a question, as written by the LLM, with its citations."""

    text: str
    citations: list[Citation]


def answer(
    question: str,
    *,
    permissions: AbstractSet[str],
    max_distance: float,
    limit: int = DEFAULT_LIMIT,
) -> Answer | None:
    """Return the LLM's answer from the at most ``limit`` chunks near ``question``.

    Return None, without calling the LLM, when no chunk is near enough.
    """
    llm = get_llm()
    chunks = retrieve(
        question, permissions=permissions, max_distance=max_distance, limit=limit
    )
    if not chunks:
        return None
    return Answer(
        text=llm.complete(_build_prompt(question, chunks), system=ANSWER_INSTRUCTIONS),
        citations=_cite(chunks),
    )


def _cite(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Return one citation per document, in the order of its nearest chunk."""
    citations = (
        Citation(title=chunk.title, url=chunk.url, source_key=chunk.source_key)
        for chunk in chunks
    )
    return list(dict.fromkeys(citations))


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Return each chunk under its document's title, then ``question``."""
    return PROMPT_PART_SEPARATOR.join(
        [*(f"{chunk.title}\n{chunk.text}" for chunk in chunks), question]
    )
