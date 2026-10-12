"""Answering questions from indexed chunks."""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from django_minimal_rag.llm import get_llm
from django_minimal_rag.retrieval import DEFAULT_LIMIT, RetrievedChunk, retrieve

PROMPT_PART_SEPARATOR = "\n"
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
    citations: tuple[Citation, ...]


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


def _cite(chunks: list[RetrievedChunk]) -> tuple[Citation, ...]:
    """Return one citation per document, in the order of its nearest chunk."""
    citations = (
        Citation(title=chunk.title, url=chunk.url, source_key=chunk.source_key)
        for chunk in chunks
    )
    return tuple(dict.fromkeys(citations))


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Return each document as an element, then ``question`` in a tag."""
    return PROMPT_PART_SEPARATOR.join(
        [
            *(
                _document(document_chunks)
                for document_chunks in _group_by_document(chunks)
            ),
            f"<question>{question}</question>",
        ]
    )


def _group_by_document(
    chunks: list[RetrievedChunk],
) -> list[list[RetrievedChunk]]:
    """Return the chunks of each document, in the order of its nearest chunk."""
    by_source_key: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        by_source_key.setdefault(chunk.source_key, []).append(chunk)
    return list(by_source_key.values())


def _document(chunks: list[RetrievedChunk]) -> str:
    """Return the chunks of one document: its title, then its extracts."""
    return PROMPT_PART_SEPARATOR.join(
        [
            "<document>",
            f"<title>{chunks[0].title}</title>",
            *(f"<extract>{chunk.text}</extract>" for chunk in chunks),
            "</document>",
        ]
    )
