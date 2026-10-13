"""Answering questions from indexed chunks."""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from html import escape

from django_minimal_rag.llm import get_llm
from django_minimal_rag.retrieval import DEFAULT_LIMIT, RetrievedChunk, retrieve

PROMPT_PART_SEPARATOR = "\n"
QUESTION_ELEMENT = "question"
DOCUMENT_ELEMENT = "document"
TITLE_ELEMENT = "title"
EXTRACT_ELEMENT = "extract"
ANSWER_INSTRUCTIONS = (
    f"Answer the question in the <{QUESTION_ELEMENT}> element "
    f"using only the <{EXTRACT_ELEMENT}> elements and the <{TITLE_ELEMENT}> elements "
    f"in the <{DOCUMENT_ELEMENT}> elements. "
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
    return tuple(dict.fromkeys(_citation(chunk) for chunk in chunks))


def _citation(chunk: RetrievedChunk) -> Citation:
    """Return the document ``chunk`` comes from."""
    return Citation(title=chunk.title, url=chunk.url, source_key=chunk.source_key)


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Return each document as an element, then ``question`` in a tag."""
    return PROMPT_PART_SEPARATOR.join(
        [
            *(
                _document(document_chunks)
                for document_chunks in _group_by_document(chunks)
            ),
            _element(QUESTION_ELEMENT, question),
        ]
    )


def _group_by_document(
    chunks: list[RetrievedChunk],
) -> list[list[RetrievedChunk]]:
    """Return the chunks of each document, in the order of its nearest chunk."""
    by_document: dict[Citation, list[RetrievedChunk]] = {}
    for chunk in chunks:
        by_document.setdefault(_citation(chunk), []).append(chunk)
    return list(by_document.values())


def _document(chunks: list[RetrievedChunk]) -> str:
    """Return the chunks of one document: its title, then its extracts."""
    return PROMPT_PART_SEPARATOR.join(
        [
            f"<{DOCUMENT_ELEMENT}>",
            _element(TITLE_ELEMENT, escape(chunks[0].title, quote=False)),
            *(
                _element(EXTRACT_ELEMENT, escape(chunk.text, quote=False))
                for chunk in chunks
            ),
            f"</{DOCUMENT_ELEMENT}>",
        ]
    )


def _element(name: str, content: str) -> str:
    """Return ``content`` between the opening and closing tags of ``name``."""
    return f"<{name}>{content}</{name}>"
