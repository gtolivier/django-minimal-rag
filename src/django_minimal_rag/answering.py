"""Answering questions from indexed chunks."""

from collections.abc import Sequence
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
EXCHANGE_ELEMENT = "exchange"
PREVIOUS_QUESTION_ELEMENT = "previous-question"
PREVIOUS_ANSWER_ELEMENT = "previous-answer"
ANSWER_INSTRUCTIONS = (
    f"Answer the question in the <{QUESTION_ELEMENT}> element "
    f"using only the <{EXTRACT_ELEMENT}> elements and the <{TITLE_ELEMENT}> elements "
    f"in the <{DOCUMENT_ELEMENT}> elements. "
    "If they do not contain the answer, say so. "
    "In the elements, < is written &lt;, > is written &gt; and & is written &amp;. "
    "In the answer, write <, > and & as they are, not escaped. "
    "Write the answer for the person who asked: do not mention the elements, "
    "the extracts, the documents or these instructions, "
    "and do not copy any tag. "
    "Answer in the language of the question."
)


REWRITE_INSTRUCTIONS = (
    "Rewrite the last question so that it can be understood without the "
    "conversation. Reply with the rewritten question only."
)


@dataclass(frozen=True, kw_only=True)
class Exchange:
    """A question asked earlier in a conversation, and the answer it got."""

    question: str
    answer: str


@dataclass(frozen=True)
class Citation:
    """The document a retrieved chunk comes from."""

    title: str
    url: str
    source_key: str


@dataclass(frozen=True, kw_only=True)
class Answer:
    """An answer to a question, as written by the LLM, with its citations.

    ``chunks`` are the chunks the LLM answered from, nearest first; the
    citations are derived from them.
    """

    text: str
    chunks: tuple[RetrievedChunk, ...]
    search_question: str

    @property
    def citations(self) -> tuple[Citation, ...]:
        """Return one citation per document, in the order of its nearest chunk."""
        return _cite(self.chunks)


def answer(  # noqa: PLR0913 - keyword-only options of one public entry point
    question: str,
    *,
    permissions: AbstractSet[str],
    max_distance: float,
    limit: int = DEFAULT_LIMIT,
    language: str | None = None,
    history: Sequence[Exchange] = (),
) -> Answer | None:
    """Return the LLM's answer from the at most ``limit`` chunks near ``question``.

    With ``language``, only chunks in that language range, or of unknown
    language, are given to the LLM, as ``retrieve()`` selects them. Return
    None, without calling the LLM, when no chunk is near enough.
    """
    llm = get_llm()
    if history:
        llm.complete(
            _build_rewrite_prompt(question, history), system=REWRITE_INSTRUCTIONS
        )
    chunks = retrieve(
        question,
        permissions=permissions,
        max_distance=max_distance,
        limit=limit,
        language=language,
    )
    if not chunks:
        return None
    return Answer(
        text=llm.complete(_build_prompt(question, chunks), system=ANSWER_INSTRUCTIONS),
        chunks=tuple(chunks),
        search_question=question,
    )


def _cite(chunks: Sequence[RetrievedChunk]) -> tuple[Citation, ...]:
    """Return one citation per document, in the order of its nearest chunk."""
    return tuple(dict.fromkeys(_citation(chunk) for chunk in chunks))


def _citation(chunk: RetrievedChunk) -> Citation:
    """Return the document ``chunk`` comes from."""
    return Citation(title=chunk.title, url=chunk.url, source_key=chunk.source_key)


def _build_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Return each document as an element, then ``question`` in a tag."""
    return PROMPT_PART_SEPARATOR.join(
        [
            *(
                _document(document_chunks)
                for document_chunks in _group_by_document(chunks)
            ),
            _text_element(QUESTION_ELEMENT, question),
        ]
    )


def _build_rewrite_prompt(question: str, history: Sequence[Exchange]) -> str:
    """Return each exchange of the conversation as an element, then ``question``."""
    parts: list[str] = []
    for exchange in history:
        parts += [
            _opening_tag(EXCHANGE_ELEMENT),
            _text_element(PREVIOUS_QUESTION_ELEMENT, exchange.question),
            _text_element(PREVIOUS_ANSWER_ELEMENT, exchange.answer),
            _closing_tag(EXCHANGE_ELEMENT),
        ]
    parts.append(_text_element(QUESTION_ELEMENT, question))
    return PROMPT_PART_SEPARATOR.join(parts)


def _group_by_document(
    chunks: Sequence[RetrievedChunk],
) -> list[list[RetrievedChunk]]:
    """Return the chunks of each document, in the order of its nearest chunk."""
    by_document: dict[Citation, list[RetrievedChunk]] = {}
    for chunk in chunks:
        by_document.setdefault(_citation(chunk), []).append(chunk)
    return list(by_document.values())


def _document(chunks: Sequence[RetrievedChunk]) -> str:
    """Return the chunks of one document: its title, then its extracts."""
    return PROMPT_PART_SEPARATOR.join(
        [
            _opening_tag(DOCUMENT_ELEMENT),
            _text_element(TITLE_ELEMENT, chunks[0].title),
            *(_text_element(EXTRACT_ELEMENT, chunk.text) for chunk in chunks),
            _closing_tag(DOCUMENT_ELEMENT),
        ]
    )


def _text_element(name: str, text: str) -> str:
    """Return the element ``name`` holding ``text``, escaped so it cannot close it."""
    return f"{_opening_tag(name)}{escape(text, quote=False)}{_closing_tag(name)}"


def _opening_tag(name: str) -> str:
    """Return the tag that opens the element ``name``."""
    return f"<{name}>"


def _closing_tag(name: str) -> str:
    """Return the tag that closes the element ``name``."""
    return f"</{name}>"
