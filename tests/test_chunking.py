import pytest

from django_minimal_rag.chunking import Chunk, chunk_group, split_text
from tests.documents import MutableDocument, SampleDocument

# A limit well above the length of the short texts below.
MAX_LENGTH = 100


def test_empty_text_gives_no_chunks() -> None:
    assert split_text("", max_length=MAX_LENGTH) == []


def test_whitespace_only_text_gives_no_chunks() -> None:
    assert split_text(" \t\n  \n", max_length=MAX_LENGTH) == []


def test_text_shorter_than_max_length_gives_one_chunk_the_text_itself() -> None:
    text = "A short document."
    assert len(text) < MAX_LENGTH

    assert split_text(text, max_length=MAX_LENGTH) == [text]


def test_whitespace_around_the_text_is_stripped_from_its_chunk() -> None:
    text = " \n\tA short document.\n  \t"

    assert split_text(text, max_length=MAX_LENGTH) == ["A short document."]


def test_two_paragraphs_exceeding_max_length_give_one_chunk_per_paragraph() -> None:
    first = "First paragraph."
    second = "Second paragraph."
    text = f"{first}\n\n{second}"
    max_length = 20
    assert len(first) <= max_length
    assert len(second) <= max_length
    assert len(text) > max_length

    assert split_text(text, max_length=max_length) == [first, second]


def test_consecutive_paragraphs_are_packed_into_one_chunk_while_they_fit() -> None:
    first = "First paragraph."
    second = "Second paragraph."
    third = "Third paragraph."
    first_two = f"{first}\n\n{second}"
    text = f"{first_two}\n\n{third}"
    max_length = 40
    assert len(first_two) <= max_length
    assert len(third) <= max_length
    assert len(text) > max_length

    assert split_text(text, max_length=max_length) == [first_two, third]


def test_paragraphs_joining_to_exactly_max_length_are_kept_in_one_chunk() -> None:
    first = "First paragraph."
    second = "Second paragraph."
    text = f"{first}\n\n{second}"
    max_length = len(text)

    assert split_text(text, max_length=max_length) == [text]


def test_paragraphs_are_joined_by_one_blank_line_whatever_separates_them() -> None:
    # Several blank lines, then a blank line holding spaces and a tab.
    text = "A.\n\n\n\nB.\n  \t\nC."

    assert split_text(text, max_length=MAX_LENGTH) == ["A.\n\nB.\n\nC."]


def test_crlf_line_endings_become_lf_between_and_within_paragraphs() -> None:
    # A line break inside the first paragraph, a blank line between the two.
    text = "First line,\r\nsecond line.\r\n\r\nNext paragraph."
    assert len(text) < MAX_LENGTH

    assert split_text(text, max_length=MAX_LENGTH) == [
        "First line,\nsecond line.\n\nNext paragraph."
    ]


def test_lone_cr_line_endings_become_lf_between_and_within_paragraphs() -> None:
    # A line break inside the first paragraph, a blank line between the two.
    text = "First line,\rsecond line.\r\rNext paragraph."
    assert len(text) < MAX_LENGTH

    assert split_text(text, max_length=MAX_LENGTH) == [
        "First line,\nsecond line.\n\nNext paragraph."
    ]


def test_cr_followed_by_crlf_counts_as_two_line_endings_between_paragraphs() -> None:
    # A lone "\r" then a "\r\n": two line endings, so a blank line.
    text = "A.\r\r\nB."

    assert split_text(text, max_length=MAX_LENGTH) == ["A.\n\nB."]


def test_spaces_and_tabs_around_each_paragraph_are_stripped() -> None:
    # Trailing spaces before each blank line, leading spaces or a tab after it.
    text = "First paragraph.  \n\n  Second paragraph. \t\n\n\tThird paragraph."
    assert len(text) < MAX_LENGTH

    assert split_text(text, max_length=MAX_LENGTH) == [
        "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    ]


def test_spaces_and_tabs_around_each_line_of_a_paragraph_are_stripped() -> None:
    # Trailing spaces or a tab before each line break, leading spaces or a tab
    # after it, inside a single paragraph.
    text = "First line.  \n  Second line. \t\n\tThird line."
    assert len(text) < MAX_LENGTH

    assert split_text(text, max_length=MAX_LENGTH) == [
        "First line.\nSecond line.\nThird line."
    ]


def test_paragraph_longer_than_max_length_is_split_between_words_packed() -> None:
    text = "alpha beta gamma delta epsilon"
    max_length = 12
    assert len(text) > max_length

    assert split_text(text, max_length=max_length) == [
        "alpha beta",
        "gamma delta",
        "epsilon",
    ]


def test_pieces_of_a_split_paragraph_are_not_merged_with_its_neighbours() -> None:
    before = "A."
    long_paragraph = "alpha beta gamma delta epsilon"
    after = "B."
    text = f"{before}\n\n{long_paragraph}\n\n{after}"
    max_length = 14
    assert len(long_paragraph) > max_length
    # Both neighbours would fit next to the adjacent piece of the split paragraph.
    assert len(f"{before}\n\nalpha beta") <= max_length
    assert len(f"epsilon\n\n{after}") <= max_length

    assert split_text(text, max_length=max_length) == [
        before,
        "alpha beta",
        "gamma delta",
        "epsilon",
        after,
    ]


def test_paragraph_longer_than_max_length_is_split_between_lines_packed() -> None:
    first = "alpha beta"
    second = "gamma"
    third = "delta epsilon"
    first_two = f"{first}\n{second}"
    text = f"{first_two}\n{third}"
    max_length = 16
    assert len(text) > max_length
    assert len(first_two) <= max_length
    assert len(third) <= max_length

    # Lines are kept whole and joined by one newline, not split into words
    # joined by spaces.
    assert split_text(text, max_length=max_length) == [first_two, third]


def test_pieces_of_a_split_line_are_not_merged_with_its_neighbour_lines() -> None:
    before = "A."
    long_line = "alpha beta gamma delta epsilon"
    after = "B."
    text = f"{before}\n{long_line}\n{after}"
    max_length = 14
    assert len(text) > max_length
    assert len(long_line) > max_length
    # Both neighbour lines would fit next to the adjacent piece of the split line.
    assert len(f"{before}\nalpha beta") <= max_length
    assert len(f"epsilon\n{after}") <= max_length

    assert split_text(text, max_length=max_length) == [
        before,
        "alpha beta",
        "gamma delta",
        "epsilon",
        after,
    ]


def test_no_piece_of_a_split_paragraph_keeps_the_spaces_around_its_lines() -> None:
    # Trailing spaces before the line break, leading spaces after it.
    text = "a   \n   b c d e"
    max_length = 6
    assert len(text) > max_length

    assert split_text(text, max_length=max_length) == ["a", "b c d", "e"]


def test_word_longer_than_max_length_is_cut_into_pieces_of_max_length() -> None:
    text = "abcdefghij"
    max_length = 4
    assert len(text) > max_length

    assert split_text(text, max_length=max_length) == ["abcd", "efgh", "ij"]


def count_words(text: str) -> int:
    return len(text.split())


def test_sizes_are_measured_by_the_length_function_given() -> None:
    text = "alpha beta gamma delta epsilon"
    max_words = 2
    # Every word is longer than max_words characters: counted in characters,
    # each would be cut.
    assert all(len(word) > max_words for word in text.split())

    assert split_text(text, max_length=max_words, length=count_words) == [
        "alpha beta",
        "gamma delta",
        "epsilon",
    ]


def count_utf8_bytes(text: str) -> int:
    return len(text.encode())


def test_word_too_long_for_length_function_is_cut_at_longest_fitting_prefix() -> None:
    # "é" takes two bytes in UTF-8, every other letter one.
    text = "aébécé"
    max_bytes = 4
    assert count_utf8_bytes(text) > max_bytes

    # Each piece is the longest prefix of what remains measuring at most
    # max_bytes, so pieces hold different numbers of characters.
    assert split_text(text, max_length=max_bytes, length=count_utf8_bytes) == [
        "aéb",  # 4 bytes: adding "é" would make 6
        "éc",  # 3 bytes: adding "é" would make 5
        "é",
    ]


def test_max_length_defaults_to_1000_characters() -> None:
    default_max_length = 1000
    text = "a" * (default_max_length + 1)

    # The longest prefix that fits holds exactly 1000 characters: 1000 fits,
    # 1001 does not.
    assert split_text(text) == ["a" * default_max_length, "a"]


def test_max_length_below_1_raises_value_error() -> None:
    # No chunk could fit a limit of 0: the call is refused rather than
    # producing chunks that exceed it.
    with pytest.raises(ValueError, match="max_length"):
        split_text("A short document.", max_length=0)


def count_one_more_than_characters(text: str) -> int:
    return len(text) + 1


def test_character_measuring_more_than_max_length_raises_value_error() -> None:
    max_length = 1
    # Even a single character measures 2: no piece of the text could fit the
    # limit, so the call is refused rather than producing chunks that exceed it.
    assert count_one_more_than_characters("a") > max_length

    with pytest.raises(ValueError):
        split_text("abc", max_length=max_length, length=count_one_more_than_characters)


def count_tab_as_ten(text: str) -> int:
    tab_size = 10
    return len(text) + (tab_size - 1) * text.count("\t")


def test_whitespace_character_measuring_more_than_max_length_does_not_raise() -> None:
    text = "ab\tcd"
    max_length = 5
    # The tab alone measures more than max_length, but it separates words and
    # never reaches a chunk: the words are joined by a space instead.
    assert count_tab_as_ten("\t") > max_length

    assert split_text(text, max_length=max_length, length=count_tab_as_ten) == ["ab cd"]


def test_cutting_a_long_word_calls_length_a_bounded_number_of_times() -> None:
    word = "a" * 10_000
    max_length = 100
    max_calls = 2_500
    calls = 0

    def recording_len(text: str) -> int:
        nonlocal calls
        calls += 1
        return len(text)

    # The word has no space: it is cut into 100 pieces, and measuring every
    # candidate prefix from the end of the word would take far more calls.
    chunks = split_text(word, max_length=max_length, length=recording_len)

    assert chunks == ["a" * max_length] * (len(word) // max_length)
    assert calls <= max_calls


def test_one_paragraph_that_fits_calls_length_exactly_once() -> None:
    text = "A short document."
    assert len(text) < MAX_LENGTH
    measured: list[str] = []

    def recording_len(measured_text: str) -> int:
        measured.append(measured_text)
        return len(measured_text)

    # The paragraph fits as a whole: measuring it once is enough, and no
    # character of it needs measuring on its own.
    chunks = split_text(text, max_length=MAX_LENGTH, length=recording_len)

    assert chunks == [text]
    assert measured == [text]


def note(number: int, *, title: str, text: str) -> SampleDocument:
    """Note ``number`` of the host project, in English and public."""
    return SampleDocument(
        text=text,
        source_key=f"app.note:{number}",
        title=title,
        url=f"https://example.com/notes/{number}/",
        language="en",
        permissions=frozenset(),
    )


def test_group_with_no_documents_gives_no_chunks() -> None:
    assert chunk_group([]) == []


def test_group_max_length_below_1_raises_value_error_even_with_no_documents() -> None:
    # The limit is refused up front, not only once a document is split by it.
    with pytest.raises(ValueError, match="max_length"):
        chunk_group([], max_length=0)


def test_each_chunk_of_one_document_holds_the_document_its_text_and_rank() -> None:
    default_max_length = 1000
    first = " ".join(["alpha"] * 100)
    second = " ".join(["beta"] * 120)
    text = f"{first}\n\n{second}"
    assert len(first) <= default_max_length
    assert len(second) <= default_max_length
    assert len(text) > default_max_length
    document = note(1, title="A note", text=text)

    chunks: list[Chunk] = chunk_group([document])

    assert [(chunk.document, chunk.text, chunk.rank) for chunk in chunks] == [
        (document, first, 0),
        (document, second, 1),
    ]


def test_ranks_continue_across_the_documents_of_a_group_in_their_order() -> None:
    default_max_length = 1000
    first = " ".join(["alpha"] * 100)
    second = " ".join(["beta"] * 120)
    third = "A short note."
    assert len(first) <= default_max_length
    assert len(second) <= default_max_length
    assert len(f"{first}\n\n{second}") > default_max_length
    first_document = note(1, title="A long note", text=f"{first}\n\n{second}")
    second_document = note(2, title="A short note", text=third)

    chunks = chunk_group([first_document, second_document])

    assert [(chunk.document, chunk.text, chunk.rank) for chunk in chunks] == [
        (first_document, first, 0),
        (first_document, second, 1),
        (second_document, third, 2),
    ]


def test_document_with_empty_text_gives_no_chunk_and_takes_no_rank() -> None:
    before = note(1, title="Before", text="A note before.")
    empty = note(2, title="Empty", text="")
    after = note(3, title="After", text="A note after.")

    chunks = chunk_group([before, empty, after])

    assert [(chunk.document, chunk.text, chunk.rank) for chunk in chunks] == [
        (before, "A note before.", 0),
        (after, "A note after.", 1),
    ]


def test_max_length_and_length_are_applied_to_every_document_of_the_group() -> None:
    max_words = 2
    first = note(1, title="First", text="alpha beta gamma")
    second = note(2, title="Second", text="delta epsilon zeta theta")
    # Every word is longer than max_words characters: counted in characters,
    # each would be cut.
    assert all(
        len(word) > max_words
        for document in (first, second)
        for word in document.text.split()
    )

    chunks = chunk_group([first, second], max_length=max_words, length=count_words)

    assert [(chunk.document, chunk.text, chunk.rank) for chunk in chunks] == [
        (first, "alpha beta", 0),
        (first, "gamma", 1),
        (second, "delta epsilon", 2),
        (second, "zeta theta", 3),
    ]


def test_chunk_of_an_unhashable_document_can_be_hashed() -> None:
    document = MutableDocument(
        text="A note.",
        source_key="app.note:1",
        title="A note",
        url="https://example.com/notes/1/",
        language="en",
        permissions=frozenset(),
    )
    # A non-frozen dataclass sets __hash__ to None: the document is unhashable.
    assert MutableDocument.__hash__ is None
    chunk = Chunk(document=document, text="A note.", rank=0)

    assert chunk in {chunk}
