from django_minimal_rag.chunking import split_text

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


def test_word_longer_than_max_length_is_cut_into_pieces_of_max_length() -> None:
    text = "abcdefghij"
    max_length = 4
    assert len(text) > max_length

    assert split_text(text, max_length=max_length) == ["abcd", "efgh", "ij"]
