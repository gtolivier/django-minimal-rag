from django_minimal_rag.chunking import split_text

# Any positive limit: an empty text has nothing to split, whatever the limit.
MAX_LENGTH = 100


def test_empty_text_gives_no_chunks() -> None:
    assert split_text("", max_length=MAX_LENGTH) == []
