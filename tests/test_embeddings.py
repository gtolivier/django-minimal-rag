from django_minimal_rag.embeddings import FakeEmbeddings


def test_fake_embeddings_of_no_texts_is_an_empty_list() -> None:
    assert FakeEmbeddings().embed([]) == []


def test_fake_embeddings_of_one_text_is_one_vector_of_eight_floats() -> None:
    vectors = FakeEmbeddings().embed(["a text"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 8
    assert all(isinstance(value, float) for value in vectors[0])


def test_fake_embeddings_with_three_dimensions_are_vectors_of_three_floats() -> None:
    vectors = FakeEmbeddings(dimensions=3).embed(["a text"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 3
    assert all(isinstance(value, float) for value in vectors[0])


def test_fake_embeddings_of_two_different_texts_are_different_vectors() -> None:
    first, second = FakeEmbeddings().embed(["a text", "another text"])

    assert first != second


def test_fake_embeddings_of_two_different_texts_of_the_same_length_differ() -> None:
    first, second = FakeEmbeddings().embed(["a text", "b text"])

    assert first != second


def test_fake_embeddings_returns_one_vector_per_text_in_the_texts_order() -> None:
    embeddings = FakeEmbeddings()
    [short_vector] = embeddings.embed(["a"])
    [long_vector] = embeddings.embed(["a longer text"])

    assert embeddings.embed(["a", "a longer text"]) == [short_vector, long_vector]
    assert embeddings.embed(["a longer text", "a"]) == [long_vector, short_vector]


def test_fake_embeddings_of_the_same_text_from_two_instances_are_equal() -> None:
    [first] = FakeEmbeddings().embed(["a text"])
    [second] = FakeEmbeddings().embed(["a text"])

    assert first == second
