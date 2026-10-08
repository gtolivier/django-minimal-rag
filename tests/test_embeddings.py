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
