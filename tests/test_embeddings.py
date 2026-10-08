from django_minimal_rag.embeddings import FakeEmbeddings


def test_fake_embeddings_of_no_texts_is_an_empty_list() -> None:
    assert FakeEmbeddings().embed([]) == []
