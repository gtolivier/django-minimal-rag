import pytest
from django.db import IntegrityError

from django_minimal_rag.models import Chunk, Document, Source


@pytest.fixture
def source() -> Source:
    return Source.objects.create(source_key="faq:opening-hours")


@pytest.fixture
def document(source: Source) -> Document:
    return Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language="en",
    )


@pytest.mark.django_db
def test_source_stored_with_a_source_key_reads_it_back_from_the_database() -> None:
    Source.objects.create(source_key="faq:opening-hours")

    stored = Source.objects.get()

    assert stored.source_key == "faq:opening-hours"


@pytest.mark.django_db
def test_source_with_a_500_character_source_key_reads_back_the_whole_key() -> None:
    base_key = "faq:opening-hours:"
    long_key = base_key + "a" * (500 - len(base_key))
    assert len(long_key) == 500
    Source.objects.create(source_key=long_key)

    stored = Source.objects.get()

    assert stored.source_key == long_key


@pytest.mark.django_db
def test_second_source_with_the_same_source_key_cannot_be_stored() -> None:
    Source.objects.create(source_key="faq:opening-hours")

    with pytest.raises(IntegrityError):
        Source.objects.create(source_key="faq:opening-hours")


@pytest.mark.django_db
def test_document_stored_for_a_source_reads_back_its_fields_from_the_database(
    source: Source,
) -> None:
    Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language="en",
    )

    stored = Document.objects.get()

    assert stored.source == source
    assert stored.title == "Opening hours"
    assert stored.url == "https://example.com/faq/opening-hours"
    assert stored.language == "en"


@pytest.mark.django_db
def test_document_stored_without_a_language_reads_back_none(source: Source) -> None:
    Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language=None,
    )

    stored = Document.objects.get()

    assert stored.language is None


@pytest.mark.django_db
def test_document_with_a_19_character_language_tag_reads_back_the_whole_tag(
    source: Source,
) -> None:
    long_tag = "sl-rozaj-biske-1994"
    assert len(long_tag) == 19
    Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language=long_tag,
    )

    stored = Document.objects.get()

    assert stored.language == long_tag


@pytest.mark.django_db
def test_document_with_a_300_character_language_tag_reads_back_the_whole_tag(
    source: Source,
) -> None:
    # A private-use sequence of subtags of at most 8 characters keeps the tag
    # valid BCP 47 however long it grows.
    long_tag = "en-x-" + "-".join(["abcdefgh"] * 32) + "-abcdefg"
    assert len(long_tag) == 300
    Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language=long_tag,
    )

    stored = Document.objects.get()

    assert stored.language == long_tag


@pytest.mark.django_db
def test_document_with_a_2000_character_title_reads_back_the_whole_title(
    source: Source,
) -> None:
    long_title = ("Opening hours " * 143)[:2000]
    assert len(long_title) == 2000
    Document.objects.create(
        source=source,
        title=long_title,
        url="https://example.com/faq/opening-hours",
        language="en",
    )

    stored = Document.objects.get()

    assert stored.title == long_title


@pytest.mark.django_db
def test_document_with_a_10000_character_title_reads_back_the_whole_title(
    source: Source,
) -> None:
    long_title = ("Opening hours " * 715)[:10000]
    assert len(long_title) == 10000
    Document.objects.create(
        source=source,
        title=long_title,
        url="https://example.com/faq/opening-hours",
        language="en",
    )

    stored = Document.objects.get()

    assert stored.title == long_title


@pytest.mark.django_db
def test_document_with_a_2000_character_url_reads_back_the_whole_url(
    source: Source,
) -> None:
    base_url = "https://example.com/faq/opening-hours?q="
    long_url = base_url + "a" * (2000 - len(base_url))
    assert len(long_url) == 2000
    Document.objects.create(
        source=source,
        title="Opening hours",
        url=long_url,
        language="en",
    )

    stored = Document.objects.get()

    assert stored.url == long_url


@pytest.mark.django_db
def test_document_with_a_10000_character_url_reads_back_the_whole_url(
    source: Source,
) -> None:
    base_url = "https://example.com/faq/opening-hours?q="
    long_url = base_url + "a" * (10000 - len(base_url))
    assert len(long_url) == 10000
    Document.objects.create(
        source=source,
        title="Opening hours",
        url=long_url,
        language="en",
    )

    stored = Document.objects.get()

    assert stored.url == long_url


@pytest.mark.django_db
def test_document_stored_with_permissions_reads_back_the_same_names(
    source: Source,
) -> None:
    Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language="en",
        permissions=["app.view_note", "app.change_note"],
    )

    stored = Document.objects.get()

    assert set(stored.permissions) == {"app.view_note", "app.change_note"}


@pytest.mark.django_db
def test_document_stored_without_permissions_reads_back_no_permissions(
    source: Source,
) -> None:
    Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language="en",
    )

    stored = Document.objects.get()

    assert set(stored.permissions) == set()


@pytest.mark.django_db
def test_deleting_a_source_deletes_its_documents(source: Source) -> None:
    Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language="en",
    )
    Document.objects.create(
        source=source,
        title="Holiday hours",
        url="https://example.com/faq/holiday-hours",
        language="en",
    )

    source.delete()

    assert not Document.objects.exists()


@pytest.mark.django_db
def test_chunk_stored_for_a_document_reads_back_its_fields_from_the_database(
    document: Document,
) -> None:
    Chunk.objects.create(
        document=document,
        rank=2,
        text="The shop opens at 9 am.",
        embedding_model="text-embedding-3-small",
        embedding=[0.5, -1.0, 0.25],
    )

    stored = Chunk.objects.get()

    assert stored.document == document
    assert stored.rank == 2
    assert stored.text == "The shop opens at 9 am."
    assert stored.embedding_model == "text-embedding-3-small"
    assert list(stored.embedding) == [0.5, -1.0, 0.25]


@pytest.mark.django_db
def test_chunks_with_embeddings_of_different_dimensions_are_stored_side_by_side(
    document: Document,
) -> None:
    Chunk.objects.create(
        document=document,
        rank=0,
        text="The shop opens at 9 am.",
        embedding_model="small-model",
        embedding=[0.5, -1.0, 0.25],
    )
    Chunk.objects.create(
        document=document,
        rank=0,
        text="The shop opens at 9 am.",
        embedding_model="large-model",
        embedding=[0.5, -1.0, 0.25, 2.0, -0.125],
    )

    small = Chunk.objects.get(embedding_model="small-model")
    large = Chunk.objects.get(embedding_model="large-model")

    assert list(small.embedding) == [0.5, -1.0, 0.25]
    assert list(large.embedding) == [0.5, -1.0, 0.25, 2.0, -0.125]


@pytest.mark.django_db
def test_chunk_without_an_embedding_cannot_be_stored(document: Document) -> None:
    with pytest.raises(IntegrityError):
        Chunk.objects.create(
            document=document,
            rank=0,
            text="The shop opens at 9 am.",
            embedding_model="text-embedding-3-small",
        )


@pytest.mark.django_db
def test_deleting_a_document_deletes_its_chunks(document: Document) -> None:
    Chunk.objects.create(
        document=document,
        rank=0,
        text="The shop opens at 9 am.",
        embedding_model="text-embedding-3-small",
        embedding=[0.5, -1.0, 0.25],
    )
    Chunk.objects.create(
        document=document,
        rank=1,
        text="The shop closes at 6 pm.",
        embedding_model="text-embedding-3-small",
        embedding=[0.25, 0.5, -1.0],
    )

    document.delete()

    assert not Chunk.objects.exists()
