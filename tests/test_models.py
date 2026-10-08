import pytest
from django.db import IntegrityError

from django_minimal_rag.models import Chunk, Document, Source


@pytest.mark.django_db
def test_source_stored_with_a_source_key_reads_it_back_from_the_database() -> None:
    Source.objects.create(source_key="faq:opening-hours")

    stored = Source.objects.get()

    assert stored.source_key == "faq:opening-hours"


@pytest.mark.django_db
def test_second_source_with_the_same_source_key_cannot_be_stored() -> None:
    Source.objects.create(source_key="faq:opening-hours")

    with pytest.raises(IntegrityError):
        Source.objects.create(source_key="faq:opening-hours")


@pytest.mark.django_db
def test_document_stored_for_a_source_reads_back_its_fields_from_the_database() -> None:
    source = Source.objects.create(source_key="faq:opening-hours")
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
def test_document_stored_without_a_language_reads_back_none() -> None:
    source = Source.objects.create(source_key="faq:opening-hours")
    Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language=None,
    )

    stored = Document.objects.get()

    assert stored.language is None


@pytest.mark.django_db
def test_document_with_a_2000_character_title_reads_back_the_whole_title() -> None:
    long_title = ("Opening hours " * 143)[:2000]
    assert len(long_title) == 2000
    source = Source.objects.create(source_key="faq:opening-hours")
    Document.objects.create(
        source=source,
        title=long_title,
        url="https://example.com/faq/opening-hours",
        language="en",
    )

    stored = Document.objects.get()

    assert stored.title == long_title


@pytest.mark.django_db
def test_document_with_a_2000_character_url_reads_back_the_whole_url() -> None:
    base_url = "https://example.com/faq/opening-hours?q="
    long_url = base_url + "a" * (2000 - len(base_url))
    assert len(long_url) == 2000
    source = Source.objects.create(source_key="faq:opening-hours")
    Document.objects.create(
        source=source,
        title="Opening hours",
        url=long_url,
        language="en",
    )

    stored = Document.objects.get()

    assert stored.url == long_url


@pytest.mark.django_db
def test_document_stored_with_permissions_reads_back_the_same_names() -> None:
    source = Source.objects.create(source_key="faq:opening-hours")
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
def test_document_stored_without_permissions_reads_back_no_permissions() -> None:
    source = Source.objects.create(source_key="faq:opening-hours")
    Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language="en",
    )

    stored = Document.objects.get()

    assert set(stored.permissions) == set()


@pytest.mark.django_db
def test_deleting_a_source_deletes_its_documents() -> None:
    source = Source.objects.create(source_key="faq:opening-hours")
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
def test_chunk_stored_for_a_document_reads_back_its_fields_from_the_database() -> None:
    source = Source.objects.create(source_key="faq:opening-hours")
    document = Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language="en",
    )
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
def test_chunks_with_embeddings_of_different_dimensions_are_stored_side_by_side() -> (
    None
):
    source = Source.objects.create(source_key="faq:opening-hours")
    document = Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language="en",
    )
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
def test_chunk_without_an_embedding_cannot_be_stored() -> None:
    source = Source.objects.create(source_key="faq:opening-hours")
    document = Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language="en",
    )

    with pytest.raises(IntegrityError):
        Chunk.objects.create(
            document=document,
            rank=0,
            text="The shop opens at 9 am.",
            embedding_model="text-embedding-3-small",
        )


@pytest.mark.django_db
def test_deleting_a_document_deletes_its_chunks() -> None:
    source = Source.objects.create(source_key="faq:opening-hours")
    document = Document.objects.create(
        source=source,
        title="Opening hours",
        url="https://example.com/faq/opening-hours",
        language="en",
    )
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
