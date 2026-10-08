import pytest
from django.db import IntegrityError

from django_minimal_rag.models import Document, Source


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
