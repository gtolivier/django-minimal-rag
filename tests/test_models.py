import pytest
from django.db import IntegrityError

from django_minimal_rag.models import Source


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
