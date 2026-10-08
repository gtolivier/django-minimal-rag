from django.db import models

from django_minimal_rag.models import Source


def test_models_get_a_big_auto_field_primary_key_in_an_auto_field_project() -> None:
    primary_key = Source._meta.pk

    assert isinstance(primary_key, models.BigAutoField)
