import pytest
from django.core.management import call_command


@pytest.mark.django_db
def test_makemigrations_finds_no_change_to_make_for_the_app() -> None:
    call_command("makemigrations", "django_minimal_rag", check=True, dry_run=True)
