import pytest
from django.core.management import call_command
from django.db import connection


@pytest.mark.django_db
def test_makemigrations_finds_no_change_to_make_for_the_app() -> None:
    call_command("makemigrations", "django_minimal_rag", check=True, dry_run=True)


def vector_extension_is_installed() -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        return cursor.fetchone() is not None


@pytest.mark.django_db
def test_rolling_the_app_back_to_zero_leaves_the_vector_extension_installed() -> None:
    # Other apps' vector columns may depend on the extension, and an
    # administrator may have created it before the app's migrations ran.
    try:
        call_command("migrate", "django_minimal_rag", "zero", verbosity=0)
        assert vector_extension_is_installed()
    finally:
        call_command("migrate", "django_minimal_rag", verbosity=0)
