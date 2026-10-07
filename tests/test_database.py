import pytest
from django.db import connection


@pytest.mark.django_db
def test_database_tests_run_against_postgresql() -> None:
    assert connection.vendor == "postgresql"


@pytest.mark.django_db
def test_pgvector_extension_is_available_on_the_database_server() -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM pg_available_extensions WHERE name = %s",
            ["vector"],
        )
        row = cursor.fetchone()
    assert row is not None
