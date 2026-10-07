import pytest
from django.db import connection


@pytest.mark.django_db
def test_database_tests_run_against_postgresql() -> None:
    assert connection.vendor == "postgresql"
