import importlib
from collections.abc import Iterator

import pytest

from tests import settings

LIBPQ_ENVIRONMENT_VARIABLES = ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD")


@pytest.fixture
def environment() -> Iterator[pytest.MonkeyPatch]:
    """Patch the environment for one test.

    The settings module is reloaded once the environment is restored, so the
    settings a test reloaded under its patched environment do not leak into
    the next tests.
    """
    try:
        with pytest.MonkeyPatch.context() as patch:
            yield patch
    finally:
        importlib.reload(settings)


def reloaded_connection_parameters() -> dict[str, object]:
    """The connection parameters of the settings, read from the current environment."""
    default = importlib.reload(settings).DATABASES["default"]
    return {key: default.get(key) for key in ("HOST", "PORT", "USER", "PASSWORD")}


def test_default_database_points_at_compose_database_without_libpq_variables(
    environment: pytest.MonkeyPatch,
) -> None:
    for name in LIBPQ_ENVIRONMENT_VARIABLES:
        environment.delenv(name, raising=False)

    connection_parameters = reloaded_connection_parameters()
    assert connection_parameters == {
        "HOST": "localhost",
        "PORT": "5433",
        "USER": "postgres",
        "PASSWORD": "postgres",
    }


def test_each_libpq_variable_overrides_its_default_database_parameter(
    environment: pytest.MonkeyPatch,
) -> None:
    libpq_variables = {
        "PGHOST": "db.example.test",
        "PGPORT": "6543",
        "PGUSER": "rag_user",
        "PGPASSWORD": "rag_password",
    }
    for name, value in libpq_variables.items():
        environment.setenv(name, value)

    connection_parameters = reloaded_connection_parameters()
    assert connection_parameters == {
        "HOST": "db.example.test",
        "PORT": "6543",
        "USER": "rag_user",
        "PASSWORD": "rag_password",
    }
