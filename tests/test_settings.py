import importlib

import pytest

from tests import settings

LIBPQ_ENVIRONMENT_VARIABLES = ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD")


def test_default_database_points_at_compose_database_without_libpq_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        with monkeypatch.context() as patch:
            for name in LIBPQ_ENVIRONMENT_VARIABLES:
                patch.delenv(name, raising=False)
            default = importlib.reload(settings).DATABASES["default"]
    finally:
        importlib.reload(settings)

    connection_parameters = {
        key: default.get(key) for key in ("HOST", "PORT", "USER", "PASSWORD")
    }
    assert connection_parameters == {
        "HOST": "localhost",
        "PORT": "5433",
        "USER": "postgres",
        "PASSWORD": "postgres",
    }


def test_each_libpq_variable_overrides_its_default_database_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "PGHOST": "db.example.test",
        "PGPORT": "6543",
        "PGUSER": "rag_user",
        "PGPASSWORD": "rag_password",
    }
    try:
        with monkeypatch.context() as patch:
            for name, value in environment.items():
                patch.setenv(name, value)
            default = importlib.reload(settings).DATABASES["default"]
    finally:
        importlib.reload(settings)

    connection_parameters = {
        key: default.get(key) for key in ("HOST", "PORT", "USER", "PASSWORD")
    }
    assert connection_parameters == {
        "HOST": "db.example.test",
        "PORT": "6543",
        "USER": "rag_user",
        "PASSWORD": "rag_password",
    }
