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
