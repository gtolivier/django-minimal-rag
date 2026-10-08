"""Django settings for the test suite, also loaded by the django-stubs mypy plugin.

The database is PostgreSQL (with pgvector). Its connection defaults to the
database compose.yaml starts, so that a plain `uv run pytest` reaches it;
the libpq environment variables (PGHOST, PGPORT, PGUSER, PGPASSWORD)
override them. The package's app is installed, so its models get tables in
the test database.
"""

import os

SECRET_KEY = "tests-only-not-secret"
INSTALLED_APPS = ["django_minimal_rag"]
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "django_minimal_rag",
        "HOST": os.environ.get("PGHOST", "localhost"),
        "PORT": os.environ.get("PGPORT", "5433"),
        "USER": os.environ.get("PGUSER", "postgres"),
        "PASSWORD": os.environ.get("PGPASSWORD", "postgres"),
    },
}
