"""Django settings for the test suite, also loaded by the django-stubs mypy plugin.

The database is PostgreSQL (with pgvector). Only its ENGINE and NAME are
declared: HOST, PORT, USER and PASSWORD are left unset so that libpq reads
them from the environment (PGHOST, PGPORT, PGUSER, PGPASSWORD). The
package's app is added with the first models.
"""

SECRET_KEY = "tests-only-not-secret"
INSTALLED_APPS: list[str] = []
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "django_minimal_rag",
        "HOST": "localhost",
        "PORT": "5433",
        "USER": "postgres",
        "PASSWORD": "postgres",
    },
}
