"""Models of the package."""

from django.contrib.postgres.fields import ArrayField
from django.db import models
from pgvector.django import VectorField

# A permission name is "<app_label>.<codename>". The bounds mirror Django's own
# columns, ContentType.app_label and Permission.codename (100 characters each);
# they are restated rather than imported because importing those models would
# require django.contrib.contenttypes and django.contrib.auth to be installed.
APP_LABEL_MAX_LENGTH = 100
CODENAME_MAX_LENGTH = 100
PERMISSION_NAME_MAX_LENGTH = APP_LABEL_MAX_LENGTH + len(".") + CODENAME_MAX_LENGTH


class Source(models.Model):
    """A document source, identified by its key."""

    source_key = models.CharField(max_length=255, unique=True)

    def __str__(self) -> str:
        return self.source_key


class Document(models.Model):
    """A document stored for a source."""

    source = models.ForeignKey(Source, on_delete=models.CASCADE)
    title = models.TextField()
    url = models.TextField()
    language = models.TextField(null=True)  # noqa: DJ001 - None means unknown language, distinct from ""
    permissions = ArrayField(
        models.CharField(max_length=PERMISSION_NAME_MAX_LENGTH), default=list
    )

    def __str__(self) -> str:
        return self.title


class Chunk(models.Model):
    """A chunk of a document, with its embedding."""

    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    rank = models.PositiveIntegerField()
    text = models.TextField()
    embedding_model = models.CharField(max_length=255)
    embedding = VectorField()

    def __str__(self) -> str:
        return self.text
