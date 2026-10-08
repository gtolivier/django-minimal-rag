"""Models of the package."""

from django.db import models


class Source(models.Model):
    """A document source, identified by its key."""

    source_key = models.CharField(max_length=255, unique=True)

    def __str__(self) -> str:
        return self.source_key


class Document(models.Model):
    """A document stored for a source."""

    source = models.ForeignKey(Source, on_delete=models.CASCADE)
    title = models.CharField(max_length=2000)
    url = models.URLField(max_length=2000)
    language = models.CharField(max_length=16, null=True)  # noqa: DJ001 - None means unknown language, distinct from ""

    def __str__(self) -> str:
        return self.title
