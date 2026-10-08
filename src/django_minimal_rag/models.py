"""Models of the package."""

from django.db import models


class Source(models.Model):
    """A document source, identified by its key."""

    source_key = models.CharField(max_length=255, unique=True)

    def __str__(self) -> str:
        return self.source_key
