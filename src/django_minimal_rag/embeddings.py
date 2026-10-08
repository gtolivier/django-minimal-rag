"""Embedding backends."""

import zlib
from collections.abc import Sequence
from typing import Any, Protocol

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

FAKE_EMBEDDINGS_DEFAULT_DIMENSION = 8
EMBEDDINGS_SETTING = "MINIMAL_RAG_EMBEDDINGS"
BACKEND_KEY = "BACKEND"
OPTIONS_KEY = "OPTIONS"


class Embeddings(Protocol):
    """An embedding backend."""

    @property
    def model(self) -> str:
        """Name of the embedding model."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text."""
        ...


class FakeEmbeddings:
    """Deterministic embeddings for tests."""

    def __init__(self, dimensions: int = FAKE_EMBEDDINGS_DEFAULT_DIMENSION) -> None:
        if dimensions <= 0:
            msg = "dimensions must be positive"
            raise ValueError(msg)
        self.dimensions = dimensions
        self.model = f"fake-{dimensions}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text."""
        return [self._embed_one(text.encode()) for text in texts]

    def _embed_one(self, encoded_text: bytes) -> list[float]:
        return [
            float(zlib.crc32(encoded_text, index)) for index in range(self.dimensions)
        ]


def get_embeddings() -> Embeddings:
    """Return the backend configured by the MINIMAL_RAG_EMBEDDINGS setting."""
    config = _embeddings_config()
    backend = _import_backend(_backend_path(config))
    return backend(**config.get(OPTIONS_KEY, {}))


def _embeddings_config() -> dict[str, Any]:
    try:
        config: dict[str, Any] = getattr(settings, EMBEDDINGS_SETTING)
    except AttributeError as error:
        msg = f"The {EMBEDDINGS_SETTING} setting is not set."
        raise ImproperlyConfigured(msg) from error
    return config


def _backend_path(config: dict[str, Any]) -> str:
    try:
        backend_path: str = config[BACKEND_KEY]
    except KeyError as error:
        msg = f"The {EMBEDDINGS_SETTING} setting has no {BACKEND_KEY}."
        raise ImproperlyConfigured(msg) from error
    return backend_path


def _import_backend(backend_path: str) -> type[Embeddings]:
    try:
        backend: type[Embeddings] = import_string(backend_path)
    except ImportError as error:
        msg = f"Cannot import the embeddings backend {backend_path!r}: {error}"
        raise ImproperlyConfigured(msg) from error
    return backend
