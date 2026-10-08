"""Embedding backends."""

import zlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

FAKE_EMBEDDINGS_DEFAULT_DIMENSION = 8
FAKE_EMBEDDINGS_MIN_DIMENSION = 2
# Keeps 24 bits: every such integer is exactly representable in float32.
FLOAT32_EXACT_MASK = 0xFFFFFF
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
        if dimensions < FAKE_EMBEDDINGS_MIN_DIMENSION:
            msg = f"dimensions must be at least {FAKE_EMBEDDINGS_MIN_DIMENSION}"
            raise ValueError(msg)
        self.dimensions = dimensions
        self.model = f"fake-{dimensions}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text."""
        return [self._embed_one(text.encode()) for text in texts]

    def _embed_one(self, encoded_text: bytes) -> list[float]:
        return [
            float(zlib.crc32(encoded_text, index) & FLOAT32_EXACT_MASK)
            for index in range(self.dimensions)
        ]


def get_embeddings() -> Embeddings:
    """Return the backend configured by the MINIMAL_RAG_EMBEDDINGS setting."""
    config = _embeddings_config()
    backend = _import_backend(_backend_path(config))
    return backend(**_options(config))


def _embeddings_config() -> Mapping[str, Any]:
    try:
        config: object = getattr(settings, EMBEDDINGS_SETTING)
    except AttributeError as error:
        msg = f"The {EMBEDDINGS_SETTING} setting is not set."
        raise ImproperlyConfigured(msg) from error
    if not isinstance(config, Mapping):
        msg = f"The {EMBEDDINGS_SETTING} setting must be a mapping."
        raise ImproperlyConfigured(msg)
    return config


def _backend_path(config: Mapping[str, Any]) -> str:
    try:
        backend_path: object = config[BACKEND_KEY]
    except KeyError as error:
        msg = f"The {EMBEDDINGS_SETTING} setting has no {BACKEND_KEY}."
        raise ImproperlyConfigured(msg) from error
    if not isinstance(backend_path, str):
        msg = f"The {BACKEND_KEY} of {EMBEDDINGS_SETTING} must be a string."
        raise ImproperlyConfigured(msg)
    return backend_path


def _import_backend(backend_path: str) -> type[Embeddings]:
    try:
        backend: type[Embeddings] = import_string(backend_path)
    except ImportError as error:
        msg = f"Cannot import the embeddings backend {backend_path!r}: {error}"
        raise ImproperlyConfigured(msg) from error
    return backend


def _options(config: Mapping[str, Any]) -> Mapping[str, Any]:
    options: object = config.get(OPTIONS_KEY, {})
    if not isinstance(options, Mapping):
        msg = f"The {OPTIONS_KEY} of {EMBEDDINGS_SETTING} must be a mapping."
        raise ImproperlyConfigured(msg)
    return options
