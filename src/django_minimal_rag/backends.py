"""Loading a backend from a setting that names its class and options."""

from collections.abc import Mapping
from typing import Any

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

BACKEND_KEY = "BACKEND"
OPTIONS_KEY = "OPTIONS"


def load_backend(setting_name: str) -> Any:
    """Return the backend a setting configures, built with its options.

    The setting is a mapping with the dotted path of the backend class under
    ``BACKEND`` and, optionally, the keyword arguments to build it with under
    ``OPTIONS``.
    """
    config = _backend_setting(setting_name)
    backend = _import_backend(_backend_path(config, setting_name), setting_name)
    return backend(**_options(config, setting_name))


def _backend_setting(setting_name: str) -> Mapping[str, Any]:
    try:
        config: object = getattr(settings, setting_name)
    except AttributeError as error:
        msg = f"The {setting_name} setting is not set."
        raise ImproperlyConfigured(msg) from error
    if not isinstance(config, Mapping):
        msg = f"The {setting_name} setting must be a mapping."
        raise ImproperlyConfigured(msg)
    return config


def _backend_path(config: Mapping[str, Any], setting_name: str) -> str:
    try:
        backend_path: object = config[BACKEND_KEY]
    except KeyError as error:
        msg = f"The {setting_name} setting has no {BACKEND_KEY}."
        raise ImproperlyConfigured(msg) from error
    if not isinstance(backend_path, str):
        msg = f"The {BACKEND_KEY} of {setting_name} must be a string."
        raise ImproperlyConfigured(msg)
    return backend_path


def _import_backend(backend_path: str, setting_name: str) -> Any:
    try:
        return import_string(backend_path)
    except ImportError as error:
        msg = (
            f"Cannot import the {BACKEND_KEY} of {setting_name}, "
            f"{backend_path!r}: {error}"
        )
        raise ImproperlyConfigured(msg) from error


def _options(config: Mapping[str, Any], setting_name: str) -> Mapping[str, Any]:
    options: object = config.get(OPTIONS_KEY, {})
    if not isinstance(options, Mapping):
        msg = f"The {OPTIONS_KEY} of {setting_name} must be a mapping."
        raise ImproperlyConfigured(msg)
    return options
