"""Indexing helpers for tests run inside a test transaction, which never commits."""

from collections.abc import Mapping, Sequence

from django.test import TestCase

from django_minimal_rag.documents import Document
from django_minimal_rag.indexing import Indexer


def replace_and_run_on_commit(groups: Mapping[str, Sequence[Document]]) -> None:
    """Call ``Indexer().replace(groups)``, then run what it deferred to the commit.

    Inside a test transaction, ``transaction.on_commit()`` callbacks never run
    on their own; an exception from ``replace()`` propagates as usual.
    """
    with TestCase.captureOnCommitCallbacks(execute=True):
        Indexer().replace(groups)
