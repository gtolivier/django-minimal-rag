"""Sequences a host project may hand to this package, beyond plain lists."""

from collections.abc import Iterable, Iterator, Sequence
from typing import TypeVar, overload

T = TypeVar("T")


class SinglePassSequence(Sequence[T]):
    """A sequence whose iteration yields its items the first time only.

    Its length and indexing stay correct; any later iteration yields nothing,
    like a sequence wrapping a one-shot source.
    """

    def __init__(self, items: Iterable[T]) -> None:
        self._items = list(items)
        self._iterated = False

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        return self._items[index]

    def __iter__(self) -> Iterator[T]:
        if self._iterated:
            return iter(())
        self._iterated = True
        return iter(self._items)
