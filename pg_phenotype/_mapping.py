"""The read-only mapping every result and error exposes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


class ReadOnlyMapping(Mapping[Any, Any]):
    """A read-only copy of a dict that, unlike ``types.MappingProxyType``, pickles.

    A worker process returns results and raises errors by pickling them
    (issues #28, #29); this unpickles back to a read-only mapping.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[Any, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __reversed__(self) -> Iterator[Any]:
        return reversed(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def __or__(self, other: object) -> dict[Any, Any]:
        if not isinstance(other, Mapping):
            return NotImplemented
        return self._data | dict(other)

    def __ror__(self, other: object) -> dict[Any, Any]:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(other) | self._data

    def copy(self) -> dict[Any, Any]:
        """A shallow, writable copy as a dict, as ``MappingProxyType.copy`` gives."""
        return self._data.copy()

    def __reduce__(self) -> tuple[object, ...]:
        return (type(self), (self._data,))
