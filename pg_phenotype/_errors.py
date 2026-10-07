"""Structured exception classes.

Each class carries a stable ``.code`` and an immutable ``.fields`` mapping;
messages are prose and not part of the contract.  Pedigree codes are
pedigree-graph's own (``duplicate_id``, ``cycle``, ...), raised here as
:class:`ValidationError`.
"""

from __future__ import annotations

__all__ = ["ParameterError", "PgPhenotypeError", "ResourceError", "ValidationError"]

from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


class PgPhenotypeError(Exception):
    """Base class: a failure with a stable code and keyword fields.

    Args:
        code: The stable error code.
        message: Human-readable prose.
        **fields: The operands that produced the error.
    """

    code: str
    fields: Mapping[str, object]

    def __init__(self, code: str, message: str, /, **fields: object) -> None:
        super().__init__(message)
        self.code = code
        self.fields = MappingProxyType(dict(fields))


class ValidationError(PgPhenotypeError, ValueError):
    """The pedigree, a trait, or the proband list violates the input contract."""


class ParameterError(PgPhenotypeError, ValueError):
    """A scoring parameter or CIP table lies outside its domain."""


class ResourceError(PgPhenotypeError, MemoryError):
    """A capacity or allocation limit was reached."""
