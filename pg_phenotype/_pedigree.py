"""A validated pedigree that methods share."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from pg_phenotype import _native
from pg_phenotype._input import pedigree_arrays

if TYPE_CHECKING:
    import numpy as np

    from pg_phenotype._input import PedigreeArrays


class Pedigree:
    """A pedigree validated once, for several methods or calls.

    ``mate_correlation`` and ``pafgrs.prepare`` take a ``Pedigree`` wherever
    they take pedigree columns.  It keeps the Mating Pairs and Mate Networks
    the first ``mate_correlation`` finds, so a later call skips validation
    and finding them.

    Row *i* of the Pedigree is row *i* of *pedigree*.  Traits, ages, strata
    and probands are matched to it by position: align values from another
    table with :attr:`ids` first.  A misaligned trait is not detected.

    A Pedigree lives in memory only; it cannot be pickled.

    Args:
        pedigree: Columns ``id``, ``mother``, ``father`` (``-1`` or NA when
            missing) and optional ``twin``, ``sex`` (``0`` female, ``1`` male,
            ``-1`` unknown), as a polars or pandas frame or a mapping of arrays.

    Raises:
        ValidationError: The pedigree is invalid.
        ResourceError: A capacity or allocation limit was reached.
    """

    __slots__ = ("_native",)

    def __init__(self, pedigree: object) -> None:
        self._native = _native.Pedigree(*pedigree_arrays(pedigree))

    def __len__(self) -> int:
        return len(self._native)

    @property
    def ids(self) -> np.ndarray:
        """The row ids in row order, as a new read-only int64 array."""
        ids = self._native.ids
        ids.flags.writeable = False
        return ids

    def __repr__(self) -> str:
        return f"<Pedigree: {len(self)} rows>"

    def __reduce__(self) -> NoReturn:
        raise TypeError("a Pedigree cannot be pickled; build it again from its columns")


def native_pedigree(pedigree: object) -> _native.Pedigree | PedigreeArrays:
    """What a native method takes: a built Pedigree, or the coerced columns."""
    return pedigree._native if isinstance(pedigree, Pedigree) else pedigree_arrays(pedigree)
