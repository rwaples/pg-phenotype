"""A validated pedigree that methods share."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from pg_phenotype import _native
from pg_phenotype._input import floats, id_array, pedigree_arrays

if TYPE_CHECKING:
    import numpy as np

    from pg_phenotype._input import PedigreeArrays


class Pedigree:
    """A pedigree validated once, for several methods or calls.

    ``mate_correlation`` and ``pafgrs.prepare`` take a ``Pedigree`` wherever
    they take pedigree columns.  It keeps the Mating Pairs the first
    ``mate_correlation`` finds and the Mate Networks the first with inference
    builds, so a later call skips validation and finding them.

    Row *i* of the Pedigree is row *i* of *pedigree*.  Traits, ages, strata
    and probands are matched to it by position: move values from another
    table onto its rows with :meth:`align` (or :meth:`rows`) first.  A
    misaligned trait is not detected.

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

    def rows(self, ids: object) -> np.ndarray:
        """The row of each of *ids*.

        Builds an id index on first use and keeps it (12 bytes per row).

        Args:
            ids: Pedigree ids, each at most once, as an array or series.

        Returns:
            A new int64 array, one row per id.

        Raises:
            ValidationError: ``invalid_integer_value`` for an id that is
                missing or not an integer, ``unknown_id`` for an id that is
                not a row's, ``duplicate_lookup_id`` for an id listed twice.
        """
        return self._native.rows(id_array(ids, "ids"))

    def align(self, ids: object, values: object) -> np.ndarray:
        """*values*, given per *ids*, on the Pedigree's rows.

        Example::

            dx = ped.align(pheno["id"], pheno["dx"])  # NaN where a row has no value
            trait = pgp.Trait(dx, kind="binary")

        Args:
            ids: Pedigree ids, each at most once, as :meth:`rows` takes them.
            values: One number per id; NA or None becomes NaN.

        Returns:
            A new float64 array, one value per row, NaN for a row no id names.

        Raises:
            ValidationError: ``pair_length_mismatch`` when *values* is not
                *ids*'s length, and the errors of :meth:`rows`.
        """
        return self._native.align(id_array(ids, "ids"), floats(values, "values"))

    def __repr__(self) -> str:
        return f"<Pedigree: {len(self)} rows>"

    def __reduce__(self) -> NoReturn:
        raise TypeError("a Pedigree cannot be pickled; build it again from its columns")


def native_pedigree(pedigree: object) -> _native.Pedigree | PedigreeArrays:
    """What a native method takes: a built Pedigree, or the coerced columns."""
    return pedigree._native if isinstance(pedigree, Pedigree) else pedigree_arrays(pedigree)
