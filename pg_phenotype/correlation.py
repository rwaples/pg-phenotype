"""Correlation estimators on their own, without a pedigree.

:func:`tetrachoric` is the tetrachoric correlation of two binary variables,
from a 2 x 2 table or from paired values.  Its fit, boundary flag and SE are
the Mate Correlation's (:mod:`pg_phenotype.assortative`): two-step maximum
likelihood, thresholds from the margins and then rho by Newton with a
bounded-Brent fallback on (-0.9999, 0.9999), and the two-step sandwich SE
with every pair its own cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pg_phenotype import _native
from pg_phenotype._errors import ValidationError
from pg_phenotype._input import floats

__all__ = ["Tetrachoric", "tetrachoric"]

#: The core's largest count (``MAX_COUNT``), the last integer a float64 holds exactly.
_MAX_COUNT = 2**53


@dataclass(frozen=True)
class Tetrachoric:
    """The tetrachoric correlation of one 2 x 2 table.

    ``table`` holds pair counts, rows by the ``x`` level and columns by the
    ``y`` level; ``n`` pairs were counted and ``n_dropped`` left out for a
    missing value.  ``value`` is ``None`` exactly when ``reason`` says why
    there is no estimate (``no_complete_pairs``, ``constant_margin``).
    ``boundary`` is set when rho sits at, or on a likelihood plateau reaching,
    a bound.  ``se`` and ``ci`` (a Wald interval on the Fisher-z scale at
    ``ci_level``) are each ``None`` with a ``*_unavailable_reason``
    (``boundary``, ``sandwich_undefined``, or the estimate's reason).
    """

    estimator: str
    table: tuple[tuple[int, int], tuple[int, int]]
    n: int
    n_dropped: int
    value: float | None
    reason: str | None
    boundary: bool | None
    se: float | None
    se_unavailable_reason: str | None
    ci: tuple[float, float] | None
    ci_unavailable_reason: str | None
    ci_method: str | None
    ci_level: float
    se_method: str


def _count_table(table: object) -> tuple[tuple[int, int], tuple[int, int]]:
    try:
        array = np.asarray(table)
    except ValueError:  # a ragged nesting
        array = np.empty(0)
    if array.shape != (2, 2):
        raise ValidationError(
            "invalid_shape", "table must be 2 x 2", field="table", expected_shape=(2, 2), actual_shape=array.shape
        )
    out = [[0, 0], [0, 0]]
    for (i, j), entry in np.ndenumerate(array):
        v = entry.item() if isinstance(entry, np.generic) else entry
        count = v if isinstance(v, int) and not isinstance(v, bool) else None
        if isinstance(v, float) and v.is_integer():
            count = int(v)
        if count is None or not 0 <= count <= _MAX_COUNT:
            raise ValidationError(
                "invalid_table",
                f"table[{i}, {j}] = {v!r} is not a count (a whole number from 0 to 2^53)",
                field="table",
                row=i,
                column=j,
                value=v,
            )
        out[i][j] = count
    return (out[0][0], out[0][1]), (out[1][0], out[1][1])


def tetrachoric(x: object = None, y: object = None, *, table: object = None) -> Tetrachoric:
    """The tetrachoric correlation of paired binary values, or of a 2 x 2 table.

    Pass either *x* and *y*, or *table*.

    Args:
        x: Binary values (0/1 or bool), NA/None/NaN for missing.
        y: Binary values paired with *x*, the same length.  Pairs with a
            missing value are dropped and counted in ``n_dropped``.
        table: Pair counts ``[[n00, n01], [n10, n11]]``, rows by the ``x``
            level and columns by the ``y`` level.

    Returns:
        A :class:`Tetrachoric`.

    Raises:
        ValidationError: ``pair_length_mismatch`` when *y* is not *x*'s
            length, ``invalid_trait_value`` for a value that is not 0, 1 or
            missing, ``invalid_shape`` for an input of the wrong shape, and
            ``invalid_table`` for a table entry that is not a whole number
            from 0 to 2^53.
        TypeError: Neither or both of (*x*, *y*) and *table*.
    """
    if table is not None:
        if x is not None or y is not None:
            raise TypeError("pass x and y, or table, not both")
        raw = _native.tetrachoric_table(_count_table(table))
    elif x is not None and y is not None:
        raw = _native.tetrachoric_pairs(floats(x, "x"), floats(y, "y"))
    else:
        raise TypeError("pass x and y, or table")
    t = raw.pop("table")
    ci = raw.pop("ci")
    return Tetrachoric(table=(tuple(t[0]), tuple(t[1])), ci=None if ci is None else tuple(ci), **raw)
