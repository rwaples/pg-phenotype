"""A NumPy reference for the Relative-Pair Correlation.

Independent of the Rust reduction: the pairs come from pedigree-graph's
Python ``relationship_pairs`` (each symmetric pair stored once), are doubled
here for a symmetric entry, and correlated with NumPy (Pearson, phi) and
``pg_phenotype.correlation.tetrachoric`` on the doubled table, which is the
same fit the method uses.

A cell reads ``x`` in each pair's first member and ``y`` in the second.  A
symmetric pair is read in both orientations, each kept where its two values
are observed, so a cross-trait cell's entries are the orientations complete;
a same-trait cell (``y`` omitted) keeps both or neither.  The Within-Person
Cross-Trait Correlation is each row paired with itself, read once.

With strata, :func:`stratified_reference` assigns the same pairs to strata
and fits the stratified form with the assortative-mating oracle's
``polychoric`` (bounded Brent, its own thresholds and bivariate normal) and
``stratified_pearson``, which share no code with the Rust fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pedigree_graph import RELATIONSHIPS, PedigreeGraph

from pg_phenotype import correlation as pgc
from tests.oracle.assortative_reference import (
    CellPairs,
    Fit,
    degenerate_strata,
    polychoric,
    shown_levels,
    stratified_pearson,
)

#: pedigree-graph's symmetric categories: those with no roles.
SYMMETRIC = frozenset(code for code, category in RELATIONSHIPS.items() if category.first_role is None)


@dataclass(frozen=True)
class Reference:
    n: int
    n_dropped: int
    #: A symmetric cross-trait cell's complete orientations; None otherwise.
    n_entries: int | None
    pearson: float | None
    tetrachoric: float | None
    phi: float | None
    #: The table the fit uses: doubled for a symmetric entry.
    table: tuple[tuple[int, int], tuple[int, int]] | None


def graph(ped: dict[str, np.ndarray]) -> PedigreeGraph:
    return PedigreeGraph.from_arrays(
        ids=ped["id"], mother_ids=ped["mother"], father_ids=ped["father"], twin_ids=ped.get("twin"), sex=ped.get("sex")
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) == 0:
        return None
    dx, dy = x - x.mean(), y - y.mean()
    sxx, syy = float(dx @ dx), float(dy @ dy)
    if sxx == 0 or syy == 0:
        return None
    return float(dx @ dy) / np.sqrt(sxx * syy)


def _phi(t: np.ndarray) -> float | None:
    r0, r1, c0, c1 = t[0].sum(), t[1].sum(), t[:, 0].sum(), t[:, 1].sum()
    if 0 in (r0, r1, c0, c1):
        return None
    return float(t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]) / (np.sqrt(float(r0 * r1)) * np.sqrt(float(c0 * c1)))


def _pairs(ped: dict[str, np.ndarray], members: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pairs = graph(ped).relationship_pairs(categories=members)
    first = np.concatenate([np.asarray(pairs[c].first_rows, dtype=np.int64) for c in members])
    second = np.concatenate([np.asarray(pairs[c].second_rows, dtype=np.int64) for c in members])
    return first, second


@dataclass(frozen=True)
class _Read:
    """How a cell reads its pairs: ``x`` in the first member, ``y`` in the second."""

    x: np.ndarray
    y: np.ndarray
    kind: str
    symmetric: bool
    cross: bool

    def orientations(self, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per pair, whether it is complete read forward and (symmetric only) reversed."""
        forward = ~np.isnan(self.x[first]) & ~np.isnan(self.y[second])
        if not self.symmetric:
            return forward, np.zeros_like(forward)
        return forward, ~np.isnan(self.x[second]) & ~np.isnan(self.y[first])

    def entries(self, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The complete orientations as (first rows, second rows, pair index)."""
        forward, reverse = self.orientations(first, second)
        index = np.arange(len(first))
        return (
            np.concatenate([first[forward], second[reverse]]),
            np.concatenate([second[forward], first[reverse]]),
            np.concatenate([index[forward], index[reverse]]),
        )


def _reader(x: np.ndarray, y: np.ndarray | None, kind: str, symmetric: bool) -> _Read:
    return _Read(x, x if y is None else y, kind, symmetric, symmetric and y is not None)


def reference(
    ped: dict[str, np.ndarray], x: np.ndarray, kind: str, members: list[str], y: np.ndarray | None = None
) -> Reference:
    """The reference cell of one entry over *members* (all symmetric or all directed).

    The first member reads *x* and the second *y*; *y* omitted reads *x* in
    both, a same-trait cell.
    """
    first, second = _pairs(ped, members)
    return _cell(first, second, _reader(x, y, kind, members[0] in SYMMETRIC))


def within_person_reference(x: np.ndarray, y: np.ndarray, kind: str) -> Reference:
    """The Within-Person Cross-Trait Correlation: each row's *x* against its *y*."""
    rows = np.arange(len(x))
    return _cell(rows, rows, _Read(x, y, kind, symmetric=False, cross=False))


def _cell(first: np.ndarray, second: np.ndarray, read: _Read) -> Reference:
    forward, reverse = read.orientations(first, second)
    complete = forward | reverse
    n, n_dropped = int(complete.sum()), int((~complete).sum())
    n_entries = int(forward.sum() + reverse.sum()) if read.cross else None
    a, b, _ = read.entries(first, second)
    x, y = read.x[a], read.y[b]
    if read.kind == "continuous":
        return Reference(n, n_dropped, n_entries, _pearson(x, y), None, None, None)
    t = np.zeros((2, 2), dtype=np.int64)
    np.add.at(t, (x.astype(np.int64), y.astype(np.int64)), 1)
    table = ((int(t[0, 0]), int(t[0, 1])), (int(t[1, 0]), int(t[1, 1])))
    tet = pgc.tetrachoric(table=table).value
    return Reference(n, n_dropped, n_entries, None, tet, _phi(t), table)


@dataclass(frozen=True)
class StratifiedReference:
    #: Each known stratum label to its cell's reference.
    strata: dict[int, Reference]
    n_cross_stratum: int
    n_unknown_stratum: int
    #: The stratified fit: unordered pairs fitted and left out, its value or why not.
    n: int
    #: A symmetric cross-trait cell's complete orientations fitted, else None.
    n_entries: int | None
    n_degenerate_stratum: int
    value: float | None
    reason: str | None


def stratified_reference(
    ped: dict[str, np.ndarray],
    x: np.ndarray,
    kind: str,
    members: list[str],
    stratum: np.ndarray,
    y: np.ndarray | None = None,
) -> StratifiedReference:
    """The reference strata of one entry, *stratum* one label per row, NaN unknown; *x*, *y* as :func:`reference`."""
    first, second = _pairs(ped, members)
    return _stratified(first, second, _reader(x, y, kind, members[0] in SYMMETRIC), stratum)


def within_person_stratified_reference(
    x: np.ndarray, y: np.ndarray, kind: str, stratum: np.ndarray
) -> StratifiedReference:
    """:func:`within_person_reference` by each row's stratum."""
    rows = np.arange(len(x))
    return _stratified(rows, rows, _Read(x, y, kind, symmetric=False, cross=False), stratum)


def _stratified(first: np.ndarray, second: np.ndarray, read: _Read, stratum: np.ndarray) -> StratifiedReference:
    known = ~np.isnan(stratum)
    labels = np.unique(stratum[known])
    s1, s2 = stratum[first], stratum[second]
    strata = {}
    for label in labels:
        assigned = (s1 == label) & (s2 == label) if read.symmetric else s1 == label
        strata[int(label)] = _cell(first[assigned], second[assigned], read)
    forward, reverse = read.orientations(first, second)
    complete = forward | reverse
    both_known = known[first] & known[second]
    if read.symmetric:
        n_cross = int((complete & both_known & (s1 != s2)).sum())
        n_unknown = int((complete & ~both_known).sum())
    else:
        n_cross, n_unknown = 0, int((complete & ~known[first]).sum())
    a, b, pair = read.entries(first[both_known], second[both_known])
    n_known = len(np.unique(pair))
    code_a, code_b = np.searchsorted(labels, stratum[a]), np.searchsorted(labels, stratum[b])
    while True:
        constant = degenerate_strata(read.x[a], code_a)[code_a] | degenerate_strata(read.y[b], code_b)[code_b]
        if not constant.any():
            break
        a, b, pair = a[~constant], b[~constant], pair[~constant]
        code_a, code_b = code_a[~constant], code_b[~constant]
    n = len(np.unique(pair))
    value = reason = None
    if n_known == 0:
        reason = "no_complete_pairs"
    elif n == 0:
        reason = "constant_value" if read.kind == "continuous" else "constant_margin"
    elif read.kind == "continuous":
        value = stratified_pearson(CellPairs(read.x[a], read.y[b], code_a, code_b))
    else:
        x, y = read.x[a], read.y[b]
        fit = polychoric(CellPairs(x, y, code_a, code_b, shown_levels(x, code_a, 2), shown_levels(y, code_b, 2)))
        assert isinstance(fit, Fit), fit
        value = fit.value
    assert value is None or isinstance(value, float), value
    n_entries = len(a) if read.cross else None
    return StratifiedReference(strata, n_cross, n_unknown, n, n_entries, n_known - n, value, reason)
