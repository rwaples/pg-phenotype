"""A NumPy reference for the Relative-Pair Correlation.

Independent of the Rust reduction: the pairs come from pedigree-graph's
Python ``relationship_pairs`` (each symmetric pair stored once), are doubled
here for a symmetric entry, and correlated with NumPy (Pearson, phi) and
``pg_phenotype.correlation.tetrachoric`` on the doubled table, which is the
same fit the method uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pedigree_graph import PedigreeGraph

from pg_phenotype import correlation as pgc

SYMMETRIC = frozenset({"MZ", "FS", "MHS", "PHS", "1C", "H1C", "2C"})


@dataclass(frozen=True)
class Reference:
    n: int
    n_dropped: int
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


def reference(ped: dict[str, np.ndarray], values: np.ndarray, kind: str, members: list[str]) -> Reference:
    """The reference cell of one entry over *members* (all symmetric or all directed)."""
    pairs = graph(ped).relationship_pairs(categories=members)
    first = np.concatenate([np.asarray(pairs[c].first_rows, dtype=np.int64) for c in members])
    second = np.concatenate([np.asarray(pairs[c].second_rows, dtype=np.int64) for c in members])
    symmetric = members[0] in SYMMETRIC
    observed = ~np.isnan(values[first]) & ~np.isnan(values[second])
    n, n_dropped = int(observed.sum()), int((~observed).sum())
    a, b = first[observed], second[observed]
    if symmetric:
        a, b = np.concatenate([a, b]), np.concatenate([b, a])
    x, y = values[a], values[b]
    if kind == "continuous":
        return Reference(n, n_dropped, _pearson(x, y), None, None, None)
    t = np.zeros((2, 2), dtype=np.int64)
    np.add.at(t, (x.astype(np.int64), y.astype(np.int64)), 1)
    table = ((int(t[0, 0]), int(t[0, 1])), (int(t[1, 0]), int(t[1, 1])))
    tet = pgc.tetrachoric(table=table).value
    return Reference(n, n_dropped, None, tet, _phi(t), table)
