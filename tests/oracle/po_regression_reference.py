"""A NumPy/SciPy reference for parent-offspring regression.

Independent of the Rust code: rows and parents are matched by id in a
dict, Mate Networks are a union-find over every row joined by each child's
mother and father (so a parent in no Mating Pair stays a set of its own),
the line is ``scipy.stats.linregress``, and the sandwich is the matrix form
``G/(G-1) (X'X)^-1 (sum_g X_g'e_g e_g'X_g) (X'X)^-1`` for X = [1, x].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

PARENTS = ("midparent", "mother", "father")


@dataclass(frozen=True)
class Reference:
    n: int
    n_mate_networks: int
    dropped: tuple[int, int, int]
    slope: float | None
    intercept: float | None
    r: float | None
    se_ols: float | None
    se: float | None


def _networks(mother_row: list[int | None], father_row: list[int | None]) -> list[int]:
    root = list(range(len(mother_row)))

    def find(a: int) -> int:
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    for m, f in zip(mother_row, father_row, strict=True):
        if m is not None and f is not None:
            root[find(m)] = find(f)
    return [find(a) for a in range(len(root))]


def _sandwich(x: np.ndarray, y: np.ndarray, clusters: list[int]) -> float | None:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    keys = sorted(set(clusters))
    g = len(keys)
    if g < 2:
        return None
    meat = np.zeros((2, 2))
    labels = np.array(clusters)
    for k in keys:
        s = X[labels == k].T @ e[labels == k]
        meat += np.outer(s, s)
    v = g / (g - 1) * bread @ meat @ bread
    return float(np.sqrt(v[1, 1]))


def regress(ped: dict[str, np.ndarray], values: np.ndarray, rows: np.ndarray | None = None) -> dict[str, Reference]:
    """The three regressions over the children in *rows* (every row when None)."""
    ids = [int(i) for i in ped["id"]]
    where = {i: r for r, i in enumerate(ids)}
    mother_row = [where.get(int(m)) for m in ped["mother"]]
    father_row = [where.get(int(f)) for f in ped["father"]]
    network = _networks(mother_row, father_row)
    rows = np.arange(len(ids)) if rows is None else rows
    out = {}
    for parent in PARENTS:
        x, y, clusters = [], [], []
        dropped = [0, 0, 0]
        for c in rows:
            m, f = mother_row[c], father_row[c]
            need = {"midparent": (m, f), "mother": (m,), "father": (f,)}[parent]
            if any(p is None for p in need):
                dropped[0] += 1
                continue
            if np.isnan(values[c]):
                dropped[1] += 1
                continue
            xv = float(np.mean([values[p] for p in need]))
            if np.isnan(xv):
                dropped[2] += 1
                continue
            x.append(xv)
            y.append(float(values[c]))
            clusters.append(network[need[0]])
        xa, ya = np.array(x), np.array(y)
        g = len(set(clusters))
        defined = len(x) > 0 and np.ptp(xa) > 0
        if not defined:
            out[parent] = Reference(len(x), g, tuple(dropped), None, None, None, None, None)
            continue
        fit = stats.linregress(xa, ya)
        constant_child = np.ptp(ya) == 0
        out[parent] = Reference(
            n=len(x),
            n_mate_networks=g,
            dropped=tuple(dropped),
            slope=float(fit.slope),
            intercept=float(fit.intercept),
            r=None if constant_child else float(fit.rvalue),
            se_ols=float(fit.stderr) if len(x) >= 3 else None,
            se=_sandwich(xa, ya, clusters),
        )
    return out
