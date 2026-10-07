"""Pedigree and trait builders for the ported pedsum #13 assortative-mating tests.

Each builder mirrors a pedsum test helper at 142adf300d5b (``tests/test_assortative_mating.py``,
``tests/test_assortative_permutation.py``); the docstring names it.  A pedigree is a mapping of
``id``, ``mother``, ``father`` int64 arrays: founder parents in id order, then one row per child.
Trait values are one float per row, NaN where unknown.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

Pedigree = dict[str, np.ndarray]


def pedigree(pairs: list[tuple[int, int]], children_per_pair: int = 1) -> Pedigree:
    """Founder parents, then ``children_per_pair`` children of each (mother, father) pair (pedsum ``_pedigree``)."""
    parents = sorted({m for m, _ in pairs} | {f for _, f in pairs})
    ids, mothers, fathers = list(parents), [-1] * len(parents), [-1] * len(parents)
    next_id = max(parents) + 1
    for m, f in pairs:
        for _ in range(children_per_pair):
            ids.append(next_id)
            mothers.append(m)
            fathers.append(f)
            next_id += 1
    return {
        name: np.array(col, dtype=np.int64) for name, col in (("id", ids), ("mother", mothers), ("father", fathers))
    }


def rows(ped: Pedigree, order: np.ndarray) -> Pedigree:
    """The pedigree with its rows in ``order``."""
    return {name: col[order] for name, col in ped.items()}


def column(ped: Pedigree, by_id: dict[int, float], default: float = np.nan) -> np.ndarray:
    """One value per row from ``by_id``, ``default`` for ids it lacks (pedsum ``_trait``)."""
    return np.array([by_id.get(int(i), default) for i in ped["id"]], dtype=np.float64)


def binary(ped: Pedigree, by_id: dict[int, float], cut: float = 0.0) -> np.ndarray:
    """``by_id`` thresholded at ``cut``, coded 0/1 (pedsum ``_binary``)."""
    values = column(ped, by_id)
    return np.where(np.isnan(values), np.nan, (values > cut).astype(np.float64))


def ordinal(ped: Pedigree, by_id: dict[int, float], cuts: list[float]) -> np.ndarray:
    """``by_id`` binned at ``cuts``, coded ``0..len(cuts)`` (pedsum ``_ordinal``)."""
    values = column(ped, by_id)
    return np.where(np.isnan(values), np.nan, np.digitize(values, cuts).astype(np.float64))


def decades(ped: Pedigree, years: dict[int, int], default: int = 2000) -> np.ndarray:
    """Birth-year strata at ``birth_year_bin=10``, ``-1`` unknown (pedsum ``_with_birth_years`` then ``strata``)."""
    year = np.array([years.get(int(i), default) for i in ped["id"]], dtype=np.int64)
    return np.where(year == -1, np.nan, year // 10 * 10)


def pair_values(rng: np.random.Generator, n_pairs: int, remate_every: int = 3) -> tuple[list[tuple[int, int]], dict]:
    """Pairs with one mother each; every ``remate_every``-th pair reuses the previous father (pedsum ``_pair_values``)."""
    pairs, father = [], 1000
    for k in range(n_pairs):
        if k % remate_every:
            father += 1
        pairs.append((k + 1, father))
    ids = sorted({p for pair in pairs for p in pair})
    return pairs, dict(zip(ids, rng.normal(size=len(ids)).tolist(), strict=True))


def one_to_one(n_pairs: int) -> list[tuple[int, int]]:
    """``n_pairs`` pairs, no remating (pedsum ``_one_to_one``)."""
    return [(k + 1, 100_000 + k) for k in range(n_pairs)]


def stratified_fixture(seed: int, n_pairs: int = 80) -> tuple[Pedigree, list[tuple[int, int]], dict, dict, np.ndarray]:
    """Remating pairs with mates born over three decades: ``(ped, pairs, values, years, strata)`` (pedsum ``_stratified_fixture``)."""
    rng = np.random.default_rng(seed)
    pairs, values = pair_values(rng, n_pairs)
    ped = pedigree(pairs)
    years = {i: int(rng.integers(1950, 1980)) for i in values}
    return ped, pairs, values, years, decades(ped, years)


def frame(seed: int, n_pairs: int, remate_every: int = 3) -> tuple[Pedigree, np.ndarray, np.ndarray, np.ndarray]:
    """Remating pairs, birth decades, a continuous ``x`` and a binary ``b``: ``(ped, x, b, strata)`` (pedsum ``_frame``)."""
    rng = np.random.default_rng(seed)
    pairs, father = [], 1000
    for k in range(n_pairs):
        if k % remate_every:
            father += 1
        pairs.append((k + 1, father))
    ped = pedigree(pairs)
    n_parents = len({m for m, _ in pairs} | {f for _, f in pairs})
    year = np.array([int(rng.integers(1950, 1980)) for _ in range(n_parents)] + [2000] * len(pairs))
    n = ped["id"].size
    x = rng.normal(size=n)
    x[rng.random(n) < 0.1] = np.nan
    b = np.where(np.isnan(x), np.nan, (x + rng.normal(size=n) > 0.5).astype(float))
    b[rng.random(n) < 0.1] = np.nan
    return ped, x, b, (year // 10 * 10).astype(np.float64)


def plain(obj: object) -> object:
    """A result as JSON-shaped data: dataclasses and mappings to dicts, tuples to lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [plain(v) for v in obj]
    return obj
