"""The Relative-Pair SE's per-entry influences (``influence=True``, issue #25) against the NumPy reference."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from pg_phenotype import ResourceError, Trait
from pg_phenotype import relative_pairs as rp
from tests.oracle.relative_pairs_reference import _reader
from tests.oracle.relative_pairs_se import parent_indices
from tests.pedigrees import random_pedigree
from tests.test_relative_pairs_se import GROUPS, _estimate_units, _pairs, _strata, _traits

#: Each entry's influence against the reference's: Pearson and phi to rounding, the tetrachoric to its fit's.
TOL = {"pearson": 1e-10, "phi": 1e-10, "tetrachoric": 1e-5}


def _keys(ped: dict[str, np.ndarray], mrca_id: np.ndarray) -> np.ndarray:
    """Each MRCA id as the reference's key: a row's index, an external parent's ``n + its rank``, -1 for none."""
    ids = np.asarray(ped["id"], dtype=np.int64)
    parents = np.concatenate([np.asarray(ped[k], dtype=np.int64) for k in ("mother", "father")])
    external = np.unique(parents[(parents >= 0) & ~np.isin(parents, ids)])
    key = {int(i): r for r, i in enumerate(ids)} | {int(p): len(ids) + r for r, p in enumerate(external)}
    return np.array([key.get(int(m), -1) for m in mrca_id], dtype=np.int64)


def _se(entries: rp.Entries, influence: np.ndarray) -> float:
    """√ of the inclusion-exclusion meat over *entries*, a pair with no MRCA its own cluster (ADR 0013)."""
    a, b, m = entries.first_id, entries.second_id, entries.mrca_id
    lone = m == -1
    cluster = np.where(lone, -1 - np.minimum(a, b), m), np.where(lone, np.maximum(a, b), -1)
    cluster = np.unique(np.stack(cluster, axis=1), axis=0, return_inverse=True)[1].ravel()

    def squares(*keys: np.ndarray, phi: np.ndarray) -> float:
        _, at = np.unique(np.stack(keys, axis=1), axis=0, return_inverse=True)
        return float((np.bincount(at.ravel(), weights=phi) ** 2).sum())

    twice = np.concatenate([influence, influence])
    member, both = np.concatenate([a, b]), np.concatenate([cluster, cluster])
    meat = squares(cluster, phi=influence) + squares(member, phi=twice) - squares(member, both, phi=twice)
    return float(np.sqrt(meat))


def _check(res: rp.RelativePairCorrelation, ped: dict[str, np.ndarray], values, kind: str) -> Counter[str]:
    """Every cell's entries and pooled influences against the reference, and every SE against its entries."""
    row = {int(i): r for r, i in enumerate(ped["id"])}
    checked: Counter[str] = Counter()
    for cell in res.cells:
        where = f"{cell.category} {cell.first_trait}x{cell.second_trait}"
        entries = cell.entries
        assert entries is not None, where
        n = cell.n_entries if cell.n_entries is not None else (2 * cell.n if cell.symmetric else cell.n)
        assert len(entries) == n, where
        first = np.array([row[int(i)] for i in entries.first_id], dtype=np.int64)
        second = np.array([row[int(i)] for i in entries.second_id], dtype=np.int64)
        # The reference's entries, keys and influences, matched to ours by (first row, second row).
        a, b, m = _pairs(ped, list(cell.members))
        x = values[cell.first_trait]
        y = None if cell.first_trait == cell.second_trait else values[cell.second_trait]
        units = _estimate_units(_reader(x, y, kind, cell.symmetric), a, b, m)
        ours = np.lexsort((second, first))
        for estimate in cell.crude:
            unit = units[estimate.estimator]
            if unit is None or estimate.influence is None:
                assert unit is None or estimate.value is None, (where, estimate)
                continue
            r_first, r_second = unit.members
            theirs = np.lexsort((r_second, r_first))
            assert np.array_equal(first[ours], r_first[theirs]), where
            assert np.array_equal(second[ours], r_second[theirs]), where
            assert np.array_equal(_keys(ped, entries.mrca_id)[ours], unit.mrca[theirs]), where
            phi, psi = estimate.influence[ours], unit.psi[theirs]
            tol = TOL[estimate.estimator] * np.abs(psi).max()
            assert np.abs(phi - psi).max() <= tol, (where, estimate.estimator)
            checked["influence"] += 1
        estimates = [*cell.crude, *(e for s in (cell.strata or {}).values() for e in s.crude)]
        if cell.stratified is not None:
            estimates.append(cell.stratified.estimate)
        for estimate in estimates:
            if estimate.influence is None:
                assert estimate.se_reason in {"no_estimate", "boundary", "sandwich_undefined"}, (where, estimate)
                continue
            assert estimate.influence.shape == (n,), where
            if estimate.se is not None:
                assert _se(entries, estimate.influence) == pytest.approx(estimate.se, rel=1e-9), (where, estimate)
                checked["se"] += 1
    return checked


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
@pytest.mark.parametrize("n_traits", [1, 2])
def test_the_entries_and_influences_match_the_reference(seed, kind, n_traits):
    ped = random_pedigree(seed, generations=5)
    n = len(ped["id"])
    assert len(parent_indices(ped)[0]) > n, "the fixture has an external parent"
    values = _traits(seed, n, kind, n_traits)
    traits = [Trait(v, kind=kind) for v in values]
    checked: Counter[str] = Counter()
    for categories in (None, GROUPS):
        res = rp.correlation(ped, traits, categories=categories, stratum=_strata(seed, n), se=True, influence=True)
        assert res.settings["influence"] is True
        checked += _check(res, ped, values, kind)
        if n_traits == 2:
            assert res.within_person.entries is None
            assert all(e.influence is None for e in res.within_person.crude)
    assert checked["influence"] >= 10, checked
    assert checked["se"] >= 40, checked


def test_influence_is_opt_in_and_leaves_the_result_unchanged():
    ped = random_pedigree(0, generations=4)
    traits = [Trait(v, kind="binary") for v in _traits(0, len(ped["id"]), "binary", 2)]
    off = rp.correlation(ped, traits, se=True, covariance=True)
    on = rp.correlation(ped, traits, se=True, covariance=True, influence=True)
    assert on.cells == off.cells
    assert np.array_equal(on.covariance.pooled, off.covariance.pooled, equal_nan=True)
    assert (off.settings["influence"], on.settings["influence"]) == (False, True)
    assert all(c.entries is None and all(e.influence is None for e in c.crude) for c in off.cells)
    cell = on.cell("FS")
    assert cell.entries.first_id.dtype == np.int64
    assert cell.primary.influence.dtype == np.float64
    for array in (cell.entries.first_id, cell.entries.mrca_id, cell.primary.influence):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0


def test_influence_needs_the_se_and_a_bool():
    ped = random_pedigree(0, generations=3)
    trait = Trait(_traits(0, len(ped["id"]), "continuous", 1)[0], kind="continuous")
    with pytest.raises(ValueError, match="influence=True needs se=True"):
        rp.correlation(ped, trait, influence=True)
    with pytest.raises(TypeError, match="influence must be a bool"):
        rp.correlation(ped, trait, se=True, influence=1)


def test_entries_that_do_not_fit_the_budget_are_refused():
    """With two binary traits and strata the entries outgrow the SE: at the smallest budget they fit, the SE
    alone fits one byte below it, and the entries are refused there with their operation."""
    ped = random_pedigree(5, n_founders=300, generations=6, size=800)
    n = len(ped["id"])
    traits = [Trait(v, kind="binary") for v in _traits(0, n, "binary", 2)]

    def call(budget: int, influence: bool) -> None:
        rp.correlation(ped, traits, stratum=_strata(0, n), se=True, influence=influence, memory_budget_bytes=budget)

    lo, hi = 1, 1 << 30
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            call(mid, influence=True)
            hi = mid
        except ResourceError:
            lo = mid
    call(lo, influence=False)
    with pytest.raises(ResourceError) as refused:
        call(lo, influence=True)
    assert refused.value.code == "memory_budget_exceeded"
    assert refused.value.fields["operation"] == "relative_pairs_influence"
