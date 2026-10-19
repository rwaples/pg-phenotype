"""The covariance of the Relative-Pair estimates the ACE Fit reads (ADR 0014) against a NumPy reference."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from pg_phenotype import ResourceError, Trait
from pg_phenotype import correlation as pgc
from pg_phenotype import relative_pairs as rp
from tests.oracle.relative_pairs_reference import _reader, graph
from tests.oracle.relative_pairs_se import (
    KERNEL,
    Units,
    brute_cross,
    cross_meat,
    kin_meat,
    meats,
    within_person_psi,
)
from tests.pedigrees import random_pedigree
from tests.test_relative_pairs_se import (
    FLOOR,
    GROUPS,
    TOL,
    _close,
    _estimate_units,
    _pairs,
    _strata,
    _stratified_pearson,
    _traits,
)

FORMS = ("pooled", "phi", "stratified")
#: Each form's within-person estimate in :func:`within_person_psi`'s keys, by estimator.
WITHIN_SCOPE = {"pooled": "pooled", "phi": "pooled", "stratified": "stratified"}


def _estimate(cell: rp.Cell, form: str) -> rp.Estimate | None:
    """The estimate of *cell* that *form*'s matrix covers; ``None`` when the cell has no such form."""
    if form == "pooled":
        return cell.crude[0]
    if form == "phi":
        return next((e for e in cell.crude if e.estimator == "phi"), None)
    return None if cell.stratified is None else cell.stratified.estimate


def _kernel_pairs(ped) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each kernel category's relative pairs (rows a, b), each listed once."""
    pairs = graph(ped).relationship_pairs(categories=list(KERNEL))
    return [
        tuple(np.asarray(getattr(pairs[c], f), dtype=np.int64) for f in ("first_rows", "second_rows")) for c in KERNEL
    ]


def _tetrachoric(table: np.ndarray) -> tuple[float | None, bool | None]:
    fit = pgc.tetrachoric(table=table.tolist())
    return fit.value, fit.boundary


def _within(values, kind, stratum) -> dict:
    return within_person_psi(values[0], values[1], kind, stratum, _tetrachoric)


def _within_units(res, values, kind, stratum) -> dict[str, tuple[Units, np.ndarray] | None]:
    """Per form, the within-person estimate's entries (rows, joined on members alone) and its influence per
    row; ``None`` where the reference has none to compare (a binary stratified fit, or one that left a
    constant stratum out)."""
    ref = _within(values, kind, stratum)
    out: dict[str, tuple[Units, np.ndarray] | None] = {}
    for form in FORMS:
        est = _estimate(res.within_person, form)
        name = "phi" if form == "phi" else ("pearson" if kind == "continuous" else "tetrachoric")
        fit = ref.get((WITHIN_SCOPE[form], name))
        degenerate = form == "stratified" and est is not None and res.within_person.stratified.n_degenerate_stratum
        if est is None or est.value is None or fit is None or degenerate:
            out[form] = None
            continue
        _, psi, on = fit
        rows = np.flatnonzero(on)
        out[form] = (Units(psi[rows], (rows,), None), psi)
    return out


def _units(res, ped, values, kind, stratum) -> dict[str, list[Units | None]]:
    """Per form, each cell's estimate's entries from the reference; ``None`` where it has none to compare.

    The reference's stratified fit keeps every block and is Pearson only, so a binary stratified fit, or one
    that left a constant stratum out, has none.
    """
    out: dict[str, list[Units | None]] = {form: [] for form in FORMS}
    for cell in res.cells:
        a, b, m = _pairs(ped, list(cell.members))
        y = None if cell.first_trait == cell.second_trait else values[cell.second_trait]
        read = _reader(values[cell.first_trait], y, kind, cell.symmetric)
        units = _estimate_units(read, a, b, m)
        primary = "pearson" if kind == "continuous" else "tetrachoric"
        out["pooled"].append(units[primary] if cell.crude[0].value is not None else None)
        out["phi"].append(units.get("phi") if kind == "binary" and cell["phi"].value is not None else None)
        fit = cell.stratified
        defined = fit is not None and fit.estimate.value is not None and fit.n_degenerate_stratum == 0
        strat = _stratified_pearson(read, a, b, m, stratum)[1] if kind == "continuous" and defined else None
        out["stratified"].append(strat)
    return out


def _check_shape(res) -> Counter[str]:
    """Each matrix's shape, bitwise symmetry, NaN exactly where an estimate has no SE, and a diagonal whose
    square root is the SE bit for bit, within_person's included."""
    cov = res.covariance
    cells = [*res.cells, *([] if res.within_person is None else [res.within_person])]
    n = len(cells)
    seen: Counter[str] = Counter()
    for form in FORMS:
        m = getattr(cov, form)
        estimates = [_estimate(c, form) for c in cells]
        if estimates[0] is None:
            assert m is None, form
            continue
        assert m.shape == (n, n), form
        assert m.dtype == np.float64, form
        assert not m.flags.writeable, form
        bits = m.view(np.uint64)
        assert np.array_equal(bits, bits.T), form
        has = np.array([e is not None and e.se is not None for e in estimates])
        assert np.array_equal(np.isnan(m), ~np.outer(has, has)), form
        for i, e in enumerate(estimates):
            if has[i]:
                assert np.sqrt(m[i, i]) == e.se, (form, i)
                seen[form] += 1
    return seen


def _cov_close(got: float, ref: float, scale: float, estimator: str, where: str) -> None:
    tol = 2 * TOL[estimator] * scale + FLOOR
    assert abs(got - ref) <= tol, f"{where}: {got} vs {ref} (scale {scale})"


def _check(res, ped, values, kind, stratum) -> Counter[str]:
    """Every entry of every matrix between two estimates with an SE against :func:`cross_meat`; the
    within-person diagonal against :func:`kin_meat`, its row against the cells' on members alone."""
    units = _units(res, ped, values, kind, stratum)
    cells = list(res.cells)
    within = None
    if res.within_person is not None:
        within = _within_units(res, values, kind, stratum)
        cells.append(res.within_person)
        pairs = _kernel_pairs(ped)
    checked: Counter[str] = Counter()
    for form in FORMS:
        m = getattr(res.covariance, form)
        if m is None:
            continue
        us = list(units[form])
        own = [None if u is None else cross_meat(u, u) for u in us]
        if within is not None:
            w = within[form]
            us.append(None if w is None else w[0])
            own.append(None if w is None else kin_meat(w[1], w[1] != 0, pairs)[0])
        for j, k in itertools.combinations_with_replacement(range(len(cells)), 2):
            uj, uk = us[j], us[k]
            if uj is None or uk is None or np.isnan(m[j, k]):
                continue
            estimator = _estimate(cells[j], form).estimator
            where = f"{form} {cells[j].category} x {cells[k].category}"
            ref = own[j] if j == k else cross_meat(uj, uk)
            _cov_close(m[j, k], ref, np.sqrt(abs(own[j] * own[k])), estimator, where)
            checked["diagonal" if j == k else "off"] += 1
            checked["within" if k == len(res.cells) else form] += 1
    return checked


def test_the_cross_meat_is_its_definition_and_on_the_diagonal_the_se_meat():
    """1a: on a tiny pedigree :func:`cross_meat`'s three key-joined sums are :func:`brute_cross`'s every two
    entries sharing an MRCA or a member, and a cell's own is the SE's ``mrca_person`` meat."""
    ped = random_pedigree(2, n_founders=8, generations=3, size=10)
    n = len(ped["id"])
    values = _traits(2, n, "continuous", 1)
    res = rp.correlation(ped, Trait(values[0]), se=True, covariance=True)
    units = [u for u in _units(res, ped, values, "continuous", None)["pooled"] if u is not None]
    assert len(units) >= 5
    shared = 0
    for uj, uk in itertools.product(units, repeat=2):
        ref = brute_cross(uj, uk)
        assert abs(cross_meat(uj, uk) - ref) <= 1e-12 * (1 + abs(ref))
        shared += ref != 0.0
    assert shared > len(units), "cells share MRCAs or members"
    for u in units:
        first, second = u.members
        own = meats(u.psi, first, second, u.mrca, u.mrca)["mrca_person"]
        # A cell whose pairs share one MRCA has a meat of zero up to rounding: bound the
        # error by the influences' scale, not only the meat's.
        assert abs(cross_meat(u, u) - own) <= 1e-12 * (abs(own) + np.abs(u.psi).sum() ** 2)
    within = Units(units[0].psi, (units[0].members[0],), None)
    ref = brute_cross(within, units[1])
    assert ref != 0.0
    scale = np.abs(within.psi).sum() * np.abs(units[1].psi).sum()
    assert abs(cross_meat(within, units[1]) - ref) <= 1e-12 * (abs(ref) + scale)


def _without_se(cell: rp.Cell | None) -> rp.Cell | None:
    """*cell* with every estimate's SE taken out."""
    if cell is None:
        return None
    bare = lambda e: replace(e, se=None, se_reason=None)  # noqa: E731
    strata = {label: replace(s, crude=tuple(map(bare, s.crude))) for label, s in (cell.strata or {}).items()}
    stratified = cell.stratified and replace(cell.stratified, estimate=bare(cell.stratified.estimate))
    return replace(cell, crude=tuple(map(bare, cell.crude)), strata=strata or cell.strata, stratified=stratified)


def _within_estimates(cell: rp.Cell) -> list[rp.Estimate]:
    strata = [e for s in (cell.strata or {}).values() for e in s.crude]
    return [*cell.crude, *strata, *([cell.stratified.estimate] if cell.stratified else [])]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
@pytest.mark.parametrize("n_traits", [1, 2])
def test_every_covariance_matches_the_reference(seed, kind, n_traits):
    """1b, 1c and 1d: every entry of the pooled, phi and (continuous) stratified matrices against the
    reference, within_person's row and column among them, over the default request and the groups, at the
    SE's tolerances; the matrices' shape and NaN."""
    ped = random_pedigree(seed, generations=5)
    n = len(ped["id"])
    values = _traits(seed, n, kind, n_traits)
    stratum = _strata(seed, n)
    traits = [Trait(v, kind=kind) for v in values]
    checked: Counter[str] = Counter()
    for categories in (None, GROUPS):
        res = rp.correlation(ped, traits, categories=categories, stratum=stratum, se=True, covariance=True)
        alone = rp.correlation(ped, traits, categories=categories, stratum=stratum, se=True)
        assert res.cells == alone.cells
        assert _without_se(res.within_person) == _without_se(alone.within_person)
        assert _check_shape(res)["pooled"] > 0
        checked += _check(res, ped, values, kind, stratum)
    assert checked["off"] > 20, checked
    assert checked["diagonal"] > 5, checked
    assert checked["phi" if kind == "binary" else "stratified"] > 0, checked
    if n_traits == 2:
        assert checked["within"] > 20, checked


@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_the_covariance_is_opt_in(kind):
    ped = random_pedigree(3, generations=5)
    n = len(ped["id"])
    traits = [Trait(v, kind=kind) for v in _traits(3, n, kind, 2)]
    stratum = _strata(3, n)
    alone = rp.correlation(ped, traits, stratum=stratum, se=True)
    both = rp.correlation(ped, traits, stratum=stratum, se=True, covariance=True)
    assert alone.covariance is None
    assert both.covariance is not None
    assert (alone.settings["covariance"], both.settings["covariance"]) == (False, True)
    assert rp.correlation(ped, traits, stratum=stratum).settings["covariance"] is False
    assert both.method["covariance"].startswith("with covariance, Cov(r_j, r_k)")
    assert both.cells == alone.cells
    assert _without_se(both.within_person) == _without_se(alone.within_person)
    assert {e.se_reason for e in _within_estimates(alone.within_person)} == {"within_person"}
    assert "within_person" not in {e.se_reason for e in _within_estimates(both.within_person)}
    assert both == rp.correlation(ped, traits, stratum=stratum, se=True, covariance=True)
    assert both != alone
    assert (both.covariance.phi is None) == (kind == "continuous")
    one = rp.correlation(ped, traits[0], se=True, covariance=True).covariance
    assert one.pooled.shape == (12, 12)
    assert one.stratified is None
    none = rp.correlation(ped, traits, categories=[], se=True, covariance=True)
    assert none.cells == ()
    assert none.covariance.pooled.shape == (1, 1)
    assert np.sqrt(none.covariance.pooled[0, 0]) == none.within_person.primary.se
    assert none.within_person == rp.correlation(ped, traits, se=True, covariance=True).within_person
    with pytest.raises(ValueError, match="covariance=True needs se=True"):
        rp.correlation(ped, traits, covariance=True)
    with pytest.raises(TypeError, match="covariance must be a bool"):
        rp.correlation(ped, traits, se=True, covariance=1)


def _within_estimate(cell: rp.Cell, scope, estimator: str) -> rp.Estimate:
    if scope == "pooled":
        return cell[estimator]
    if scope == "stratified":
        return cell.stratified.estimate
    return cell.strata[int(scope)][estimator]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
@pytest.mark.parametrize("strata", [False, True])
def test_the_within_person_se_is_the_kinship_kernel(seed, kind, strata):
    """1c: every within-person estimate's SE, pooled, stratum cells and stratified, against
    :func:`kin_meat` over the rows' relative pairs of degree at most 2, at the SE's tolerances.  The
    reference's stratified fit is Pearson only, so a binary stratified fit has an SE but no reference."""
    ped = random_pedigree(seed, generations=5)
    n = len(ped["id"])
    values = _traits(seed, n, kind, 2)
    stratum = _strata(seed, n) if strata else None
    res = rp.correlation(ped, [Trait(v, kind=kind) for v in values], stratum=stratum, se=True, covariance=True)
    cell = res.within_person
    pairs = _kernel_pairs(ped)
    checked: Counter[str] = Counter()
    for (scope, estimator), fit in _within(values, kind, stratum).items():
        estimate = _within_estimate(cell, scope, estimator)
        where = f"within_person {scope} {estimator}"
        if estimate.value is None:
            assert (fit, estimate.se, estimate.se_reason) == (None, None, "no_estimate"), where
            continue
        if scope == "stratified" and cell.stratified.n_degenerate_stratum:
            continue
        if fit is not None:
            assert abs(estimate.value - fit[0]) <= 1e-12, where
        checked[_close(estimate, None if fit is None else kin_meat(fit[1], fit[2], pairs), where)] += 1
    if kind == "binary" and strata:
        assert cell.stratified.estimate.se is not None
    assert checked["se"] >= (4 if strata else 1 + (kind == "binary")), checked


@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_a_narrow_call_traverses_the_kernel_and_leaves_its_cells(kind):
    """A call for MZ alone gives the default call's within-person SEs bit for bit (the kernel's other
    categories are traversed for it alone) and the same cells as the SE alone; with the SE alone the
    within-person cell keeps the reason ``within_person``."""
    ped = random_pedigree(4, generations=5)
    n = len(ped["id"])
    traits = [Trait(v, kind=kind) for v in _traits(4, n, kind, 2)]
    stratum = _strata(4, n)
    wide = rp.correlation(ped, traits, stratum=stratum, se=True, covariance=True)
    for categories in (["MZ"], GROUPS, ["1C"]):
        narrow = rp.correlation(ped, traits, categories=categories, stratum=stratum, se=True, covariance=True)
        alone = rp.correlation(ped, traits, categories=categories, stratum=stratum, se=True)
        assert narrow.within_person == wide.within_person, categories
        assert narrow.cells == alone.cells, categories
        assert {e.se_reason for e in _within_estimates(alone.within_person)} == {"within_person"}
        for form in FORMS:
            m, w = getattr(narrow.covariance, form), getattr(wide.covariance, form)
            if m is not None:
                assert m[-1, -1] == w[-1, -1] or (np.isnan(m[-1, -1]) and np.isnan(w[-1, -1])), form
    assert any(e.se is not None for e in _within_estimates(wide.within_person))


def test_a_grandmother_star_gives_a_negative_kernel_meat():
    """A grandmother whose six grandchildren are in six sibships: her GP pairs are the kernel's only pairs
    among the rows, and her influence against theirs outweighs the squares, so the meat is negative."""
    m = 6
    mother = [-1, -1, *[1] * m, *[-1] * m, *range(3, 3 + m), *[-1] * 6]
    father = [-1, -1, *[2] * m, *[-1] * m, *range(9, 9 + m), *[-1] * 6]
    ped = {"id": np.arange(1, 27), "mother": np.array(mother), "father": np.array(father)}
    nan = [np.nan] * 13
    x = np.array([2.17, *nan, 1.64, 1.63, 1.75, 1.74, 1.61, 1.71, 1.74, -0.45, 1.05, -0.21, -0.25, 1.4])
    y = np.array([2.17, *nan, -1.64, -1.72, -1.65, -1.62, -1.68, -1.66, 0.43, -0.75, -0.49, -0.67, -1.17, 2.61])
    res = rp.correlation(ped, [Trait(x), Trait(y)], se=True, covariance=True)
    _, psi, on = _within([x, y], "continuous", None)["pooled", "pearson"]
    pairs = _kernel_pairs(ped)
    meat, scale, entries = kin_meat(psi, on, pairs)
    assert (entries, scale) == (13, (13 + 2 * m) / 13**2)
    assert meat < -0.5 * float(psi @ psi), meat
    assert (res.within_person.primary.se, res.within_person.primary.se_reason) == (None, "negative_meat")
    assert np.isnan(res.covariance.pooled[-1]).all()


_RUN = """
import sys
sys.path.insert(0, {root!r})
from pg_phenotype import Trait, relative_pairs as rp
from tests.pedigrees import random_pedigree
from tests.test_relative_pairs_se import _strata, _traits

ped = random_pedigree(5, n_founders=300, generations=6, size=800)
n = len(ped["id"])
for kind in ("continuous", "binary"):
    traits = [Trait(v, kind=kind) for v in _traits(5, n, kind, 2)]
    for categories in (None, {groups!r}):
        res = rp.correlation(ped, traits, categories=categories, stratum=_strata(5, n), se=True, covariance=True)
        print(repr(res.within_person))
        for m in (res.covariance.pooled, res.covariance.phi, res.covariance.stratified):
            print(None if m is None else m.tobytes().hex())
"""


def test_the_covariance_does_not_depend_on_the_thread_count():
    """2: bit for bit at 1, 3 and 8 threads, the within-person SEs among them (Rust's test covers
    batching); the groups leave out kernel categories, which the pass traverses for the kernel alone."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = [
        subprocess.run(
            [sys.executable, "-c", _RUN.format(root=root, groups=GROUPS)],
            env={**os.environ, "PG_PHENOTYPE_THREADS": str(t)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for t in (1, 3, 8)
    ]
    assert out[0].count("\n") == 16
    assert "se_reason='within_person'" not in out[0]
    assert "None" in out[0]
    assert out[0] == out[1] == out[2]


def _smallest(call, hi: int = 2**30) -> tuple[int, int]:
    """The budget *call* first passes at, to 0.1%, and one it fails at below it."""
    lo = 1
    while hi - lo > hi // 1000:
        mid = (lo + hi) // 2
        try:
            call(mid)
            hi = mid
        except ResourceError:
            lo = mid
    return lo, hi


def test_the_tallies_are_bounded_before_the_sweeps():
    """3: two binary traits' tallies outgrow what the pass frees, so just below the smallest budget the
    covariance passes, the SE alone passes and the covariance is refused (``relative_pairs_covariance``,
    after the pass and before any sweep); at the smallest budget it is the ample budget's, bit for bit."""
    ped = random_pedigree(5, n_founders=300, generations=6, size=800)
    n = len(ped["id"])
    traits = [Trait(v, kind="binary") for v in _traits(5, n, "binary", 2)]
    stratum = _strata(5, n)

    def call(budget: int, covariance: bool = True):
        return rp.correlation(ped, traits, stratum=stratum, se=True, covariance=covariance, memory_budget_bytes=budget)

    lo, hi = _smallest(call)
    with pytest.raises(ResourceError) as err:
        call(lo)
    assert err.value.code == "memory_budget_exceeded"
    fields = err.value.fields
    assert fields["operation"] == "relative_pairs_covariance"
    assert fields["estimated_bytes"] > fields["budget_bytes"] == lo
    ample = call(2**30)
    assert call(lo, covariance=False).cells == ample.cells
    assert call(hi).covariance == ample.covariance
