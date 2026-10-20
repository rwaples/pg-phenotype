"""The Relative-Pair SE's cluster design (ADR 0013) against a NumPy reference."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from collections import Counter
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pedigree_graph import RELATIONSHIPS

from pg_phenotype import Trait, _native
from pg_phenotype import correlation as pgc
from pg_phenotype import relative_pairs as rp
from tests.oracle.relative_pairs_reference import SYMMETRIC, _reader, graph
from tests.oracle.relative_pairs_se import (
    Units,
    degenerate,
    meats,
    parent_indices,
    pearson_if,
    tetrachoric_if,
    top_keys,
)
from tests.pedigrees import random_pedigree
from tests.test_relative_pairs import _family

if TYPE_CHECKING:
    from tests.oracle.relative_pairs_reference import _Read

GROUPS = [{"PO": ["MO", "FO"]}, {"HS": ["MHS", "PHS"]}]
#: The SE against the reference: Pearson and phi to rounding, the tetrachoric to its fit's influence (AS241 and
#: the observed Hessian against SciPy's ndtri and the analytic one).
TOL = {"pearson": 1e-10, "phi": 1e-10, "tetrachoric": 1e-5}
#: Meats below this are rounding: a cell whose influences cancel (r = ±1 on one pair) has a true meat of 0.
FLOOR = 1e-14


def test_meioses_match_the_registry():
    meioses = getattr(_native, "_meioses", None)
    if meioses is None:
        if os.environ.get("PG_PHENOTYPE_REQUIRE_TEST_HOOKS") == "1":
            pytest.fail("the extension was built without the test-hooks feature")
        pytest.skip("installed wheel without test hooks")
    assert {code: (up, down) for code, up, down in meioses()} == {
        code: (category.up, category.down) for code, category in RELATIONSHIPS.items()
    }
    assert [code for code, _, _ in meioses()] == list(RELATIONSHIPS)


def test_the_degenerate_cutoff_mirrors_rusts():
    """Rust's ``a_meat_within_rounding_of_zero_is_degenerate_and_one_below_it_negative``, on the mirror."""
    floor = 1024 * float(np.finfo(np.float64).eps)
    assert [degenerate(n) for n in (0, 1, 2**16)] == [floor] * 3
    assert degenerate(2**20) == 4 * floor


def _traits(seed: int, n: int, kind: str, n_traits: int) -> list[np.ndarray]:
    """Traits sharing a family signal, each missing on its own 15% of rows."""
    rng = np.random.default_rng(seed + 100)
    shared = rng.normal(size=n)
    out = []
    for _ in range(n_traits):
        v = 0.6 * shared + rng.normal(size=n)
        if kind == "binary":
            v = (v > 0.4).astype(float)
        v[rng.random(n) < 0.15] = np.nan
        out.append(v)
    return out


def _strata(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 200)
    s = rng.integers(0, 3, n).astype(float)
    s[rng.random(n) < 0.1] = np.nan
    return s


def _pairs(ped: dict[str, np.ndarray], members: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each pair's (first, second) rows, a symmetric pair's lower row first, and its MRCA key."""
    pairs = graph(ped).relationship_pairs(categories=members)
    mother, father = parent_indices(ped)
    net = np.zeros(len(mother), dtype=np.int64)
    out = []
    for code in members:
        a = np.asarray(pairs[code].first_rows, dtype=np.int64)
        b = np.asarray(pairs[code].second_rows, dtype=np.int64)
        if code in SYMMETRIC:
            a, b = np.minimum(a, b), np.maximum(a, b)
        out.append((a, b, top_keys(code, a, b, mother, father, net)[1]))
    return tuple(np.concatenate(parts) for parts in zip(*out, strict=True))


def _meat(units: Units) -> tuple[float, float, int]:
    """The meat, its scale, and its entries."""
    first, second = units.members
    out = meats(units.psi, first, second, units.mrca, units.mrca)
    return out["mrca_person"], out["mrca_person_scale"], len(units.psi)


def _oracle(read: _Read, a: np.ndarray, b: np.ndarray, m: np.ndarray) -> dict[str, tuple[float, float, int] | None]:
    """Each estimator's meat over the entries of pairs (a, b) with MRCA m."""
    return {est: None if units is None else _meat(units) for est, units in _estimate_units(read, a, b, m).items()}


def _estimate_units(read: _Read, a: np.ndarray, b: np.ndarray, m: np.ndarray) -> dict[str, Units | None]:
    """Each estimator's entries over pairs (a, b) with MRCA m, influence / entries each; ``None`` at a boundary."""
    first, second, pair = read.entries(a, b)
    x, y = read.x[first], read.y[second]
    if read.kind == "continuous":
        fits = {"pearson": pearson_if(x, y)}
    else:
        xi, yi = x.astype(np.int64), y.astype(np.int64)
        t = np.zeros((2, 2), dtype=np.int64)
        np.add.at(t, (xi, yi), 1)
        tet = pgc.tetrachoric(table=t.tolist())
        defined = tet.value is not None and not tet.boundary
        fits = {
            "tetrachoric": (tet.value, tetrachoric_if(t.astype(float), tet.value)[xi, yi]) if defined else None,
            "phi": pearson_if(x, y),
        }
    return {est: None if fit is None else Units(fit[1] / len(x), (first, second), m[pair]) for est, fit in fits.items()}


def _stratified_pearson(read: _Read, a: np.ndarray, b: np.ndarray, m: np.ndarray, stratum: np.ndarray):
    """ρ and its entries over the pairs with both strata known, each side standardised within its stratum.

    The influence is assortative's ``kernels::pearson_influence``: ψ = z₁z₂ − ρ less each side's mean
    and variance terms, over the entries.
    """
    known = ~np.isnan(stratum[a]) & ~np.isnan(stratum[b])
    first, second, pair = read.entries(a[known], b[known])
    x, y = read.x[first], read.y[second]
    s, u = stratum[first].astype(np.int64), stratum[second].astype(np.int64)

    def z(v: np.ndarray, code: np.ndarray) -> np.ndarray:
        out = np.empty_like(v)
        for c in np.unique(code):
            at = code == c
            out[at] = (v[at] - v[at].mean()) / v[at].std()
        return out

    z1, z2 = z(x, s), z(y, u)
    n = len(x)
    rho = float(z1 @ z2) / n
    psi = z1 * z2 - rho
    for code, za, zb in ((s, z1, z2), (u, z2, z1)):
        for c in np.unique(code):
            at = code == c
            big_a, big_b = zb[at].sum(), (za * zb)[at].sum()
            psi[at] -= big_a * za[at] / at.sum() + big_b * (za[at] ** 2 - 1) / (2 * at.sum())
    return rho, Units(psi / n, (first, second), m[known][pair])


def _close(estimate, reference: tuple[float, float, int] | None, where: str) -> str:
    """*estimate*'s SE against the reference (meat, scale, entries), compared as meats; its reason or ``"se"``.

    A meat within :func:`degenerate` of its scale has no SE (``degenerate_meat``), nor has a negative one.
    """
    if reference is None:
        assert (estimate.se, estimate.se_reason) == (None, "boundary"), where
        return "boundary"
    meat, scale, entries = reference
    if abs(meat) <= degenerate(entries) * scale:
        assert (estimate.se, estimate.se_reason) == (None, "degenerate_meat"), (where, estimate, meat, scale)
        return "degenerate_meat"
    if estimate.se is None:
        assert estimate.se_reason == "negative_meat", (where, estimate)
        assert meat < FLOOR, (where, meat)
        return "negative_meat"
    tol = 2 * TOL[estimate.estimator] * abs(meat) + FLOOR
    assert abs(estimate.se**2 - meat) <= tol, f"{where}: {estimate.se} vs {math.sqrt(max(meat, 0))}"
    return "se"


def _check(res, ped, values, kind, stratum) -> Counter[str]:
    """Every SE of every cell of *res* against the reference; how many compared, by outcome."""
    checked: Counter[str] = Counter()
    for cell in res.cells:
        a, b, m = _pairs(ped, list(cell.members))
        x = values[cell.first_trait]
        y = None if cell.first_trait == cell.second_trait else values[cell.second_trait]
        read = _reader(x, y, kind, cell.symmetric)
        where = f"{cell.category} {cell.first_trait}x{cell.second_trait}"
        scopes = [("pooled", cell.crude, np.ones(len(a), dtype=bool))]
        for label, s in (cell.strata or {}).items():
            assigned = (stratum[a] == label) & ((stratum[b] == label) if cell.symmetric else True)
            scopes.append((f"stratum {label}", s.crude, assigned))
        for scope, crude, sel in scopes:
            expected = _oracle(read, a[sel], b[sel], m[sel])
            for e in crude:
                if e.value is None:
                    assert (e.se, e.se_reason) == (None, "no_estimate"), where
                    continue
                checked[_close(e, expected[e.estimator], f"{where} {scope} {e.estimator}")] += 1
        fit = cell.stratified
        # The reference keeps every block: fits that left a constant stratum out are the Rust test's.
        if kind == "continuous" and fit.estimate.value is not None and fit.n_degenerate_stratum == 0:
            rho, units = _stratified_pearson(read, a, b, m, stratum)
            assert abs(fit.estimate.value - rho) < 1e-12, where
            checked[_close(fit.estimate, _meat(units), f"{where} stratified")] += 1
    return checked


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
@pytest.mark.parametrize("n_traits", [1, 2])
def test_every_cell_se_matches_the_reference(seed, kind, n_traits):
    ped = random_pedigree(seed, generations=5)
    n = len(ped["id"])
    assert len(parent_indices(ped)[0]) > n, "the fixture has an external parent"
    values = _traits(seed, n, kind, n_traits)
    stratum = _strata(seed, n)
    traits = [Trait(v, kind=kind) for v in values]
    checked: Counter[str] = Counter()
    for categories in (None, GROUPS):
        res = rp.correlation(ped, traits, categories=categories, stratum=stratum, se=True)
        checked += _check(res, ped, values, kind, stratum)
        if n_traits == 2:
            assert all(e.se_reason == "within_person" for e in res.within_person.crude)
            partial = [c for c in res.cells if c.n_entries is not None and c.n < c.n_entries < 2 * c.n]
            assert partial, "a symmetric cross-trait cell has pairs complete in one orientation"
    assert checked.total() > 50, checked
    assert checked["se"] > 0, checked


@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_the_se_is_opt_in_and_leaves_the_estimates_unchanged(kind):
    ped = random_pedigree(3, generations=5)
    n = len(ped["id"])
    values = _traits(3, n, kind, 2)
    traits = [Trait(v, kind=kind) for v in values]
    stratum = _strata(3, n)
    plain = rp.correlation(ped, traits, stratum=stratum)
    with_se = rp.correlation(ped, traits, stratum=stratum, se=True)
    assert plain == rp.correlation(ped, traits, stratum=stratum, se=False)

    def estimates(res):
        cells = [*res.cells, res.within_person]
        crude = [e for c in cells for e in (*c.crude, *(e for s in c.strata.values() for e in s.crude))]
        return crude + [c.stratified.estimate for c in cells]

    assert all((e.se, e.se_reason) == (None, None) for e in estimates(plain))
    assert any(e.se is not None for e in estimates(with_se))
    assert [replace(e, se=None, se_reason=None) for e in estimates(with_se)] == estimates(plain)
    assert (plain.settings["se"], with_se.settings["se"]) == (False, True)
    settings = MappingProxyType({**with_se.settings, "se": False})
    assert replace(with_se, cells=plain.cells, within_person=plain.within_person, settings=settings) == plain

    def raw(se):
        entries = rp._entries(None)
        native = rp.native_pedigree(ped)
        return _native.relative_pairs_correlation(native, traits, rp.strata(stratum), entries, 2**30, se, False)

    def keys(tree, out):
        if isinstance(tree, dict):
            out.update(tree)
            for k, v in tree.items():
                if k != "method":
                    keys(v, out)
        elif isinstance(tree, list | tuple):
            for v in tree:
                keys(v, out)
        return out

    assert {"se", "se_reason"}.isdisjoint(keys(raw(False)["cells"], {}))
    assert {"se", "se_reason"} <= set(keys(raw(True)["cells"], {}))
    with pytest.raises(TypeError, match="se must be a bool"):
        rp.correlation(ped, traits, se=1)
    with pytest.raises(TypeError, match="se must be a bool"):
        rp.correlation({"id": [0, 0]}, traits, stratum=[1.5], se="yes")


@pytest.mark.parametrize(
    ("kind", "values", "estimator"),
    [
        ("continuous", [0.3, 1.2, -0.5, 0.9, 0.1, 2.0, -1.1, 0.4, 0.7], "pearson"),
        ("binary", [0.0, 1, 1, 0, 1, 1, 0, 1, 0], "phi"),
    ],
)
def test_a_meat_of_rounding_noise_or_below_zero_has_no_se(kind, values, estimator):
    # FS: sibs 5-7 share one MRCA, so their influences cancel inside it.  MO: mother 3's own entry as
    # daughter of 1 cancels her four as mother.  GP: the grandchildren's sums net to less than their
    # per-grandparent parts.
    ped = _family()
    v = np.array(values)
    res = rp.correlation(ped, Trait(v, kind=kind), stratum=np.zeros(len(v)), se=True)
    assert {c.category: (c[estimator].se is None, c[estimator].se_reason) for c in res.cells if c.n} == {
        "MO": (True, "degenerate_meat"),
        "FO": (False, None),
        "FS": (True, "degenerate_meat"),
        "MHS": (True, "degenerate_meat"),
        "GP": (True, "negative_meat"),
    }
    assert _check(res, ped, [v], kind, np.zeros(len(v))).keys() >= {"se", "degenerate_meat", "negative_meat"}


def test_a_boundary_tetrachoric_has_no_se():
    # Two sibships, one all 1 and one all 0: the FS table is concordant and the fit stops at 1.
    ped = {
        "id": np.arange(1, 9),
        "mother": np.array([-1, -1, 1, 1, -1, -1, 5, 5]),
        "father": np.array([-1, -1, 2, 2, -1, -1, 6, 6]),
    }
    values = np.array([np.nan, np.nan, 1, 1, np.nan, np.nan, 0, 0])
    fs = rp.correlation(ped, Trait(values, kind="binary"), categories=["FS"], se=True).cell("FS")
    assert fs["tetrachoric"].boundary
    assert (fs["tetrachoric"].se, fs["tetrachoric"].se_reason) == (None, "boundary")
    assert fs["phi"].value == 1.0


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
        res = rp.correlation(ped, traits, categories=categories, stratum=_strata(5, n), se=True, **{options!r})
{report}
"""


def at_thread_counts(report: str, **options: object) -> list[str]:
    """What *report* prints about each two-trait SE fit of one pedigree, at 1, 3 and 8 threads.

    The fits cover both kinds, all categories and :data:`GROUPS`, with strata; *options* go to the fit and
    *report* sees it as ``res``.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = _RUN.format(root=root, groups=GROUPS, options=options, report=textwrap.indent(report, " " * 8))
    return [
        subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PG_PHENOTYPE_THREADS": str(t)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for t in (1, 3, 8)
    ]


def test_the_se_does_not_depend_on_the_thread_count():
    out = at_thread_counts("print(repr((res.cells, res.within_person)))")
    assert "se=0." in out[0]
    assert out[0] == out[1] == out[2]
