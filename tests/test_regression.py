"""Parent-offspring regression against an independent NumPy/SciPy reference."""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.stats import norm

import pg_phenotype as pgp
from pg_phenotype.regression import parent_offspring
from tests.oracle.po_regression_reference import PARENTS, regress
from tests.pedigrees import random_pedigree
from tests.po_regression_golden import REL, comparisons, gaps, load, simace_validate_float32, tolerance

if TYPE_CHECKING:
    from pg_phenotype.regression import Fit
    from tests.oracle.po_regression_reference import Reference


def _close(a: float | None, b: float | None, scale: float = 1.0) -> None:
    assert (a is None) == (b is None), (a, b)
    if a is not None:
        assert abs(a - b) <= REL * (scale + abs(b)), (a, b)


def _check(fit: Fit, ref: Reference) -> None:
    assert (fit.n, fit.n_mate_networks) == (ref.n, ref.n_mate_networks)
    assert (fit.dropped.parent_not_in_pedigree, fit.dropped.child_value_missing, fit.dropped.parent_value_missing) == (
        ref.dropped
    )
    _close(fit.slope, ref.slope)
    _close(fit.intercept, ref.intercept)
    _close(fit.r, ref.r)
    if fit.r is not None:
        assert fit.r2 == pytest.approx(fit.r**2, rel=1e-15)
    _close(fit.se_ols, ref.se_ols)
    if ref.slope is None:
        assert fit.reason in ("no_complete_children", "constant_parent_value")
        return
    if fit.p_unavailable_reason == "zero_se":
        # A score sum within rounding: the reference SE is rounding noise.
        assert fit.se == 0.0
        assert ref.se <= 1e-12 * (1 + abs(ref.slope))
        return
    _close(fit.se, ref.se)
    if fit.se is not None:
        z = abs(fit.slope) / fit.se
        _close(fit.p, 2 * norm.sf(z), scale=0.0)
        half = 1.959963984540054 * fit.se
        assert fit.ci == pytest.approx((fit.slope - half, fit.slope + half), rel=1e-14, abs=1e-14)


def _values(seed: int, n: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    v = rng.normal(size=n)
    if kind == "binary":
        v = (v > 0.4).astype(float)
    elif kind == "ordinal":
        v = np.digitize(v, [-0.5, 0.5]).astype(float)
    v[rng.random(n) < 0.1] = np.nan
    return v


@pytest.mark.parametrize("kind", ["continuous", "binary", "ordinal"])
@pytest.mark.parametrize("seed", range(12))
def test_pooled_matches_the_reference(seed: int, kind: str) -> None:
    ped = random_pedigree(seed, size=30)
    v = _values(seed, len(ped["id"]), kind)
    res = parent_offspring(ped, pgp.Trait(v, kind=kind))
    ref = regress(ped, v)
    for parent in PARENTS:
        _check(getattr(res.pooled, parent), ref[parent])
    assert res.strata is None
    assert res.n_unknown_stratum == 0


@pytest.mark.parametrize("seed", range(12))
def test_each_stratum_matches_the_reference(seed: int) -> None:
    ped = random_pedigree(seed, size=30)
    n = len(ped["id"])
    v = _values(seed, n, "continuous")
    rng = np.random.default_rng(seed)
    stratum = rng.integers(0, 3, n).astype(float) * 10 - 5
    stratum[rng.random(n) < 0.1] = np.nan
    res = parent_offspring(ped, pgp.Trait(v), stratum=stratum)
    assert res.strata is not None
    assert list(res.strata) == sorted({int(s) for s in stratum[~np.isnan(stratum)]})
    assert res.n_unknown_stratum == int(np.isnan(stratum).sum())
    for label, subset in res.strata.items():
        ref = regress(ped, v, np.flatnonzero(stratum == label))
        for parent in PARENTS:
            _check(getattr(subset, parent), ref[parent])
    for parent in PARENTS:
        fits = [getattr(s, parent) for s in res.strata.values()]
        rows = sum(f.n + sum(vars(f.dropped).values()) for f in fits)
        assert rows + res.n_unknown_stratum == n
        pooled = getattr(res.pooled, parent)
        assert pooled.n + sum(vars(pooled.dropped).values()) == n


def test_a_pedigree_and_its_columns_agree() -> None:
    ped = random_pedigree(3, size=30)
    v = _values(3, len(ped["id"]), "continuous")
    built = pgp.Pedigree(ped)
    a = parent_offspring(built, pgp.Trait(v))
    b = parent_offspring(ped, pgp.Trait(v))
    assert a.pooled == b.pooled
    assert a.metadata["pg_phenotype_version"] == pgp.__version__


def test_an_all_missing_trait_gives_reasons() -> None:
    ped = random_pedigree(1)
    res = parent_offspring(ped, pgp.Trait(np.full(len(ped["id"]), np.nan), kind="continuous"))
    for parent in PARENTS:
        fit = getattr(res.pooled, parent)
        assert fit.slope is None
        assert fit.reason == "no_complete_children"
        assert fit.se_unavailable_reason == fit.p_unavailable_reason == fit.ci_unavailable_reason == fit.reason


def test_a_perfect_fit_has_an_se_of_zero_and_no_wald_test() -> None:
    # Two sibships, one midparent each: the line passes through both means.
    ped = {
        "id": np.arange(8),
        "mother": np.array([-1, -1, -1, -1, 0, 0, 2, 2]),
        "father": np.array([-1, -1, -1, -1, 1, 1, 3, 3]),
    }
    v = np.array([0.0, 1.0, 2.0, 2.4, 0.1, 0.7, 2.5, 2.0])
    fit = parent_offspring(ped, pgp.Trait(v)).pooled.midparent
    assert fit.n_mate_networks == 2
    assert fit.se == 0.0
    assert fit.p is None
    assert fit.ci is None
    assert fit.p_unavailable_reason == fit.ci_unavailable_reason == "zero_se"
    assert fit.se_ols is not None
    assert fit.se_ols > 0
    assert math.isfinite(fit.slope)


@pytest.mark.parametrize(
    ("make", "code"),
    [
        (lambda n: (pgp.Trait(np.array(["a", "b"] * (n // 2) + ["a"] * (n % 2))), None), "unsupported_trait_kind"),
        (lambda n: (pgp.Trait(np.zeros(n - 1), kind="continuous"), None), "trait_length_mismatch"),
        (lambda n: (pgp.Trait(np.zeros(n), kind="continuous"), np.zeros(n - 1)), "stratum_length_mismatch"),
    ],
)
def test_invalid_input_raises(make, code: str) -> None:
    ped = random_pedigree(2)
    trait, stratum = make(len(ped["id"]))
    with pytest.raises(pgp.ValidationError) as err:
        parent_offspring(ped, trait, stratum=stratum)
    assert err.value.code == code


def test_trait_must_be_a_trait() -> None:
    ped = random_pedigree(2)
    with pytest.raises(TypeError):
        parent_offspring(ped, np.zeros(len(ped["id"])))  # type: ignore[arg-type]


# --- The simACE gate (docs/gates/po-regression) ------------------------------


def test_gate_golden_records_the_full_simace_revision() -> None:
    assert load()["simace_rev"] == "6355ffa6aebe37238cb2724dfcde288830fdcb6c"


def test_gate_every_outcome_is_explained() -> None:
    outcomes = Counter(c.outcome for c in comparisons())
    assert set(outcomes) <= {"both", "both_undefined", "simace_min_10", "simace_min_100", "simace_sd_floor"}
    # Each intended difference shows up in the golden at least once.
    for kind in ("both", "both_undefined", "simace_min_10", "simace_min_100", "simace_sd_floor"):
        assert outcomes[kind] > 0, kind


def test_gate_defined_lines_agree() -> None:
    for c in comparisons():
        if c.outcome != "both":
            continue
        if tolerance(c) == REL:
            # simACE's float64 paths: bit for bit, as the gate report says.
            assert set(gaps(c).values()) == {0.0}, (c.scenario, c.label, gaps(c))
        else:
            assert max(gaps(c).values()) <= tolerance(c), (c.scenario, c.label, gaps(c))
        if "n_pairs" in c.simace:
            assert c.pg.n == c.simace["n_pairs"]


def test_gate_simace_undefined_cases_are_its_rules() -> None:
    for c in comparisons():
        if c.outcome == "simace_min_10":
            assert c.pg.n == c.simace["n_pairs"] < 10
        elif c.outcome == "simace_sd_floor":
            # pg-phenotype estimates the same slope as the unscaled A1 line.
            unscaled = next(d for d in comparisons() if d.scenario == c.scenario and d.label == "validate.A1")
            assert c.pg.slope == pytest.approx(unscaled.pg.slope, rel=1e-6)
        elif c.outcome == "simace_min_100":
            n_mothered, n_trios = c.validation_counts
            assert min(n_mothered, n_trios) <= 100
            assert c.pg.n == n_trios
        elif c.outcome == "both_undefined":
            assert c.pg.reason == "constant_parent_value"


def test_gate_validation_gaps_are_simaces_float32_midparent() -> None:
    for sc in load()["scenarios"]:
        for key in ("A1", "P1"):
            redone = simace_validate_float32(sc, key)
            want = sc["simace"]["validate"][key]
            if redone is None:
                assert "slope" not in want
                continue
            fit = (want["slope"], want["intercept"], want["r_squared"])
            # simACE fits in float32: every value it stored is a float32.
            assert all(float(np.float32(v)) == v for v in fit), (sc["name"], key, fit)
            # The redo is bit for bit on the AVX2 machine that wrote the golden; AVX-512
            # orders NumPy's float32 sums differently, which moves a value by an ulp.
            assert redone == pytest.approx(fit, rel=0, abs=4 * np.finfo(np.float32).eps)
