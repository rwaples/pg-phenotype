"""pedsum #13's black-box assortative-mating tests, ported onto :func:`pg_phenotype.assortative.mate_correlation`.

Each test names the pedsum test it ports (pedsum 142adf300d5b, ``tests/test_assortative_*.py``) and
keeps its data and assertions; a payload key becomes the equivalent result field.  Where pedsum
checks against one of its own estimators, the check runs against the standalone reference
(``tests/oracle/assortative_reference.py``, pedsum's NumPy oracles) at pedsum's kernel-vs-oracle
tolerance.  The input-boundary tests at the end cover the facade's own errors.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from pg_phenotype import ParameterError, PgPhenotypeError, Trait, ValidationError
from pg_phenotype.assortative import MateCorrelation, WithinPerson, mate_correlation
from tests import assortative_builders as b
from tests.oracle import assortative_reference as ref

ROOT = Path(__file__).resolve().parent.parent

#: pedsum ``tests/test_assortative_kernels.py`` TOL: its kernels against the Brent oracles (xatol 1e-7).
ORACLE_TOL = 1e-7

#: pedsum ``INFERENCE`` (``pedsum/assortative_mating.py:39-55``), the keys its payload tests compare.
PERMUTATION_STATISTIC = (
    "score_at_zero: the score of the latent-correlation log-likelihood at rho = 0 with "
    "every margin refit on the permuted pairs (polychoric, tetrachoric, polyserial, biserial); "
    "pearson: Pearson r (continuous x continuous)"
)
BOOTSTRAP_METHOD = (
    "one_step: each draw refits the first step (thresholds, stratum moments) and every "
    "closed-form estimator exactly on the resampled Mate Networks, and takes one Newton step for rho from "
    "the observed estimate with the draw-weighted score and the full-sample Hessian"
)


#: pg_phenotype/assortative.py:223 builds an undefined estimate (``reason`` set) from 4 values and
#: ``[None] * 9`` for a 12-field ``EstimatorResult``, so every result holding an undefined estimator
#: raises ``TypeError: EstimatorResult.__init__() takes 13 positional arguments but 14 were given``.
#: With ``[None] * 8`` each marked test passes; strict, so the fix turns them into failures to unmark.


def _continuous(ped: b.Pedigree, by_id: dict) -> Trait:
    return Trait(b.column(ped, by_id), kind="continuous")


def _binary(ped: b.Pedigree, by_id: dict, cut: float = 0.0) -> Trait:
    return Trait(b.binary(ped, by_id, cut), kind="binary")


def _manual_standardise(x: np.ndarray, code: np.ndarray) -> np.ndarray:
    z = np.empty_like(x)
    for c in np.unique(code):
        sel = code == c
        z[sel] = (x[sel] - x[sel].mean()) / x[sel].std()
    return z


def _wald_ci(value: float, se: float, scale: str) -> tuple[float, float]:
    """pedsum ``wald_ci`` (``assortative_mating.py:1346-1352``) at the 95% level."""
    z = float(norm.ppf(0.975))
    if scale == "log":
        return math.exp(math.log(value) - z * se), math.exp(math.log(value) + z * se)
    centre, half = math.atanh(value), z * se / ((1 - value) * (1 + value))
    return math.tanh(centre - half), math.tanh(centre + half)


def _floats(obj: object) -> list[float]:
    data = b.plain(obj)
    out: list[float] = []

    def walk(x: object) -> None:
        if isinstance(x, float):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return out


# ---------------------------------------------------------------------------
# pedsum tests/test_assortative_mating.py: the payload
# ---------------------------------------------------------------------------


def test_crude_pearson_is_corrcoef_over_mating_pairs():
    """pedsum test_crude_pearson_is_corrcoef_over_mating_pairs: r is np.corrcoef over one observation per Mating Pair."""
    rng = np.random.default_rng(0)
    pairs, values = b.pair_values(rng, 30)
    ped = b.pedigree(pairs, children_per_pair=3)
    res = mate_correlation(ped, _continuous(ped, values), permutations=19, bootstrap=50, seed=0)
    m = [values[mo] for mo, _ in pairs]
    f = [values[fa] for _, fa in pairs]
    cell = res.cells[0]
    assert cell.n == 30
    assert cell["pearson"].value == pytest.approx(np.corrcoef(m, f)[0, 1], abs=1e-12)
    assert res.sample.n_total == 30
    assert res.sample.n_fathers_multiple_mates == sum(n > 1 for n in Counter(f for _, f in pairs).values())


def test_one_network_withholds_the_ci():
    """pedsum test_one_network_withholds_the_ci: a cell whose pairs form one Mate Network publishes no CI."""
    pairs = [(1, 100), (1, 101), (2, 101), (2, 102), (3, 102)]
    ped = b.pedigree(pairs)
    trait = _continuous(ped, {1: 0.0, 2: 1.0, 3: 3.0, 100: 0.5, 101: 2.0, 102: 1.0})
    for bootstrap in (100, 0):
        cell = mate_correlation(ped, trait, permutations=0, bootstrap=bootstrap, seed=0).cells[0]
        assert (cell.n_mate_networks, cell.largest_mate_network_share) == (1, 1.0)
        record = cell["pearson"]
        assert isinstance(record.value, float)
        assert (record.se, record.ci) == (None, None)
        assert record.ci_unavailable_reason == "single_mate_network"
        assert record.permutation is not None
        assert record.permutation.p_unavailable_reason == "not_requested"


def test_constant_draws_are_counted_as_failures():
    """pedsum test_constant_draws_are_counted_as_failures: one-network draws are constant, fail, and withhold the CI."""
    ped = b.pedigree([(1, 11), (2, 12), (3, 13)])
    trait = _continuous(ped, {1: 0.0, 2: 1.0, 3: 2.0, 11: 0.5, 12: 0.0, 13: 2.0})
    record = mate_correlation(ped, trait, permutations=0, bootstrap=400, seed=0).cells[0]["pearson"]
    boot = record.bootstrap
    assert boot is not None
    assert set(boot.failure_reasons) == {"constant_margin"}
    assert boot.failed == boot.failure_reasons["constant_margin"] > 0
    assert boot.valid + boot.failed == 400
    assert record.ci is None
    assert record.ci_unavailable_reason == "too_many_failed_draws"


def test_all_fixed_fathers_have_no_informative_permutations():
    """pedsum test_all_fixed_fathers_have_no_informative_permutations: one father per missingness block moves nothing."""
    ped = b.pedigree([(1, 11), (2, 12), (3, 12)])
    x = _continuous(ped, {1: 0.0, 2: 1.0, 3: 2.0, 11: 0.5, 12: 1.5})
    y = _continuous(ped, {1: 1.0, 2: 0.0, 3: 2.0, 11: 1.0})
    record = mate_correlation(ped, [x, y], permutations=50, bootstrap=0, seed=0).cell(0, 0)["pearson"]
    assert record.permutation is not None
    assert record.permutation.p is None
    assert record.permutation.p_unavailable_reason == "no_informative_permutations"
    assert record.permutation.n_fixed_fathers == 2


def test_unknown_stratum_pairs_are_dropped():
    """pedsum test_unknown_stratum_pairs_are_dropped: a pair with a mate of unknown birth year leaves every cell."""
    rng = np.random.default_rng(2)
    pairs, values = b.pair_values(rng, 12)
    ped = b.pedigree(pairs)
    years = {int(i): -1 if i == 1 else 1990 + int(i) % 20 for i in ped["id"]}
    res = mate_correlation(
        ped, _continuous(ped, values), stratum=b.decades(ped, years), permutations=9, bootstrap=9, seed=0
    )
    assert res.sample.n_total == 12
    assert res.sample.n_dropped_unknown_stratum == 1
    cell = res.cells[0]
    assert cell.n + cell.n_dropped.small_stratum + cell.n_dropped.degenerate_stratum == 11
    assert cell.stratified is not None
    assert cell.stratified.result.estimator == "pearson"


@pytest.mark.parametrize("binary", [False, True])
def test_determinism_per_seed(binary):
    """pedsum test_determinism_per_seed: the same seed reproduces every resample; another seed changes them."""
    rng = np.random.default_rng(5)
    pairs, values = b.pair_values(rng, 40)
    ped = b.pedigree(pairs)
    trait = _binary(ped, values) if binary else _continuous(ped, values)
    run = [mate_correlation(ped, trait, permutations=99, bootstrap=99, seed=s) for s in (3, 3, 4)]
    assert run[0] == run[1]
    name = "tetrachoric" if binary else "pearson"
    first, other = run[0].cells[0][name], run[2].cells[0][name]
    assert first.value == other.value
    assert first.ci != other.ci


def _outcome(ped: b.Pedigree, trait: Trait, stratum: np.ndarray | None, draws: int, seed: int) -> object:
    """The result, or the error code where the facade rejects the trait (pedsum has no such boundary)."""
    try:
        return mate_correlation(ped, trait, stratum=stratum, permutations=draws, bootstrap=draws, seed=seed)
    except PgPhenotypeError as exc:
        return exc.code


@settings(deadline=None, max_examples=25)
@given(
    n_pairs=st.integers(2, 25),
    remate_every=st.integers(1, 4),
    seed=st.integers(0, 10_000),
    stratify=st.booleans(),
    binary=st.booleans(),
)
def test_row_order_invariance(n_pairs, remate_every, seed, stratify, binary):
    """pedsum test_row_order_invariance: shuffling the pedigree rows changes nothing, resamples and strata included."""
    rng = np.random.default_rng(seed)
    pairs, values = b.pair_values(rng, n_pairs, remate_every)
    ped = b.pedigree(pairs, children_per_pair=2)
    missing = {i for i in values if rng.random() < 0.15}
    values = {i: v for i, v in values.items() if i not in missing}
    years = {int(i): int(rng.integers(1950, 1980)) for i in ped["id"]}
    order = rng.permutation(ped["id"].size)
    draws = 9 if binary else 29

    def run(frame: b.Pedigree) -> object:
        trait = _binary(frame, values) if binary else _continuous(frame, values)
        return _outcome(frame, trait, b.decades(frame, years) if stratify else None, draws, seed)

    shuffled, original = run(b.rows(ped, order)), run(ped)
    assert isinstance(original, (MateCorrelation, str))
    assert shuffled == original


def test_full_r_mf_recovers_an_asymmetric_matrix():
    """pedsum test_full_r_mf_recovers_an_asymmetric_matrix: every R_mf cell and both within-person r recover."""
    within_m, within_f = 0.4, 0.3
    r_mf = np.array([[0.30, 0.15], [0.05, 0.25]])
    cov = np.block(
        [[np.array([[1, within_m], [within_m, 1]]), r_mf], [r_mf.T, np.array([[1, within_f], [within_f, 1]])]]
    )
    n = 20_000
    draws = np.random.default_rng(11).multivariate_normal(np.zeros(4), cov, size=n)
    pairs = b.one_to_one(n)
    ped = b.pedigree(pairs)
    t1 = {m: draws[k, 0] for k, (m, _) in enumerate(pairs)} | {f: draws[k, 2] for k, (_, f) in enumerate(pairs)}
    t2 = {m: draws[k, 1] for k, (m, _) in enumerate(pairs)} | {f: draws[k, 3] for k, (_, f) in enumerate(pairs)}
    res = mate_correlation(ped, [_continuous(ped, t1), _continuous(ped, t2)], permutations=0, bootstrap=0, seed=0)
    # SE of r is about (1 - r^2) / sqrt(n) < 0.0075; 0.03 is four of them.
    for i in range(2):
        for j in range(2):
            assert res.cell(i, j).n == n
            assert res.cell(i, j)["pearson"].value == pytest.approx(r_mf[i, j], abs=0.03)
    assert res.within_person is not None
    assert res.within_person["mothers"].value == pytest.approx(within_m, abs=0.03)
    assert res.within_person["fathers"].value == pytest.approx(within_f, abs=0.03)
    assert res.within_person["mothers"].estimator == "pearson"


def test_each_cell_uses_its_pairwise_complete_pairs():
    """pedsum test_each_cell_uses_its_pairwise_complete_pairs: each cell keeps the pairs with its own two values."""
    rng = np.random.default_rng(12)
    pairs = b.one_to_one(60)
    ped = b.pedigree(pairs)
    ids = [p for pair in pairs for p in pair]
    x = {i: v for i, v in zip(ids, rng.normal(size=len(ids)), strict=True) if rng.random() > 0.2}
    y = {i: v for i, v in zip(ids, rng.normal(size=len(ids)), strict=True) if rng.random() > 0.3}
    res = mate_correlation(ped, [_continuous(ped, x), _continuous(ped, y)], permutations=0, bootstrap=0, seed=0)
    for cell in res.cells:
        mv, fv = (x, y)[cell.mother_trait], (x, y)[cell.father_trait]
        complete = [(mv[m], fv[f]) for m, f in pairs if m in mv and f in fv]
        assert cell.n == len(complete)
        assert cell.n_dropped.mother_missing == sum(m not in mv and f in fv for m, f in pairs)
        assert cell.n_dropped.father_missing == sum(m in mv and f not in fv for m, f in pairs)
        assert cell.n_dropped.both_missing == sum(m not in mv and f not in fv for m, f in pairs)
        assert (cell.n_dropped.small_stratum, cell.n_dropped.degenerate_stratum) == (0, 0)
        assert cell["pearson"].value == pytest.approx(np.corrcoef(np.array(complete).T)[0, 1], abs=1e-12)


def test_within_person_counts_each_eligible_individual_once():
    """pedsum test_within_person_counts_each_eligible_individual_once: a remating father counts once."""
    ped = b.pedigree([(1, 10), (2, 10), (3, 10), (4, 11), (5, 12), (6, 13)])
    x = {1: 0.1, 2: 0.9, 3: 0.4, 4: 1.2, 5: -0.3, 6: 0.7, 10: 1.0, 11: 0.2, 12: -1.0, 13: 0.5, 14: 3.0}
    y = {1: 0.3, 2: 0.5, 3: -0.2, 4: 1.0, 5: 0.0, 10: 2.0, 11: 0.1, 12: -0.5, 13: 1.5, 14: -3.0}
    res = mate_correlation(ped, [_continuous(ped, x), _continuous(ped, y)], permutations=0, bootstrap=0, seed=0)
    mothers, fathers = [1, 2, 3, 4, 5], [10, 11, 12, 13]

    def r(people: list[int]) -> object:
        return pytest.approx(np.corrcoef([x[i] for i in people], [y[i] for i in people])[0, 1], abs=1e-12)

    assert res.within_person is not None
    assert dict(res.within_person) == {
        "mothers": WithinPerson("pearson", 5, r(mothers), None, None),
        "fathers": WithinPerson("pearson", 4, r(fathers), None, None),
    }


def test_stratified_pearson_uses_pair_weighted_margins_and_the_crude_sample():
    """pedsum test_stratified_pearson_uses_pair_weighted_margins_and_the_crude_sample: one entry per pair, crude sample."""
    ped, pairs, values, years, strata = b.stratified_fixture(14)
    res = mate_correlation(ped, _continuous(ped, values), stratum=strata, permutations=19, bootstrap=19, seed=0)
    cell = res.cells[0]
    assert cell.n_dropped.degenerate_stratum == 0
    m = np.array([values[mo] for mo, _ in pairs])
    f = np.array([values[fa] for _, fa in pairs])
    ms = np.array([years[mo] // 10 for mo, _ in pairs])
    fs = np.array([years[fa] // 10 for _, fa in pairs])
    assert cell.stratified is not None
    stratified = cell.stratified.result
    assert cell.n == len(pairs)
    assert cell["pearson"].value == pytest.approx(np.corrcoef(m, f)[0, 1], abs=1e-12)
    assert stratified.value == pytest.approx(
        np.corrcoef(_manual_standardise(m, ms), _manual_standardise(f, fs))[0, 1], abs=1e-12
    )
    assert (cell.stratified.n_strata_mothers, cell.stratified.n_strata_fathers) == (3, 3)
    assert stratified.bootstrap is not None
    assert stratified.permutation is not None
    assert stratified.bootstrap.requested == stratified.permutation.draws.requested == 19
    assert stratified.permutation.p is not None


@pytest.mark.parametrize(("min_networks", "n_small", "n_degenerate"), [(1, 0, 3), (10, 3, 0)])
def test_degenerate_strata_leave_both_crude_and_stratified(min_networks, n_small, n_degenerate):
    """pedsum test_degenerate_strata_leave_both_crude_and_stratified: thin or constant strata leave both samples."""
    rng = np.random.default_rng(15)
    pairs = b.one_to_one(30)
    ped = b.pedigree(pairs)
    values = {p: float(v) for p, v in zip([p for pair in pairs for p in pair], rng.normal(size=60), strict=True)}
    years = {m: 1960 + 10 * (k % 2) for k, (m, _) in enumerate(pairs)} | {
        f: 1960 + 10 * (k % 2) for k, (_, f) in enumerate(pairs)
    }
    years[pairs[0][0]] = 1990
    for _, f in pairs[1:3]:
        years[f] = 2000
        values[f] = 0.25
    res = mate_correlation(
        ped,
        _continuous(ped, values),
        stratum=b.decades(ped, years),
        permutations=0,
        bootstrap=0,
        seed=0,
        min_stratum_networks=min_networks,
    )
    cell = res.cells[0]
    kept = pairs[3:]
    assert (cell.n_dropped.small_stratum, cell.n_dropped.degenerate_stratum) == (n_small, n_degenerate)
    assert cell.n == len(kept)
    assert (res.sample.n_total, res.sample.n_dropped_unknown_stratum) == (30, 0)
    m = [values[mo] for mo, _ in kept]
    f = [values[fa] for _, fa in kept]
    assert cell["pearson"].value == pytest.approx(np.corrcoef(m, f)[0, 1], abs=1e-12)
    assert cell.stratified is not None
    assert cell.stratified.n_strata_mothers == 2


def test_small_strata_leave_both_crude_and_stratified():
    """pedsum test_small_strata_leave_both_crude_and_stratified: a stratum under the network minimum counts as small."""
    rng = np.random.default_rng(20)
    pairs = b.one_to_one(30)
    ped = b.pedigree(pairs)
    values = {p: float(v) for p, v in zip([p for pair in pairs for p in pair], rng.normal(size=60), strict=True)}
    years = {p: 1960 + 10 * (k % 2) for k, pair in enumerate(pairs) for p in pair}
    for m, _ in pairs[:2]:
        years[m] = 1990
    for _, f in pairs[2:5]:
        years[f] = 2000
        values[f] = 0.25
    cell = mate_correlation(
        ped,
        _continuous(ped, values),
        stratum=b.decades(ped, years),
        permutations=0,
        bootstrap=0,
        seed=0,
        min_stratum_networks=3,
    ).cells[0]
    kept = pairs[5:]
    assert (cell.n_dropped.small_stratum, cell.n_dropped.degenerate_stratum) == (2, 3)
    assert cell.n == len(kept)
    m = np.array([values[mo] for mo, _ in kept])
    f = np.array([values[fa] for _, fa in kept])
    code = np.array([k % 2 for k in range(5, 30)])
    assert cell["pearson"].value == pytest.approx(np.corrcoef(m, f)[0, 1], abs=1e-12)
    assert cell.stratified is not None
    assert cell.stratified.result.value == pytest.approx(
        np.corrcoef(_manual_standardise(m, code), _manual_standardise(f, code))[0, 1], abs=1e-12
    )
    assert cell.stratified.n_strata_mothers == 2


def test_min_stratum_networks_is_recorded_only_when_stratified():
    """pedsum test_min_stratum_networks_is_recorded_only_when_stratified: default 10 with strata, None without."""
    ped, _, values, _, strata = b.stratified_fixture(21, n_pairs=20)
    trait = _continuous(ped, values)
    stratified = mate_correlation(ped, trait, stratum=strata, permutations=0, bootstrap=0, seed=0)
    crude = mate_correlation(ped, trait, permutations=0, bootstrap=0, seed=0)
    assert stratified.settings["min_stratum_networks"] == 10
    assert crude.settings["min_stratum_networks"] is None
    assert crude.cells[0].n_dropped.small_stratum == 0


def test_degenerate_stratum_in_a_draw_fails_that_draw():
    """pedsum test_degenerate_stratum_in_a_draw_fails_that_draw: a draw leaving a stratum constant fails the stratified form."""
    rng = np.random.default_rng(16)
    pairs = b.one_to_one(24)
    ped = b.pedigree(pairs)
    values = {p: float(v) for p, v in zip([p for pair in pairs for p in pair], rng.normal(size=48), strict=True)}
    years = {p: 1900 + 10 * (k // 2) for k, pair in enumerate(pairs) for p in pair}
    cell = mate_correlation(
        ped,
        _continuous(ped, values),
        stratum=b.decades(ped, years),
        permutations=0,
        bootstrap=200,
        seed=0,
        min_stratum_networks=1,
    ).cells[0]
    assert cell.stratified is not None
    stratified = cell.stratified.result.bootstrap
    crude = cell["pearson"].bootstrap
    assert stratified is not None
    assert crude is not None
    assert stratified.failure_reasons["degenerate_stratum"] == stratified.failed > 0
    assert "degenerate_stratum" not in crude.failure_reasons


def test_permutations_stay_inside_father_strata():
    """pedsum test_permutations_stay_inside_father_strata: one father per stratum x missingness block turns them off."""
    rng = np.random.default_rng(17)
    pairs = b.one_to_one(40)
    ped = b.pedigree(pairs)
    x = {p: float(v) for p, v in zip([p for pair in pairs for p in pair], rng.normal(size=80), strict=True)}
    years = {f: 1800 + 10 * (k // 2) for k, (_, f) in enumerate(pairs)} | {m: 2000 for m, _ in pairs}
    y = {m: 1.0 + k for k, (m, _) in enumerate(pairs)} | {f: 1.0 for k, (_, f) in enumerate(pairs) if k % 2}
    traits = [_continuous(ped, x), _continuous(ped, y)]

    def cell(**options):
        return mate_correlation(ped, traits, permutations=49, bootstrap=0, seed=0, **options).cell(0, 0)

    crude_only = cell()["pearson"].permutation
    assert crude_only is not None
    assert crude_only.p is not None
    stratified_cell = cell(stratum=b.decades(ped, years), min_stratum_networks=1)
    assert stratified_cell.stratified is not None
    for record in (stratified_cell["pearson"], stratified_cell.stratified.result):
        assert record.permutation is not None
        assert record.permutation.p is None
        assert record.permutation.p_unavailable_reason == "no_informative_permutations"
        assert record.permutation.n_fixed_fathers == 40


def test_cohort_trend_without_assortment_vanishes_when_stratified():
    """pedsum test_cohort_trend_without_assortment_vanishes_when_stratified: crude r picks up a cohort trend, stratified does not."""
    rng = np.random.default_rng(18)
    n = 12_000
    pairs = b.one_to_one(n)
    ped = b.pedigree(pairs)
    cohort = rng.integers(0, 6, n)
    m = cohort + rng.normal(size=n)
    f = cohort + rng.normal(size=n)
    values = {mo: m[k] for k, (mo, _) in enumerate(pairs)} | {fa: f[k] for k, (_, fa) in enumerate(pairs)}
    years = {p: 1950 + 10 * int(cohort[k]) for k, pair in enumerate(pairs) for p in pair}
    cell = mate_correlation(
        ped, _continuous(ped, values), stratum=b.decades(ped, years), permutations=0, bootstrap=0, seed=0
    ).cells[0]
    # Crude r is var(cohort) / (var(cohort) + 1) = 0.74 in expectation; the stratified SE is about 0.009.
    assert cell["pearson"].value > 0.6
    assert cell.stratified is not None
    assert abs(cell.stratified.result.value) < 0.04


def test_stratified_permutations_refit_the_pair_weighted_father_margins():
    """pedsum test_stratified_permutations_refit_the_pair_weighted_father_margins, the parts the API can observe.

    pedsum also recomputes the p from its own donor stream (``permutation_donors``), which the
    API does not expose; the goldens (tests/test_assortative_golden.py) pin that p.  Here the
    stratified r is the reference over one entry per pair and the permutation test ran.
    """
    ped, pairs, values, years, strata = b.stratified_fixture(19, n_pairs=60)
    permutations = 49
    record = mate_correlation(
        ped, _continuous(ped, values), stratum=strata, permutations=permutations, bootstrap=0, seed=3
    ).cells[0]
    assert record.stratified is not None
    result = record.stratified.result
    m = np.array([values[mo] for mo, _ in pairs])
    f = np.array([values[fa] for _, fa in pairs])
    ms = np.array([years[mo] // 10 for mo, _ in pairs])
    fs = np.array([years[fa] // 10 for _, fa in pairs])
    _, ms_code = np.unique(ms, return_inverse=True)
    _, fs_code = np.unique(fs, return_inverse=True)
    observed = ref.stratified_pearson(ref.CellPairs(m, f, ms_code, fs_code))
    assert result.value == pytest.approx(observed, abs=1e-12)
    assert result.permutation is not None
    assert result.permutation.p is not None
    assert result.permutation.draws.requested == permutations
    assert result.permutation.seed == 3


def test_constant_margin_cell_in_the_payload():
    """pedsum test_constant_margin_cell_in_the_payload: mothers all at one level make every table estimator undefined."""
    pairs = b.one_to_one(20)
    ped = b.pedigree(pairs)
    values = {m: 0.0 for m, _ in pairs} | {f: float(k % 2) for k, (_, f) in enumerate(pairs)}
    cell = mate_correlation(
        ped, Trait(b.column(ped, values), kind="binary"), permutations=9, bootstrap=9, seed=0
    ).cells[0]
    assert cell.table == ((10, 10), (0, 0))
    for name in ("tetrachoric", "odds_ratio", "phi"):
        record = cell[name]
        assert (record.value, record.reason) == (None, "constant_margin")
        assert (record.se, record.ci, record.bootstrap, record.permutation) == (None, None, None, None)


def _mates(rng, r_mf, n, within_m=0.4, within_f=0.3) -> tuple[b.Pedigree, dict, dict]:
    """pedsum ``_mates``: (mother t1, mother t2, father t1, father t2) from a 4-variate normal."""
    cov = np.block(
        [[np.array([[1, within_m], [within_m, 1]]), r_mf], [r_mf.T, np.array([[1, within_f], [within_f, 1]])]]
    )
    draws = rng.multivariate_normal(np.zeros(4), cov, size=n)
    pairs = b.one_to_one(n)
    t1 = {m: draws[k, 0] for k, (m, _) in enumerate(pairs)} | {f: draws[k, 2] for k, (_, f) in enumerate(pairs)}
    t2 = {m: draws[k, 1] for k, (m, _) in enumerate(pairs)} | {f: draws[k, 3] for k, (_, f) in enumerate(pairs)}
    return b.pedigree(pairs), t1, t2


LATENT = ("tetrachoric", "polychoric", "biserial", "polyserial")


def test_large_n_recovers_rho_from_binary_and_ordinal_mates():
    """pedsum test_large_n_recovers_rho_from_binary_and_ordinal_mates: latent R_mf within 0.04 at n = 20000."""
    r_mf = np.array([[0.30, 0.15], [0.05, 0.25]])
    ped, t1, t2 = _mates(np.random.default_rng(21), r_mf, 20_000)
    traits = [_binary(ped, t1, cut=0.5), Trait(b.ordinal(ped, t2, [-1.0, 0.0, 1.0]), kind="ordinal")]
    res = mate_correlation(ped, traits, permutations=0, bootstrap=0, seed=0)
    # Latent-correlation SEs at n = 20000 are below 0.02; 0.04 is at least two of them.
    assert res.cell(0, 0)["tetrachoric"].value == pytest.approx(0.30, abs=0.04)
    assert res.cell(0, 1)["polychoric"].value == pytest.approx(0.15, abs=0.04)
    assert res.cell(1, 0)["polychoric"].value == pytest.approx(0.05, abs=0.04)
    assert res.cell(1, 1)["polychoric"].value == pytest.approx(0.25, abs=0.04)
    assert all(r.boundary is False for c in res.cells for r in c.crude if r.estimator in LATENT)
    bb = res.cell(0, 0)
    assert bb["phi"].value < bb["tetrachoric"].value
    assert bb["odds_ratio"].value > 1
    assert bb.table is not None
    assert all(isinstance(v, int) for row in bb.table for v in row)
    assert sum(map(sum, bb.table)) == 20_000
    assert res.within_person is not None
    assert res.within_person["mothers"].estimator == "polychoric"
    assert res.within_person["mothers"].value == pytest.approx(0.4, abs=0.04)
    assert res.within_person["fathers"].value == pytest.approx(0.3, abs=0.04)


def test_large_n_recovers_rho_from_mixed_mates():
    """pedsum test_large_n_recovers_rho_from_mixed_mates: biserial and polyserial in both orientations within 0.04."""
    r_mf = np.array([[0.30, 0.15], [0.05, 0.25]])
    ped, t1, t2 = _mates(np.random.default_rng(22), r_mf, 20_000)
    binary = mate_correlation(
        ped, [_continuous(ped, t1), _binary(ped, t2, cut=0.8)], permutations=0, bootstrap=0, seed=0
    )
    assert binary.cell(0, 1)["biserial"].value == pytest.approx(0.15, abs=0.04)
    assert binary.cell(1, 0)["biserial"].value == pytest.approx(0.05, abs=0.04)
    assert binary.cell(1, 1)["tetrachoric"].value == pytest.approx(0.25, abs=0.04)
    assert abs(binary.cell(0, 1)["point_biserial"].value) < abs(binary.cell(0, 1)["biserial"].value)
    assert binary.within_person is not None
    assert binary.within_person["mothers"].estimator == "biserial"
    assert binary.within_person["mothers"].value == pytest.approx(0.4, abs=0.04)

    ordinal = mate_correlation(
        ped,
        [Trait(b.ordinal(ped, t1, [-0.5, 0.5]), kind="ordinal"), _continuous(ped, t2)],
        permutations=0,
        bootstrap=0,
        seed=0,
    )
    assert ordinal.cell(0, 1)["polyserial"].value == pytest.approx(0.15, abs=0.04)
    assert ordinal.cell(1, 0)["polyserial"].value == pytest.approx(0.05, abs=0.04)
    assert [r.estimator for r in ordinal.cell(0, 1).crude] == ["polyserial"]
    assert ordinal.within_person is not None
    assert ordinal.within_person["fathers"].estimator == "polyserial"
    assert ordinal.within_person["fathers"].value == pytest.approx(0.3, abs=0.04)


def test_within_person_mixed_kinds_count_a_remating_individual_once():
    """pedsum test_within_person_mixed_kinds_count_a_remating_individual_once: father 10 with three mates counts once."""
    ped = b.pedigree([(1, 10), (2, 10), (3, 10), (4, 11), (5, 12), (6, 13)])
    x = {1: 0.1, 2: 0.9, 3: 0.4, 4: 1.2, 5: -0.3, 6: 0.7, 10: 1.0, 11: 0.2, 12: -1.0, 13: 0.5}
    bv = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 10: 1, 11: 0, 12: 0, 13: 1}
    res = mate_correlation(ped, [_continuous(ped, x), _binary(ped, bv, cut=0.5)], permutations=0, bootstrap=0, seed=0)
    fathers = [10, 11, 12, 13]
    one_stratum = np.zeros(4, dtype=np.int64)
    direct = ref.polyserial(
        np.array([x[i] for i in fathers]),
        np.array([bv[i] for i in fathers], float),
        one_stratum,
        one_stratum,
        np.ones((1, 2), bool),
    )
    assert isinstance(direct, ref.Fit)
    assert res.within_person is not None
    assert res.within_person["fathers"] == WithinPerson(
        "biserial", 4, pytest.approx(direct.value, abs=ORACLE_TOL), direct.boundary, None
    )
    assert res.within_person["mothers"].n == 6


def test_empty_category_in_a_draw_is_counted():
    """pedsum test_empty_category_in_a_draw_is_counted: a level absent from a draw's margin fails that draw."""
    pairs = b.one_to_one(30)
    ped = b.pedigree(pairs)
    rng = np.random.default_rng(23)
    values = {p: float(rng.integers(0, 2)) for pair in pairs for p in pair}
    values[pairs[0][1]] = 2.0
    trait = Trait(b.column(ped, values), kind="ordinal")
    boot = mate_correlation(ped, trait, permutations=0, bootstrap=200, seed=0).cells[0]["polychoric"].bootstrap
    assert boot is not None
    assert boot.failure_reasons["empty_category"] == boot.failed > 0
    assert boot.valid + boot.failed == 200


def test_odds_ratio_inf_is_a_value_and_the_ci_is_never_nan():
    """pedsum test_odds_ratio_inf_is_a_value_and_the_ci_is_never_nan, its payload half.

    pedsum's first half calls ``odds_ratio``, ``table_2x2`` and ``bootstrap_record`` directly and
    its last lines read the written YAML; here no float anywhere in the cell is NaN.
    """
    pairs = b.one_to_one(60)
    ped = b.pedigree(pairs)
    rows = [(0, 0)] * 30 + [(0, 1)] * 10 + [(1, 1)] * 20
    values = {m: float(mv) for (m, _), (mv, _) in zip(pairs, rows, strict=True)} | {
        f: float(fv) for (_, f), (_, fv) in zip(pairs, rows, strict=True)
    }
    cell = mate_correlation(
        ped, Trait(b.column(ped, values), kind="binary"), permutations=0, bootstrap=200, seed=0
    ).cells[0]
    odds = cell["odds_ratio"]
    assert cell.table == ((30, 10), (0, 20))
    assert odds.value == np.inf
    # No pair has (mother 1, father 0), so no draw can either: every draw is inf and so is the whole CI.
    assert odds.ci == (np.inf, np.inf)
    assert odds.bootstrap is not None
    assert odds.bootstrap.failed == 0
    assert cell["tetrachoric"].boundary is True
    assert odds.se is None
    floats = _floats(cell)
    assert np.inf in floats
    assert not any(math.isnan(v) for v in floats)


def test_stratified_tetrachoric_in_the_payload():
    """pedsum test_stratified_tetrachoric_in_the_payload: per-stratum thresholds and one shared rho."""
    rng = np.random.default_rng(24)
    n = 400
    pairs = b.one_to_one(n)
    ped = b.pedigree(pairs)
    cohort = rng.integers(0, 2, n)
    draws = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], size=n) + cohort[:, None]
    values = {m: draws[k, 0] for k, (m, _) in enumerate(pairs)} | {f: draws[k, 1] for k, (_, f) in enumerate(pairs)}
    years = {p: 1950 + 10 * int(cohort[k]) for k, pair in enumerate(pairs) for p in pair}
    codes = b.binary(ped, values, cut=0.5)
    cell = mate_correlation(
        ped, Trait(codes, kind="binary"), stratum=b.decades(ped, years), permutations=19, bootstrap=19, seed=0
    ).cells[0]
    assert cell.stratified is not None
    stratified = cell.stratified.result
    assert stratified.estimator == "tetrachoric"
    assert -1 < stratified.value < 1
    assert stratified.boundary is False
    assert (cell.stratified.n_strata_mothers, cell.stratified.n_strata_fathers) == (2, 2)
    assert stratified.permutation is not None
    assert stratified.permutation.p is not None
    assert stratified.value < cell["tetrachoric"].value

    row = {int(i): k for k, i in enumerate(ped["id"])}
    m = codes[[row[mo] for mo, _ in pairs]]
    f = codes[[row[fa] for _, fa in pairs]]
    levels = ref.shown_levels(m, cohort, 2), ref.shown_levels(f, cohort, 2)
    direct = ref.polychoric(ref.CellPairs(m, f, cohort, cohort, *levels))
    assert isinstance(direct, ref.Fit)
    assert stratified.value == pytest.approx(direct.value, abs=ORACLE_TOL)


# ---------------------------------------------------------------------------
# pedsum tests/test_assortative_permutation.py and test_assortative_bootstrap.py: through the payload
# ---------------------------------------------------------------------------


def _frame_traits(seed: int, n_pairs: int) -> tuple[b.Pedigree, list[Trait], np.ndarray]:
    ped, x, bv, strata = b.frame(seed, n_pairs)
    return ped, [Trait(x, kind="continuous"), Trait(bv, kind="binary")], strata


def permutation_payload() -> object:
    """pedsum test_assortative_permutation ``_payload``: run in a subprocess with a set thread budget."""
    ped, traits, strata = _frame_traits(21, 300)
    return b.plain(mate_correlation(ped, traits, stratum=strata, permutations=199, bootstrap=0, seed=4))


def bootstrap_payload() -> object:
    """pedsum test_assortative_bootstrap ``_payload``: run in a subprocess with a set thread budget."""
    ped, traits, strata = _frame_traits(21, 300)
    return b.plain(mate_correlation(ped, traits, stratum=strata, permutations=0, bootstrap=150, seed=4))


def _payload_with_threads(name: str, threads: int) -> dict:
    code = f"import json; from tests.test_assortative_api import {name}; print(json.dumps({name}()))"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=os.environ | {"PG_PHENOTYPE_THREADS": str(threads)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _primaries(payload: dict) -> list[dict]:
    return [r for cell in payload["cells"] for r in cell["crude"] if r["permutation"] is not None]


def test_permutation_thread_count_invariance():
    """pedsum test_assortative_permutation test_thread_count_invariance: 1 and 4 threads give identical records."""
    one, four = (_payload_with_threads("permutation_payload", t) for t in (1, 4))
    assert (one["settings"].pop("threads"), four["settings"].pop("threads")) == (1, 4)
    assert one == four
    primaries = _primaries(four)
    assert len(primaries) == 4
    assert all(r["permutation"]["draws"]["valid"] > 0 for r in primaries)
    stopped = [r["permutation"] for r in primaries if r["permutation"]["stopped_early"]]
    assert stopped, "null cells should stop early, so the 4-thread run exercised the batch scan"
    assert all(block["draws_used"] < 199 for block in stopped)


def test_permutation_statistic_is_recorded_per_cell():
    """pedsum test_permutation_statistic_is_recorded_per_cell: pearson cells record pearson, latent cells score_at_zero."""
    ped, traits, strata = _frame_traits(8, 200)
    res = mate_correlation(ped, traits, stratum=strata, permutations=49, bootstrap=0, seed=1)
    assert res.method["permutation_statistic"] == PERMUTATION_STATISTIC
    expected = {
        (0, 0): ("pearson", "pearson"),
        (0, 1): ("biserial", "score_at_zero"),
        (1, 0): ("biserial", "score_at_zero"),
        (1, 1): ("tetrachoric", "score_at_zero"),
    }
    for cell in res.cells:
        name, statistic = expected[cell.mother_trait, cell.father_trait]
        assert cell.stratified is not None
        for record in (cell[name], cell.stratified.result):
            assert record.estimator == name
            assert record.permutation is not None
            assert record.permutation.statistic == statistic
            assert record.permutation.p is not None
        for secondary in cell.crude[1:]:
            assert secondary.permutation is None


def test_payload_records_stopping_per_cell():
    """pedsum test_payload_records_stopping_per_cell: a correlated x x x cell runs every draw, the null x x b cell stops."""
    ped, x, bv, _ = b.frame(5, 300)
    row = {int(i): k for k, i in enumerate(ped["id"])}
    rng = np.random.default_rng(5)
    for m, f in zip(ped["mother"], ped["father"], strict=True):
        if m > 0 and not np.isnan(x[row[int(m)]]):
            x[row[int(f)]] = 0.9 * x[row[int(m)]] + 0.3 * rng.normal()
    res = mate_correlation(
        ped, [Trait(x, kind="continuous"), Trait(bv, kind="binary")], permutations=199, bootstrap=0, seed=2
    )
    assert res.method["permutation_stopping"] == "besag_clifford_closed"
    assert res.method["permutation_stop_h"] == 20
    xx = res.cell(0, 0)["pearson"].permutation
    assert xx is not None
    assert (xx.stopped_early, xx.draws_used, xx.draws.requested, xx.sequential_h) == (False, 199, 199, 20)
    xb = res.cell(0, 1)["biserial"].permutation
    assert xb is not None
    assert xb.stopped_early
    assert xb.draws_used < 199
    assert xb.p == 20 / xb.draws.valid


def test_spearman_ci_needs_the_bootstrap():
    """pedsum test_spearman_ci_needs_the_bootstrap: no sandwich for Spearman, the bootstrap gives it a percentile CI."""
    ped, traits, _ = _frame_traits(2, 120)
    without, with_bootstrap = (
        mate_correlation(ped, traits[:1], permutations=0, bootstrap=n, seed=0).cells[0] for n in (0, 50)
    )
    assert without["spearman"].ci is None
    assert without["spearman"].ci_unavailable_reason == "bootstrap_not_requested"
    spearman = with_bootstrap["spearman"]
    assert spearman.ci_method == "bootstrap"
    assert spearman.bootstrap is not None
    assert spearman.bootstrap.valid == 50
    assert spearman.se is None
    assert spearman.ci is not None
    assert len(spearman.ci) == 2
    assert with_bootstrap["pearson"].se == without["pearson"].se


def test_bootstrap_thread_count_invariance():
    """pedsum test_assortative_bootstrap test_thread_count_invariance: 1 and 4 threads give identical draws."""
    one, four = (_payload_with_threads("bootstrap_payload", t) for t in (1, 4))
    assert (one["settings"].pop("threads"), four["settings"].pop("threads")) == (1, 4)
    assert one == four
    with_ci = [r for cell in four["cells"] for r in cell["crude"]]
    with_ci += [cell["stratified"]["result"] for cell in four["cells"]]
    assert len(with_ci) == 13
    assert all(r["ci_method"] == "bootstrap" and r["bootstrap"]["valid"] > 100 for r in with_ci)
    assert four["method"]["bootstrap_method"] == BOOTSTRAP_METHOD


# ---------------------------------------------------------------------------
# pedsum tests/test_assortative_sandwich.py: reasons and the ci_method switch
# ---------------------------------------------------------------------------


def _binary_cell(rows: list[tuple[int, int]], bootstrap: int) -> MateCorrelation:
    """pedsum ``_binary_cell_payload``: one-to-one pairs with the given (mother, father) codes."""
    pairs = [(k + 1, 1000 + k) for k in range(len(rows))]
    values = {m: float(mv) for (m, _), (mv, _) in zip(pairs, rows, strict=True)}
    values |= {f: float(fv) for (_, f), (_, fv) in zip(pairs, rows, strict=True)}
    ped = b.pedigree(pairs)
    return mate_correlation(ped, _binary(ped, values), permutations=0, bootstrap=bootstrap, seed=0)


def test_boundary_and_infinite_odds_ratio_withhold_the_sandwich_ci():
    """pedsum test_boundary_and_infinite_odds_ratio_withhold_the_sandwich_ci: null CIs with reasons; phi keeps its CI.

    pedsum's ``settings.ci_method == "sandwich"`` has no settings field here; ``bootstrap == 0``
    selects it, and each record's ``ci_method`` says so.
    """
    res = _binary_cell([(0, 0)] * 30 + [(0, 1)] * 10 + [(1, 1)] * 20, bootstrap=0)
    cell = res.cells[0]
    assert res.settings["bootstrap"] == 0
    tetrachoric, odds, phi = cell["tetrachoric"], cell["odds_ratio"], cell["phi"]
    assert tetrachoric.boundary is True
    assert (tetrachoric.se, tetrachoric.ci, tetrachoric.ci_method) == (None, None, None)
    assert tetrachoric.ci_unavailable_reason == "boundary"
    assert odds.value == np.inf
    assert (odds.se, odds.ci) == (None, None)
    assert odds.ci_unavailable_reason == "infinite_odds_ratio"
    assert phi.ci_method == "sandwich"
    assert phi.ci is not None
    assert phi.ci[0] < phi.value < phi.ci[1]
    assert phi.bootstrap is None


def test_sandwich_ci_is_the_fisher_z_wald_interval():
    """pedsum test_sandwich_ci_is_the_fisher_z_wald_interval: Wald CIs on the Fisher-z and log scales."""
    latent = np.random.default_rng(4).multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], 200)
    rows = [(int(a > 0), int(c > 0.2)) for a, c in latent]
    cell = _binary_cell(rows, bootstrap=0).cells[0]
    for name in ("tetrachoric", "phi"):
        record = cell[name]
        assert (record.ci_method, record.ci_unavailable_reason) == ("sandwich", None)
        assert record.ci == pytest.approx(_wald_ci(record.value, record.se, "fisher_z"))
    odds = cell["odds_ratio"]
    assert odds.ci == pytest.approx(_wald_ci(odds.value, odds.se, "log"))


def test_bootstrap_request_keeps_the_percentile_ci_and_reports_the_sandwich_se():
    """pedsum test_bootstrap_request_keeps_the_percentile_ci_and_reports_the_sandwich_se."""
    latent = np.random.default_rng(6).multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], 150)
    rows = [(int(a > 0), int(c > 0.2)) for a, c in latent]
    with_bootstrap, without = _binary_cell(rows, bootstrap=50), _binary_cell(rows, bootstrap=0)
    assert with_bootstrap.settings["bootstrap"] == 50
    boot, sand = (r.cells[0]["tetrachoric"] for r in (with_bootstrap, without))
    assert boot.bootstrap is not None
    assert (boot.ci_method, boot.bootstrap.requested) == ("bootstrap", 50)
    assert boot.se == sand.se
    assert boot.ci != sand.ci


def test_single_network_and_spearman_reasons():
    """pedsum test_single_network_and_spearman_reasons: one network withholds every CI; Spearman needs the bootstrap."""
    chain = [(1, 100), (1, 101), (2, 101), (2, 102), (3, 102)]
    ped = b.pedigree(chain)
    trait = _continuous(ped, {1: 0.0, 2: 1.0, 3: 3.0, 100: 0.5, 101: 2.0, 102: 1.0})
    pearson = mate_correlation(ped, trait, permutations=0, bootstrap=0, seed=0).cells[0]["pearson"]
    assert pearson.ci_unavailable_reason == "single_mate_network"
    assert pearson.se is None

    pairs = [(k + 1, 1000 + k) for k in range(40)]
    ped = b.pedigree(pairs)
    rng = np.random.default_rng(9)
    values = {p: float(v) for p, v in zip([p for pair in pairs for p in pair], rng.normal(size=80), strict=True)}
    cell = mate_correlation(ped, _continuous(ped, values), permutations=0, bootstrap=0, seed=0).cells[0]
    spearman = cell["spearman"]
    assert isinstance(spearman.value, float)
    assert (spearman.se, spearman.ci, spearman.ci_method) == (None, None, None)
    assert spearman.ci_unavailable_reason == "bootstrap_not_requested"
    assert (spearman.bootstrap, spearman.permutation) == (None, None)
    assert cell["pearson"].ci_method == "sandwich"


# ---------------------------------------------------------------------------
# Input boundary: the facade's errors
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small() -> tuple[b.Pedigree, np.ndarray]:
    """Ten one-to-one pairs and a continuous trait on every row."""
    ped = b.pedigree(b.one_to_one(10))
    return ped, np.random.default_rng(0).normal(size=ped["id"].size)


@pytest.mark.parametrize("n_traits", [0, 3])
def test_trait_count(small, n_traits):
    ped, x = small
    with pytest.raises(ValidationError) as exc:
        mate_correlation(ped, [Trait(x, kind="continuous")] * n_traits)
    assert exc.value.code == "trait_count"


def test_trait_length_mismatch(small):
    ped, x = small
    with pytest.raises(ValidationError) as exc:
        mate_correlation(ped, Trait(x[:-1], kind="continuous"))
    assert exc.value.code == "trait_length_mismatch"


def test_unsupported_trait_kind(small):
    ped, x = small
    labels = np.where(x > 0, "high", "low")
    with pytest.raises(ValidationError) as exc:
        mate_correlation(ped, Trait(labels, kind="categorical"))
    assert exc.value.code == "unsupported_trait_kind"


@pytest.mark.parametrize(("kind", "bad"), [("binary", 2.0), ("ordinal", 0.5), ("continuous", np.inf)])
def test_invalid_trait_value(small, kind, bad):
    ped, x = small
    values = (x > 0).astype(np.float64) if kind != "continuous" else x.copy()
    values[3] = bad
    with pytest.raises(ValidationError) as exc:
        mate_correlation(ped, Trait(values, kind=kind))
    assert exc.value.code == "invalid_trait_value"


def test_all_missing_trait(small):
    ped, x = small
    with pytest.raises(ValidationError) as exc:
        mate_correlation(ped, Trait(np.full(x.size, np.nan), kind="continuous"))
    assert exc.value.code == "all_missing_trait"


def test_constant_trait(small):
    ped, x = small
    with pytest.raises(ValidationError) as exc:
        mate_correlation(ped, Trait(np.ones(x.size), kind="continuous"))
    assert exc.value.code == "constant_trait"


def test_sparse_ordinal_codes():
    ped = b.pedigree([(1, 2)])
    with pytest.raises(ValidationError) as exc:
        mate_correlation(ped, Trait([10, 20, 40], kind="ordinal"))
    assert exc.value.code == "sparse_ordinal_codes"


def test_unused_level(small):
    ped, x = small
    labels = np.where(x > 0, "high", "low")
    values = pd.Series(pd.Categorical(labels, categories=["low", "mid", "high"], ordered=True))
    with pytest.raises(ValidationError) as exc:
        mate_correlation(ped, Trait(values, kind="ordinal"))
    assert exc.value.code == "unused_level"


def test_stratum_length_mismatch(small):
    ped, x = small
    with pytest.raises(ValidationError) as exc:
        mate_correlation(ped, Trait(x, kind="continuous"), stratum=np.zeros(x.size - 1))
    assert exc.value.code == "stratum_length_mismatch"


@pytest.mark.parametrize(("name", "value"), [("permutations", -1), ("bootstrap", -1), ("min_stratum_networks", 0)])
@pytest.mark.parametrize("stratified", [False, True])
def test_parameter_out_of_range(small, name, value, stratified):
    ped, x = small
    stratum = np.zeros(x.size) if stratified else None
    with pytest.raises(ParameterError) as exc:
        mate_correlation(ped, Trait(x, kind="continuous"), stratum=stratum, **{name: value})
    assert exc.value.code == "parameter_out_of_range"
