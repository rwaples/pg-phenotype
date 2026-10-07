"""pedsum #13's repeated-dataset calibration (``tests/test_assortative_calibration.py``), ported onto the public API.

Each test simulates ``R`` independent Mating-Pair datasets with a known mate correlation and counts
rejections (``R_mf = 0``) or CI hits (``R_mf = 0.3``).  Monte Carlo tolerance is ``TOL_SD`` binomial
standard deviations around the nominal rate.  Size is checked one-sided (a rate at or below nominal is
valid for discrete, tied and sequential p-values); coverage fails when too low and is reported, not
failed, when mildly high.  Run with ``-s`` to see the achieved rates with their binomial SDs.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.stats import norm

from pg_phenotype import Trait
from pg_phenotype.assortative import mate_correlation
from tests.oracle import assortative_reference as ref

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pg_phenotype.assortative import EstimatorResult, MateCorrelation

ALPHA = 0.05
NOMINAL_COVERAGE = 0.95
#: Binomial Monte Carlo tolerance in standard deviations of the achieved rate.
TOL_SD = 3.0
#: Coverage this high means the CI is far wider than it should be; above the "over" flag, it fails.
MAX_COVERAGE = 0.995
R_MF = 0.3
ORDINAL_CUTS = (-0.5, 0.7)
PRIMARIES = frozenset({"pearson", "tetrachoric", "biserial", "polychoric", "polyserial"})


@dataclass(frozen=True)
class Design:
    """One synthetic population: mothers mate once, fathers take ``mates_per_father`` mothers in a row.

    Every trait of a person is a function of one latent normal; the mother's latent is ``r_mf``
    times her mate's plus independent noise, so the latent mate correlation is ``r_mf`` in every
    cell and pairs sharing a father are dependent (the Mate-Network clustering the sandwich must
    absorb).  ``cohort_trend`` shifts both mates' latents by the father's birth decade, which
    stratifying by birth decade removes.
    """

    n_pairs: int
    mates_per_father: int
    r_mf: float = 0.0
    prevalence: float = 0.5
    cohort_trend: float = 0.0
    kinds: tuple[str, ...] = ("continuous", "binary")

    @property
    def n_fathers(self) -> int:
        """Distinct fathers; the last one may have fewer mates."""
        return -(-self.n_pairs // self.mates_per_father)

    def dataset(self, seed: int) -> tuple[dict[str, np.ndarray], list[Trait], np.ndarray]:
        """One simulated pedigree (parents then one child per pair), its traits and birth-decade strata."""
        rng = np.random.default_rng(seed)
        n_m, n_f = self.n_pairs, self.n_fathers
        father_of = np.arange(n_m) // self.mates_per_father
        decade_f = rng.integers(195, 198, n_f)
        latent_f = rng.normal(size=n_f)
        latent_m = self.r_mf * latent_f[father_of] + math.sqrt(1 - self.r_mf**2) * rng.normal(size=n_m)
        shift = self.cohort_trend * (decade_f - 196)
        latent = np.concatenate([latent_m + shift[father_of], latent_f + shift, np.full(n_m, np.nan)])
        n_parents = n_m + n_f
        ped = {
            "id": np.arange(1, n_parents + n_m + 1, dtype=np.int64),
            "mother": np.concatenate([np.full(n_parents, -1), np.arange(1, n_m + 1)]).astype(np.int64),
            "father": np.concatenate([np.full(n_parents, -1), n_m + 1 + father_of]).astype(np.int64),
        }
        birth_year = np.concatenate([decade_f[father_of], decade_f, np.full(n_m, 200)]) * 10
        return ped, [self.trait(kind, latent) for kind in self.kinds], (birth_year // 10 * 10).astype(np.float64)

    @property
    def cut(self) -> float:
        """Latent threshold of the binary trait."""
        return float(norm.isf(self.prevalence))

    def trait(self, kind: str, latent: np.ndarray) -> Trait:
        """The trait of ``kind`` as a function of the per-row latent."""
        if kind == "continuous":
            return Trait(latent, kind="continuous")
        coded = (latent > self.cut) if kind == "binary" else np.digitize(latent, ORDINAL_CUTS)
        return Trait(np.where(np.isnan(latent), np.nan, coded.astype(np.float64)), kind=kind)

    def truth(self, estimator: str) -> float | None:
        """The population value each estimator targets; None when it has no closed form here (spearman)."""
        if estimator in PRIMARIES:
            return self.r_mf
        c, p = self.cut, self.prevalence
        if estimator == "point_biserial":
            return self.r_mf * norm.pdf(c) / math.sqrt(p * (1 - p))
        p11 = float(ref.bvn_cdf(np.array([-c]), np.array([-c]), self.r_mf)[0])
        if estimator == "phi":
            return (p11 - p * p) / (p * (1 - p))
        if estimator == "odds_ratio":
            p10 = p - p11
            return p11 * (1 - 2 * p + p11) / (p10 * p10)
        return None


@dataclass
class Rate:
    """Hits out of ``n`` replicates with the binomial SD of the rate."""

    hits: int = 0
    n: int = 0

    @property
    def rate(self) -> float:
        """Achieved rate."""
        return self.hits / self.n

    @property
    def sd(self) -> float:
        """Binomial SD at the achieved rate."""
        return math.sqrt(self.rate * (1 - self.rate) / self.n)


Key = tuple[str, str, str]


def _binomial_sd(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def _cells(res: MateCorrelation, kinds: tuple[str, ...]) -> Iterator[tuple[str, str, EstimatorResult]]:
    """``(cell, form, record)`` for every defined estimate; a cell is named by its kinds' initials."""
    for cell in res.cells:
        name = kinds[cell.mother_trait][0] + kinds[cell.father_trait][0]
        for record in cell.crude:
            if record.value is not None:
                yield name, "crude", record
        if cell.stratified is not None and cell.stratified.result.value is not None:
            yield name, "stratified", cell.stratified.result


def _replicates(design: Design, seed: int, n_reps: int, **kwargs) -> Iterator[MateCorrelation]:
    for rep in range(n_reps):
        ped, traits, strata = design.dataset(seed + rep)
        yield mate_correlation(ped, traits, stratum=strata, seed=seed + rep, **kwargs)


def _table(title: str, nominal: float, rows: list[tuple[str, str, str, Rate, float, bool]]) -> None:
    print(f"\n{title}: nominal {nominal}, tolerance {TOL_SD} binomial SD")
    print(f"{'cell':<5}{'form':<12}{'estimator':<16}{'rate':>7}{'sd':>7}{'bound':>7}  ok")
    for cell, form, estimator, rate, bound, ok in rows:
        print(
            f"{cell:<5}{form:<12}{estimator:<16}{rate.rate:>7.3f}{rate.sd:>7.3f}{bound:>7.3f}  {'pass' if ok else 'FAIL'}"
        )


# ---------------------------------------------------------------------------
# 1. Size of the sequential score-statistic permutation p at alpha = 0.05 under R_mf = 0
# ---------------------------------------------------------------------------

SIZE_REPS = 500
SIZE_DESIGNS = {
    "no_remating": Design(600, 1),
    "heavy_remating": Design(600, 5),
    "sparse_binary": Design(1000, 3, prevalence=0.05),
}


def _rejections(design: Design, seed: int) -> dict[Key, Rate]:
    rates: dict[Key, Rate] = {}
    for res in _replicates(design, seed, SIZE_REPS, permutations=999, bootstrap=0):
        for cell, form, record in _cells(res, design.kinds):
            if record.estimator not in PRIMARIES:
                continue
            rate = rates.setdefault((cell, form, record.estimator), Rate())
            assert record.permutation is not None
            assert record.permutation.p is not None, record.permutation.p_unavailable_reason
            rate.n += 1
            rate.hits += record.permutation.p < ALPHA
    return rates


def _check_size(title: str, rates: dict[Key, Rate], forms: tuple[str, ...]) -> None:
    bound = ALPHA + TOL_SD * _binomial_sd(ALPHA, SIZE_REPS)
    rows = [(c, f, e, r, bound, r.rate <= bound) for (c, f, e), r in rates.items() if f in forms]
    _table(title, ALPHA, rows)
    assert all(ok for *_, ok in rows), [row[:3] for row in rows if not row[-1]]
    assert all(r.n == SIZE_REPS for *_, r, _b, _ok in rows)


@pytest.mark.parametrize(
    "setting", ["heavy_remating", "no_remating", pytest.param("sparse_binary", marks=pytest.mark.slow)]
)
def test_permutation_size_under_the_null(setting):
    """pedsum test_permutation_size_under_the_null: primaries, crude and stratified, reject at most alpha + 3 SD."""
    _check_size(f"size {setting}", _rejections(SIZE_DESIGNS[setting], 100_000), ("crude", "stratified"))


def test_permutation_size_with_a_cohort_trend():
    """pedsum test_permutation_size_with_a_cohort_trend: a shared birth-decade trend does not inflate the stratified test.

    The trend is real: the crude Pearson sandwich CI excludes zero in most replicates.  The crude
    permutation p is not checked because its donors are drawn within father strata too, so it is
    conditional on the cohort structure by construction.
    """
    design = Design(600, 4, cohort_trend=0.5)
    _check_size("size cohort_trend", _rejections(design, 200_000), ("stratified",))
    crude_excludes_zero = Rate()
    for res in _replicates(design, 200_000, 100, permutations=0, bootstrap=0):
        ci = res.cells[0]["pearson"].ci
        assert ci is not None
        lo, hi = ci
        crude_excludes_zero.n += 1
        crude_excludes_zero.hits += lo > 0 or hi < 0
    print(f"positive control: crude pearson CI excludes 0 in {crude_excludes_zero.rate:.2f} of replicates")
    assert crude_excludes_zero.rate >= 0.5


# ---------------------------------------------------------------------------
# 2. Coverage of the nominal 95% CI at R_mf = 0.3
# ---------------------------------------------------------------------------

COVERAGE_REPS = 500
COVERAGE_DESIGNS = {
    "remating": Design(600, 4, r_mf=R_MF, kinds=("continuous", "binary", "ordinal")),
    "no_remating": Design(600, 1, r_mf=R_MF, kinds=("continuous", "binary", "ordinal")),
}


def _coverage(design: Design, seed: int, n_reps: int, **kwargs) -> dict[Key, Rate]:
    """Coverage over every cell of the design's traits, two traits per call (the API's limit)."""
    rates: dict[Key, Rate] = {}
    for rep in range(n_reps):
        ped, traits, strata = design.dataset(seed + rep)
        records: dict[Key, EstimatorResult] = {}
        for pair in itertools.combinations(range(len(traits)), 2):
            res = mate_correlation(
                ped, [traits[k] for k in pair], stratum=strata, seed=seed + rep, permutations=0, **kwargs
            )
            for cell, form, record in _cells(res, tuple(design.kinds[k] for k in pair)):
                records.setdefault((cell, form, record.estimator), record)
        for key, record in records.items():
            truth = design.truth(key[2])
            if truth is None:
                continue
            assert record.ci is not None, (key, record.ci_unavailable_reason)
            rate = rates.setdefault(key, Rate())
            rate.n += 1
            rate.hits += record.ci[0] <= truth <= record.ci[1]
    return rates


def _check_coverage(title: str, rates: dict[Key, Rate], n_reps: int) -> None:
    sd = _binomial_sd(NOMINAL_COVERAGE, n_reps)
    low, high = NOMINAL_COVERAGE - TOL_SD * sd, NOMINAL_COVERAGE + TOL_SD * sd
    rows = [(c, f, e, r, low, low <= r.rate <= MAX_COVERAGE) for (c, f, e), r in rates.items()]
    _table(title, NOMINAL_COVERAGE, rows)
    over = [(c, f, e) for c, f, e, r, *_ in rows if r.rate > high]
    if over:
        print(f"over-coverage above {high:.3f} (reported, not failed): {over}")
    assert all(ok for *_, ok in rows), [row[:3] for row in rows if not row[-1]]
    assert all(r.n == n_reps for *_, r, _b, _ok in rows)


@pytest.mark.slow
@pytest.mark.parametrize("setting", sorted(COVERAGE_DESIGNS))
def test_sandwich_ci_coverage(setting):
    """pedsum test_sandwich_ci_coverage: every sandwich CI covers its truth at the nominal rate."""
    rates = _coverage(COVERAGE_DESIGNS[setting], 300_000, COVERAGE_REPS, bootstrap=0)
    assert {e for _c, _f, e in rates} == PRIMARIES | {"phi", "point_biserial", "odds_ratio"}
    _check_coverage(f"sandwich coverage {setting}", rates, COVERAGE_REPS)


# ---------------------------------------------------------------------------
# 3. Opt-in one-step bootstrap coverage
# ---------------------------------------------------------------------------

BOOTSTRAP_REPS = 200
BOOTSTRAP_DRAWS = 399


@pytest.mark.slow
def test_bootstrap_ci_coverage_with_remating():
    """pedsum test_bootstrap_ci_coverage_with_remating: the one-step percentile bootstrap covers tetrachoric and Pearson."""
    design = Design(1000, 4, r_mf=R_MF)
    rates = _coverage(design, 400_000, BOOTSTRAP_REPS, bootstrap=BOOTSTRAP_DRAWS)
    checked = {k: r for k, r in rates.items() if k[2] in ("tetrachoric", "pearson")}
    _check_coverage("bootstrap coverage remating", checked, BOOTSTRAP_REPS)
