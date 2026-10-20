"""Calibrate the ACE Fit's SE and its weights (issue #19) on simACE replicates.

    pixi run python -B tools/relative_pairs_ace_se_calibration.py --self-test
    PG_PHENOTYPE_THREADS=1 RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 pixi run python -B tools/relative_pairs_ace_se_calibration.py \\
        <simACE>/results/se_calibration --jobs 8 --out docs/gates/relative-pairs-ace-se --reps <scratch>/reps.tsv
    pixi run python -B tools/relative_pairs_ace_se_calibration.py --from-reps <scratch>/reps.tsv \\
        --out docs/gates/relative-pairs-ace-se

For every rep of every scenario under the results folder, and for each
kind (Pearson on ``liability{t}``, the tetrachoric and phi on
``affected{t}``, both over the rows whose case status is observed, as
``relative_pairs_se_calibration.load_rep`` gives them), call
``relative_pairs.correlation`` once with both traits, the default categories,
``stratum`` the generation, ``se=True`` and ``covariance=True``.  Then fit
``relative_pairs.ace(..., shared_environment=["MZ", "FS", "MHS"])`` (simACE
assigns the household by mother) for each form the fit can read (``pooled``
primary, ``pooled`` phi for binary traits, ``stratified`` primary) under each
of the ``weights`` ``pairs``, ``inverse_variance`` and ``gls``.  Each fit
gives a per-rep row per quantity (A, C, E per trait; sA, sC, sE, rA, rC, rE)
with its value, SE and both reasons, plus the fit's count of rows left out
for want of an SE (``no_se``) and, per SE reason, the cells of that form
whose estimate has a value and no SE.  Each form's within-person estimate is
a row too (``weights`` ``kernel``, quantity ``r_wp``): its SE is the kinship
kernel's.

The nominal truths come from each rep's ``params.yaml`` (the resolved
config), on the liability scale: A, C, E per trait, ``s = r sqrt(X_1 X_2)``,
the r's (rC undefined when a trait's C is 0) and ``r_wp = sA + sC + sE``.
Phi is on the observed scale, so its A, C and E have no nominal truth.

On the first rep of each scenario (and in ``--self-test``) every fit's
values and SEs are recomputed with NumPy from the result's cells and
covariance: weighted least squares per solve (``pairs``: the row's weight;
``inverse_variance``: 1 / its variance; ``gls``: the inverse of the solve's
rows' covariance), ``G Sigma G'`` and the delta method for E and the r's.

``calibration.tsv``: one row per scenario x kind x estimate x estimator x
weights x quantity (see ``summarise``).  ``summary.tsv``: the medians across
scenarios.  ``run.json``: the command, tree state, wall time and the NumPy
check's largest gaps.  ``--from-reps`` rebuilds both tables from the per-rep
rows.  Run it with the thread variables at 1: the workers are the
parallelism.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from pg_phenotype import Trait
from pg_phenotype import relative_pairs as rp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import _calibration_runner as runner
from tools.relative_pairs_se_calibration import Rep, load_rep, require

if TYPE_CHECKING:
    from pg_phenotype.relative_pairs import AceFit, AceRow, Cell, Estimate, RelativePairCorrelation

ROOT = Path(__file__).resolve().parents[1]
SHARED_ENVIRONMENT = ["MZ", "FS", "MHS"]
WEIGHTS = ("pairs", "inverse_variance", "gls")
TRAIT_QUANTITIES = ("A1", "C1", "E1", "A2", "C2", "E2")
CROSS_QUANTITIES = ("sA", "sC", "sE", "rA", "rC", "rE")
QUANTITIES = (*TRAIT_QUANTITIES, *CROSS_QUANTITIES)
#: Why a cell's estimate with a value has no SE (``Estimate.se_reason``), as per-fit counts.
CELL_SE_REASONS = ("boundary", "sandwich_undefined", "degenerate_meat", "negative_meat")
Z95 = 1.959964
#: The NumPy check's tolerance, relative to max(1, |x|) for values and to |x| for SEs.
TOL_VALUE, TOL_SE = 1e-9, 1e-7


def forms(kind: str) -> list[tuple[str, str]]:
    """The (estimate, estimator) pairs ``ace`` can read for *kind*."""
    out = [("pooled", "primary"), ("stratified", "primary")]
    return [*out[:1], ("pooled", "phi"), *out[1:]] if kind == "binary" else out


def read(cell: Cell, estimate: str, estimator: str) -> Estimate:
    """The estimate of *cell* that ``ace`` reads for the form."""
    if estimate == "stratified":
        require(cell.stratified is not None, f"{cell.category}: no stratified fit")
        assert cell.stratified is not None
        return cell.stratified.estimate
    return cell.primary if estimator == "primary" else cell["phi"]


def form_matrix(res: RelativePairCorrelation, estimate: str, estimator: str) -> np.ndarray:
    cov = res.covariance
    require(cov is not None, "no covariance in the result")
    assert cov is not None
    m = cov.stratified if estimate == "stratified" else cov.pooled if estimator == "primary" else cov.phi
    require(m is not None, f"no {estimate} {estimator} covariance")
    assert m is not None
    return m


# --- nominal truth ---------------------------------------------------------------------------


def nominal(rep_dir: Path) -> dict[str, float]:
    """The rep's nominal truths on the liability scale, from its flat ``params.yaml``."""
    params = {}
    for line in (rep_dir / "params.yaml").read_text().splitlines():
        if m := re.fullmatch(r"(\w+): (\S+)", line):
            params[m[1]] = m[2]
    p = {k: float(params[k]) for k in ("A1", "C1", "E1", "A2", "C2", "E2", "rA", "rC", "rE")}
    # Otherwise r = 2 phi A + s C with s in {0, 1} is not the model's expectation.  A rep from
    # before C transmission (simACE ADR 0025) has no Ctm_ keys, and transmits none.
    for k in ("assort1", "assort2", "Ctm_maternal1", "Ctm_maternal2", "Ctm_paternal1", "Ctm_paternal2"):
        require(float(params.get(k, "0" if k.startswith("Ctm_") else "nan")) == 0, f"{rep_dir}: {k} not 0")
    return nominal_truths(p)


def nominal_truths(p: dict[str, float]) -> dict[str, float]:
    out = {q: p[q] for q in TRAIT_QUANTITIES}
    for x in "ACE":
        scale = math.sqrt(p[f"{x}1"] * p[f"{x}2"])
        out[f"s{x}"] = p[f"r{x}"] * scale
        out[f"r{x}"] = p[f"r{x}"] if scale > 0 else math.nan
    out["r_wp"] = out["sA"] + out["sC"] + out["sE"]
    return out


def truth_of(truths: dict[str, float], estimator: str, quantity: str) -> float | None:
    # Phi's shares are on the observed scale; a cross-trait 0 stays 0 there.
    if estimator == "phi" and quantity in TRAIT_QUANTITIES:
        return None
    t = truths[quantity]
    if estimator == "phi" and t != 0:
        return None
    return None if math.isnan(t) else t


# --- one rep ---------------------------------------------------------------------------------


Solved = tuple[float | None, str | None, float | None, str | None]


def solved(shares: object, name: str) -> Solved:
    """(value, reason, se, se_reason) of *name* in a ``TraitShares`` or ``CrossShares``."""
    return tuple(getattr(shares, f"{name}{s}") for s in ("", "_reason", "_se", "_se_reason"))  # type: ignore[return-value]


def quantities(fit: AceFit) -> dict[str, Solved]:
    """Each quantity's (value, reason, se, se_reason)."""
    out = {f"{x}{shares.trait + 1}": solved(shares, x) for shares in fit.traits for x in "ACE"}
    require(fit.cross is not None, "no cross shares")
    return out | {q: solved(fit.cross, q) for q in CROSS_QUANTITIES}


@dataclass
class Gaps:
    """The NumPy check's largest relative gaps and its counts of fits and SEs checked."""

    value: float = 0.0
    se: float = 0.0
    fits: int = 0
    se_checked: int = 0

    def add(self, other: Gaps) -> None:
        self.value = max(self.value, other.value)
        self.se = max(self.se, other.se)
        self.fits += other.fits
        self.se_checked += other.se_checked


def rep_rows(rep: Rep, truths: dict[str, float], check: bool) -> tuple[list[dict], Gaps]:
    """The rep's per-quantity rows, and with *check* the NumPy check's largest gaps."""
    rows: list[dict] = []
    gaps = Gaps()
    for kind, values in (("continuous", rep.liability), ("binary", rep.affected)):
        traits = [Trait(values[t], kind) for t in (1, 2)]
        res = rp.correlation(rep.ped, traits, stratum=rep.generation, se=True, covariance=True)
        within = res.within_person
        require(within is not None, "no within-person cell")
        assert within is not None
        for estimate, estimator in forms(kind):
            cells = [read(c, estimate, estimator) for c in res.cells]
            no_se = Counter(e.se_reason for e in cells if e.value is not None and e.se is None)
            require(set(no_se) <= set(CELL_SE_REASONS), f"unexpected cell SE reasons {no_se}")
            base = {
                "scenario": rep.scenario,
                "rep": rep.rep,
                "kind": kind,
                "estimate": estimate,
                "estimator": estimator,
            }
            counts = {f"cell_se_{r}": no_se.get(r, 0) for r in CELL_SE_REASONS}
            wp = read(within, estimate, estimator)
            rows.append(
                base
                | {"weights": "kernel", "quantity": "r_wp", "truth": truth_of(truths, estimator, "r_wp")}
                | {"value": wp.value, "reason": wp.reason, "se": wp.se, "se_reason": wp.se_reason}
                | {"rows_used": None, "rows_no_se": None}
                | counts
            )
            for weights in WEIGHTS:
                fit = rp.ace(
                    res, shared_environment=SHARED_ENVIRONMENT, estimate=estimate, estimator=estimator, weights=weights
                )
                if check:
                    gaps.add(check_fit(res, fit, estimate, estimator, weights))
                fit_counts = {
                    "rows_used": sum(r.used for r in fit.rows),
                    "rows_no_se": sum(r.reason == "no_se" for r in fit.rows),
                }
                for q, (value, reason, se, se_reason) in quantities(fit).items():
                    rows.append(
                        base
                        | {"weights": weights, "quantity": q, "truth": truth_of(truths, estimator, q)}
                        | {"value": value, "reason": reason, "se": se, "se_reason": se_reason}
                        | fit_counts
                        | counts
                    )
    return rows, gaps


def rep_job(rep_dir: Path, check: bool) -> tuple[list[dict], Gaps]:
    return rep_rows(load_rep(rep_dir), nominal(rep_dir), check)


# --- the NumPy check -------------------------------------------------------------------------

#: The linear unknowns of the stacked solves, in order: trait 1's, trait 2's, the cross solve's.
LINEAR = ("A1", "C1", "A2", "C2", "sA", "sC", "sE")
SOLVE_OF = {"A1": 0, "C1": 0, "A2": 1, "C2": 1, "sA": 2, "sC": 2, "sE": 2}
#: The linear unknowns each quantity is a function of.
DEPENDS = {name: (name,) for name in LINEAR} | {
    "E1": ("A1", "C1"),
    "E2": ("A2", "C2"),
    "rA": ("sA", "A1", "A2"),
    "rC": ("sC", "C1", "C2"),
    "rE": ("sE", "A1", "C1", "A2", "C2"),
}


def known(x: float | None) -> float:
    require(x is not None, "a row the solve used has no kinship or variance")
    assert x is not None
    return x


def design(row: AceRow, cross: bool) -> list[float]:
    """The row of the design matrix: [2 phi, s] in a trait solve, [2 phi, s, 0] or [1, 1, 1] in the cross solve."""
    if row.within_person:
        return [1.0, 1.0, 1.0]
    out = [2 * known(row.kinship), float(bool(row.shared))]
    return [*out, 0.0] if cross else out


def check_fit(res: RelativePairCorrelation, fit: AceFit, estimate: str, estimator: str, weights: str) -> Gaps:
    """Recompute *fit*'s values and SEs with NumPy; raise on a gap, else return the largest gaps."""
    where = f"{estimate} {estimator} {weights}"
    sigma = form_matrix(res, estimate, estimator)
    n = sigma.shape[0]
    require(len(fit.rows) == n == len(res.cells) + 1, f"{where}: rows {len(fit.rows)} vs covariance {n}")
    for j, row in enumerate(fit.rows):
        cell = res.within_person if j == n - 1 else res.cells[j]
        assert cell is not None
        require(row.within_person == (j == n - 1), f"{where}: row {j} order")
        require((row.first_trait, row.second_trait) == (cell.first_trait, cell.second_trait), f"{where}: row {j}")
        e = read(cell, estimate, estimator)
        if weights != "pairs" and e.value is not None and e.se is None:
            require(row.value is None and row.reason == "no_se", f"{where}: row {j} without an SE is {row.reason}")
        else:
            require(row.value == e.value, f"{where}: row {j} value")
        var = sigma[j, j]
        require((row.variance is None and math.isnan(var)) or row.variance == var, f"{where}: row {j} variance {var}")
        expect_used = row.value is not None and row.weight > 0 and (weights == "pairs" or row.variance is not None)
        require(row.used == expect_used, f"{where}: row {j} used {row.used}")
    solves: list[list[int]] = [[], [], []]
    for j, row in enumerate(fit.rows):
        if row.used:
            same = row.first_trait == row.second_trait and not row.within_person
            solves[row.first_trait if same else 2].append(j)
    g_full = np.zeros((len(LINEAR), n))
    theta = np.full(len(LINEAR), np.nan)
    failed = [False, False, False]
    for s, idx in enumerate(solves):
        rows = [fit.rows[j] for j in idx]
        x = np.array([design(r, cross=s == 2) for r in rows])
        if weights == "gls":
            try:
                chol = np.linalg.cholesky(sigma[np.ix_(idx, idx)])
            except np.linalg.LinAlgError:
                failed[s] = True
                continue
            whiten = np.linalg.inv(chol)
        else:
            w = np.array([r.weight if weights == "pairs" else 1 / known(r.variance) for r in rows], dtype=float)
            whiten = np.diag(np.sqrt(w))
        g = np.linalg.pinv(whiten @ x) @ whiten
        y = np.array([r.value for r in rows], dtype=float)
        first = 0 if s == 0 else 2 if s == 1 else 4
        g_full[first : first + g.shape[0], idx] = g
        theta[first : first + g.shape[0]] = g @ y
    q = dict(zip(LINEAR, theta, strict=True))
    grads: dict[str, dict[str, float]] = {name: {name: 1.0} for name in LINEAR}
    for t in "12":
        q[f"E{t}"] = 1 - q[f"A{t}"] - q[f"C{t}"]
        grads[f"E{t}"] = {f"A{t}": -1.0, f"C{t}": -1.0}
    for x in "ACE":
        c1, c2 = q[f"{x}1"], q[f"{x}2"]
        if c1 > 0 and c2 > 0:
            r = q[f"s{x}"] / math.sqrt(c1 * c2)
            q[f"r{x}"] = r
            grad = {f"s{x}": 1 / math.sqrt(c1 * c2)}
            for t, ct in (("1", c1), ("2", c2)):
                for name, sign in grads[f"{x}{t}"].items():
                    grad[name] = grad.get(name, 0.0) - sign * r / (2 * ct)
            grads[f"r{x}"] = grad
        else:
            q[f"r{x}"] = math.nan
    gaps = Gaps(fits=1)
    for name, (value, reason, se, se_reason) in quantities(fit).items():
        touched = {SOLVE_OF[k] for k in DEPENDS[name]}
        at = f"{where} {name}"
        if any(failed[s] for s in touched):
            require(reason == "covariance_not_positive_definite", f"{at}: {reason}, NumPy has no Cholesky factor")
            continue
        if value is None:
            require(reason == "nonpositive_component" and name[0] == "r", f"{at}: {reason}")
            require(math.isnan(q[name]), f"{at}: NumPy gives {q[name]}")
            continue
        gaps.value = max(gaps.value, abs(value - q[name]) / max(1.0, abs(q[name])))
        h = sum(coef * g_full[LINEAR.index(k)] for k, coef in grads[name].items())
        support = sorted({j for s in touched for j in solves[s]})
        block = sigma[np.ix_(support, support)]
        if np.isnan(block).any():
            require(se is None and se_reason == "row_without_se", f"{at}: {se_reason}, a row has no SE")
            continue
        var = float(h[support] @ block @ h[support])
        if var < 0:
            require(se is None and se_reason == "negative_variance", f"{at}: {se_reason}, NumPy var {var}")
            continue
        require(se is not None, f"{at}: no SE ({se_reason}), NumPy var {var}")
        assert se is not None
        gaps.se = max(gaps.se, abs(se - math.sqrt(var)) / math.sqrt(var))
        gaps.se_checked += 1
    require(gaps.value <= TOL_VALUE, f"{where}: value gap {gaps.value}")
    require(gaps.se <= TOL_SE, f"{where}: SE gap {gaps.se}")
    return gaps


# --- summaries -------------------------------------------------------------------------------

KEYS = ["scenario", "kind", "estimate", "estimator", "weights", "quantity"]


def summarise(reps: pl.DataFrame) -> pl.DataFrame:
    """Per scenario x kind x estimate x estimator x weights x quantity, the SE's calibration.

    ``n_reps`` rows; ``n_defined`` with a value; ``n_se`` with an SE too.
    ``mean_value`` and ``sd_value`` over the reps with a value; ``sd_value_se``,
    ``mean_se``, ``median_se`` and ``cv_se`` over the reps with an SE;
    ``ratio`` = sd_value_se / mean_se with ``ratio_mc_se`` = ratio sqrt(1 / (2 (n_se - 1)) + cv_se² / n_se);
    ``cover_mean`` = Wald 95% coverage of mean_value over the reps with an SE,
    ``cover_truth`` the same around the nominal ``truth``, ``z_truth`` =
    (mean_value - truth) / (sd_value / sqrt(n_defined)); ``efficiency`` =
    SD(value) / SD(value under ``pairs``) over the reps both define;
    ``reason_<r>`` counts reps without a value, ``se_reason_<r>`` reps with a
    value and no SE; ``rows_no_se`` and ``cell_se_<r>`` sum the fits' counts.
    """
    reps = reps.with_columns(pl.col("value", "se").fill_nan(None))
    truths = reps.group_by(KEYS).agg(pl.col("truth").n_unique().alias("truths"))
    require(truths["truths"].max() == 1, "a nominal truth differs between reps of one scenario")
    has_value, has_se = pl.col("value").is_not_null(), pl.col("se").is_not_null()
    value_se = pl.col("value").filter(has_se)

    def cover(centre: pl.Expr) -> pl.Expr:
        return ((pl.col("value") - centre).abs() <= Z95 * pl.col("se")).filter(has_se).mean()

    aggs = [
        pl.len().alias("n_reps"),
        has_value.sum().alias("n_defined"),
        has_se.sum().alias("n_se"),
        pl.col("truth").first(),
        pl.col("value").mean().alias("mean_value"),
        pl.col("value").std().alias("sd_value"),
        value_se.std().alias("sd_value_se"),
        pl.col("se").mean().alias("mean_se"),
        pl.col("se").median().alias("median_se"),
        (pl.col("se").std() / pl.col("se").mean()).alias("cv_se"),
        cover(pl.col("value").mean()).alias("cover_mean"),
        cover(pl.col("truth")).alias("cover_truth"),
        pl.col("rows_no_se").sum(),
        *(pl.col(f"cell_se_{r}").sum() for r in CELL_SE_REASONS),
    ]
    aggs += [(pl.col("reason") == r).sum().alias(f"reason_{r}") for r in sorted(reps["reason"].drop_nulls().unique())]
    se_reasons = sorted(reps.filter(has_value)["se_reason"].drop_nulls().unique())
    aggs += [((pl.col("se_reason") == r) & has_value).sum().alias(f"se_reason_{r}") for r in se_reasons]
    out = reps.group_by(KEYS).agg(aggs)
    out = out.with_columns(
        (pl.col("sd_value_se") / pl.col("mean_se")).alias("ratio"),
        ((pl.col("mean_value") - pl.col("truth")) / (pl.col("sd_value") / pl.col("n_defined").sqrt())).alias("z_truth"),
    ).with_columns(
        (pl.col("ratio") * (1 / (2 * (pl.col("n_se") - 1)) + pl.col("cv_se") ** 2 / pl.col("n_se")).sqrt()).alias(
            "ratio_mc_se"
        )
    )
    pairs = reps.filter(pl.col("weights") == "pairs").select(
        *[k for k in KEYS if k != "weights"], "rep", pl.col("value").alias("value_pairs")
    )
    eff = (
        reps.join(pairs, on=[*[k for k in KEYS if k != "weights"], "rep"])
        .filter(has_value, pl.col("value_pairs").is_not_null())
        .group_by(KEYS)
        .agg((pl.col("value").std() / pl.col("value_pairs").std()).alias("efficiency"))
    )
    out = out.join(eff, on=KEYS, how="left")
    first = [*KEYS, "truth", "n_reps", "n_defined", "n_se", "mean_value", "sd_value", "mean_se", "median_se"]
    first += ["ratio", "ratio_mc_se", "cover_mean", "cover_truth", "z_truth", "efficiency"]
    return out.select(first + [c for c in out.columns if c not in first]).sort(KEYS)


def across_scenarios(cal: pl.DataFrame) -> pl.DataFrame:
    """The medians (and extremes) over scenarios per kind x estimate x estimator x weights x quantity."""
    keys = [k for k in KEYS if k != "scenario"]
    reasons = [c for c in cal.columns if c.startswith(("reason_", "se_reason_"))]
    return (
        cal.group_by(keys)
        .agg(
            pl.len().alias("scenarios"),
            pl.col("n_reps").sum().alias("reps"),
            pl.col("n_defined").sum().alias("defined"),
            pl.col("n_se").sum().alias("with_se"),
            pl.col("ratio").median().alias("ratio_median"),
            pl.col("ratio").min().alias("ratio_min"),
            pl.col("ratio").max().alias("ratio_max"),
            pl.col("cover_mean").median().alias("cover_median"),
            pl.col("cover_mean").min().alias("cover_min"),
            pl.col("efficiency").median().alias("efficiency_median"),
            pl.col("efficiency").min().alias("efficiency_min"),
            pl.col("efficiency").max().alias("efficiency_max"),
            pl.col("cover_truth").median().alias("cover_truth_median"),
            *(pl.col(c).sum() for c in reasons),
        )
        .sort(keys)
    )


def write_summaries(reps: pl.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cal = summarise(reps)
    cal.write_csv(out / "calibration.tsv", separator="\t", float_precision=4)
    across_scenarios(cal).write_csv(out / "summary.tsv", separator="\t", float_precision=3)


# --- self-test -------------------------------------------------------------------------------


def simulated_rep(seed: int = 19, n: int = 600, generations: int = 4) -> Rep:
    """A small two-trait ACE pedigree: households by mother, some half sibs and MZ twins, generation 0 unphenotyped."""
    rng = np.random.default_rng(seed)
    a, c, e = np.array([0.5, 0.4]), np.array([0.15, 0.2]), np.array([0.35, 0.4])

    def cov(var: np.ndarray, r: float) -> np.ndarray:
        s = math.sqrt(var[0] * var[1]) * r
        return np.array([[var[0], s], [s, var[1]]])

    sa, sc, se = cov(a, 0.5), cov(c, 0.6), cov(e, 0.2)
    total = n * generations
    mother, father, twin = (np.full(total, -1, dtype=np.int64) for _ in range(3))
    sex = rng.integers(0, 2, total)
    gen = np.repeat(np.arange(generations), n)
    av = np.zeros((total, 2))
    av[:n] = rng.multivariate_normal([0, 0], sa, n)
    household: dict[int, np.ndarray] = {}
    for g in range(1, generations):
        prev = np.arange((g - 1) * n, g * n)
        moms, dads = prev[sex[prev] == 0], prev[sex[prev] == 1]
        partner = dict(zip(moms.tolist(), rng.choice(dads, len(moms)).tolist(), strict=True))
        for i in range(g * n, (g + 1) * n):
            if i > g * n and twin[i - 1] < 0 and rng.random() < 0.04:
                mother[i], father[i], sex[i], twin[i], twin[i - 1] = mother[i - 1], father[i - 1], sex[i - 1], i - 1, i
                av[i] = av[i - 1]
                continue
            m = int(rng.choice(moms))
            mother[i] = m
            father[i] = partner[m] if rng.random() < 0.7 else int(rng.choice(dads))
            av[i] = (av[m] + av[father[i]]) / 2 + rng.multivariate_normal([0, 0], sa / 2)
    for m in np.unique(mother[mother >= 0]):
        household[int(m)] = rng.multivariate_normal([0, 0], sc)
    cv = np.array([household[int(m)] if m >= 0 else rng.multivariate_normal([0, 0], sc) for m in mother])
    liab = av + cv + rng.multivariate_normal([0, 0], se, total)
    liability, affected = {}, {}
    for t in (1, 2):
        lv = np.where(gen > 0, liab[:, t - 1], np.nan)
        liability[t] = lv
        affected[t] = np.where(gen > 0, (lv > np.nanquantile(lv, 0.8)).astype(float), np.nan)
    ped = {"id": np.arange(total), "mother": mother, "father": father, "twin": twin, "sex": sex}
    return Rep("self_test", seed, ped, gen, liability, affected)


def self_test() -> None:
    start = time.perf_counter()
    truths = nominal_truths(
        {"A1": 0.5, "C1": 0.15, "E1": 0.35, "A2": 0.4, "C2": 0.2, "E2": 0.4, "rA": 0.5, "rC": 0.6, "rE": 0.2}
    )
    rows, gaps = rep_rows(simulated_rep(), truths, check=True)
    require(gaps.fits == 15, f"checked {gaps.fits} fits, expected 15")
    require(gaps.se_checked > 0, "no SE checked")
    per_fit = len(QUANTITIES) * len(WEIGHTS) + 1
    require(len(rows) == 5 * per_fit, f"{len(rows)} rows")
    require(any(r["se"] is not None for r in rows if r["quantity"] == "r_wp"), "no within-person SE")
    print(f"NumPy check on a simulated pedigree: {gaps}")
    # The summaries on rows whose statistics are known exactly.
    synthetic = []
    for i in range(200):
        sign = 1.0 if i % 2 else -1.0
        for weights, value, se in (("pairs", sign, 1.0), ("gls", 0.5 * sign, None if i == 0 else 0.25)):
            synthetic.append(
                {"scenario": "s", "rep": i, "kind": "continuous", "estimate": "pooled", "estimator": "primary"}
                | {"weights": weights, "quantity": "A1", "truth": 1.5, "value": value, "reason": None}
                | {"se": se, "se_reason": "row_without_se" if se is None else None, "rows_used": 1, "rows_no_se": 0}
                | {f"cell_se_{r}": 0 for r in CELL_SE_REASONS}
            )
    cal = summarise(pl.DataFrame(synthetic, infer_schema_length=None))
    pairs, gls = (
        cal.filter(pl.col("weights") == "pairs").row(0, named=True),
        cal.filter(pl.col("weights") == "gls").row(0, named=True),
    )
    sd = math.sqrt(200 / 199)
    require(abs(pairs["ratio"] - sd) < 1e-12 and pairs["cover_mean"] == 1.0, f"pairs summary {pairs}")
    require(pairs["cover_truth"] == 0.5 and abs(pairs["efficiency"] - 1) < 1e-12, f"pairs summary {pairs}")
    # gls: rep 0 (value -0.5) has no SE; every value is 0.5 from the mean, 0, beyond 1.96 * 0.25.
    require(gls["n_se"] == 199 and gls["se_reason_row_without_se"] == 1, f"gls summary {gls}")
    require(gls["cover_mean"] == 0.0 and abs(gls["efficiency"] - 0.5) < 1e-12, f"gls summary {gls}")
    print(f"self-test passed in {time.perf_counter() - start:.1f} s")


# --- driver ----------------------------------------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def tree_state(out: Path) -> dict[str, str]:
    """``git describe`` and a SHA-256 of the tracked diff plus the untracked files, this tool and *out* aside."""
    tool, out = Path(__file__).resolve(), out.resolve()
    digest = hashlib.sha256(git("diff", "HEAD", "--binary").encode())
    for name in sorted(git("ls-files", "--others", "--exclude-standard").splitlines()):
        path = (ROOT / name).resolve()
        if path == tool or path.is_relative_to(out):
            continue
        digest.update(name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).hexdigest().encode())
    return {
        "describe": git("describe", "--always", "--dirty").strip(),
        "head": git("rev-parse", "HEAD").strip(),
        "tree_diff_sha256": digest.hexdigest(),
        "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = runner.parser(__doc__, jobs=8)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    runner.check_args(parser, args)
    if args.from_reps is not None:
        write_summaries(pl.read_csv(args.from_reps, separator="\t", infer_schema_length=None), args.out)
        return
    start = time.perf_counter()
    work = runner.rep_dirs(args.results, args.max_reps)
    rows: list[dict] = []
    gaps = Gaps()
    for out, g in runner.run(rep_job, work, args.jobs):
        rows += out
        gaps.add(g)
    reps = pl.DataFrame(rows, infer_schema_length=None)
    if args.reps is not None:
        reps.write_csv(args.reps, separator="\t")
    write_summaries(reps, args.out)
    run = {
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "command": [str(Path(sys.argv[0])), *sys.argv[1:]],
        "threads": runner.threads(),
        "jobs": args.jobs,
        "reps": len(work),
        "rows": len(reps),
        "wall_seconds": round(time.perf_counter() - start, 1),
        "numpy_check": {"reps": sum(c for _, c in work), **dataclasses.asdict(gaps)},
        "tree": tree_state(args.out),
        "results": str(args.results),
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
