"""ACE Fit truth recovery on simACE reps (plan D10; pg-phenotype #17).

    pixi run python -B tools/relative_pairs_ace_validation.py \\
        --results <simACE>/results/relative_pairs_ace --config <simACE>/config/relative_pairs_ace.yaml \\
        --out docs/gates/relative-pairs-ace/results.json

Each rep's full pedigree (``pedigree.parquet``) carries the two traits on the
phenotyped rows (``cohort.parquet`` rows with a ``t1``), NaN elsewhere, and
``stratum`` is the generation on those rows.  ``relative_pairs.correlation``
runs over the default categories and ``relative_pairs.ace`` fits A and C with
MZ, FS and MHS sharing C (simACE assigns the household by mother), once with
``estimate="stratified"`` and once with ``"pooled"``.

Outcomes: ``liability`` (``liability1/2``, Pearson) in every scenario;
``uncensored_status`` (``t < 1e6``, simple_ltm status before censoring) in a
scenario whose two phenotype models are simple_ltm; ``affected``
(``affected1/2``, after censoring) in any other.  The stratified liability and
uncensored-status fits of the simple_ltm scenarios are gated: A1, A2, C1, C2,
rA and rE always, rC when both traits' C is above 0.  An uncensored-status
target's z is (mean - nominal truth) / MC SE, the MC SE being the rep SD
over sqrt(reps).  A liability target is held to its realized truth
instead: its z is the mean of fit - realized truth over that difference's
MC SE.  A target passes when every rep gives a value (and, for liability, a
realized truth) and |z| is at most the Bonferroni critical value
``Phi^-1(1 - 0.05 / (2 G))``, G the number of gated targets: familywise 5%.
Each target also reports ``within_2_mc_se``, |z| <= 2.

The realized truth is the ACE Fit of each rep's realized truth cells, with
the liability fit's rows, weights and design.  A Pearson cell is
``(1/N) sum z1 z2`` over its entries, each side standardised by its stratum's
mean and 1/N SD over them (one stratum when pooled), so with
``L = A + C + E + R`` (R the float32 rounding of the stored components) it is
exactly a sum over component pairs.  The realized truth cell keeps the A x A
and C x C terms, and E x E within person; the dropped terms (E between
relatives, cross-component, R) have expectation 0.  The tool stops if the
full sum misses any cell by more than 1e-9.  In the other scenarios the stratified ``affected``
rA must be defined in every rep and closer to the truth than fitACE's
``estimate_rg`` gave in fitACE#5 (0.308).  Every fit also reports
``estimate_rg``'s formula, ``2 (r_wp - r_FS,01) / sqrt(2 r_FS,00 * 2 r_FS,11)``,
on the same cells.  Exits non-zero when a gated check fails.

Four diagnostics are reported for every scenario.  ``numpy_wls_max_gap``
is each fit's largest gap in A and C from NumPy's weighted least squares
over the rows it read.  ``genetic_value`` fits the simulated additive
genetic values ``A1``, ``A2``, whose truth is A = 1 and C = 0, so its cells
are each category's realized genetic correlation against ``2 phi``.
``realized_shares`` is each rep's variance shares and cross-trait correlations of
the simulated components on the phenotyped rows, within generation and
averaged by generation size.  ``exact_kinship`` is each category's mean
pedigree kinship over every path (pedigree-graph's ``pair_kinship``) on
the phenotyped pairs, over its nominal ``2^-(degree+1)``, and the share of
its pairs above nominal.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pedigree_graph import PedigreeGraph

from pg_phenotype import Pedigree, Trait, relative_pairs

if TYPE_CHECKING:
    from collections.abc import Callable

    from pedigree_graph import PedigreeView

    from pg_phenotype._trait import TraitKind

ROOT = Path(__file__).resolve().parent.parent
SHARED_ENVIRONMENT = ("MZ", "FS", "MHS")
ESTIMATES = ("stratified", "pooled")
GATED_ESTIMATE = "stratified"
MC_SE_BOUND = 2.0
FAMILYWISE_ALPHA = 0.05
ESTIMATE_RG_FITACE_5 = 0.308
LTM = "simple_ltm"
UNCENSORED = 1e6
DECOMPOSITION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Truth:
    """A scenario's simulated variance shares and cross-trait correlations."""

    A: tuple[float, float]
    C: tuple[float, float]
    E: tuple[float, float]
    rA: float
    rC: float
    rE: float

    def targets(self) -> dict[str, float]:
        return {
            "A1": self.A[0],
            "A2": self.A[1],
            "C1": self.C[0],
            "C2": self.C[1],
            "rA": self.rA,
            "rC": self.rC,
            "rE": self.rE,
            "estimate_rg_formula": self.rA,
        }

    def genetic_value(self) -> Truth:
        """The truth of the additive genetic values themselves: all of each trait's variance is A."""
        return Truth(A=(1.0, 1.0), C=(0.0, 0.0), E=(0.0, 0.0), rA=self.rA, rC=0.0, rE=0.0)

    def gated(self) -> set[str]:
        """D10's targets: rC only where both traits have C."""
        return {"A1", "A2", "C1", "C2", "rA", "rE"} | ({"rC"} if min(self.C) > 0 else set())

    def expected(self, row: relative_pairs.AceRow) -> float:
        """A cell's correlation under the model: ``2 phi A + s C``, or its cross-trait form."""
        sA = self.rA * math.sqrt(self.A[0] * self.A[1])
        sC = self.rC * math.sqrt(self.C[0] * self.C[1])
        sE = self.rE * math.sqrt(self.E[0] * self.E[1])
        if row.within_person:
            return sA + sC + sE
        assert row.kinship is not None
        s = 1.0 if row.shared else 0.0
        if row.first_trait == row.second_trait:
            t = row.first_trait
            return 2 * row.kinship * self.A[t] + s * self.C[t]
        return 2 * row.kinship * sA + s * sC


@dataclass(frozen=True)
class Rep:
    path: Path
    pedigree: pl.DataFrame
    cohort: pl.DataFrame
    rows: np.ndarray
    run: dict[str, str]


@dataclass(frozen=True)
class Outcome:
    name: str
    kind: TraitKind
    source: str
    values: Callable[[Rep, int], np.ndarray]
    truth: Callable[[Truth], Truth]
    realized_truth: bool = False


def _liability(rep: Rep, trait: int) -> np.ndarray:
    return rep.pedigree[f"liability{trait}"].to_numpy()[rep.rows].astype(np.float64)


def _uncensored(rep: Rep, trait: int) -> np.ndarray:
    return (rep.cohort[f"t{trait}"].to_numpy() < UNCENSORED).astype(np.float64)


def _affected(rep: Rep, trait: int) -> np.ndarray:
    return rep.cohort[f"affected{trait}"].to_numpy().astype(np.float64)


def _genetic_value(rep: Rep, trait: int) -> np.ndarray:
    return rep.pedigree[f"A{trait}"].to_numpy()[rep.rows].astype(np.float64)


def _as_simulated(truth: Truth) -> Truth:
    return truth


LIABILITY = Outcome(
    "liability", "continuous", "pedigree liability1, liability2 (Pearson)", _liability, _as_simulated, True
)
UNCENSORED_STATUS = Outcome(
    "uncensored_status",
    "binary",
    "t1, t2 < 1e6: simple_ltm status before censoring (tetrachoric)",
    _uncensored,
    _as_simulated,
)
AFFECTED = Outcome("affected", "binary", "affected1, affected2 after censoring (tetrachoric)", _affected, _as_simulated)
GENETIC_VALUE = Outcome(
    "genetic_value",
    "continuous",
    "pedigree A1, A2, the simulated additive genetic values (Pearson; truth A = 1, C = 0)",
    _genetic_value,
    Truth.genetic_value,
)

_KEY = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_]\w*):(?:[ \t]+(?P<value>[^#]*?))?[ \t]*(?:#.*)?$")


def _yaml_lines(path: Path) -> list[tuple[int, str, str | None]]:
    """``(indent, key, value or None)`` for each ``key: value`` line; other lines (list items) are skipped.

    The pg-phenotype env has no YAML parser; the two files read here are
    block mappings, and every value read is checked against the other file.
    """
    out = []
    for line in path.read_text().splitlines():
        m = _KEY.match(line)
        if m:
            out.append((len(m["indent"]), m["key"], m["value"] or None))
    return out


def _flow(value: str) -> dict[str, float]:
    if not (value.startswith("{") and value.endswith("}")):
        raise ValueError(f"expected a flow mapping, got {value!r}")
    items = (item.split(":", 1) for item in value[1:-1].split(","))
    return {k.strip(): float(v) for k, v in items}


def read_config(path: Path) -> dict[str, tuple[Truth, int]]:
    """Each scenario's :class:`Truth` (its ``pedigree`` block) and ``replicates``."""
    pedigree: dict[str, dict[str, str]] = {}
    replicates: dict[str, int] = {}
    scenario, block = "", None
    for indent, key, value in _yaml_lines(path):
        if indent == 0:
            scenario, block = key, None
            pedigree[scenario] = {}
        elif indent == 2:
            block = key if value is None else None
            if key == "replicates" and value is not None:
                replicates[scenario] = int(value)
        elif indent == 4 and block == "pedigree" and value is not None:
            pedigree[scenario][key] = value
    scenarios = {}
    for name, values in pedigree.items():
        try:
            t1, t2 = _flow(values["trait1"]), _flow(values["trait2"])
            truth = Truth(
                A=(t1["A"], t2["A"]),
                C=(t1["C"], t2["C"]),
                E=(t1["E"], t2["E"]),
                rA=float(values["rA"]),
                rC=float(values["rC"]),
                rE=float(values["rE"]),
            )
            scenarios[name] = (truth, replicates[name])
        except (KeyError, ValueError) as error:
            sys.exit(f"{path}: scenario {name} lacks its pedigree truth or replicates ({error!r})")
    return scenarios


def read_run(path: Path) -> dict[str, str]:
    """A ``run.yaml``'s top-level scalars and its ``resolved`` scalars, the latter as ``resolved.<key>``."""
    out, block = {}, None
    for indent, key, value in _yaml_lines(path):
        if indent == 0:
            block = key if value is None else None
            if value is not None:
                out[key] = value
        elif indent == 2 and block == "resolved" and value is not None:
            out[f"resolved.{key}"] = value
    return out


def check_run(run: dict[str, str], truth: Truth, path: Path) -> None:
    resolved = {
        "A1": truth.A[0],
        "A2": truth.A[1],
        "C1": truth.C[0],
        "C2": truth.C[1],
        "E1": truth.E[0],
        "E2": truth.E[1],
        "rA": truth.rA,
        "rC": truth.rC,
        "rE": truth.rE,
    }
    for key, value in resolved.items():
        if float(run[f"resolved.{key}"]) != value:
            sys.exit(f"{path}: resolved {key} {run[f'resolved.{key}']} differs from the config's {value}")


def load_rep(path: Path) -> Rep:
    pedigree = pl.read_parquet(path / "pedigree.parquet")
    cohort = pl.read_parquet(path / "cohort.parquet").filter(pl.col("t1").is_not_null())
    for column in ("t2", "affected1", "affected2"):
        if cohort[column].null_count():
            sys.exit(f"{path}: cohort rows with t1 lack {column}")
    index = pedigree.select("id").with_row_index("row")
    joined = cohort.select("id").join(index, on="id", how="left", maintain_order="left")
    if joined["row"].null_count():
        sys.exit(f"{path}: cohort ids missing from the pedigree")
    return Rep(path, pedigree, cohort, joined["row"].to_numpy(), read_run(path / "run.yaml"))


def on_pedigree(rep: Rep, values: np.ndarray) -> np.ndarray:
    out = np.full(len(rep.pedigree), np.nan)
    out[rep.rows] = values
    return out


def estimate_rg_formula(fit: relative_pairs.AceFit) -> tuple[float | None, str | None, dict[str, float | None]]:
    """fitACE ``estimate_rg``'s rA on the cells this fit read, unclamped (``estimate_rg`` clamps to [-1, 1])."""
    value = {(r.entry, r.first_trait, r.second_trait): r.value for r in fit.rows}
    within, cross = value[("within_person", 0, 1)], value[("FS", 0, 1)]
    fs1, fs2 = value[("FS", 0, 0)], value[("FS", 1, 1)]
    h1 = None if fs1 is None else 2 * fs1
    h2 = None if fs2 is None else 2 * fs2
    parts = {"rho_within": within, "rho_cross_FS": cross, "h2_1": h1, "h2_2": h2}
    if within is None or cross is None or h1 is None or h2 is None:
        return None, "undefined_cell", parts
    if h1 * h2 <= 0:
        return None, "nonpositive_h2", parts
    return 2 * (within - cross) / math.sqrt(h1 * h2), None, parts


def wls_gap(fit: relative_pairs.AceFit) -> float:
    """The largest gap between the fit's A and C and NumPy's weighted least squares over the same rows."""
    gaps = []
    for shares in fit.traits:
        rows = [
            r for r in fit.rows if r.used and not r.within_person and r.first_trait == r.second_trait == shares.trait
        ]
        X = np.array([[2 * (r.kinship or 0.0), 1.0 if r.shared else 0.0] for r in rows])
        y = np.array([r.value for r in rows], dtype=np.float64)
        w = np.sqrt(np.array([r.weight for r in rows], dtype=np.float64))
        A, C = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        assert shares.A is not None
        assert shares.C is not None
        gaps += [abs(A - shares.A), abs(C - shares.C)]
    return float(max(gaps))


def fit_values(fit: relative_pairs.AceFit) -> dict[str, dict[str, object]]:
    """Each target's ``value``, ``reason`` and ``out_of_range`` in one fit."""
    t1, t2 = fit.traits
    cross = fit.cross
    assert cross is not None
    out: dict[str, dict[str, object]] = {
        "A1": {"value": t1.A, "reason": t1.A_reason},
        "A2": {"value": t2.A, "reason": t2.A_reason},
        "C1": {"value": t1.C, "reason": t1.C_reason},
        "C2": {"value": t2.C, "reason": t2.C_reason},
        "rA": {"value": cross.rA, "reason": cross.rA_reason, "out_of_range": cross.rA_out_of_range},
        "rC": {"value": cross.rC, "reason": cross.rC_reason, "out_of_range": cross.rC_out_of_range},
        "rE": {"value": cross.rE, "reason": cross.rE_reason, "out_of_range": cross.rE_out_of_range},
    }
    value, reason, parts = estimate_rg_formula(fit)
    out["estimate_rg_formula"] = {"value": value, "reason": reason, **parts}
    return out


def _side(columns: list[np.ndarray], liability: np.ndarray, strata: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Each column at *rows*, centred on its mean within the entries' stratum, over the liability's 1/N SD there."""
    _, ids = np.unique(strata[rows], return_inverse=True)
    n = np.bincount(ids)

    def centred(x: np.ndarray) -> np.ndarray:
        values = x[rows]
        return values - (np.bincount(ids, values) / n)[ids]

    deviation = centred(liability)
    sd = np.sqrt(np.bincount(ids, deviation * deviation) / n)[ids]
    return np.stack([centred(c) / sd for c in columns])


def decomposition(
    rep: Rep, res: relative_pairs.RelativePairCorrelation, view: PedigreeView
) -> dict[tuple[str, int, int, str], np.ndarray]:
    """Each liability cell, per estimate, as its 4 x 4 component-pair terms (A, C, E, R).

    A Pearson cell is ``(1/N) sum z1 z2`` over its entries, each side
    standardised by its stratum's mean and 1/N SD over the cell's entries
    (one stratum when pooled).  With ``L = A + C + E + R`` (R the float32
    rounding of the stored components), the cell is exactly the sum over
    component pairs X, Y of ``(1/N) sum (X_i - m_X)(Y_j - m_Y) / (sd_L sd_L)``.
    """
    rows = rep.pedigree[rep.rows]
    generation = rows["generation"].to_numpy()
    strata = {"pooled": np.zeros_like(generation), "stratified": generation}
    parts = {}
    for t in (0, 1):
        liability = rows[f"liability{t + 1}"].to_numpy().astype(np.float64)
        components = [rows[f"{X}{t + 1}"].to_numpy().astype(np.float64) for X in "ACE"]
        parts[t] = (liability, [*components, liability - sum(components)])
    pairs = view.relationship_pairs(categories=sorted({c.category for c in res.cells}))
    out = {}
    assert res.within_person is not None
    for cell in (*res.cells, res.within_person):
        if cell.category == "within_person":
            first = second = np.arange(len(rows))
        elif cell.n == 0:
            continue
        else:
            a, b = (np.asarray(x, dtype=np.int64) for x in pairs[cell.category])
            first, second = (np.concatenate([a, b]), np.concatenate([b, a])) if cell.symmetric else (a, b)
        t, u = cell.first_trait, cell.second_trait
        for estimate in ESTIMATES:
            U = _side(parts[t][1], parts[t][0], strata[estimate], first)
            W = _side(parts[u][1], parts[u][0], strata[estimate], second)
            out[(cell.category, t, u, estimate)] = U @ W.T / len(first)
    return out


def realized_result(
    res: relative_pairs.RelativePairCorrelation, terms: dict[tuple[str, int, int, str], np.ndarray], estimate: str
) -> tuple[relative_pairs.RelativePairCorrelation, float, float]:
    """*res* with each cell's *estimate* replaced by its realized truth, the decomposition's gap, and the largest R term.

    The realized truth keeps the A x A and C x C terms, and E x E in the
    within-person cell; the dropped terms (E between relatives, X x Y
    across components, R) have expectation 0.
    """
    gaps, residual = [0.0], [0.0]

    def realized(cell: relative_pairs.Cell) -> relative_pairs.Cell:
        key = (cell.category, cell.first_trait, cell.second_trait, estimate)
        stratified = cell.stratified
        if estimate == "pooled":
            observed = cell.primary.value
        else:
            observed = None if stratified is None else stratified.estimate.value
        if key not in terms or observed is None:
            return cell
        T = terms[key]
        gaps.append(abs(float(T.sum()) - observed))
        residual.append(float(max(np.abs(T[3]).max(), np.abs(T[:, 3]).max())))
        value = float(T[0, 0] + T[1, 1] + (T[2, 2] if cell.category == "within_person" else 0.0))
        if estimate == "pooled":
            return replace(cell, crude=(replace(cell.primary, value=value, reason=None), *cell.crude[1:]))
        assert stratified is not None
        return replace(
            cell, stratified=replace(stratified, estimate=replace(stratified.estimate, value=value, reason=None))
        )

    within = res.within_person
    out = replace(
        res, cells=tuple(realized(c) for c in res.cells), within_person=None if within is None else realized(within)
    )
    return out, max(gaps), max(residual)


def run_rep(rep: Rep, outcomes: list[Outcome], view: PedigreeView) -> dict[str, dict[str, dict[str, object]]]:
    """Per outcome and estimate: the targets, every row the fit read, and the realized truth's targets."""
    native = Pedigree({c: rep.pedigree[c].to_numpy() for c in ("id", "mother", "father", "twin", "sex")})
    generation = on_pedigree(rep, rep.pedigree["generation"].to_numpy()[rep.rows].astype(np.float64))
    out: dict[str, dict[str, dict[str, object]]] = {}
    for outcome in outcomes:
        traits = [Trait(on_pedigree(rep, outcome.values(rep, t)), kind=outcome.kind) for t in (1, 2)]
        res = relative_pairs.correlation(native, traits, stratum=generation)
        out[outcome.name] = {}
        terms = decomposition(rep, res, view) if outcome.realized_truth else {}
        for estimate in ESTIMATES:
            fit = relative_pairs.ace(res, shared_environment=list(SHARED_ENVIRONMENT), estimate=estimate)
            out[outcome.name][estimate] = {
                "targets": fit_values(fit),
                "rows": fit.rows,
                "numpy_wls_gap": wls_gap(fit),
                "metadata": dict(res.metadata),
            }
            if outcome.realized_truth:
                truth_res, gap, residual = realized_result(res, terms, estimate)
                if gap > DECOMPOSITION_TOLERANCE:
                    sys.exit(f"{rep.path} {outcome.name} {estimate}: component terms miss the cell by {gap:.1e}")
                truth_fit = relative_pairs.ace(
                    truth_res, shared_environment=list(SHARED_ENVIRONMENT), estimate=estimate
                )
                out[outcome.name][estimate] |= {
                    "realized_targets": fit_values(truth_fit),
                    "decomposition_gap": gap,
                    "residual_term": residual,
                }
    return out


def summarise(values: list[float | None], reasons: list[str | None], truth: float, reps: list[int]) -> dict:
    defined = np.array([v for v in values if v is not None], dtype=np.float64)
    n = len(defined)
    mean = float(defined.mean()) if n else None
    sd = float(defined.std(ddof=1)) if n > 1 else None
    mc_se = None if sd is None else sd / math.sqrt(n)
    within = None if mean is None or mc_se is None else abs(mean - truth) <= MC_SE_BOUND * mc_se
    return {
        "truth": truth,
        "mean": mean,
        "sd": sd,
        "mc_se": mc_se,
        "z": None if mean is None or not mc_se else (mean - truth) / mc_se,
        "within_2_mc_se": within,
        "n_defined": n,
        "undefined_reps": [
            {"rep": r, "reason": why} for r, v, why in zip(reps, values, reasons, strict=True) if v is None
        ],
        "per_rep": dict(zip(map(str, reps), values, strict=True)),
    }


def cell_diagnostics(fits: list[dict], truth: Truth, reps: list[int]) -> list[dict]:
    """Each cell's rep-mean correlation against ``2 phi A + s C`` (or its cross-trait form)."""
    first: tuple[relative_pairs.AceRow, ...] = fits[0]["rows"]
    out = []
    for i, row in enumerate(first):
        values = [f["rows"][i].value for f in fits]
        reasons = [f["rows"][i].reason for f in fits]
        summary = summarise(values, reasons, truth.expected(row), reps)
        out.append(
            {
                "entry": row.entry,
                "first_trait": row.first_trait,
                "second_trait": row.second_trait,
                "kinship": row.kinship,
                "shared": row.shared,
                "mean_weight": float(np.mean([f["rows"][i].weight for f in fits])),
                "expected": summary.pop("truth"),
                **{k: summary[k] for k in ("mean", "mc_se", "z", "n_defined")},
            }
        )
    return out


def scenario_outcomes(models: set[str]) -> list[tuple[Outcome, bool]]:
    """The outcomes of a scenario and whether D10 gates them."""
    if models == {LTM}:
        return [(LIABILITY, True), (UNCENSORED_STATUS, True), (GENETIC_VALUE, False)]
    return [(LIABILITY, False), (AFFECTED, False), (GENETIC_VALUE, False)]


def brief(summary: dict) -> dict:
    """A summary without its per-rep values."""
    return {k: v for k, v in summary.items() if k not in ("per_rep", "undefined_reps")}


def exact_kinship(view: PedigreeView, nominal: dict[str, float]) -> dict[str, dict[str, float]]:
    """Each category's mean exact kinship over nominal, and the share of its pairs above nominal."""
    kinship = view.pair_kinship(view.relationship_pairs(categories=list(nominal)))
    assert not isinstance(kinship, np.ndarray)
    out = {}
    for code, phi in nominal.items():
        values = np.asarray(kinship[code], dtype=np.float64) if code in kinship else np.empty(0)
        if len(values):
            out[code] = {"ratio": float(values.mean() / phi), "share_above": float((values > phi).mean())}
    return out


def realized(rep: Rep) -> dict[str, float | None]:
    """The simulated components' shares and cross-trait correlations on the phenotyped rows.

    Within each generation, then averaged weighted by its rows; a correlation
    of a constant component is ``None``.
    """
    rows = rep.pedigree[rep.rows]
    out: dict[str, list[tuple[float, float]]] = {}
    for _, gen in rows.group_by("generation"):
        weight = len(gen) / len(rows)
        cols = {c: gen[c].to_numpy().astype(np.float64) for c in gen.columns}
        for t in (1, 2):
            var = cols[f"liability{t}"].var()
            for c in "ACE":
                out.setdefault(f"{c}{t}", []).append((weight, cols[f"{c}{t}"].var() / var))
        for c in "ACE":
            x, y = cols[f"{c}1"], cols[f"{c}2"]
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else math.nan
            out.setdefault(f"r{c}", []).append((weight, r))
    shares = {k: sum(w * v for w, v in pairs) for k, pairs in out.items()}
    return {k: None if math.isnan(v) else float(v) for k, v in shares.items()}


def run_scenario(directory: Path, truth: Truth, replicates: int) -> dict:
    paths = sorted(directory.glob("rep*"), key=lambda p: int(p.name[3:]))
    reps = [int(p.name[3:]) for p in paths]
    if reps != list(range(1, replicates + 1)):
        sys.exit(f"{directory}: reps {reps}, config replicates {replicates}")
    per_rep, runs, walls, sizes, realized_shares, kinships, plan = [], [], [], [], [], [], None
    for path in paths:
        start = time.perf_counter()
        rep = load_rep(path)
        check_run(rep.run, truth, path)
        models = {rep.run["resolved.phenotype_model1"], rep.run["resolved.phenotype_model2"]}
        plan = plan or scenario_outcomes(models)
        view = PedigreeGraph.from_frame(rep.pedigree).view(ids=rep.cohort["id"].to_numpy())
        per_rep.append(run_rep(rep, [o for o, _ in plan], view))
        realized_shares.append(realized(rep))
        first = per_rep[-1][plan[0][0].name][GATED_ESTIMATE]["rows"]
        nominal = {r.entry: r.kinship for r in first if r.kinship is not None and r.first_trait == r.second_trait == 0}
        kinships.append(exact_kinship(view, nominal))
        walls.append(time.perf_counter() - start)
        runs.append({k: rep.run.get(k) for k in ("simace_version", "source", "seed", "finished")})
        sizes.append({"rep": rep.run["rep"], "n_rows": len(rep.pedigree), "n_phenotyped": len(rep.rows)})
        print(f"  {path.name}: {walls[-1]:.1f} s", file=sys.stderr, flush=True)
    assert plan is not None
    outcomes = {}
    for outcome, gated in plan:
        outcome_truth = outcome.truth(truth)
        estimates = {}
        for estimate in ESTIMATES:
            fits = [r[outcome.name][estimate] for r in per_rep]
            targets = {}
            for name, truth_value in outcome_truth.targets().items():
                values = [f["targets"][name]["value"] for f in fits]
                reasons = [f["targets"][name]["reason"] for f in fits]
                summary = summarise(values, reasons, truth_value, reps)
                summary["out_of_range_reps"] = [
                    r for r, f in zip(reps, fits, strict=True) if f["targets"][name].get("out_of_range") is True
                ]
                summary["gated"] = gated and estimate == GATED_ESTIMATE and name in outcome_truth.gated()
                summary["gate_against"] = "realized" if outcome.realized_truth else "nominal"
                verdict = summary
                if outcome.realized_truth:
                    truths = [f["realized_targets"][name]["value"] for f in fits]
                    defined = [x for x in truths if x is not None]
                    differences = [
                        None if v is None or x is None else v - x for v, x in zip(values, truths, strict=True)
                    ]
                    why = [
                        r
                        if v is None
                        else f"realized truth {f['realized_targets'][name]['reason']}"
                        if x is None
                        else None
                        for v, x, r, f in zip(values, truths, reasons, fits, strict=True)
                    ]
                    verdict = summarise(differences, why, 0.0, reps)
                    summary["realized_truth"] = {
                        "per_rep": dict(zip(map(str, reps), truths, strict=True)),
                        "mean": float(np.mean(defined)) if defined else None,
                        "difference": {k: v for k, v in verdict.items() if k != "per_rep"},
                    }
                summary["gate_z"] = verdict["z"]
                summary["gate_every_rep_defined"] = not verdict["undefined_reps"]
                summary["pass"] = None
                if name == "estimate_rg_formula":
                    summary["parts_mean"] = {
                        k: float(np.mean([f["targets"][name][k] for f in fits if f["targets"][name][k] is not None]))
                        for k in ("rho_within", "rho_cross_FS", "h2_1", "h2_2")
                    }
                targets[name] = summary
            estimates[estimate] = {
                "targets": targets,
                "numpy_wls_max_gap": max(f["numpy_wls_gap"] for f in fits),
            }
            if estimate == GATED_ESTIMATE:
                estimates[estimate]["cells"] = cell_diagnostics(fits, outcome_truth, reps)
            if outcome.realized_truth:
                estimates[estimate]["decomposition"] = {
                    "max_gap": max(f["decomposition_gap"] for f in fits),
                    "max_residual_term": max(f["residual_term"] for f in fits),
                    "tolerance": DECOMPOSITION_TOLERANCE,
                }
        outcomes[outcome.name] = {
            "kind": outcome.kind,
            "source": outcome.source,
            "truth": outcome_truth.targets(),
            "gated": gated,
            "estimates": estimates,
        }
    return {
        "truth": truth.targets() | {"E1": truth.E[0], "E2": truth.E[1]},
        "gated_targets": sorted(truth.gated()),
        "reps": [r | s for r, s in zip(runs, sizes, strict=True)],
        "wall_seconds": sum(walls),
        "exact_kinship": {
            code: {
                "ratio": brief(
                    summarise([k.get(code, {}).get("ratio") for k in kinships], ["no_pairs"] * len(reps), 1.0, reps)
                ),
                "share_above_mean": float(np.mean([k[code]["share_above"] for k in kinships if code in k])),
            }
            for code in sorted({c for k in kinships for c in k})
        },
        "realized_shares": {
            name: brief(summarise([r[name] for r in realized_shares], ["constant_component"] * len(reps), value, reps))
            for name, value in (truth.targets() | {"E1": truth.E[0], "E2": truth.E[1]}).items()
            if name != "estimate_rg_formula"
        },
        "pg_phenotype_metadata": per_rep[0][plan[0][0].name][GATED_ESTIMATE]["metadata"],
        "outcomes": outcomes,
    }


def stress(scenario: dict) -> dict:
    """D10's stress criterion: stratified affected rA closer to the truth than fitACE#5's 0.308."""
    est = scenario["outcomes"]["affected"]["estimates"]
    ra = est[GATED_ESTIMATE]["targets"]["rA"]
    truth = ra["truth"]
    improves = (
        ra["mean"] is not None
        and not ra["undefined_reps"]
        and abs(ra["mean"] - truth) < abs(ESTIMATE_RG_FITACE_5 - truth)
    )
    return {
        "criterion": f"stratified ACE rA defined in every rep and |mean - {truth}| < |{ESTIMATE_RG_FITACE_5} - {truth}|",
        "estimate_rg_fitace_5": ESTIMATE_RG_FITACE_5,
        "truth_rA": truth,
        "pass": improves,
        "rA": {e: est[e]["targets"]["rA"]["mean"] for e in ESTIMATES},
        "estimate_rg_formula": {e: est[e]["targets"]["estimate_rg_formula"]["mean"] for e in ESTIMATES},
        "attenuation": {
            e: {k: est[e]["targets"][k]["mean"] for k in ("A1", "A2", "C1", "C2", "rA", "rC", "rE")} for e in ESTIMATES
        },
    }


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def fmt(x: object, digits: int = 4) -> str:
    return "-" if x is None else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def print_tables(scenarios: dict) -> None:
    for name, scenario in scenarios.items():
        for outcome_name, outcome in scenario["outcomes"].items():
            print(f"\n### {name}: {outcome_name}\n")
            if any("realized_truth" in s for s in outcome["estimates"]["stratified"]["targets"].values()):
                print(
                    "| target | nominal truth | stratified mean (MC SE) | z vs nominal | realized truth "
                    "| fit - realized (MC SE) | z | within 2 MC SE | undefined | pass | pooled fit - realized |"
                )
                print("|---|---|---|---|---|---|---|---|---|---|---|")
                for target, s in outcome["estimates"]["stratified"]["targets"].items():
                    d = s["realized_truth"]["difference"]
                    pd = outcome["estimates"]["pooled"]["targets"][target]["realized_truth"]["difference"]
                    undefined = ",".join(str(u["rep"]) for u in d["undefined_reps"]) or "-"
                    verdict = {True: "pass", False: "**FAIL**", None: "reported"}[s["pass"]]
                    print(
                        f"| {target} | {fmt(s['truth'])} | {fmt(s['mean'])} ({fmt(s['mc_se'])}) | {fmt(s['z'], 1)} "
                        f"| {fmt(s['realized_truth']['mean'])} | {fmt(d['mean'])} ({fmt(d['mc_se'])}) | {fmt(d['z'], 1)} "
                        f"| {fmt(d['within_2_mc_se'])} | {undefined} | {verdict} | {fmt(pd['mean'])} ({fmt(pd['mc_se'])}) |"
                    )
                continue
            print(
                "| target | truth | stratified mean (MC SE) | z | within 2 MC SE | undefined | pass | pooled mean (MC SE) |"
            )
            print("|---|---|---|---|---|---|---|---|")
            for target, s in outcome["estimates"]["stratified"]["targets"].items():
                p = outcome["estimates"]["pooled"]["targets"][target]
                undefined = ",".join(str(u["rep"]) for u in s["undefined_reps"]) or "-"
                verdict = {True: "pass", False: "**FAIL**", None: "reported"}[s["pass"]]
                print(
                    f"| {target} | {fmt(s['truth'])} | {fmt(s['mean'])} ({fmt(s['mc_se'])}) | {fmt(s['z'], 1)} "
                    f"| {fmt(s['within_2_mc_se'])} | {undefined} | {verdict} | {fmt(p['mean'])} ({fmt(p['mc_se'])}) |"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    scenarios = {}
    for name, (truth, replicates) in read_config(args.config).items():
        print(f"{name}", file=sys.stderr, flush=True)
        scenarios[name] = run_scenario(args.results / name, truth, replicates)
    stress_checks = {name: stress(s) for name, s in scenarios.items() if "affected" in s["outcomes"]}
    gated = [
        t
        for s in scenarios.values()
        for o in s["outcomes"].values()
        for t in o["estimates"][GATED_ESTIMATE]["targets"].values()
        if t["gated"]
    ]
    critical_z = NormalDist().inv_cdf(1 - FAMILYWISE_ALPHA / (2 * len(gated)))
    for t in gated:
        t["pass"] = t["gate_every_rep_defined"] and t["gate_z"] is not None and abs(t["gate_z"]) <= critical_z
    failures = [
        f"{name} {outcome_name} {target}"
        for name, s in scenarios.items()
        for outcome_name, o in s["outcomes"].items()
        for target, t in o["estimates"][GATED_ESTIMATE]["targets"].items()
        if t["pass"] is False
    ] + [f"{name} stress rA" for name, s in stress_checks.items() if not s["pass"]]
    tool = Path(__file__).resolve()
    report = {
        "result": "fail" if failures else "pass",
        "failures": failures,
        "run": {
            "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "command": [str(Path(sys.argv[0])), *sys.argv[1:]],
            "tool": str(tool.relative_to(ROOT)),
            "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
            "pg_phenotype_rev": git("rev-parse", "HEAD"),
            "pg_phenotype_describe": git("describe", "--always", "--dirty"),
            "pg_phenotype_tree_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
            "results": str(args.results),
            "config": str(args.config),
            "wall_seconds": time.perf_counter() - start,
        },
        "gate": {
            "estimate": GATED_ESTIMATE,
            "rule": (
                "every rep defined and |z| <= the Bonferroni critical value; z is (mean - truth) / (SD / sqrt(reps)) "
                "against nominal truth, or mean(fit - realized) / (SD / sqrt(reps)) against realized truth"
            ),
            "familywise_alpha": FAMILYWISE_ALPHA,
            "gated_targets": len(gated),
            "critical_z": critical_z,
            "reported_flag": f"within_2_mc_se: |mean - truth| <= {MC_SE_BOUND} * SD / sqrt(reps)",
            "shared_environment": list(SHARED_ENVIRONMENT),
        },
        "stress": stress_checks,
        "scenarios": scenarios,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print_tables(scenarios)
    print("\nstress:", json.dumps(stress_checks, indent=1))
    print(f"\ngated targets {len(gated)}, critical |z| {critical_z:.3f}")
    print(f"\nresult: {report['result']}; wall {report['run']['wall_seconds']:.0f} s")
    if failures:
        sys.exit("gated targets failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
