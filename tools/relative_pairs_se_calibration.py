"""Calibrate cluster designs for the Relative-Pair Correlation SE (issue #18) on simACE replicates.

    pixi run python -B tools/relative_pairs_se_calibration.py --self-test
    PG_PHENOTYPE_THREADS=1 RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 pixi run python -B tools/relative_pairs_se_calibration.py \\
        <simACE>/results/se_calibration [--rust] --jobs 11 --out docs/gates/relative-pairs-se

For every rep of every scenario under the results folder, compute each
Relative-Pair Correlation cell (ADR 0009) of each trait: Pearson on
``liability{t}`` and the tetrachoric and phi on ``affected{t}``, both
restricted to the cohort's rows.  A cell is a category or Category Group,
pooled or in one generation's stratum.  For each cell, form each unordered
pair's influence φ_d on r̂ (r̂ - r ≈ Σ_d φ_d / n) and the sandwich variance
Σ meat / n² under six cluster designs (``docs/relative-pairs-se-research.md``
Q2, Q3, Q5):

``pair``         every unordered pair its own cluster: Σ_d φ_d²
``person``       dyadic, pairs sharing a member: Σ_i S_i² - Σ_d φ_d²
``mn``           the Mate Network of the pair's top mating: Σ_g Φ_g²
``mn_person``    same Mate Network or a shared member, by inclusion-exclusion:
                 mn + person - (Σ_(i,g) S_(i,g)² - Σ_d φ_d²)
``mrca``         the pair's most recent common ancestor, one person
``mrca_person``  same MRCA or a shared member, as ``mn_person``

A pair's top mating is the mating of its most recent common ancestor (or
ancestors) whose child x lies on the path down to the first member: the
first member's own parents for MZ, FS and the half sibs, the offspring's
parents for MO and FO, the grandparents for 1C and Av.  Mate Networks are
the whole pedigree's, read at that mating's parents (ADR 0007's policy).
The MRCA is the ancestor member of a lineal pair, the shared parent of x and
its counterpart on the second member's side for a half relationship, and
x's mother for a full one (the couple, merged with the mother's other
matings).  Unlike a Mate Network it cannot chain through serial mates.
No small-sample factor is applied: every cell has thousands of clusters.

On the first rep of each scenario, every pooled r̂ is checked against
``pg_phenotype.relative_pairs.correlation`` and the MRCA of 300 sampled pairs
per category against an independent walk up the pedigree.  Writes ``calibration.tsv`` (per scenario and
cell: SD of r̂ across reps over each design's mean SE, the SE's spread, and
Wald coverage) and ``summary.tsv`` (medians over cells per scenario and
category type) into ``--out``; ``--reps`` also keeps the per-rep rows, and
``--from-reps`` rebuilds both tables from them.

``--rust`` calibrates the shipped SE instead, through the public API: per
rep, ``relative_pairs.correlation(..., se=True)`` with both traits and the
generation strata, for the same cells plus each cell's stratified fit, and
beside each pooled and stratum cell this script's ``mrca_person`` SE and r̂
for the same rep.  Writes ``rust_calibration.tsv`` (as ``calibration.tsv``,
one design ``rust``, with ``no_se_<reason>`` counting reps whose estimate
has no SE, and ``mean_n11``, the binary cell's mean count of pairs with both
affected) and ``rust_summary.tsv`` (medians per scenario, category type,
estimator and fit: ``cells`` for pooled and stratum cells, ``stratified``),
and prints the largest gap between the two SEs.  A boundary tetrachoric
has no influence function and is left out of the r̂ statistics, as above.
Run it with the thread variables at 1: the workers are the parallelism.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from pedigree_graph import RELATIONSHIPS, PedigreeGraph
from scipy import integrate, optimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import ndtr, ndtri

from pg_phenotype import Trait
from pg_phenotype import correlation as pgc
from pg_phenotype import relative_pairs as rp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.oracle.relative_pairs_se import bvn_pdf, meats, pearson_if, tetrachoric_if, top_keys
from tools import _calibration_runner as runner

DESIGNS = ("pair", "person", "mn", "mn_person", "mrca", "mrca_person")
CODES = tuple(code for code, c in RELATIONSHIPS.items() if c.degree <= 3)
#: Cells: name -> member codes.  Groups as fitACE's Falconer step reports them.
CELLS: dict[str, tuple[str, ...]] = {code: (code,) for code in CODES} | {"PO": ("MO", "FO"), "HS": ("MHS", "PHS")}
SYMMETRIC = frozenset(code for code, c in RELATIONSHIPS.items() if c.first_role is None)
#: Pooled r̂ against pg-phenotype: Pearson and phi exactly, the tetrachoric to its fit's tolerance.
TOL = {"pearson": 1e-10, "phi": 1e-10, "tetrachoric": 1e-8}


# --- influence functions ---------------------------------------------------------------------


def _bvn_upper(h: float, k: float, rho: float) -> float:
    """P(X > h, Y > k) for a standard bivariate normal (Plackett's identity, quadrature to ~1e-14)."""
    if rho == 0:
        return float(ndtr(-h) * ndtr(-k))
    integral, _ = integrate.quad(lambda t: bvn_pdf(h, k, t), 0.0, rho, epsabs=1e-14, epsrel=1e-12)
    return float(ndtr(-h) * ndtr(-k)) + integral


def two_step_tetrachoric(p11: float, pa: float, pb: float) -> float:
    """The 2 x 2 two-step tetrachoric from proportions: thresholds from margins, then ρ matching p11."""
    h, k = -float(ndtri(pa)), -float(ndtri(pb))
    return optimize.brentq(lambda r: _bvn_upper(h, k, r) - p11, -0.9999, 0.9999, xtol=1e-15, rtol=1e-15)


def binary_ifs(x: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, np.ndarray]]:
    """Tetrachoric and phi r̂ with each entry's influence, on 0/1 entries."""
    out: dict[str, tuple[float, np.ndarray]] = {}
    xi, yi = x.astype(np.int64), y.astype(np.int64)
    t = np.zeros((2, 2), dtype=np.int64)
    np.add.at(t, (xi, yi), 1)
    tet = pgc.tetrachoric(table=t.tolist())
    if tet.value is not None and not tet.boundary:
        out["tetrachoric"] = (tet.value, tetrachoric_if(t.astype(float), tet.value)[xi, yi])
    phi = pearson_if(x, y)
    if phi is not None:
        out["phi"] = phi
    return out


def pair_fits(xs: list[np.ndarray], ys: list[np.ndarray], kind: str) -> dict[str, tuple[float, np.ndarray]]:
    """Each estimator's r̂ and each unordered pair's influence φ_d, from its entries in one or two orientations.

    Over k·n entries r̂ - r ≈ Σ_e IF_e / (k n) = Σ_d φ_d / n, with φ_d the mean of the pair's k entries.
    """
    x, y, k = np.concatenate(xs), np.concatenate(ys), len(xs)
    if kind == "continuous":
        res = pearson_if(x, y)
        fits = {"pearson": res} if res is not None else {}
    else:
        fits = binary_ifs(x, y)
    return {est: (r, entry_if.reshape(k, -1).mean(axis=0)) for est, (r, entry_if) in fits.items()}


# --- pedigree structure ----------------------------------------------------------------------


@dataclass(frozen=True)
class Rep:
    scenario: str
    rep: int
    ped: dict[str, np.ndarray]
    generation: np.ndarray
    liability: dict[int, np.ndarray]
    affected: dict[int, np.ndarray]


def load_rep(rep_dir: Path) -> Rep:
    for name in ("pedigree.parquet", "cohort.parquet"):
        if pl.read_parquet_metadata(rep_dir / name).get("simace_layout") != "2":
            raise ValueError(f"{rep_dir / name}: not simACE results layout 2 (ADR 0021)")
    ped = pl.read_parquet(rep_dir / "pedigree.parquet")
    cohort = pl.read_parquet(rep_dir / "cohort.parquet")
    if not (ped["id"] == pl.int_range(len(ped), eager=True)).all():
        raise ValueError(f"{rep_dir}: ids are not rows")
    rows = cohort["id"].to_numpy()
    # Pairs come from the recorded pedigree, which matches the analysis pedigree only when
    # the cohort keeps every recorded parent of its rows: no dropout or N_sample selection.
    kept = np.zeros(len(ped), dtype=bool)
    kept[rows] = True
    for col in ("mother", "father"):
        parent = ped[col].to_numpy()[rows]
        if not kept[parent[parent >= 0]].all():
            raise ValueError(f"{rep_dir}: the cohort drops a recorded parent of its rows (dropout or selection)")
    liability, affected = {}, {}
    for t in (1, 2):
        lv = np.full(len(ped), np.nan)
        # Liability on the rows whose case status is observed, so both kinds share one pair set.
        av = np.full(len(ped), np.nan)
        av[rows] = cohort[f"affected{t}"].cast(pl.Float64).to_numpy()
        seen = ~np.isnan(av)
        lv[seen] = ped[f"liability{t}"].to_numpy()[seen]
        liability[t], affected[t] = lv, av
    cols = {k: ped[k].to_numpy() for k in ("id", "mother", "father", "twin", "sex")}
    return Rep(
        rep_dir.parent.name,
        int(rep_dir.name.removeprefix("rep")),
        cols,
        ped["generation"].to_numpy(),
        liability,
        affected,
    )


def mate_networks(mother: np.ndarray, father: np.ndarray) -> np.ndarray:
    """Each row's Mate Network label: components of the graph joining every child's parents."""
    n = len(mother)
    both = (mother >= 0) & (father >= 0)
    m, f = mother[both], father[both]
    adj = coo_matrix((np.ones(len(m), dtype=np.int8), (m, f)), shape=(n, n))
    return connected_components(adj, directed=False)[1]


@dataclass(frozen=True)
class CellPairs:
    a: np.ndarray
    b: np.ndarray
    g: np.ndarray
    m: np.ndarray
    symmetric: bool


def cell_pairs(rep: Rep) -> dict[str, CellPairs]:
    ped = rep.ped
    graph = PedigreeGraph.from_arrays(
        ids=ped["id"], mother_ids=ped["mother"], father_ids=ped["father"], twin_ids=ped["twin"], sex=ped["sex"]
    )
    pairs = graph.relationship_pairs(categories=list(CODES))
    net = mate_networks(ped["mother"], ped["father"])
    by_code = {}
    for code in CODES:
        a = np.asarray(pairs[code].first_rows, dtype=np.int64)
        b = np.asarray(pairs[code].second_rows, dtype=np.int64)
        by_code[code] = (a, b, *top_keys(code, a, b, ped["mother"], ped["father"], net))
    out = {}
    for name, members in CELLS.items():
        a, b, g, m = (np.concatenate([by_code[c][i] for c in members]) for i in range(4))
        out[name] = CellPairs(a, b, g, m, members[0] in SYMMETRIC)
    return out


# --- one rep ---------------------------------------------------------------------------------


def rep_rows(rep_dir: Path, check: bool) -> list[dict]:
    rep = load_rep(rep_dir)
    cells = cell_pairs(rep)
    rows = []
    for name, cp in cells.items():
        gen_a, gen_b = rep.generation[cp.a], rep.generation[cp.b]
        # A directed pair's stratum is its first role's generation; a symmetric pair's, the one both share.
        strata = [("pooled", np.ones(len(cp.a), dtype=bool))] + [
            (f"gen{gen}", (gen_a == gen) & ((gen_b == gen) if cp.symmetric else True)) for gen in np.unique(gen_a)
        ]
        # (first member's trait, second member's): a symmetric cell's cross-trait combination enters both orientations.
        combos = [(1, 1), (2, 2), (1, 2)] + ([] if cp.symmetric else [(2, 1)])
        for (t1, t2), (stratum, sel) in itertools.product(combos, strata):
            trait = f"trait{t1}" if t1 == t2 else f"trait{t1}_x_trait{t2}"
            for kind, values in (("continuous", rep.liability), ("binary", rep.affected)):
                a, b = cp.a[sel], cp.b[sel]
                xs, ys = [values[t1][a]], [values[t2][b]]
                if cp.symmetric:
                    xs, ys = [*xs, values[t1][b]], [*ys, values[t2][a]]
                # A pair enters only with every orientation complete.
                ok = np.logical_and.reduce([~np.isnan(v) for v in (*xs, *ys)])
                fits = pair_fits([v[ok] for v in xs], [v[ok] for v in ys], kind)
                a, b, g, mrca = (v[sel][ok] for v in (cp.a, cp.b, cp.g, cp.m))
                n = int(ok.sum())
                for estimator, (r, phi) in fits.items():
                    m = meats(phi, a, b, g, mrca)
                    sizes = np.bincount(g)
                    row = {
                        "scenario": rep.scenario,
                        "rep": rep.rep,
                        "trait": trait,
                        "cell": name,
                        "stratum": stratum,
                        "estimator": estimator,
                        "n": n,
                        "n_networks": int(np.count_nonzero(sizes)),
                        "largest_network_share": float(sizes.max() / n),
                        "r": r,
                        "mean_if": float(phi.mean()),
                    }
                    for d in DESIGNS:
                        row[f"se_{d}"] = math.sqrt(m[d]) / n if m[d] > 0 else math.nan
                    rows.append(row)
    if check:
        for t in (1, 2):
            check_against_pg(rep, t, rows)
        check_mrca(rep, cells)
    return rows


#: Why an estimate with a value has no SE (``Estimate.se_reason``), but ``boundary``, whose rows
#: ``rust_summaries`` counts and then leaves out.
SE_REASONS = ("sandwich_undefined", "degenerate_meat", "negative_meat")


def rust_rep_rows(rep_dir: Path) -> list[dict]:
    """Each estimate's r̂ and Rust SE, with this script's r̂ and ``mrca_person`` SE beside it."""
    rep = load_rep(rep_dir)
    tool = {(r["trait"], r["cell"], r["stratum"], r["estimator"]): r for r in rep_rows(rep_dir, check=False)}
    groups = {name: list(members) for name, members in CELLS.items() if len(members) > 1}
    rows = []
    for kind, values in (("continuous", rep.liability), ("binary", rep.affected)):
        traits = [Trait(values[t], kind) for t in (1, 2)]
        # A code may not sit in two entries of one call: the codes, then the groups.
        for entries in (list(CODES), [groups]):
            result = rp.correlation(rep.ped, traits, categories=entries, stratum=rep.generation, se=True)
            for cell in result.cells:
                t1, t2 = cell.first_trait + 1, cell.second_trait + 1
                trait = f"trait{t1}" if t1 == t2 else f"trait{t1}_x_trait{t2}"
                fits = [("pooled", cell.n, cell, cell.crude)]
                fits += [(f"gen{label}", s.n, s, s.crude) for label, s in cell.strata.items()]
                # The stratified fit's concordance is read from the pooled cell's.
                fits.append(("stratified", cell.stratified.n, cell, (cell.stratified.estimate,)))
                for stratum, n, counts, estimates in fits:
                    if counts.concordance is not None:
                        n11 = counts.concordance[2]
                    elif counts.table is not None:
                        n11 = counts.table[1][1]
                    else:
                        n11 = None
                    for est in estimates:
                        if est.value is None:
                            continue
                        mine = tool.get((trait, cell.category, stratum, est.estimator), {})
                        rows.append(
                            {
                                "scenario": rep.scenario,
                                "rep": rep.rep,
                                "trait": trait,
                                "cell": cell.category,
                                "stratum": stratum,
                                "estimator": est.estimator,
                                "n": n,
                                "n11": n11,
                                "r": est.value,
                                "se_rust": est.se,
                                "se_reason": est.se_reason,
                                "r_tool": mine.get("r"),
                                "se_tool": mine.get("se_mrca_person"),
                            }
                        )
    return rows


def require(ok: bool, what: str) -> None:
    """A check that survives ``python -O``."""
    if not ok:
        raise RuntimeError(what)


def check_mrca(rep: Rep, cells: dict[str, CellPairs], sample: int = 300) -> None:
    """Each sampled pair's MRCA is an ancestor of both members at the category's depths.

    An independent per-pair walk over parent rows.  Parentless MZ twins are their own
    cluster, keyed past the last row.
    """
    mother, father = rep.ped["mother"], rep.ped["father"]

    def up(row: int, depth: int) -> set[int]:
        level = {row}
        for _ in range(depth):
            level = {p for i in level for p in (mother[i], father[i]) if p >= 0}
        return level

    rng = np.random.default_rng(rep.rep)
    for code in CODES:
        c, cp = RELATIONSHIPS[code], cells[code]
        for d in rng.choice(len(cp.a), size=min(sample, len(cp.a)), replace=False):
            a, b, m = int(cp.a[d]), int(cp.b[d]), int(cp.m[d])
            where = f"{rep.scenario} rep{rep.rep} {code} pair ({a}, {b}) MRCA {m}"
            if m >= len(mother):
                require(code == "MZ" and mother[a] < 0 and father[a] < 0 and m == len(mother) + a, where)
                continue
            if c.down == 0 and c.up > 0:
                require(m == b and b in up(a, c.up), where)
                continue
            require(m in up(a, max(c.up, 1)) and m in up(b, max(c.down, 1)), where)


def check_against_pg(rep: Rep, t: int, rows: list[dict]) -> None:
    """Every pooled r̂ of trait t matches pg-phenotype's Relative-Pair Correlation."""
    # A code may not sit in two entries of one call: the codes, then the groups.
    calls = [list(CODES), [{name: list(members)} for name, members in CELLS.items() if len(members) > 1]]
    mine = {(r["cell"], r["estimator"]): r["r"] for r in rows if r["trait"] == f"trait{t}" and r["stratum"] == "pooled"}
    checked = 0
    for values, kind in ((rep.liability[t], "continuous"), (rep.affected[t], "binary")):
        for entries in calls:
            for cell in rp.correlation(rep.ped, Trait(values, kind), categories=entries).cells:
                for est in cell.crude:
                    ours = mine.get((cell.category, est.estimator))
                    where = (
                        f"{rep.scenario} rep{rep.rep} trait{t} {cell.category} {est.estimator}: {ours} vs {est.value}"
                    )
                    if ours is None:
                        # Undefined, or a boundary tetrachoric, which has no influence function.
                        require(est.value is None or est.estimator == "tetrachoric", where)
                        continue
                    require(est.value is not None, where)
                    require(abs(ours - est.value) <= TOL[est.estimator], where)
                    checked += 1
    require(checked > 0, f"{rep.scenario} rep{rep.rep} trait{t}: no cell checked")


# --- self-test -------------------------------------------------------------------------------


def self_test() -> None:
    rng = np.random.default_rng(18)
    # Tetrachoric: our fit equals pg-phenotype's, and the analytic IF equals n ∂r̂/∂n_c.
    for _ in range(20):
        t = rng.integers(20, 2000, size=(2, 2)).astype(float)
        n = t.sum()
        fit = lambda tt: two_step_tetrachoric(tt[1, 1] / tt.sum(), tt[1].sum() / tt.sum(), tt[:, 1].sum() / tt.sum())  # noqa: E731
        rho = fit(t)
        assert abs(rho - pgc.tetrachoric(table=t.astype(int).tolist()).value) < 1e-7, (
            "tetrachoric fit"
        )  # pg-phenotype converges to ~2e-8 near rho = -0.94
        cell = tetrachoric_if(t, rho)
        for i, j in itertools.product((0, 1), repeat=2):
            e = np.zeros((2, 2))
            e[i, j] = 1e-3 * t[i, j]
            fd = n * (fit(t + e) - fit(t - e)) / (2 * e[i, j])
            assert abs(fd - cell[i, j]) < 1e-5 * max(1, abs(cell[i, j])), ("tetrachoric IF", i, j, fd, cell[i, j])
        assert abs((t * cell).sum()) < 1e-8 * n, "tetrachoric IF centred"
    # Per-pair influence: φ_d = n ∂r̂/∂w_d for pair weights w, in each entry layout.
    n = 60
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y1, y2 = 0.4 * x1 + rng.normal(size=n), 0.3 * x2 + 0.5 * x1 + rng.normal(size=n)
    layouts = {
        "directed": ([x1], [y2]),
        "symmetric": ([x1, y1], [y1, x1]),
        "symmetric cross-trait": ([x1, y1], [y2, x2]),
    }

    def weighted_pearson(xs: list[np.ndarray], ys: list[np.ndarray], w: np.ndarray) -> float:
        x, y, ww = np.concatenate(xs), np.concatenate(ys), np.tile(w, len(xs))
        mx, my = np.average(x, weights=ww), np.average(y, weights=ww)
        cov = np.average((x - mx) * (y - my), weights=ww)
        return cov / math.sqrt(np.average((x - mx) ** 2, weights=ww) * np.average((y - my) ** 2, weights=ww))

    def weighted_tetrachoric(xs: list[np.ndarray], ys: list[np.ndarray], w: np.ndarray) -> float:
        t = np.zeros((2, 2))
        for x, y in zip(xs, ys, strict=True):
            np.add.at(t, (x.astype(int), y.astype(int)), w)
        return two_step_tetrachoric(t[1, 1] / t.sum(), t[1].sum() / t.sum(), t[:, 1].sum() / t.sum())

    for layout, (xs, ys) in layouts.items():
        cut = [(v > 0.8).astype(float) for v in xs], [(v > 0.8).astype(float) for v in ys]
        for kind, est, (xx, yy), fit in (
            ("continuous", "pearson", (xs, ys), weighted_pearson),
            ("binary", "tetrachoric", cut, weighted_tetrachoric),
            ("binary", "phi", cut, weighted_pearson),
        ):
            r, phi = pair_fits(xx, yy, kind)[est]
            assert abs(r - fit(xx, yy, np.ones(n))) < 1e-9, (layout, est, "r")
            for d in range(n):
                e = np.zeros(n)
                e[d] = 1e-5
                fd = n * (fit(xx, yy, 1 + e) - fit(xx, yy, 1 - e)) / 2e-5
                assert abs(fd - phi[d]) < 1e-5 * max(1, abs(phi[d])), (layout, est, d, fd, phi[d])
    # Meats against brute force over pairs of pairs.
    n_people, n_pairs = 30, 80
    a, b = rng.integers(0, n_people, n_pairs), rng.integers(0, n_people, n_pairs)
    keep = a != b
    a, b = np.minimum(a, b)[keep], np.maximum(a, b)[keep]
    key = a * n_people + b
    _, first = np.unique(key, return_index=True)
    a, b = a[first], b[first]
    g, mrca, phi = rng.integers(0, 6, len(a)), rng.integers(0, 9, len(a)), rng.normal(size=len(a))
    brute = dict.fromkeys(DESIGNS, 0.0)
    for d, e in itertools.product(range(len(a)), repeat=2):
        share = d == e or len({a[d], b[d]} & {a[e], b[e]}) > 0
        term = phi[d] * phi[e]
        brute["pair"] += term * (d == e)
        brute["person"] += term * share
        for name, key in (("mn", g), ("mrca", mrca)):
            brute[name] += term * (key[d] == key[e])
            brute[f"{name}_person"] += term * (share or key[d] == key[e])
    got = meats(phi, a, b, g, mrca)
    for d in DESIGNS:
        assert abs(got[d] - brute[d]) < 1e-9, ("meat", d, got[d], brute[d])
    print("self-test passed")


# --- driver ----------------------------------------------------------------------------------


#: One calibration row per scenario, trait, estimator, cell and stratum.
KEYS = ["scenario", "trait", "estimator", "cell", "stratum"]


def summarise(reps: pl.DataFrame, designs: tuple[str, ...] = DESIGNS) -> pl.DataFrame:
    networks = "n_networks" in reps.columns
    aggs = [pl.len().alias("n_reps"), pl.col("n").mean().alias("mean_n")]
    if networks:
        aggs.append(pl.col("n_networks").mean().alias("mean_networks"))
    if "n11" in reps.columns:
        aggs.append(pl.col("n11").mean().alias("mean_n11"))
    aggs += [pl.col("r").mean().alias("mean_r"), pl.col("r").std().alias("sd_r")]
    aggs += [pl.col(f"se_{d}").mean().alias(f"se_{d}") for d in designs]
    if networks:
        aggs.append(pl.col("largest_network_share").mean())
    aggs += [pl.col(f"se_{d}").is_null().sum().alias(f"undefined_{d}") for d in designs]
    if "se_reason" in reps.columns:
        aggs += [(pl.col("se_reason") == reason).sum().alias(f"no_se_{reason}") for reason in SE_REASONS]
    # The SE's own spread across reps: a noisy SE can have the right mean and still undercover.
    aggs += [(pl.col(f"se_{d}").std() / pl.col(f"se_{d}").mean()).alias(f"cv_se_{d}") for d in designs]
    # Wald 95% coverage of the mean r across reps, which stands in for the truth (its SE is sd_r / sqrt(reps)).
    aggs += [
        ((pl.col("r") - pl.col("r").mean()).abs() <= 1.959964 * pl.col(f"se_{d}"))
        .fill_null(False)
        .mean()
        .alias(f"cover_{d}")
        for d in designs
    ]
    # An undefined (negative) meat is missing, not NaN, so it leaves the means and counts as uncovered.
    reps = reps.with_columns([pl.col(f"se_{d}").fill_nan(None) for d in designs])
    out = reps.group_by(KEYS).agg(aggs)
    out = out.with_columns([(pl.col("sd_r") / pl.col(f"se_{d}")).alias(f"ratio_{d}") for d in designs])
    return out.sort(KEYS)


#: Category types for the medians: whom a person shares a cell's pairs with.
CATEGORY_TYPE = {
    **dict.fromkeys(["MZ", "FS", "MHS", "PHS", "HS"], "sibling"),
    **dict.fromkeys(["MO", "FO", "PO", "GP", "GGP"], "lineal"),
    **dict.fromkeys(["Av", "HAv", "GAv", "1C"], "collateral"),
}


def medians(cal: pl.DataFrame, designs: tuple[str, ...] = DESIGNS, by: tuple[str, ...] = ("type",)) -> pl.DataFrame:
    """Per scenario and *by* (category type first), the median over cells of each design's SD(r̂)/SE and coverage.

    Only cells estimated in every rep of their scenario, with at least 1000 pairs on average.
    """
    cal = cal.filter(pl.col("n_reps") == pl.col("n_reps").max().over("scenario"), pl.col("mean_n") >= 1000)
    cal = cal.with_columns(pl.col("cell").replace_strict(CATEGORY_TYPE).alias("type"))
    aggs = [pl.len().alias("cells")]
    if "largest_network_share" in cal.columns:
        aggs.append(pl.col("largest_network_share").median())
    aggs += [pl.col(f"ratio_{d}").median() for d in designs] + [pl.col(f"cover_{d}").median() for d in designs]
    by_scenario = cal.group_by("scenario", *by).agg(aggs)
    overall = cal.group_by(*by).agg(aggs).with_columns(pl.lit("all").alias("scenario"))
    return pl.concat([by_scenario, overall.select(by_scenario.columns)]).sort("scenario", *by)


def rust_summaries(reps: pl.DataFrame, out: Path) -> None:
    """``rust_calibration.tsv`` and ``rust_summary.tsv``, and the gap between the Rust SE and this script's."""
    boundary = reps.group_by(KEYS).agg((pl.col("se_reason") == "boundary").sum().alias("no_se_boundary"))
    # A boundary tetrachoric has no influence function: out of r̂'s statistics, as in rep_rows.
    cal = summarise(reps.filter(pl.col("se_reason").ne_missing("boundary")), ("rust",))
    cal = cal.join(boundary, on=KEYS, how="full", coalesce=True).sort(KEYS)
    cal.write_csv(out / "rust_calibration.tsv", separator="\t", float_precision=4)
    cal = cal.filter(pl.col("n_reps").is_not_null()).with_columns(
        pl.when(pl.col("stratum") == "stratified").then(pl.lit("stratified")).otherwise(pl.lit("cells")).alias("fit")
    )
    medians(cal, ("rust",), ("type", "estimator", "fit")).write_csv(
        out / "rust_summary.tsv", separator="\t", float_precision=3
    )
    both = reps.filter(pl.col("se_rust").is_not_null(), pl.col("se_tool").is_not_nan())
    gap = both.select(
        pl.len().alias("estimates"),
        ((pl.col("se_rust") - pl.col("se_tool")).abs() / pl.col("se_tool")).max().alias("max_rel_se_gap"),
        (pl.col("r") - pl.col("r_tool")).abs().max().alias("max_r_gap"),
    )
    cells = reps.filter(pl.col("stratum") != "stratified")
    one_sided = cells.filter(pl.col("se_rust").is_null() != (pl.col("se_tool").is_null() | pl.col("se_tool").is_nan()))
    print(f"Rust SE against mrca_person: {gap.to_dicts()[0]}; one has an SE and the other not: {len(one_sided)}")


def main() -> None:
    parser = runner.parser(__doc__, jobs=os.cpu_count(), reps_help="also write the per-rep rows here (about 40 MB)")
    parser.add_argument("--rust", action="store_true", help="calibrate the shipped SE (rust_*.tsv)")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    runner.check_args(parser, args)
    if args.from_reps is not None:
        write_summaries(pl.read_csv(args.from_reps, separator="\t", infer_schema_length=None), args.out, args.rust)
        return
    work = runner.rep_dirs(args.results, args.max_reps)
    if args.rust:
        work = [(p,) for p, _ in work]
    rows: list[dict] = []
    for out in runner.run(rust_rep_rows if args.rust else rep_rows, work, args.jobs):
        rows += out
    reps = pl.DataFrame(rows, infer_schema_length=None)
    if args.reps is not None:
        reps.write_csv(args.reps, separator="\t", float_precision=8)
    write_summaries(reps, args.out, args.rust)
    print(f"{len(work)} reps, {len(reps)} rows -> {args.out}")


def write_summaries(reps: pl.DataFrame, out: Path, rust: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if rust:
        rust_summaries(reps, out)
        return
    cal = summarise(reps)
    cal.write_csv(out / "calibration.tsv", separator="\t", float_precision=4)
    medians(cal).write_csv(out / "summary.tsv", separator="\t", float_precision=3)


if __name__ == "__main__":
    main()
