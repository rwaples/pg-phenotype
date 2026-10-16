"""Calibrate cluster designs for the Relative-Pair Correlation SE (issue #18) on simACE replicates.

    pixi run python -B tools/relative_pairs_se_calibration.py --self-test
    pixi run python -B tools/relative_pairs_se_calibration.py \\
        <simACE>/results/se_calibration --jobs 11 --out docs/gates/relative-pairs-se

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
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
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

DESIGNS = ("pair", "person", "mn", "mn_person", "mrca", "mrca_person")
CODES = tuple(code for code, c in RELATIONSHIPS.items() if c.degree <= 3)
#: Cells: name -> member codes.  Groups as fitACE's Falconer step reports them.
CELLS: dict[str, tuple[str, ...]] = {code: (code,) for code in CODES} | {"PO": ("MO", "FO"), "HS": ("MHS", "PHS")}
SYMMETRIC = frozenset(code for code, c in RELATIONSHIPS.items() if c.first_role is None)
#: Pooled r̂ against pg-phenotype: Pearson and phi exactly, the tetrachoric to its fit's tolerance.
TOL = {"pearson": 1e-10, "phi": 1e-10, "tetrachoric": 1e-8}


# --- influence functions ---------------------------------------------------------------------


def pearson_if(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray] | None:
    """r̂ and each entry's influence.  On a same-trait symmetric cell's doubled entries the two sides'
    means and spreads coincide, so this is the intraclass Pearson."""
    if len(x) == 0:
        return None
    mx, my = x.mean(), y.mean()
    sx, sy = math.sqrt(((x - mx) ** 2).mean()), math.sqrt(((y - my) ** 2).mean())
    if sx == 0 or sy == 0:
        return None
    u, v = (x - mx) / sx, (y - my) / sy
    r = float((u * v).mean())
    return r, u * v - r / 2 * (u * u + v * v)


def _bvn_upper(h: float, k: float, rho: float) -> float:
    """P(X > h, Y > k) for a standard bivariate normal (Plackett's identity, quadrature to ~1e-14)."""
    if rho == 0:
        return float(ndtr(-h) * ndtr(-k))
    integral, _ = integrate.quad(lambda t: _bvn_pdf(h, k, t), 0.0, rho, epsabs=1e-14, epsrel=1e-12)
    return float(ndtr(-h) * ndtr(-k)) + integral


def _bvn_pdf(h: float, k: float, rho: float) -> float:
    q = 1 - rho * rho
    return math.exp(-(h * h - 2 * rho * h * k + k * k) / (2 * q)) / (2 * math.pi * math.sqrt(q))


def two_step_tetrachoric(p11: float, pa: float, pb: float) -> float:
    """The 2 x 2 two-step tetrachoric from proportions: thresholds from margins, then ρ matching p11."""
    h, k = -float(ndtri(pa)), -float(ndtri(pb))
    return optimize.brentq(lambda r: _bvn_upper(h, k, r) - p11, -0.9999, 0.9999, xtol=1e-15, rtol=1e-15)


def tetrachoric_if(table: np.ndarray, rho: float) -> np.ndarray:
    """Influence of one entry in cell (i, j) of a directed 2 x 2 two-step tetrachoric at ρ̂.

    dρ = (dp11 - Qa dpa - Qb dpb) / φ2(h, k; ρ), with Qa = P(Y > k | X = h).
    """
    n = table.sum()
    p11, pa, pb = table[1, 1] / n, table[1].sum() / n, table[:, 1].sum() / n
    h, k = -float(ndtri(pa)), -float(ndtri(pb))
    s = math.sqrt(1 - rho * rho)
    qa, qb = float(ndtr(-(k - rho * h) / s)), float(ndtr(-(h - rho * k) / s))
    phi2 = _bvn_pdf(h, k, rho)
    i, j = np.meshgrid([0.0, 1.0], [0.0, 1.0], indexing="ij")
    return ((i * j - p11) - qa * (i - pa) - qb * (j - pb)) / phi2


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


# --- cluster designs -------------------------------------------------------------------------


def _group_sq(keys: np.ndarray, w: np.ndarray) -> float:
    """Σ over distinct keys of (Σ w at that key)², by sort (NumPy's hash unique is slow on int64)."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    sums = np.add.reduceat(w[order], starts)
    return float(sums @ sums)


def meats(phi: np.ndarray, a: np.ndarray, b: np.ndarray, g: np.ndarray, m: np.ndarray) -> dict[str, float]:
    """Each design's meat for influences φ_d of pairs (a_d, b_d) with Mate Network g_d and MRCA m_d."""
    pair = float(phi @ phi)
    twice = np.concatenate([phi, phi])
    person = _group_sq(np.concatenate([a, b]), twice) - pair
    out = {"pair": pair, "person": person}
    for name, key in (("mn", g), ("mrca", m)):
        stride = int(key.max()) + 1
        by_person = _group_sq(np.concatenate([a * stride + key, b * stride + key]), twice) - pair
        out[name] = _group_sq(key, phi)
        out[f"{name}_person"] = out[name] + person - by_person
    return out


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


def top_keys(
    code: str, a: np.ndarray, b: np.ndarray, mother: np.ndarray, father: np.ndarray, net: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each pair's top-mating Mate Network and MRCA; x is a's ancestor up-1 meioses up the path."""
    c = RELATIONSHIPS[code]
    parents = (mother, father)

    def ancestors(start: np.ndarray, depth: int) -> list[np.ndarray]:
        out = []
        for path in itertools.product((0, 1), repeat=depth):
            x = start.copy()
            for step in path:
                x = np.where(x >= 0, parents[step][np.maximum(x, 0)], -1)
            out.append(x)
        return out

    if c.up == 0:
        x = a
        mrca = np.where(mother[x] >= 0, mother[x], father[x])
    else:
        x = np.full(len(a), -1)
        mrca = np.full(len(a), -1)
        mine = ancestors(a, c.up - 1)
        theirs = [b] if c.down == 0 else ancestors(b, c.down - 1)
        for xa in mine:
            ok = xa >= 0
            ma, fa = np.where(ok, mother[np.maximum(xa, 0)], -1), np.where(ok, father[np.maximum(xa, 0)], -1)
            for y in theirs:
                if c.down == 0:
                    hit, shared = ok & ((ma == y) | (fa == y)), y
                else:
                    yy = np.maximum(y, 0)
                    same_m = ok & (y >= 0) & (y != xa) & (ma >= 0) & (ma == mother[yy])
                    same_f = ok & (y >= 0) & (y != xa) & (fa >= 0) & (fa == father[yy])
                    hit, shared = same_m | same_f, np.where(same_m, ma, fa)
                new = (x < 0) & hit
                x, mrca = np.where(new, xa, x), np.where(new, shared, mrca)
        if (x < 0).any():
            raise RuntimeError(f"{code}: {int((x < 0).sum())} pairs have no top mating")
    top = np.where(mother[x] >= 0, mother[x], father[x])
    # Founder MZ twins have no parents: such a pair is its own cluster under both keys.
    lone = top < 0
    return np.where(lone, net.max() + 1 + a, net[top]), np.where(lone, len(mother) + a, mrca)


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


def summarise(reps: pl.DataFrame) -> pl.DataFrame:
    keys = ["scenario", "trait", "estimator", "cell", "stratum"]
    aggs = [
        pl.len().alias("n_reps"),
        pl.col("n").mean().alias("mean_n"),
        pl.col("n_networks").mean().alias("mean_networks"),
        pl.col("r").mean().alias("mean_r"),
        pl.col("r").std().alias("sd_r"),
    ]
    aggs += [pl.col(f"se_{d}").mean().alias(f"se_{d}") for d in DESIGNS]
    aggs.append(pl.col("largest_network_share").mean())
    aggs += [pl.col(f"se_{d}").is_null().sum().alias(f"undefined_{d}") for d in DESIGNS]
    # The SE's own spread across reps: a noisy SE can have the right mean and still undercover.
    aggs += [(pl.col(f"se_{d}").std() / pl.col(f"se_{d}").mean()).alias(f"cv_se_{d}") for d in DESIGNS]
    # Wald 95% coverage of the mean r across reps, which stands in for the truth (its SE is sd_r / sqrt(reps)).
    aggs += [
        ((pl.col("r") - pl.col("r").mean()).abs() <= 1.959964 * pl.col(f"se_{d}"))
        .fill_null(False)
        .mean()
        .alias(f"cover_{d}")
        for d in DESIGNS
    ]
    # An undefined (negative) meat is missing, not NaN, so it leaves the means and counts as uncovered.
    reps = reps.with_columns([pl.col(f"se_{d}").fill_nan(None) for d in DESIGNS])
    out = reps.group_by(keys).agg(aggs)
    out = out.with_columns([(pl.col("sd_r") / pl.col(f"se_{d}")).alias(f"ratio_{d}") for d in DESIGNS])
    return out.sort(keys)


#: Category types for the medians: whom a person shares a cell's pairs with.
CATEGORY_TYPE = {
    **dict.fromkeys(["MZ", "FS", "MHS", "PHS", "HS"], "sibling"),
    **dict.fromkeys(["MO", "FO", "PO", "GP", "GGP"], "lineal"),
    **dict.fromkeys(["Av", "HAv", "GAv", "1C"], "collateral"),
}


def medians(cal: pl.DataFrame) -> pl.DataFrame:
    """Per scenario and category type, the median over cells of each design's SD(r̂)/SE and coverage.

    Only cells estimated in every rep of their scenario, with at least 1000 pairs on average.
    """
    cal = cal.filter(pl.col("n_reps") == pl.col("n_reps").max().over("scenario"), pl.col("mean_n") >= 1000)
    cal = cal.with_columns(pl.col("cell").replace_strict(CATEGORY_TYPE).alias("type"))
    aggs = [pl.len().alias("cells"), pl.col("largest_network_share").median()]
    aggs += [pl.col(f"ratio_{d}").median() for d in DESIGNS] + [pl.col(f"cover_{d}").median() for d in DESIGNS]
    by_scenario = cal.group_by("scenario", "type").agg(aggs)
    overall = cal.group_by("type").agg(aggs).with_columns(pl.lit("all").alias("scenario"))
    return pl.concat([by_scenario, overall.select(by_scenario.columns)]).sort("scenario", "type")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="?", type=Path, help="simACE results folder, e.g. results/se_calibration")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--max-reps", type=int)
    parser.add_argument("--reps", type=Path, help="also write the per-rep rows here (about 40 MB)")
    parser.add_argument("--from-reps", type=Path, help="summarise these per-rep rows instead of computing them")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out is None or (args.results is None) == (args.from_reps is None):
        parser.error("--out and one of results or --from-reps are required")
    if args.from_reps is not None:
        write_summaries(pl.read_csv(args.from_reps, separator="\t"), args.out)
        return
    work = []
    for scen in sorted(p for p in args.results.iterdir() if p.is_dir()):
        numbered = ((int(m[1]), p) for p in scen.iterdir() if (m := re.fullmatch(r"rep(\d+)", p.name)))
        reps = [p for _, p in sorted(numbered) if (p / "cohort.parquet").exists()]
        work += [(p, i == 0) for i, p in enumerate(reps[: args.max_reps])]
    rows: list[dict] = []
    with ProcessPoolExecutor(args.jobs) as pool:
        for done, out in enumerate(pool.map(rep_rows, *zip(*work, strict=True)), 1):
            rows += out
            if done % 50 == 0:
                print(f"{done}/{len(work)} reps", file=sys.stderr)
    reps = pl.DataFrame(rows)
    if args.reps is not None:
        reps.write_csv(args.reps, separator="\t", float_precision=8)
    write_summaries(reps, args.out)
    print(f"{len(work)} reps, {len(reps)} rows -> {args.out}")


def write_summaries(reps: pl.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cal = summarise(reps)
    cal.write_csv(out / "calibration.tsv", separator="\t", float_precision=4)
    medians(cal).write_csv(out / "summary.tsv", separator="\t", float_precision=3)


if __name__ == "__main__":
    main()
