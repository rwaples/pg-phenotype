"""Probe why the ACE Fit's continuous SEs are short of the across-rep spread (issue #22).

    PG_PHENOTYPE_THREADS=1 RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 pixi run python -B tools/relative_pairs_ace_se_probe.py \\
        <simACE>/results/se_calibration --jobs 8 --out docs/gates/relative-pairs-ace-se/probe --reps <scratch>/probe.tsv
    pixi run python -B tools/relative_pairs_ace_se_probe.py --from-reps <scratch>/probe.tsv \\
        --out docs/gates/relative-pairs-ace-se/probe

For each rep (``relative_pairs_se_calibration.load_rep``: Pearson on
``liability{t}`` over the rows whose case status is observed, ``stratum`` the
generation), ``relative_pairs.correlation(..., se=True, covariance=True,
influence=True)`` and ``relative_pairs.ace`` with MZ, FS and MHS sharing C,
stratified primary, under each ``weights``.  Three probes per fit:

``realized``: the same fit on the rep's realized-truth cells (A x A and
C x C terms, E x E within person; ``relative_pairs_ace_validation``'s
``decomposition`` and ``realized_result``), so ``d = θ̂ − θ_realized``
splits the error into the part around the rep's own realized truth, the
realized truth's spread across reps, and their covariance.

``level``: the cells' covariance rebuilt from the per-entry influences
(issue #25), and the fit refitted on it.  ``base`` is ADR 0013's design: two
entries count together when they share an MRCA or a member (each entry's
MRCA, or the pair when it has none), by inclusion-exclusion.  It must give
the result's own covariance.  The pedigree is one connected component, so
a partition coarser than the MRCA keeps only a few clusters; the levels
extend the union instead.  ``deg<d>`` also counts two entries together when
their MRCAs are distinct relatives of degree at most ``d`` (pedigree-graph's
categories, MZ at degree 0): it adds ``Σ Φ_m K_mm' Φ_m'`` over related MRCA
rows and subtracts ``Σ_i Σ S_(i,m) K_mm' S_(i,m')``, the related-MRCA pairs
of entries that already count through a shared member, so each pair of
entries counts once.  ``kin3`` weights the degree-at-most-3 pairs by
``2 φ``, ``φ`` their nominal kinship.  ``<level>_diag`` and
``<level>_off`` add that level's extra term to the diagonal alone or to the
off-diagonal alone.  The within-person row and column keep the result's.
Each level records the smallest eigenvalue of the stratified covariance
over the cells with an SE (relative to the largest), since neither kernel is
positive semidefinite.

``shock``: each rep's realized variances of A, C and E per trait and their
cross-trait covariances on the phenotyped rows (within generation, averaged
by size), and the same for A in recorded generation 0, for regressing θ̂ on
them across a scenario's reps.  The R² is reported in sample and by 5-fold
cross-validation, for the phenotyped covariates (which are partly a product
of the same dependence) and for generation 0's alone.

Each cell's own stratified SE at each level is a row too (``weights``
``cell``, ``quantity`` ``<category>:<first trait><second trait>``), with its
realized truth.

Per-rep rows (``--reps``) hold scenario, rep, weights, level, quantity,
value, SE, the realized value, Cov(Â_t, Ĉ_t) from the SEs, the shock
variables, the kernel's MRCA pairs and the smallest eigenvalue.  ``--out``
gets ``probe.tsv`` (per scenario, weights, level and quantity: the fits
that gave a value and an SE and those that did not, SD(θ̂)/mean SE,
SD(d)/mean SE, SD(θ_realized)/mean SE, 2 Cov(d, θ_realized)/mean SE², the
shock R²s and the median smallest eigenvalue) and ``run.json``.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from pedigree_graph import RELATIONSHIPS, PedigreeGraph

from pg_phenotype import Trait
from pg_phenotype import relative_pairs as rp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import _calibration_runner as runner
from tools import relative_pairs_ace_validation as val
from tools.relative_pairs_ace_se_calibration import quantities
from tools.relative_pairs_se_calibration import load_rep

ROOT = Path(__file__).resolve().parents[1]
SHARED_ENVIRONMENT = ["MZ", "FS", "MHS"]
WEIGHTS = ("pairs", "inverse_variance", "gls")
ESTIMATE = "stratified"
#: The kernels: each MRCA relative pair up to a degree at weight 1, or up to degree 3 at 2 φ.
KERNELS = ("deg0", "deg1", "deg2", "deg3", "kin3")
LEVELS = ("base", *KERNELS, "deg1_diag", "deg1_off", "kin3_diag", "kin3_off")
#: Six phenotyped generations of 20k rows need about 1.4 GB for the entries.
MEMORY_BUDGET_BYTES = 4 << 30
#: The base level's rebuilt covariance against the result's, relative to the largest |entry|.
TOL_BASE = 1e-9
PHENOTYPED = ("varA1", "varC1", "varE1", "varA2", "varC2", "varE2", "covA12", "covC12", "covE12")
GENERATION0 = ("varA1_g0", "varA2_g0", "covA12_g0")
FOLDS = 5


def mrca_kernels(ped: pl.DataFrame, mrcas: np.ndarray) -> dict[str, sp.csr_matrix]:
    """Per kernel, the symmetric matrix over the *mrcas* rows (sorted, in that order): 1 for two relatives of
    degree at most ``d`` at ``deg<d>``, ``2 φ`` for degree at most 3 at ``kin3``."""
    view = PedigreeGraph.from_frame(ped).view(ids=ped["id"].to_numpy()[mrcas])
    codes = [c for c, r in RELATIONSHIPS.items() if r.degree <= 3]
    pairs = view.relationship_pairs(categories=codes)
    n = len(mrcas)
    out = {}
    for kernel in KERNELS:
        top = 3 if kernel == "kin3" else int(kernel.removeprefix("deg"))
        rows, cols, data = [], [], []
        for code in codes:
            r = RELATIONSHIPS[code]
            if r.degree > top:
                continue
            a, b = (np.asarray(x, dtype=np.int64) for x in pairs[code])
            w = 2 * r.nominal_kinship if kernel == "kin3" else 1.0
            rows += [a, b]
            cols += [b, a]
            data += [np.full(len(a), w), np.full(len(a), w)]
        out[kernel] = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return out


def estimates(cell: rp.Cell) -> tuple[rp.Estimate, rp.Estimate | None]:
    """The cell's pooled primary and stratified estimates: the covariance's two forms."""
    return cell.primary, None if cell.stratified is None else cell.stratified.estimate


def rebuilt(
    res: rp.RelativePairCorrelation, mrcas: np.ndarray, kernels: dict[str, sp.csr_matrix]
) -> dict[str, rp.CellCovariance]:
    """*res*'s covariance from its entries at each level (see the module docstring).

    A pair with no MRCA (``-1``) is its own cluster, keyed by its unordered pair; an MRCA outside *mrcas*
    (an external parent) is its own and takes no kernel term.
    """
    assert res.covariance is not None
    n_cells = len(res.cells)
    keys = []
    for cell in res.cells:
        e = cell.entries
        assert e is not None
        a, b, m = (np.asarray(x, dtype=np.int64) for x in (e.first_id, e.second_id, e.mrca_id))
        lone = -1 - (np.minimum(a, b) * (1 << 31) + np.maximum(a, b))
        keys.append((a, b, np.where(m >= 0, m, lone)))
    clusters, at = np.unique(np.concatenate([k for _, _, k in keys]), return_inverse=True)
    starts = np.cumsum([0, *(len(k) for _, _, k in keys)])
    # Each cluster's MRCA index into *mrcas*, -1 for a lone pair or an external parent.
    where = np.searchsorted(mrcas, clusters).clip(0, max(len(mrcas) - 1, 0))
    mrca_of = np.where((len(mrcas) > 0) & (mrcas[where] == clusters), where, -1)
    out: dict[str, dict[str, np.ndarray | None]] = {level: {} for level in LEVELS}
    for form, pick in (("pooled", 0), ("stratified", 1)):
        original = getattr(res.covariance, form)
        coo: dict[str, tuple[list, list, list]] = {k: ([], [], []) for k in ("phi", "person", "group")}
        for j, ((a, b, _), cell) in enumerate(zip(keys, res.cells, strict=True)):
            estimate = estimates(cell)[pick]
            if original is None or estimate is None or estimate.influence is None:
                continue
            w = estimate.influence
            c = at[starts[j] : starts[j + 1]]
            ab, cc, ww = np.concatenate([a, b]), np.concatenate([c, c]), np.concatenate([w, w])
            for name, rows, data in (("phi", c, w), ("person", ab, ww), ("group", ab * len(clusters) + cc, ww)):
                coo[name][0].append(rows)
                coo[name][1].append(np.full(len(rows), j))
                coo[name][2].append(data)
        if not coo["phi"][0]:
            for level in LEVELS:
                out[level][form] = original
            continue
        assert original is not None

        def matrix(parts: tuple[list, list, list]) -> tuple[sp.csc_matrix, np.ndarray]:
            rows, cols, data = (np.concatenate(x) for x in parts)
            keys_, rows = np.unique(rows, return_inverse=True)
            return sp.csc_matrix((data, (rows, cols)), shape=(len(keys_), n_cells)), keys_

        phi, phi_keys = matrix(coo["phi"])
        person, _ = matrix(coo["person"])
        group, group_keys = matrix(coo["group"])
        base = (phi.T @ phi + person.T @ person - group.T @ group).toarray()
        # Φ over the MRCA rows, and the (member, MRCA) groups' members and MRCA indices.
        phi_mrca = mrca_of[phi_keys]
        phi_m = (
            sp.csc_matrix(
                (np.ones((phi_mrca >= 0).sum()), (phi_mrca[phi_mrca >= 0], np.flatnonzero(phi_mrca >= 0))),
                shape=(len(mrcas), len(phi_keys)),
            )
            @ phi
        )
        g_member, g_mrca = group_keys // len(clusters), mrca_of[group_keys % len(clusters)]
        extra = {}
        for kernel, K in kernels.items():
            # The (member, MRCA) groups of one member whose MRCAs are related: their entries already count
            # together through the member.
            g = pl.DataFrame({"i": g_member, "m": g_mrca, "row": np.arange(len(group_keys))}).filter(pl.col("m") >= 0)
            both = g.join(g, on="i", suffix="_b").filter(pl.col("m") != pl.col("m_b"))
            ma, mb = both["m"].to_numpy(), both["m_b"].to_numpy()
            w = np.asarray(K[ma, mb]).ravel() if len(ma) else np.zeros(0)
            keep = w != 0
            M = sp.csr_matrix(
                (w[keep], (both["row"].to_numpy()[keep], both["row_b"].to_numpy()[keep])),
                shape=(len(group_keys), len(group_keys)),
            )
            extra[kernel] = (phi_m.T @ (K @ phi_m) - group.T @ (M @ group)).toarray()
        known = ~np.isnan(np.diag(original))[:n_cells]
        block = np.ix_(known, known)
        for level in LEVELS:
            inner = base.copy()
            if level != "base":
                kernel, _, part = level.partition("_")
                add = extra[kernel]
                if part == "diag":
                    add = np.diag(np.diag(add))
                elif part == "off":
                    add = add - np.diag(np.diag(add))
                inner += add
            cov = np.array(original, copy=True)
            cov[:n_cells, :n_cells][block] = inner[block]
            cov.flags.writeable = False
            out[level][form] = cov
    return {level: dataclasses.replace(res.covariance, **forms) for level, forms in out.items()}


def smallest_eigenvalue(cov: np.ndarray) -> float:
    """The smallest eigenvalue of *cov* over its rows with an SE, over the largest."""
    known = ~np.isnan(np.diag(cov))
    values = np.linalg.eigvalsh(cov[np.ix_(known, known)])
    return float(values[0] / values[-1])


def shock(ped: pl.DataFrame, rows: np.ndarray) -> dict[str, float]:
    """The rep's realized component variances and cross-trait covariances on the phenotyped *rows*, within
    generation, and A's in recorded generation 0."""
    generation = ped["generation"].to_numpy()
    _, ids, n = np.unique(generation[rows], return_inverse=True, return_counts=True)

    def centred(x: np.ndarray) -> np.ndarray:
        return x - (np.bincount(ids, x) / n)[ids]

    out = {}
    for X in "ACE":
        x1, x2 = (centred(ped[f"{X}{t}"].to_numpy().astype(np.float64)[rows]) for t in (1, 2))
        out[f"var{X}1"], out[f"var{X}2"] = float(x1 @ x1 / len(rows)), float(x2 @ x2 / len(rows))
        out[f"cov{X}12"] = float(x1 @ x2 / len(rows))
    g0 = generation == 0
    a1, a2 = (ped[f"A{t}"].to_numpy().astype(np.float64)[g0] for t in (1, 2))
    a1, a2 = a1 - a1.mean(), a2 - a2.mean()
    out["varA1_g0"], out["varA2_g0"], out["covA12_g0"] = (
        float(a1 @ a1 / len(a1)),
        float(a2 @ a2 / len(a2)),
        float(a1 @ a2 / len(a1)),
    )
    return out


def rep_job(rep_dir: Path, _first: bool) -> list[dict]:
    rep = load_rep(rep_dir)
    ped = pl.read_parquet(rep_dir / "pedigree.parquet")
    rows = np.flatnonzero(~np.isnan(rep.liability[1]))
    if not np.array_equal(rows, np.flatnonzero(~np.isnan(rep.liability[2]))):
        sys.exit(f"{rep_dir}: the traits are observed on different rows")
    traits = [Trait(rep.liability[t], "continuous") for t in (1, 2)]
    res = rp.correlation(
        rep.ped,
        traits,
        stratum=rep.generation,
        se=True,
        covariance=True,
        influence=True,
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
    )
    vrep = val.Rep(rep_dir, ped, ped[rows], rows, {})
    view = PedigreeGraph.from_frame(ped).view(ids=ped["id"].to_numpy()[rows])
    truth, gap, _ = val.realized_result(res, val.decomposition(vrep, res, view), ESTIMATE)
    if gap > val.DECOMPOSITION_TOLERANCE:
        sys.exit(f"{rep_dir}: component terms miss a cell by {gap:.1e}")
    mrcas = np.unique(np.concatenate([np.asarray(c.entries.mrca_id) for c in res.cells if c.entries is not None]))
    mrcas = mrcas[(mrcas >= 0) & (mrcas < len(ped))]
    kernels = mrca_kernels(ped, mrcas)
    covariances = rebuilt(res, mrcas, kernels)
    assert res.covariance is not None
    for form in ("pooled", "stratified"):
        a, b = getattr(covariances["base"], form), getattr(res.covariance, form)
        if np.nanmax(np.abs(a - b)) > TOL_BASE * np.nanmax(np.abs(b)):
            sys.exit(f"{rep_dir}: the base level misses the result's {form} covariance")
    base = {"scenario": rep.scenario, "rep": rep.rep} | shock(ped, rows)
    out = []
    for level in LEVELS:
        res_k = dataclasses.replace(res, covariance=covariances[level])
        truth_k = dataclasses.replace(truth, covariance=covariances[level])
        stratified = covariances[level].stratified
        assert stratified is not None
        kernel = level.partition("_")[0]
        meta = base | {
            "level": level,
            "mrca_links": kernels[kernel].nnz // 2 if kernel in kernels else 0,
            "min_eig": smallest_eigenvalue(stratified),
        }
        for j, (cell, real_cell) in enumerate(zip(res.cells, truth.cells, strict=True)):
            if cell.stratified is None or real_cell.stratified is None or np.isnan(stratified[j, j]):
                continue
            out.append(
                meta
                | {"weights": "cell", "quantity": f"{cell.category}:{cell.first_trait}{cell.second_trait}"}
                | {"value": cell.stratified.estimate.value, "se": float(np.sqrt(stratified[j, j]))}
                | {"realized": real_cell.stratified.estimate.value, "cov_AC": None}
            )
        for weights in WEIGHTS:
            fit, real = (
                rp.ace(r, shared_environment=SHARED_ENVIRONMENT, estimate=ESTIMATE, weights=weights)
                for r in (res_k, truth_k)
            )
            q, r = quantities(fit), quantities(real)
            cov_ac = {}
            for t, shares in enumerate(fit.traits, 1):
                if shares.A_se is not None and shares.C_se is not None and shares.E_se is not None:
                    cov_ac[t] = (shares.E_se**2 - shares.A_se**2 - shares.C_se**2) / 2
            for name, (value, _, se, _) in q.items():
                out.append(
                    meta
                    | {"weights": weights, "quantity": name}
                    | {"value": value, "se": se, "realized": r[name][0]}
                    | {"cov_AC": cov_ac.get(int(name[1])) if name[0] in "ACE" else None}
                )
    return out


def r2(value: np.ndarray, covariates: np.ndarray) -> tuple[float, float]:
    """The R² of *value* on *covariates* and an intercept, in sample and by ``FOLDS``-fold cross-validation
    (folds by rep order)."""
    X = np.column_stack([np.ones(len(value)), covariates])
    total = ((value - value.mean()) ** 2).sum()
    fitted = X @ np.linalg.lstsq(X, value, rcond=None)[0]
    held = np.empty_like(value)
    for fold in np.array_split(np.arange(len(value)), FOLDS):
        train = np.setdiff1d(np.arange(len(value)), fold)
        held[fold] = X[fold] @ np.linalg.lstsq(X[train], value[train], rcond=None)[0]
    return 1 - ((value - fitted) ** 2).sum() / total, 1 - ((value - held) ** 2).sum() / total


def summarise(reps: pl.DataFrame) -> pl.DataFrame:
    """Per scenario, weights, level and quantity: the ratios to the mean SE and the shock regressions."""
    out = []
    keys = ["scenario", "weights", "level", "quantity"]
    for key, g in reps.sort("rep").group_by(keys, maintain_order=True):
        # A fit fails when it gives no value, SE or realized value, or a nonfinite one (a negative rebuilt
        # diagonal gives a NaN SE).
        ok = g.drop_nulls(["value", "se", "realized"]).filter(
            pl.all_horizontal(pl.col(c).is_finite() for c in ("value", "se", "realized"))
        )
        record = dict(zip(keys, key, strict=True)) | {"reps": len(ok), "failed": len(g) - len(ok)}
        if len(ok) > FOLDS + len(PHENOTYPED) + 1:
            value, realized, se = (ok[c].to_numpy() for c in ("value", "realized", "se"))
            mean_se, d = se.mean(), value - realized
            phen = r2(value, ok.select(PHENOTYPED).to_numpy())
            gen0 = r2(value, ok.select(GENERATION0).to_numpy())
            record |= {
                "mrca_links": float(np.median(ok["mrca_links"].to_numpy())),
                "min_eig": float(np.median(ok["min_eig"].to_numpy())),
                "sd_over_se": value.std(ddof=1) / mean_se,
                "sd_d_over_se": d.std(ddof=1) / mean_se,
                "sd_realized_over_se": realized.std(ddof=1) / mean_se,
                "cov_d_realized_over_se2": 2 * np.cov(d, realized)[0, 1] / mean_se**2,
                "shock_r2": phen[0],
                "shock_r2_cv": phen[1],
                "g0_r2": gen0[0],
                "g0_r2_cv": gen0[1],
            }
        out.append(record)
    return pl.DataFrame(out)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    parser = runner.parser(__doc__, jobs=8)
    args = parser.parse_args()
    runner.check_args(parser, args)
    start = time.perf_counter()
    if args.from_reps is not None:
        reps = pl.read_csv(args.from_reps, separator="\t", schema_overrides={"level": pl.String})
    else:
        work = runner.rep_dirs(args.results, args.max_reps)
        reps = pl.DataFrame([row for rows in runner.run(rep_job, work, args.jobs) for row in rows])
        if args.reps is not None:
            reps.write_csv(args.reps, separator="\t")
    args.out.mkdir(parents=True, exist_ok=True)
    summarise(reps).write_csv(args.out / "probe.tsv", separator="\t", float_precision=4)
    run = {
        "command": " ".join(sys.argv),
        "head": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "finished": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "wall_s": round(time.perf_counter() - start, 1),
    }
    if args.from_reps is None:
        run["threads"] = runner.threads()
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
