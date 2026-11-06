"""Issue #21 probe: per-rep parent-offspring Pearson in each child generation, decomposed.

For each rep of the se_calibration grid, each trait, each child generation g (4, 5) and each
parent role (FO, MO), over the children of generation g whose case status is observed and whose
parent is too (the gate's cell): r̂ = Pearson(L_parent, L_child), the parent-cluster sandwich SE
(the MRCA design for a lineal stratum cell), and the covariance split by the child's liability
components:  L_c = A_f/2 + A_m/2 + M_c + C_c + E_c,  M_c the Mendelian sampling term.

Also `mate_r` / `mate_rA`, the correlation of the two parents' liability / A over generation g's
children (one entry per child, so a couple counts once per child: an offspring-weighted mate
correlation, not a per-couple one), and mates per parent.  Writes one row per
rep x trait x gen x role to the TSV named by --out.

Reps are the `se_cal_*/rep*` directories with a run.yaml (complete simACE reps).  With
--pedigree-only they are those with a pedigree.parquet and no cohort is read: every row counts as
observed, as it is in generations 4-5 of these scenarios (n = 20000 per cell).

    pixi run python -B docs/gates/relative-pairs-se/issue-21/fo_probe.py <simACE>/results/se_calibration \
        --out docs/gates/relative-pairs-se/issue-21/fo_probe.tsv --jobs 10
"""

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl


def cluster_se(phi: np.ndarray, key: np.ndarray) -> float:
    """SE of r̂ from the per-cluster sums of the influences."""
    _, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=phi)
    return math.sqrt((s**2).sum()) / len(phi)


def rep_rows(rep_dir: Path, pedigree_only: bool = False) -> list[dict]:
    """One row per trait x child generation x parent role of one rep."""
    ped = pl.read_parquet(rep_dir / "pedigree.parquet")
    n = len(ped)
    # Ids index rows below, as in the gate tool's load_rep.
    if not (ped["id"] == pl.int_range(n, eager=True)).all():
        raise ValueError(f"{rep_dir}: ids are not rows")
    cohort = None if pedigree_only else pl.read_parquet(rep_dir / "cohort.parquet")
    if cohort is not None:
        kept = np.zeros(n, dtype=bool)
        kept[cohort["id"].to_numpy()] = True
        for col in ("mother", "father"):
            parent = ped[col].to_numpy()[cohort["id"].to_numpy()]
            if not kept[parent[parent >= 0]].all():
                raise ValueError(f"{rep_dir}: the cohort drops a recorded parent of its rows")
    gen = ped["generation"].to_numpy()
    mother, father = ped["mother"].to_numpy(), ped["father"].to_numpy()
    rows_out = []
    for t in (1, 2):
        seen = np.zeros(n, dtype=bool)
        if cohort is None:
            seen[:] = True
        else:
            aff = cohort[f"affected{t}"]
            seen[cohort["id"].to_numpy()[aff.is_not_null().to_numpy()]] = True
        L = ped[f"liability{t}"].to_numpy().astype(np.float64)
        A = ped[f"A{t}"].to_numpy().astype(np.float64)
        C = ped[f"C{t}"].to_numpy().astype(np.float64)
        E = ped[f"E{t}"].to_numpy().astype(np.float64)
        for g in (4, 5):
            kids = np.flatnonzero((gen == g) & seen & (mother >= 0) & (father >= 0))
            m, f = mother[kids], father[kids]
            mate_r = float(np.corrcoef(L[f], L[m])[0, 1])
            mate_rA = float(np.corrcoef(A[f], A[m])[0, 1])
            for role, par, other in (("FO", f, m), ("MO", m, f)):
                ok = seen[par]
                c, p, o = kids[ok], par[ok], other[ok]
                x, y = L[p], L[c]
                mx, my = x.mean(), y.mean()
                sx, sy = x.std(), y.std()
                u, v = (x - mx) / sx, (y - my) / sy
                r = float((u * v).mean())
                phi = u * v - r / 2 * (u * u + v * v)
                M = A[c] - (A[p] + A[o]) / 2

                def part(z: np.ndarray, u: np.ndarray = u, sy: float = sy) -> float:
                    return float((u * (z - z.mean())).mean() / sy)

                rows_out.append(
                    {
                        "scenario": rep_dir.parent.name,
                        "rep": int(rep_dir.name.removeprefix("rep")),
                        "trait": f"trait{t}",
                        "gen": g,
                        "cell": role,
                        "n": len(c),
                        "r": r,
                        "se_parent": cluster_se(phi, p),
                        "part_own_A": part(A[p] / 2),
                        "part_mate_A": part(A[o] / 2),
                        "part_mendel": part(M),
                        "part_C": part(C[c]),
                        "part_E": part(E[c]),
                        "sd_parent": float(sx),
                        "sd_child": float(sy),
                        "var_A_parent": float(A[p].var()),
                        "mate_r": mate_r,
                        "mate_rA": mate_rA,
                        "children_per_parent_var": float(np.bincount(np.unique(p, return_inverse=True)[1]).var()),
                        "mates_per_parent": float(len(np.unique(np.stack([p, o], 1), axis=0)) / len(np.unique(p))),
                    }
                )
    return rows_out


def main() -> None:
    """Probe every rep under the results folder and write the TSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--pedigree-only", action="store_true", help="reps without a cohort: every row observed")
    a = ap.parse_args()
    marker = "pedigree.parquet" if a.pedigree_only else "run.yaml"
    reps = sorted(p for p in a.root.glob("se_cal_*/rep*") if (p / marker).exists())
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = [row for rr in ex.map(rep_rows, reps, [a.pedigree_only] * len(reps)) for row in rr]
    pl.DataFrame(rows).write_csv(a.out, separator="\t", float_precision=6)
    print(f"{len(reps)} reps -> {a.out}")


if __name__ == "__main__":
    main()
