"""The Relative-Pair Correlation gate's comparisons on simACE ``small_test`` reps.

``tools/make_simace_relative_pairs_golden.py`` wrote the golden.  Per rep,
every default category is compared with the NumPy reference
(``tests/oracle/relative_pairs_reference.py``) on ``liability1`` (Pearson)
and ``affected1`` (tetrachoric, phi), and each of simACE's categories with
simACE: a directed one estimate for estimate, a symmetric one as the
difference between double and single entry, which has no tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from pg_phenotype import Trait
from pg_phenotype import relative_pairs as rp
from tests.oracle.relative_pairs_reference import SYMMETRIC, reference

GOLDEN = Path(__file__).resolve().parent / "golden" / "simace_relative_pairs_6355ffa6aebe"

#: Pearson and phi against the reference.
REL_ORACLE = 1e-12
#: A directed category's Pearson and phi against simACE.
ABS_SIMACE = 1e-12
#: A directed category's tetrachoric against simACE (ADR 0006's gate: 1.13e-7).
ABS_SIMACE_TETRACHORIC = 1.2e-7


@cache
def load() -> tuple[dict, dict[str, np.ndarray]]:
    golden = json.loads((GOLDEN / "golden.json").read_text())
    with np.load(GOLDEN / "inputs.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    return golden, arrays


@dataclass(frozen=True)
class Rep:
    name: str
    pedigree: dict[str, np.ndarray]
    liability: np.ndarray
    affected: np.ndarray
    simace: dict


def reps() -> list[Rep]:
    golden, arrays = load()
    out = []
    for r in golden["reps"]:
        col = lambda k, n=r["rep"]: arrays[f"{n}.{k}"]  # noqa: E731
        ped = {k: col(k) for k in ("id", "mother", "father", "twin", "sex")}
        out.append(Rep(r["rep"], ped, col("liability1"), col("affected1"), r["simace"]))
    return out


@cache
def fitted(name: str) -> tuple[rp.RelativePairCorrelation, rp.RelativePairCorrelation]:
    rep = next(r for r in reps() if r.name == name)
    return (
        rp.correlation(rep.pedigree, Trait(rep.liability, kind="continuous")),
        rp.correlation(rep.pedigree, Trait(rep.affected, kind="binary")),
    )


def _gap(a: float | None, b: float | None) -> float | None:
    """|a - b|; 0 when both are None; None when only one is."""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return None
    return abs(a - b)


def oracle_rows() -> list[dict]:
    """Per rep x default category: counts and each estimator's gap to the reference."""
    rows = []
    for rep in reps():
        cont, binary = fitted(rep.name)
        for c, b in zip(cont.cells, binary.cells, strict=True):
            rl = reference(rep.pedigree, rep.liability, "continuous", [c.category])
            ra = reference(rep.pedigree, rep.affected, "binary", [b.category])
            rows.append(
                {
                    "rep": rep.name,
                    "category": c.category,
                    "n_liability": c.n,
                    "n_affected": b.n,
                    "counts_equal": (c.n, c.n_dropped, b.n, b.n_dropped) == (rl.n, rl.n_dropped, ra.n, ra.n_dropped),
                    "pearson_gap": _gap(c["pearson"].value, rl.pearson),
                    "phi_gap": _gap(b["phi"].value, ra.phi),
                    "tetrachoric_equal": b["tetrachoric"].value == ra.tetrachoric,
                }
            )
    return rows


def simace_rows() -> list[dict]:
    """Per rep x simACE category: pg-phenotype and simACE side by side."""
    rows = []
    for rep in reps():
        cont, binary = fitted(rep.name)
        s = rep.simace
        for code in s["categories"]:
            c, b = cont.cell(code), binary.cell(code)
            t = s["table"][code]
            simace_tet = s["tetrachoric"][code]["r"]
            row = {
                "rep": rep.name,
                "category": code,
                "symmetric": code in SYMMETRIC,
                "n": c.n,
                "simace_n_pairs": s["n_pairs"][code],
                "pearson": c["pearson"].value,
                "simace_liability": s["liability"][code],
                "tetrachoric": b["tetrachoric"].value,
                "tetrachoric_boundary": b["tetrachoric"].boundary,
                "simace_tetrachoric": simace_tet,
                "phi": b["phi"].value,
                "simace_phi": s["phi"][code],
            }
            if code in SYMMETRIC:
                row["counts_match"] = b.concordance == (t[0][0], t[0][1] + t[1][0], t[1][1])
            else:
                row["counts_match"] = b.table == ((t[0][0], t[0][1]), (t[1][0], t[1][1]))
            for est, theirs in (
                ("pearson", "simace_liability"),
                ("tetrachoric", "simace_tetrachoric"),
                ("phi", "simace_phi"),
            ):
                row[f"{est}_gap"] = _gap(row[est], row[theirs])
            rows.append(row)
    return rows
