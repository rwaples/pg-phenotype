"""simACE's parent-offspring regressions (``tools/make_simace_po_regression_golden.py``) beside pg-phenotype's.

:func:`comparisons` maps each simACE result onto the pg-phenotype midparent
fit it computes:

- ``by_generation`` gen g: the stratum ``generation == g`` on ``liability1``;
- ``by_sex`` sex s, gen g: the stratum ``2 g + s``;
- ``affected``, ``affected_zero``: pooled, on ``affected1`` (binary) or zeros;
- ``validate`` A1 / P1 and ``validate_tiny`` A1: pooled, on A1, A1 + C1 + E1,
  and A1 x 1e-11.

Each comparison's ``outcome`` says how the two line up: ``both`` defined,
``both_undefined``, or the side that alone has an estimate with simACE's
rule that withheld it (``simace_min_10``: fewer than 10 trios;
``simace_min_100``: the validation's 100-offspring floor;
``simace_sd_floor``: parent SD under ``safe_linregress``'s 1e-10).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import pg_phenotype as pgp
from pg_phenotype.regression import parent_offspring

if TYPE_CHECKING:
    from pg_phenotype.regression import Fit

GOLDEN = Path(__file__).parent / "golden" / "simace_po_regression_6355ffa6aebe" / "golden.json"
#: simACE's and pg-phenotype's lines agree to this, absolute plus relative.
ABS, REL = 1e-12, 1e-10
#: The validation path's tolerance: simACE stores A, C and E as float32 and
#: forms the midparent in float32 (:func:`simace_validate_float32`), where
#: pg-phenotype works in float64.  About 8 float32 ulps.
REL_F32 = 1e-6


@dataclass(frozen=True)
class Comparison:
    scenario: str
    label: str
    simace: dict
    pg: Fit
    outcome: str


@cache
def load() -> dict:
    return json.loads(GOLDEN.read_text())


def _outcome(label: str, simace: dict, pg: Fit, n_nonfounders: int) -> str:
    s_defined = simace.get("slope") is not None
    p_defined = pg.slope is not None
    if s_defined and p_defined:
        return "both"
    if not s_defined and not p_defined:
        return "both_undefined"
    if s_defined:
        return "pg_only_undefined"
    if label.startswith("validate_tiny"):
        return "simace_sd_floor"
    if label.startswith("validate"):
        return "simace_min_100" if n_nonfounders <= 100 else "unexplained"
    return "simace_min_10" if simace["n_pairs"] < 10 else "unexplained"


def comparisons() -> list[Comparison]:
    out = []
    for sc in load()["scenarios"]:
        ped = pgp.Pedigree({k: np.asarray(sc["pedigree"][k]) for k in ("id", "mother", "father")})
        gen = np.asarray(sc["pedigree"]["generation"])
        sex = np.asarray(sc["pedigree"]["sex"])
        values = {k: np.asarray(v, dtype=float) for k, v in sc["values"].items()}
        n_nonfounders = int((np.asarray(sc["pedigree"]["mother"]) >= 0).sum())
        liability = pgp.Trait(values["liability1"], kind="continuous")
        by_gen = parent_offspring(ped, liability, stratum=gen)
        by_sex = parent_offspring(ped, liability, stratum=2 * gen + sex)
        pooled = {
            "affected": parent_offspring(ped, pgp.Trait(values["affected1"], kind="binary")),
            "affected_zero": parent_offspring(ped, pgp.Trait(np.zeros(len(gen)), kind="binary")),
            "validate.A1": parent_offspring(ped, pgp.Trait(values["A1"], kind="continuous")),
            "validate.P1": parent_offspring(ped, pgp.Trait(values["P1"], kind="continuous")),
            "validate_tiny.A1": parent_offspring(ped, pgp.Trait(values["A1"] * 1e-11, kind="continuous")),
        }
        s = sc["simace"]
        pairs = [
            (f"by_generation.{g}", entry, by_gen.strata[int(g[3:])].midparent)
            for g, entry in s["by_generation"].items()
        ]
        for sex_label, code in (("female", 0), ("male", 1)):
            pairs += [
                (f"by_sex.{sex_label}.{g}", entry, by_sex.strata[2 * int(g[3:]) + code].midparent)
                for g, entry in s["by_sex"][sex_label].items()
            ]
        pairs += [
            ("affected", s["affected"], pooled["affected"].pooled.midparent),
            ("affected_zero", s["affected_zero"], pooled["affected_zero"].pooled.midparent),
            ("validate.A1", s["validate"]["A1"], pooled["validate.A1"].pooled.midparent),
            ("validate.P1", s["validate"]["P1"], pooled["validate.P1"].pooled.midparent),
            ("validate_tiny.A1", s["validate_tiny"]["A1"], pooled["validate_tiny.A1"].pooled.midparent),
        ]
        out += [
            Comparison(sc["name"], label, entry, fit, _outcome(label, entry, fit, n_nonfounders))
            for label, entry, fit in pairs
        ]
    return out


def tolerance(c: Comparison) -> float:
    return REL_F32 if c.label.startswith("validate") else REL


def gaps(c: Comparison) -> dict[str, float]:
    """``|pg - simACE| / (ABS / REL + |simACE|)`` per field both report; at most :func:`tolerance` passes."""
    s, p = c.simace, c.pg
    fields = {"slope": p.slope, "intercept": p.intercept}
    if "r_squared" in s:
        fields["r_squared"] = p.r2
    else:
        fields |= {"r": p.r, "r2": p.r2, "stderr": p.se_ols}
    return {
        k: abs(v - s[k]) / (ABS / REL + abs(s[k])) for k, v in fields.items() if s.get(k) is not None and v is not None
    }


def simace_validate_float32(scenario: dict, key: str) -> tuple[float, float, float] | None:
    """simACE's validation regression redone: a float32 midparent, then ``linregress``.

    Returns (slope, intercept, r^2), or ``None`` under the 100-offspring floor.
    """
    from scipy.stats import linregress

    p = scenario["pedigree"]
    row = {int(i): r for r, i in enumerate(p["id"])}
    kids = [r for r, (m, f) in enumerate(zip(p["mother"], p["father"], strict=True)) if m in row and f in row]
    if len(kids) <= 100:
        return None
    v = np.asarray(scenario["values"][key], dtype=np.float32)
    mothers = [row[p["mother"][r]] for r in kids]
    fathers = [row[p["father"][r]] for r in kids]
    fit = linregress((v[mothers] + v[fathers]) / 2, v[kids])
    return float(fit.slope), float(fit.intercept), float(fit.rvalue**2)
