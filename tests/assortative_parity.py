"""Run a pedsum assortative golden through pg-phenotype and diff the result against pedsum's payload.

``to_pedsum`` maps the native result onto pedsum's ``compute_assortative_mating``
records (the shape unit 14's adapter writes); ``diff`` flattens both and splits
the differences into discrete mismatches and numeric gaps.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pg_phenotype import _native
from tests.assortative_golden import load

if TYPE_CHECKING:
    from tests.assortative_golden import Golden

KEYS = {
    "pearson": "r", "spearman": "r", "phi": "r", "point_biserial": "r", "odds_ratio": "value",
    "tetrachoric": "rho", "polychoric": "rho", "biserial": "rho", "polyserial": "rho",
}  # fmt: skip
LATENT = {"tetrachoric", "polychoric", "biserial", "polyserial"}


def run(g: Golden, threads: int = 1) -> dict:
    """The native result of fixture ``g`` with its settings."""
    traits = [(np.ascontiguousarray(v), k, n) for v, k, n in zip(g.values, g.kinds, g.n_levels, strict=True)]
    labels, known = (g.stratum_labels, g.stratum_known) if g.stratified else (None, None)
    s = g.settings
    return _native.mate_correlation(
        g.id, g.mother, g.father, None, None, traits, labels, known,
        permutations=s["permutations"], bootstrap=s["bootstrap"], seed=s["seed"],
        min_stratum_networks=s["min_stratum_networks"], threads=threads,
    )  # fmt: skip


def _record(r: dict, stratified: bool) -> dict:
    name = r["estimator"]
    key = KEYS[name]
    strata = (
        {"n_strata_mothers": r["n_strata_mothers"], "n_strata_fathers": r["n_strata_fathers"]} if stratified else {}
    )
    if r["reason"] is not None:
        return {key: None, "reason": r["reason"], **strata}
    out: dict = {key: r["value"]}
    if name in LATENT:
        out["boundary"] = r["boundary"]
    out["se"] = r["se"]
    out["ci"] = None if r["ci"] is None else list(r["ci"])
    out["ci_method"] = r["ci_method"]
    out["ci_unavailable_reason"] = r["ci_unavailable_reason"]
    if r["bootstrap"] is not None:
        out["bootstrap"] = dict(r["bootstrap"])
    p = r["permutation"]
    if p is not None:
        out["p_perm"] = p["p"]
        out["p_perm_unavailable_reason"] = p["p_unavailable_reason"]
        out["permutation_statistic"] = p["statistic"]
        out["permutations"] = {
            **p["draws"],
            "seed": p["seed"],
            "n_fixed_fathers": p["n_fixed_fathers"],
            "stopped_early": p["stopped_early"],
            "draws_used": p["draws_used"],
            "sequential_h": p["sequential_h"],
        }
    return out | strata


def to_pedsum(result: dict, n_traits: int) -> dict:
    """The parts of pedsum's payload the computation determines."""
    s = result["sample"]
    records = []
    for c in result["cells"]:
        crude: dict = {} if c["table"] is None else {"table": c["table"]}
        crude |= {r["estimator"]: _record(r, False) for r in c["crude"]}
        rec = {
            "mother": f"t{c['mother_trait']}",
            "father": f"t{c['father_trait']}",
            "n": c["n"],
            "n_dropped": c["n_dropped"],
            "n_mate_networks": c["n_mate_networks"],
            "largest_mate_network_share": c["largest_mate_network_share"],
            "crude": crude,
        }
        if c["stratified"] is not None:
            rec["stratified"] = {c["stratified"]["estimator"]: _record(c["stratified"], True)}
        records.append(rec)
    out = {
        "mating_pairs": {
            "n_total": s["n_total"],
            "n_dropped": {"unknown_stratum": s["n_dropped_unknown_stratum"]},
            "n_mate_networks": s["n_mate_networks"],
            "largest_mate_network_share": s["largest_mate_network_share"],
            "n_mothers_multiple_mates": s["n_mothers_multiple_mates"],
            "n_fathers_multiple_mates": s["n_fathers_multiple_mates"],
        },
        "mate_correlation": records,
        "inference": result["method"],
    }
    if n_traits == 2:
        wp = {}
        for sex, w in result["within_person"].items():
            key = KEYS[w["estimator"]]
            rec = {"estimator": w["estimator"], "n": w["n"]}
            if w["reason"] is not None:
                rec |= {key: None, "reason": w["reason"]}
            else:
                rec[key] = w["value"]
                if w["estimator"] in LATENT:
                    rec["boundary"] = w["boundary"]
            wp[sex] = rec
        out["within_person"] = wp
    return out


def _flatten(value: object, path: str, out: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _flatten(v, f"{path}[{i}]", out)
    else:
        out[path] = value


def diff(name: str, threads: int = 1) -> tuple[list[str], list[tuple[str, float, float, float]]]:
    """``(mismatches, gaps)``: discrete differences, and ``(path, ours, pedsum, |gap|)`` of every float."""
    g = load(name)
    ours, want = {}, {}
    _flatten(to_pedsum(run(g, threads), len(g.kinds)), "", ours)
    expected = {
        k: g.payload[k] for k in ("mating_pairs", "mate_correlation", "inference", "within_person") if k in g.payload
    }
    _flatten(expected, "", want)
    mismatches, gaps = [], []
    for path in sorted(set(ours) | set(want)):
        if path not in ours or path not in want:
            mismatches.append(f"{path}: ours {ours.get(path, '<absent>')!r} pedsum {want.get(path, '<absent>')!r}")
            continue
        a, b = ours[path], want[path]
        if isinstance(b, float) and not isinstance(a, bool) and isinstance(a, (int, float)):
            if math.isinf(b) or math.isinf(a):
                if a != b:
                    mismatches.append(f"{path}: ours {a!r} pedsum {b!r}")
            else:
                gaps.append((path, float(a), b, abs(a - b)))
        elif a != b or type(a) is not type(b):
            mismatches.append(f"{path}: ours {a!r} pedsum {b!r}")
    return mismatches, gaps
