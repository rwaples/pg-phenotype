"""Write simACE's parent-offspring regressions over the gate's pedigrees (issue #3).

Runs in simACE's pixi env against a simACE checkout at ``_simace.SIMACE_REV``,
whose ``simace`` it imports ahead of the env's own install:

    pixi run --manifest-path <simACE>/pixi.toml python -B tools/make_simace_po_regression_golden.py \\
        --simace <checkout> --out tests/golden/simace_po_regression_6355ffa6aebe

Each scenario is a ``run_simulation`` pedigree with affected status at 10%
prevalence (as simACE's stats tests build it).  ``golden.json`` holds, per
scenario, the pedigree columns, the values, and simACE's results:

- ``by_generation``, ``by_sex``: ``compute_parent_offspring_corr`` and
  ``compute_parent_offspring_corr_by_sex`` on ``liability1``.
- ``affected``: ``compute_parent_offspring_affected_corr`` on ``affected1``;
  ``affected_zero`` the same with ``affected1`` all 0.
- ``validate``: ``_validate_parent_offspring`` on A1 and A1 + C1 + E1;
  ``validate_tiny`` the same with A1 scaled by 1e-11 and C1 = E1 = 0, so the
  parent SD falls below ``safe_linregress``'s 1e-10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _simace import SIMACE_REV, import_simace

BASE = {
    "p_mztwin": 0.02,
    "A1": 0.5,
    "C1": 0.2,
    "E1": 0.3,
    "A2": 0.5,
    "C2": 0.2,
    "E2": 0.3,
    "rA": 0.3,
    "rC": 0.5,
    "assort1": 0.0,
    "assort2": 0.0,
}
SCENARIOS = {
    "n2000_g4": {"seed": 1, "N": 2000, "G_ped": 4, "mating_lambda": 0.5},
    "n2000_g4_am": {"seed": 2, "N": 2000, "G_ped": 4, "mating_lambda": 0.5, "assort1": 0.3},
    "n500_g3_remating": {"seed": 3, "N": 500, "G_ped": 3, "mating_lambda": 1.5},
    "n200_g3_noc": {"seed": 4, "N": 200, "G_ped": 3, "mating_lambda": 0.5, "C1": 0.0, "E1": 0.5},
    "n30_g3": {"seed": 5, "N": 30, "G_ped": 3, "mating_lambda": 0.5},
    "n12_g3": {"seed": 6, "N": 12, "G_ped": 3, "mating_lambda": 0.5},
}


def plain(obj: object) -> object:
    """JSON-ready: NumPy scalars to Python, NaN to None, messages dropped."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items() if k not in ("details", "informational")}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def scenario(name: str, params: dict) -> dict:
    import polars as pl
    from simace.analysis.stats.correlations import (
        compute_parent_offspring_affected_corr,
        compute_parent_offspring_corr,
        compute_parent_offspring_corr_by_sex,
    )
    from simace.analysis.validate.heritability import _validate_parent_offspring
    from simace.core.pedigree_arrays import PedigreeArrays
    from simace.phenotype.models._prevalence import case_status_from_liability
    from simace.simulation.simulate import run_simulation

    df = run_simulation(**{**BASE, **params})
    gen = df["generation"].to_numpy()
    aff = np.asarray(case_status_from_liability(df["liability1"].to_numpy(), 0.10, None, gen, "global"))
    df = df.with_columns(pl.Series("affected1", aff), pl.Series("affected2", aff))
    comp = {c: df[c].to_numpy() for t in (1, 2) for c in (f"A{t}", f"C{t}", f"E{t}")}
    ped = PedigreeArrays.from_frame(df)
    validate: dict = {}
    _validate_parent_offspring(df, comp, ped, validate)
    tiny = {**comp, "A1": comp["A1"] * 1e-11, "C1": np.zeros(len(df)), "E1": np.zeros(len(df))}
    validate_tiny: dict = {}
    _validate_parent_offspring(df, tiny, ped, validate_tiny)
    zero = df.with_columns(pl.Series("affected1", np.zeros(len(df), dtype=bool)))
    return {
        "name": name,
        "params": params,
        "pedigree": {c: df[c].to_numpy().tolist() for c in ("id", "mother", "father", "sex", "generation")},
        "values": {
            "liability1": df["liability1"].to_numpy().tolist(),
            "A1": comp["A1"].tolist(),
            "P1": (comp["A1"] + comp["C1"] + comp["E1"]).tolist(),
            "affected1": aff.astype(float).tolist(),
        },
        "simace": plain(
            {
                "by_generation": compute_parent_offspring_corr(df)["trait1"],
                "by_sex": {s: v["trait1"] for s, v in compute_parent_offspring_corr_by_sex(df).items()},
                "affected": compute_parent_offspring_affected_corr(df)["trait1"],
                "affected_zero": compute_parent_offspring_affected_corr(zero)["trait1"],
                "validate": {
                    "A1": validate["parent_offspring_A1_regression"],
                    "P1": validate["parent_offspring_liability1_regression"],
                },
                "validate_tiny": {"A1": validate_tiny["parent_offspring_A1_regression"]},
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import_simace(args.simace)
    args.out.mkdir(parents=True, exist_ok=True)
    golden = {
        "simace_rev": SIMACE_REV,
        "scenarios": [scenario(name, params) for name, params in SCENARIOS.items()],
    }
    (args.out / "golden.json").write_text(json.dumps(golden))


if __name__ == "__main__":
    main()
