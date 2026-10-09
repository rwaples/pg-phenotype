"""The parent-offspring regression gate (issue #3): simACE vs pg-phenotype.

    pixi run python -B tools/po_regression_gate.py --out docs/gates/po-regression/gate.json

Reads the golden (``tools/make_simace_po_regression_golden.py``), fits each
comparison with :func:`pg_phenotype.regression.parent_offspring`
(``tests/po_regression_golden.py``), and writes:

- the outcome counts, with every comparison where the two sides differ in
  being defined;
- the largest gap per field where both are defined, on the float64 paths and
  on simACE's float32 validation path, and the float32 emulation's gap;
- the ratio of pg-phenotype's Mate Network sandwich SE to its OLS SE.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.po_regression_golden import (  # noqa: E402
    REL,
    REL_F32,
    comparisons,
    gaps,
    load,
    simace_validate_float32,
    tolerance,
)


def worst(rows: list[tuple[float, str, str]]) -> dict[str, object] | None:
    if not rows:
        return None
    gap, scenario, label = max(rows)
    return {"gap": gap, "scenario": scenario, "label": label, "n": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cs = comparisons()
    by_path: dict[str, dict[str, list]] = {"float64": {}, "float32_validation": {}}
    for c in cs:
        if c.outcome != "both":
            continue
        path = "float32_validation" if tolerance(c) == REL_F32 else "float64"
        for field, gap in gaps(c).items():
            by_path[path].setdefault(field, []).append((gap, c.scenario, c.label))
    emulation = 0.0
    for sc in load()["scenarios"]:
        for key in ("A1", "P1"):
            redone = simace_validate_float32(sc, key)
            if redone is not None:
                want = sc["simace"]["validate"][key]
                got = dict(zip(("slope", "intercept", "r_squared"), redone, strict=True))
                emulation = max(emulation, *(abs(got[k] - want[k]) / abs(want[k]) for k in got))
    ratios = np.array([c.pg.se / c.pg.se_ols for c in cs if c.outcome == "both" and c.pg.se and c.pg.se_ols])
    rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    gate = {
        "simace_rev": load()["simace_rev"],
        "pg_phenotype_base_rev": rev.stdout.strip(),
        "tolerance": {"relative": REL, "relative_float32_validation": REL_F32},
        "outcomes": dict(Counter(c.outcome for c in cs)),
        "differ_in_definition": [
            {
                "scenario": c.scenario,
                "label": c.label,
                "outcome": c.outcome,
                "pg_n": c.pg.n,
                "pg_reason": c.pg.reason,
                "simace_n": c.simace.get("n_pairs"),
            }
            for c in cs
            if c.outcome != "both"
        ],
        "max_gap": {path: {f: worst(rows) for f, rows in fields.items()} for path, fields in by_path.items()},
        "float32_emulation_max_relative_gap": emulation,
        "sandwich_over_ols_se": {
            "n": len(ratios),
            "percentiles": dict(
                zip(
                    ("min", "p10", "median", "p90", "max"),
                    np.percentile(ratios, [0, 10, 50, 90, 100]).round(4).tolist(),
                    strict=True,
                )
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate, indent=1) + "\n")
    print(
        json.dumps(
            {k: gate[k] for k in ("outcomes", "max_gap", "float32_emulation_max_relative_gap", "sandwich_over_ols_se")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
