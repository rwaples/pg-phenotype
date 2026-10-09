"""The tetrachoric gate (issue #2): simACE's ``tetrachoric_from_table`` vs pg-phenotype.

    pixi run python -B tools/tetrachoric_gate.py --out docs/gates/tetrachoric/gate.json

Reads the golden (``tools/make_simace_tetrachoric_golden.py``), fits every
table with :func:`pg_phenotype.correlation.tetrachoric` and the SciPy oracle
(``tests/oracle/tetrachoric_reference.py``), and prints:

- the discrete outcomes side by side (pg reason or boundary, simACE NaN);
- the rho gaps, pg to simACE and pg to the oracle, where both fit inside the bound;
- the SEs: pg against the oracle's two-step SE, and each against the
  known-thresholds SE.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pg_phenotype.correlation import tetrachoric  # noqa: E402
from tests.oracle.tetrachoric_reference import reference  # noqa: E402
from tests.tetrachoric_golden import load  # noqa: E402


def summary(values: list[float], where: list[object]) -> dict[str, object]:
    if not values:
        return {"n": 0}
    a = np.asarray(values)
    worst = int(np.argmax(a))
    return {
        "n": len(values),
        "median": float(np.median(a)),
        "q99": float(np.quantile(a, 0.99)),
        "max": float(a[worst]),
        "at": where[worst],
    }


def ratio_summary(values: list[float]) -> dict[str, object]:
    a = np.asarray(values)
    return {"n": len(values), "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    golden = load()
    outcomes: Counter[tuple[str, str]] = Counter()
    gaps: dict[str, tuple[list[float], list[object]]] = {
        k: ([], []) for k in ("rho_simace", "rho_oracle", "rho_simace_near_bound", "se_oracle")
    }
    boundary_simace_r: list[float] = []
    simace_over_known: list[float] = []
    pg_over_known: list[float] = []
    for case in golden.cases:
        pg = tetrachoric(table=case.table)
        ref = reference(case.table)
        pg_kind = pg.reason or ("boundary" if pg.boundary else "interior")
        outcomes[(pg_kind, "nan" if case.simace_r is None else "r")] += 1
        where = {"set": case.set, "index": case.index, "table": case.table}
        if pg_kind == "boundary" and case.simace_r is not None:
            boundary_simace_r.append(case.simace_r)
        if pg_kind != "interior":
            continue
        assert pg.value is not None
        n = sum(map(sum, case.table))
        if case.simace_r is not None:
            key = "rho_simace" if abs(pg.value) < 0.99 else "rho_simace_near_bound"
            gaps[key][0].append(abs(pg.value - case.simace_r))
            gaps[key][1].append({**where, "pg": pg.value, "simace": case.simace_r})
        if ref.rho is not None:
            gaps["rho_oracle"][0].append(abs(pg.value - ref.rho))
            gaps["rho_oracle"][1].append({**where, "pg": pg.value, "oracle": ref.rho})
            assert ref.se_two_step is not None
            assert ref.se_known_thresholds is not None
            if pg.se is not None:
                want = ref.se_two_step * math.sqrt(n / (n - 1))
                gaps["se_oracle"][0].append(abs(pg.se / want - 1))
                gaps["se_oracle"][1].append({**where, "pg": pg.se, "oracle": want, "rho": pg.value})
                pg_over_known.append(pg.se / ref.se_known_thresholds)
            if case.simace_se is not None:
                simace_over_known.append(case.simace_se / ref.se_known_thresholds)
    report = {
        "simace_rev": golden.simace_rev,
        "cases": len(golden.cases),
        "outcomes": [{"pg": k[0], "simace": k[1], "count": v} for k, v in sorted(outcomes.items())],
        "gaps": {k: summary(*v) for k, v in gaps.items()},
        "boundary_simace_abs_r": ratio_summary([abs(r) for r in boundary_simace_r]) if boundary_simace_r else None,
        "se_ratio_simace_to_known_thresholds": ratio_summary(simace_over_known),
        "se_ratio_pg_to_known_thresholds": ratio_summary(pg_over_known),
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
