"""The Aalen-Johansen CIP gate (issue #4): pg-phenotype vs simACE and survfit.

    pixi run python -B tools/aj_gate.py --out docs/gates/aalen-johansen/gate.json

Reads the golden (``tools/make_simace_aj_golden.py``,
``tools/make_survival_aj_golden.py``), estimates each case with
:meth:`pg_phenotype.pafgrs.Cip.from_aalen_johansen` (``tests/aj_golden.py``),
and writes per case: the rows, cases and deaths, whether the step ages match
each reference, the largest gap to simACE (absolute) and to survfit
(relative), the gap to simACE with entries moved one ulp earlier (its
``entry <= t`` risk set), and ``K`` beside the cases-over-rows CIP fitACE's
``compute_empirical_cip`` gives.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pg_phenotype.pafgrs import Cip  # noqa: E402
from tests.aj_golden import load  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    golden = load()
    rows = {}
    for case in golden.cases:
        cip = Cip.from_aalen_johansen(case.age, case.status, entry=case.entry)
        row = {
            "n": len(case.age),
            "n_cases": case.n_cases,
            "n_deaths": int((case.status == 2).sum()),
            "n_steps": len(cip.ages),
            "ages_match_simace": bool(np.array_equal(cip.ages, case.simace_ages)),
            "ages_match_survfit": bool(np.array_equal(cip.ages, case.survfit_ages)),
            "max_abs_gap_simace": float(np.max(np.abs(cip.cip - case.simace_cif))),
            "max_rel_gap_survfit": float(np.max(np.abs(cip.cip / case.survfit_cif - 1))),
            "k": cip.prevalence,
        }
        if case.entry is not None:
            shifted = np.nextafter(case.entry, -np.inf).clip(0)
            emulated = Cip.from_aalen_johansen(case.age, case.status, entry=shifted)
            row["max_abs_gap_simace_entry_rule"] = float(np.max(np.abs(emulated.cip - case.simace_cif)))
        if case.n_dropped is not None:
            row["k_cases_over_rows"] = case.n_cases / (len(case.age) + case.n_dropped)
        rows[case.key] = row
    rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    payload = {
        "pg_phenotype_rev": rev.stdout.strip(),
        "simace_rev": golden.simace_rev,
        "survival": golden.survival_versions,
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    for key, row in rows.items():
        print(
            f"{key:28s} n={row['n']:5d} steps={row['n_steps']:5d} "
            f"simace={row['max_abs_gap_simace']:.1e} survfit={row['max_rel_gap_survfit']:.1e} "
            f"rule={row.get('max_abs_gap_simace_entry_rule', float('nan')):.1e} K={row['k']:.4f} "
            f"naive={row.get('k_cases_over_rows', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
