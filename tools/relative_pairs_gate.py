"""The Relative-Pair Correlation gate (ADR 0009): pg-phenotype vs a NumPy reference and simACE.

    pixi run python -B tools/relative_pairs_gate.py --out docs/gates/relative-pairs/gate.json

Reads the golden (``tools/make_simace_relative_pairs_golden.py``), runs the
comparisons of ``tests/relative_pairs_golden.py``, and writes the largest
gap per estimator against the reference and against simACE's directed
categories, and every symmetric category's double- minus single-entry
difference.  Exits non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.relative_pairs_golden import (  # noqa: E402
    ABS_SIMACE,
    ABS_SIMACE_TETRACHORIC,
    REL_ORACLE,
    SIMACE_MIN_PAIRS,
    checks,
    load,
    oracle_rows,
    simace_rows,
)


def worst(rows: list[dict], key: str) -> dict[str, object] | None:
    defined = [r for r in rows if r[key] is not None]
    if not defined:
        return None
    r = max(defined, key=lambda r: r[key])
    return {"gap": r[key], "rep": r["rep"], "category": r["category"], "n": len(defined)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    golden, _ = load()
    oracle, simace = oracle_rows(), simace_rows()
    directed = [r for r in simace if not r["symmetric"] and not r["simace_floor"]]
    symmetric = [r for r in simace if r["symmetric"]]
    passed = checks(oracle, simace)
    rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    gate = {
        "simace_rev": golden["simace_rev"],
        "reps": [
            {"rep": r["rep"], **r["run"], "n_rows": r["n_rows"], "n_phenotyped": r["n_phenotyped"]}
            for r in golden["reps"]
        ],
        "pg_phenotype_base_rev": rev.stdout.strip(),
        "tolerance": {
            "oracle_pearson_phi_abs": REL_ORACLE,
            "simace_directed_pearson_phi_abs": ABS_SIMACE,
            "simace_directed_tetrachoric_abs": ABS_SIMACE_TETRACHORIC,
            "simace_min_pairs": SIMACE_MIN_PAIRS,
        },
        "checks": passed,
        "pass": all(passed.values()),
        "oracle_max_gap": {k: worst(oracle, f"{k}_gap") for k in ("pearson", "phi")},
        "simace_directed_max_gap": {k: worst(directed, f"{k}_gap") for k in ("pearson", "tetrachoric", "phi")},
        "symmetric_double_minus_single": [
            {
                "rep": r["rep"],
                "category": r["category"],
                "n": r["n"],
                **{
                    est: {
                        "pg": r[est],
                        "simace": r[theirs],
                        "difference": None if r[est] is None or r[theirs] is None else r[est] - r[theirs],
                    }
                    for est, theirs in (
                        ("pearson", "simace_liability"),
                        ("tetrachoric", "simace_tetrachoric"),
                        ("phi", "simace_phi"),
                    )
                },
                "tetrachoric_boundary": r["tetrachoric_boundary"],
            }
            for r in symmetric
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate, indent=1) + "\n")
    print(json.dumps({k: gate[k] for k in ("checks", "pass", "oracle_max_gap", "simace_directed_max_gap")}, indent=1))
    if not gate["pass"]:
        sys.exit("the relative-pairs gate failed")


if __name__ == "__main__":
    main()
