"""The unit 9 parity gate: every golden (fixtures and gate set) through pg-phenotype, gaps per estimator.

    pixi run python tools/pedsum_am_gate.py --out docs/gates/assortative-mating/gate.json

Prints, per estimator and quantity (value, se, ci, p), the max absolute and
relative gap to pedsum and where it is, and every discrete mismatch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.assortative_golden import GATE_NAMES, NAMES
from tests.assortative_parity import diff

FIELD = re.compile(r"(crude|stratified)\.(\w+)\.(r|rho|value|se|ci\[\d\]|p_perm)$")
WITHIN = re.compile(r"within_person\.\w+\.(r|rho)$")


def quantity(path: str) -> tuple[str, str] | None:
    m = FIELD.search(path)
    if m:
        q = m.group(3)
        q = "value" if q in ("r", "rho", "value") else "ci" if q.startswith("ci") else q
        return f"{m.group(2)} ({m.group(1)})", q
    m = WITHIN.search(path)
    return ("within_person", "value") if m else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    worst: dict = defaultdict(lambda: {"abs": 0.0, "rel": 0.0, "n": 0, "at": None})
    mismatches = {}
    for name in NAMES + GATE_NAMES:
        mm, gaps = diff(name)
        if mm:
            mismatches[name] = mm
        for path, _ours, want, gap in gaps:
            key = quantity(path)
            if key is None:
                key = ("sample", "share")
            w = worst[key]
            w["n"] += 1
            rel = gap / max(abs(want), 1e-300) if want else gap
            if gap > w["abs"]:
                w["abs"], w["at"] = gap, f"{name}:{path}"
            w["rel"] = max(w["rel"], rel if want else 0.0)
    rows = [{"estimator": e, "quantity": q, **v} for (e, q), v in sorted(worst.items())]
    for r in rows:
        print(f"{r['estimator']:28s} {r['quantity']:6s} n={r['n']:5d} abs={r['abs']:.2e} rel={r['rel']:.2e}  {r['at']}")
    print(f"cases: {len(NAMES)} fixtures + {len(GATE_NAMES)} gate; discrete mismatches in {len(mismatches)}")
    for name, mm in mismatches.items():
        for m in mm:
            print(f"  {name}: {m}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"rows": rows, "mismatches": mismatches, "cases": NAMES + GATE_NAMES}, indent=1) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
