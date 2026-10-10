"""Write survfit's Aalen-Johansen CIFs over the simACE golden's cases (issue #4).

    pixi run python -B tools/make_survival_aj_golden.py --golden tests/golden/simace_aj_<sha12>

Hands each case of ``cases.npz`` to ``tools/survival_aj.R`` in the ``r``
environment as C99 hex floats, so R reads and writes them exactly, and
writes ``survival.json`` beside it: per case, the case ages and survfit's
cause-1 probability at each (``timefix = FALSE``), and the R and survival
versions.  survfit counts a row at risk at ``t`` when ``entry < t <= age``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--golden", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((args.golden / "simace.json").read_text())["cases"]
    arrays = np.load(args.golden / "cases.npz")
    with tempfile.TemporaryDirectory() as tmp:
        keys = list(cases)
        for i, key in enumerate(keys):
            entry = arrays[f"{key}/entry"] if cases[key]["entry"] else None
            rows = ["entry,age,status"]
            for j, (a, s) in enumerate(zip(arrays[f"{key}/age"], arrays[f"{key}/status"], strict=True)):
                e = "" if entry is None else float(entry[j]).hex()
                rows.append(f"{e},{float(a).hex()},{int(s)}")
            (Path(tmp) / f"{i}.csv").write_text("\n".join(rows) + "\n")
        run = subprocess.run(
            ["pixi", "run", "-e", "r", "Rscript", str(ROOT / "tools/survival_aj.R"), tmp],
            capture_output=True,
            text=True,
            check=False,
            cwd=ROOT,
        )
        if run.returncode:
            raise SystemExit(run.stderr)
        out = {}
        for i, key in enumerate(keys):
            pairs = [line.split(",") for line in (Path(tmp) / f"{i}.out").read_text().split()]
            out[key] = {
                "ages": [float.fromhex(a) for a, _ in pairs],
                "cif": [float.fromhex(c) for _, c in pairs],
            }
    payload = {"versions": run.stdout.strip(), "cases": out}
    (args.golden / "survival.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{payload['versions']}: wrote {len(out)} cases")


if __name__ == "__main__":
    main()
