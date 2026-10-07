"""Regenerate the R PAFGRS golden scores (univariate).

    pixi run python tools/make_r_golden.py <PAFGRS checkout> [--rscript "conda run -n pafgrs_r Rscript"]

Writes tests/golden/r_pafgrs_<sha>/ with the fixture inputs (pedigree,
traits, CIP, parameters) and R's ``pa_fgrs`` scores per case and proband.
Each proband's problem is its relatives with ``w > 0`` in ascending row
order and the covariance ``2 h2 phi`` from pg-phenotype's PA-FGRS prep, whose kinship
tests/test_prep.py holds to pedigree-graph.  ``h2`` is 0.25, 0.5 or 0.75 so
every covariance entry is dyadic: R's floating row sums are then exact and
its stable ``order`` breaks ties by row, as the canonical order does.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pg_phenotype import pafgrs  # noqa: E402
from tests.pedigrees import random_pedigree  # noqa: E402
from tests.scoring_fixtures import observe, proband_view, random_trait  # noqa: E402

CASES = [
    {"seed": 0, "ndegree": 2, "h2": 0.5},
    {"seed": 1, "ndegree": 3, "h2": 0.25},
    {"seed": 2, "ndegree": 1, "h2": 0.75},
    {"seed": 3, "ndegree": 3, "h2": 0.5},
]


def _problems(prep, trait, h2: float) -> tuple[list[int], list[str]]:
    affected, w, thr = observe(trait)
    ids, lines = [], []
    for i in range(prep.n_probands):
        rows, kin_p, kin = proband_view(prep, i)
        valid = w[rows] > 0
        if not valid.any():
            continue
        rows, kin_p, kin = rows[valid], kin_p[valid], kin[np.ix_(valid, valid)]
        d = len(rows)
        cov = np.empty((d + 1, d + 1))
        cov[0, 0] = h2
        cov[0, 1:] = cov[1:, 0] = 2.0 * kin_p * h2
        cov[1:, 1:] = 2.0 * kin * h2
        np.fill_diagonal(cov[1:, 1:], 1.0)
        fmt = lambda a: " ".join(f"{v:.17g}" for v in np.ravel(a))  # noqa: E731
        lines += [f"{d} {thr:.17g}", fmt(affected[rows].astype(float)), fmt(w[rows]), fmt(cov)]
        ids.append(i)
    return ids, lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pafgrs", type=Path)
    parser.add_argument("--rscript", default="conda run -n pafgrs_r Rscript")
    args = parser.parse_args()
    sha = subprocess.run(
        ["git", "-C", str(args.pafgrs), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    out = ROOT / "tests" / "golden" / f"r_pafgrs_{sha}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"pafgrs_commit": sha, "cases": []}
    for case in CASES:
        ped = random_pedigree(case["seed"], generations=5)
        trait = random_trait(len(ped["id"]), case["seed"])
        prep = pafgrs.prepare(ped, ndegree=case["ndegree"])
        ids, lines = _problems(prep, trait, case["h2"])
        with tempfile.TemporaryDirectory() as tmp:
            problems, scores = Path(tmp) / "problems.txt", Path(tmp) / "scores.txt"
            problems.write_text(f"{len(ids)}\n" + "\n".join(lines) + "\n")
            subprocess.run(
                [
                    *shlex.split(args.rscript),
                    str(ROOT / "tools" / "make_r_golden.R"),
                    str(args.pafgrs),
                    str(problems),
                    str(scores),
                ],
                check=True,
            )
            est_var = np.loadtxt(scores, ndmin=2)
        name = f"seed{case['seed']}_ndeg{case['ndegree']}_h2_{case['h2']}"
        np.savez_compressed(
            out / f"{name}.npz",
            **ped,
            affected=trait.affected,
            age=trait.age,
            cip_ages=trait.cip.ages,
            cip_values=trait.cip.cip,
            proband_index=np.array(ids),
            est=est_var[:, 0],
            var=est_var[:, 1],
        )
        manifest["cases"].append({**case, "file": f"{name}.npz", "n_scored": len(ids)})
        print(f"{name}: {len(ids)} probands scored by R")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
