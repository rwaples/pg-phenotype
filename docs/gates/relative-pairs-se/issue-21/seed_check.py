"""Issue #21: which simACE reproduces the gate's stored pedigrees seed for seed?

Compares the stored reps 1-3 of se_cal_lam3_K10 (seeds 55030-55032; run.yaml: simace 0.1.1.dev38,
source b2181f9-dirty) with the same seeds re-simulated, one directory per re-simulation, each holding
rep{1,2,3}/pedigree.parquet:

    pixi run python -B docs/gates/relative-pairs-se/issue-21/seed_check.py \
        <simACE>/results/se_calibration/se_cal_lam3_K10 <current-simACE dir> <b2181f9 dir>
"""

import sys
from pathlib import Path

import polars as pl

stored, *runs = (Path(p) for p in sys.argv[1:])
cols = ["id", "sex", "mother", "father", "twin", "generation", "A1", "E1", "liability1", "A2", "C2", "E2"]
for root in runs:
    for k in (1, 2, 3):
        new = pl.read_parquet(root / f"rep{k}/pedigree.parquet").select(cols)
        old = pl.read_parquet(stored / f"rep{k}/pedigree.parquet").select(cols)
        print(f"{root} rep{k}: identical to the stored pedigree: {new.equals(old)}")
