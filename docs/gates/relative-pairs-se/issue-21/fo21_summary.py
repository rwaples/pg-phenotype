"""Summarise fo21_probe.tsv (400 disjoint-seed lam-3 reps): SD(r̂)/SE per cell.

Over all reps with a
bootstrap 95% interval, and over each block of 100 reps (four independent 100-rep 'scenarios',
the gate's per-scenario sample size).

    pixi run python -B docs/gates/relative-pairs-se/issue-21/fo21_summary.py docs/gates/relative-pairs-se/issue-21/fo21_probe.tsv
"""

import sys

import numpy as np
import polars as pl

d = pl.read_csv(sys.argv[1], separator="\t")
rng = np.random.default_rng(21)
print("trait  gen cell reps  ratio  boot95          blocks of 100")
for (t, g, c), x in d.sort("rep").group_by("trait", "gen", "cell", maintain_order=True):
    r, se = x["r"].to_numpy(), x["se_parent"].to_numpy()
    ratio = r.std(ddof=1) / se.mean()
    idx = rng.integers(0, len(r), size=(2000, len(r)))
    boot = r[idx].std(axis=1, ddof=1) / se[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    blocks = [r[i : i + 100].std(ddof=1) / se[i : i + 100].mean() for i in range(0, len(r), 100)]
    print(f"{t} {g}   {c}   {len(r)}  {ratio:.3f}  [{lo:.3f}, {hi:.3f}]  " + " ".join(f"{b:.3f}" for b in blocks))
