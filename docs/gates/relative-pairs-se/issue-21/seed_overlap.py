"""Issue #21: the se_calibration grid's rep seeds overlap across scenarios.

simACE seeds rep k of a scenario with `seed + k - 1`; the grid's six base seeds are 10 apart, so the
scenarios share most seeds.  This script (1) lists the seeds shared between scenarios, from each
rep's params.yaml, (2) checks that a shared seed at equal pedigree parameters gives the same r̂
(fo_probe.tsv, matched by seed), (3) recomputes trait1 FO generation 5's SD(r̂)/SE over the
distinct seeds, and (4) puts that value against a calibrated cell's sampling spread and the gate's
other Pearson cells.

    pixi run python -B docs/gates/relative-pairs-se/issue-21/seed_overlap.py \
        <simACE>/results/se_calibration docs/gates/relative-pairs-se/issue-21/fo_probe.tsv
"""

import itertools
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

root, probe = Path(sys.argv[1]), Path(sys.argv[2])
seeds = pl.DataFrame(
    [
        {
            "scenario": p.parent.name,
            "rep": int(p.name[3:]),
            "seed": int(re.search(r"^seed: (\d+)$", (p / "params.yaml").read_text(), re.M)[1]),
        }
        for p in sorted(root.glob("se_cal_*/rep*"))
        if (p / "run.yaml").exists()
    ]
)
by = {s: set(g["seed"]) for (s,), g in seeds.group_by("scenario")}
print("seed range per scenario:")
print(
    seeds.group_by("scenario")
    .agg(pl.col("seed").min().alias("min"), pl.col("seed").max().alias("max"), pl.len())
    .sort("min")
)
print("\nshared seeds between scenarios:")
for a, b in itertools.combinations(sorted(by), 2):
    if by[a] & by[b]:
        print(f"  {a} & {b}: {len(by[a] & by[b])}")
print(f"\ndistinct seeds over the {len(seeds)} reps: {seeds['seed'].n_unique()}")
lam3 = ["se_cal_lam3_K10", "se_cal_lam3_K02", "se_cal_lam3_K10_hiA"]
print(
    f"distinct seeds over the three lam-3 scenarios: {seeds.filter(pl.col('scenario').is_in(lam3))['seed'].n_unique()}"
)

d = pl.read_csv(probe, separator="\t").join(seeds, on=["scenario", "rep"])
for trait, cell in (("trait1", "FO"), ("trait1", "MO"), ("trait2", "FO")):
    c = d.filter(trait=trait, gen=5, cell=cell)
    k10 = c.filter(scenario="se_cal_lam3_K10").select("seed", r10="r")
    k02 = c.filter(scenario="se_cal_lam3_K02").select("seed", r02="r")
    m = k10.join(k02, on="seed")
    print(
        f"\n{trait} {cell} gen5: K10 vs K02 on {len(m)} shared seeds, max |r10 - r02| = {(m['r10'] - m['r02']).abs().max():.2e}"
    )
    hia = c.filter(scenario="se_cal_lam3_K10_hiA").select("seed", rh="r").join(k10, on="seed")
    print(f"  hiA vs K10 on {len(hia)} shared seeds: cor(r) = {np.corrcoef(hia['rh'], hia['r10'])[0, 1]:.3f}")
    same_rep = (
        c.filter(scenario="se_cal_lam3_K10")
        .select("rep", a="r")
        .join(c.filter(scenario="se_cal_lam3_K02").select("rep", b="r"), on="rep")
    )
    print(f"  K10 vs K02 matched by rep index: cor(r) = {np.corrcoef(same_rep['a'], same_rep['b'])[0, 1]:.3f}")
    # K10 and K02 share pedigrees; pool them over distinct seeds.
    pool = c.filter(pl.col("scenario").is_in(lam3[:2])).unique("seed", keep="first")
    print(
        f"  lam3, A 0.5, distinct seeds: {len(pool)}, SD(r)/mean SE = {pool['r'].std() / pool['se_parent'].mean():.3f}"
    )
    for s in lam3:
        own = c.filter(scenario=s)
        print(f"  {s}: {len(own)} reps, SD(r)/mean SE = {own['r'].std() / own['se_parent'].mean():.3f}")

# A calibrated cell's SD(r̂)/SE over k reps is about the SD of k standard normals (the SE's own CV,
# 2-7%, is ignored).  Its tail probability, and the gate's Pearson cells (pooled and stratum,
# mean n >= 1000) as far out.
gate = Path(__file__).resolve().parents[1] / "rust_calibration.tsv"
cal = pl.read_csv(gate, separator="\t", infer_schema_length=None).filter(
    pl.col("stratum") != "stratified", pl.col("mean_n") >= 1000, estimator="pearson"
)
sim = np.random.default_rng(21).standard_normal((200_000, 110)).std(axis=1, ddof=1)
print(f"\nP(SD of 110 N(0,1) >= 1.258) = {(sim >= 1.258).mean():.2e}")
print(f"gate Pearson cells: {len(cal)}; with ratio >= 1.18: {len(cal.filter(pl.col('ratio_rust') >= 1.18))}")
print(
    cal.filter(pl.col("ratio_rust") >= 1.18)
    .select("scenario", "trait", "cell", "stratum", "ratio_rust")
    .sort("ratio_rust")
)

# Is the 110-pedigree value carried by a few pedigrees?  Drop the k reps farthest from the mean.
c = (
    d.filter(trait="trait1", gen=5, cell="FO", scenario="se_cal_lam3_K10")
    .vstack(d.filter(trait="trait1", gen=5, cell="FO", scenario="se_cal_lam3_K02"))
    .unique("seed", keep="first")
)
z = ((c["r"] - c["r"].mean()) / c["se_parent"]).to_numpy()
order = np.argsort(-np.abs(z))
print(f"\ntrait1 FO gen5, 110 lam-3 pedigrees: largest |r - mean| / SE: {np.round(np.abs(z[order[:5]]), 2).tolist()}")
for k in (0, 1, 2, 3, 5):
    keep = c[np.sort(order[k:])]
    print(f"  dropping the {k} farthest: SD(r)/mean SE = {keep['r'].std() / keep['se_parent'].mean():.3f}")
