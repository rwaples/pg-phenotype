# simACE consumer experiment (simACE plan pg-phenotype-pedigree-v2, unit 7, gate 4)

Result: **pass**.  On both reps, every estimate of simACE #47's planned
three calls matches simACE's NumPy code within 1e-12, with a Pedigree and
with columns.  Every pair count is equal.  Wall time and memory are
reported only.

## How it was run

`tools/consumer_simace.py` runs in simACE's env from the simACE root.
Commit B's release wheel is first on `PYTHONPATH`; simACE has no commit
and no relock.  Each run is a fresh process in its own systemd scope, with
one thread for pg-phenotype, polars and OpenMP.  All three paths' modules
load before the timer.  Each run reads the rep's `pedigree.parquet` and
`params.yaml`.  The three paths alternate order across repeats.

- **today**: simACE's NumPy code.
  - `compute_mate_correlation` gives the stats 2x2.
  - `PedigreeArrays.from_frame` and `validate_assortative_mating` give
    liability per trait, μ_A per trait, and both cross-trait cells.
  - `observed_mate_correlations` for both traits gives μ_A and r_ho.
- **columns**: the three calls on the polars frame: liability for stats,
  liability for validation, then A for validation.
- **pedigree**: the same three calls on one `Pedigree` built from the frame.

r_ho is reported, not gated.  simACE computes it from `A + C + E`, which are
stored as Float32.  The three calls read the stored Float64 `liability`
column, which differs from that sum by up to 2.8e-7 (coverage rep).  So
#47's design changes simACE's r_ho by about 1e-9, by changing the input,
not by any pg-phenotype difference.

```bash
PYTHONPATH=<B site> pixi run python -P external/pg-phenotype/tools/consumer_simace.py bench \
    --rep results/bench_scale/bench1M_am/rep1 --repeats 7 --pixi <absolute pixi> --out <jsonl>
pixi run python -P external/pg-phenotype/tools/consumer_simace.py summary <jsonl>...
```

## Results

Medians of 5 repeats (coverage rep) and 7 repeats (1M rep), 2026-10-09,
`load1` 1.1 to 1.5.  Data:
[`consumer-coverage-2026-10-09.jsonl`](consumer-coverage-2026-10-09.jsonl),
[`consumer-1M-2026-10-09.jsonl`](consumer-1M-2026-10-09.jsonl).

| Rep | Pairs | Path | Wall (ms) | memory.peak (MiB) | ru_maxrss (MiB) | Max abs diff | r_ho diff | Equal |
|---|---|---|---|---|---|---|---|---|
| test/coverage_scenario/rep1 | 2,982 | today | 151 | 109 | 196 |  |  |  |
| test/coverage_scenario/rep1 | 2,982 | columns | 261 | 127 | 224 | 0.0e+00 | 2.3e-09 | yes |
| test/coverage_scenario/rep1 | 2,982 | pedigree | 288 | 126 | 224 | 0.0e+00 | 2.3e-09 | yes |
| bench_scale/bench1M_am/rep1 | 401,523 | today | 1301 | 295 | 383 |  |  |  |
| bench_scale/bench1M_am/rep1 | 401,523 | columns | 1714 | 356 | 454 | 8.2e-15 | 6.0e-11 | yes |
| bench_scale/bench1M_am/rep1 | 401,523 | pedigree | 1374 | 324 | 423 | 8.2e-15 | 6.0e-11 | yes |

At 1M rows, one Pedigree takes the three calls from 1714 ms to 1374 ms
(20% less) and from 356 MiB to 324 MiB.  That is still 6% slower than
simACE's NumPy code (1301 ms) and uses 29 MiB more.  The three calls also
return SEs and CIs that simACE does not read today.  On the coverage rep's
7,632 rows, fixed per-call costs dominate.
