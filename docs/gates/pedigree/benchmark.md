# Pedigree benchmark: columns path and warm calls (simACE plan pg-phenotype-pedigree-v2, unit 7)

Result: **gate 2 fails at 1M rows; every other gate passes.**

- Gate 1: the columns path is not slower or larger.  Commit B over commit A
  is at most 1.05 in every mode at both sizes, for wall time,
  `memory.peak` and `ru_maxrss`.
- Gate 2: a warm call saves enough.  It passes at 5.36M rows (1444 ms
  against a 1393 ms floor).  At 1M rows it saves 162 ms in the first run
  and 160 ms in a 20-pair recheck, against a 170 ms floor.
- Gate 3: building a Pedigree first costs nothing extra.  It passes at
  both sizes (1.036 and 0.999).

## How it was run

`tools/bench_pedigree.py` compares two builds, each a release wheel from
`maturin build --release` (no test hooks), unpacked into its own directory.

- A is commit A: the #7 work, snapshotted with `git stash create`.
- B is commit B: A plus this change.

Each run is a fresh process in its own systemd scope (`MemoryMax=12G`, no
swap), with `PG_PHENOTYPE_THREADS=1`.  Each run checks which native module
it imported.  Pairs interleave A and B and alternate which goes first.  The
`ped` mode runs on B only.

Every run uses the same input: two continuous traits (`liability1` and
`liability2` of a simACE rep's `pedigree.parquet`, with its `twin` and `sex`
columns) and `permutations=0`.  Wall time is `perf_counter` around each
public call, including the Python coercion the call does.  Memory is the
scope's `memory.peak`; `ru_maxrss` is in the data too.

Inputs:

- 1M rows (401,523 Mating Pairs): simACE `results/bench_scale/bench1M_am/rep1`.
- 5.36M rows (2,421,895 Mating Pairs): simACE `results/base/baseline1M/rep1`.

`gen` writes each input once as an `.npz`.

Machine: i7-9750H, `platform_profile` performance, turbo on,
`scaling_max_freq` 4.5 GHz, 3.3 GHz on a busy loop before the runs.  Load
from other sessions kept `load1` at 1.1 to 1.9 (medians 1.47 to 1.69).
Single pairs vary by up to about 15%, so the medians are what count.

```bash
pixi run python tools/bench_pedigree.py bench --site-a <A> --site-b <B> --data <npz> --pairs 10 \
    --pixi <absolute pixi> --out <jsonl>
pixi run python tools/bench_pedigree.py summary <jsonl>...
```

## 1M rows

From [`bench-1M-2026-10-09.jsonl`](bench-1M-2026-10-09.jsonl): 10 pairs per mode, `load1` median 1.47.

| Mode | A wall (ms) | B wall (ms) | B/A wall | B/A memory.peak | B/A ru_maxrss | Gate 1 |
|---|---|---|---|---|---|---|
| cols | 458 | 459 | 0.981 | 0.961 | 0.963 | pass |
| strat_unknown | 815 | 811 | 1.004 | 0.979 | 0.982 | pass |
| strat_known | 981 | 932 | 0.967 | 0.937 | 0.940 | pass |
| prep | 10103 | 10194 | 1.017 | 0.987 | 0.988 | pass |

| B, ms | T_cols | T_ctor | T_first | T_warm |
|---|---|---|---|---|
| median | 459 | 136 | 340 | 297 |

- Gate 2: `T_cols - T_warm` = 162 ms; the floor is 170 ms.  **Fail.**
- Gate 3: `(T_ctor + T_first) / T_cols` = 1.036; the limit is 1.05.  Pass.
- Three calls: `3 x T_cols` = 1376 ms, and `T_ctor + T_first + 2 x T_warm`
  = 1069 ms, 22% less.

Recheck of `cols` and `ped`, from
[`bench-1M-gate2-recheck-2026-10-09.jsonl`](bench-1M-gate2-recheck-2026-10-09.jsonl):
20 pairs, `load1` median 1.53.

- `cols`: B/A wall 0.965, `memory.peak` 0.960.
- B medians: `T_cols` 454 ms, `T_ctor` 138 ms, `T_first` 334 ms, `T_warm` 294 ms.
- Gate 2: 160 ms against 170 ms.  **Fail.**
- Gate 3: 1.039.  Pass.

### Why gate 2 misses at 1M

The floor is 80% of the stage table's validation plus pairs
(148 + 64 = 212 ms).  The parts a warm call skips measure about 178 ms:

- `T_ctor` is 136 to 138 ms.  It covers the Python coercion and the
  validation.
- Finding the pairs and networks (`T_first - T_warm`) is about 40 ms.

End to end, though, calls on a held Pedigree run about 17 ms slower than
the columns path predicts.  Gate 3's 1.036 to 1.039 shows the same gap.
The miss is that gap.

**Refuted:** the gap is not the wrapper's `malloc_trim(0)`.  A sequential
B-only probe (7 runs of each mode, without scopes) gave a 205 ms saving
with the trim on and 208 ms with it off.  That probe also clears the floor,
so at 1M rows gate 2 sits within run-to-run variation.  Under the gate's
protocol it failed twice.  The gap's cause is not known.

### Recheck after the review fixes

After the review, `polyserial_influence`'s scratch buffer was resized and
`analyse_cell` now reads stratification from its pairs.  Gate 1 was rerun
at 1M rows from a rebuilt B, from
[`bench-1M-review-recheck-2026-10-10.jsonl`](bench-1M-review-recheck-2026-10-10.jsonl):
10 pairs, `load1` median 1.77.

| Mode | A wall (ms) | B wall (ms) | B/A wall | B/A memory.peak | B/A ru_maxrss | Gate 1 |
|---|---|---|---|---|---|---|
| cols | 508 | 506 | 1.031 | 0.960 | 0.964 | pass |
| strat_unknown | 871 | 893 | 1.035 | 0.978 | 0.980 | pass |
| strat_known | 1029 | 1003 | 0.974 | 0.935 | 0.941 | pass |

## 5.36M rows

From [`bench-5M-2026-10-09.jsonl`](bench-5M-2026-10-09.jsonl) (8 pairs) and
[`bench-5M-prep-2026-10-09.jsonl`](bench-5M-prep-2026-10-09.jsonl)
(5 pairs).  `load1` median 1.69.  PA-FGRS's prep fits: about 80 s and
3.7 GiB per run.

| Mode | A wall (ms) | B wall (ms) | B/A wall | B/A memory.peak | B/A ru_maxrss | Gate 1 |
|---|---|---|---|---|---|---|
| cols | 3780 | 3886 | 1.048 | 0.951 | 0.951 | pass |
| strat_unknown | 6298 | 6076 | 0.956 | 1.008 | 1.008 | pass |
| strat_known | 7351 | 7399 | 0.981 | 0.947 | 0.948 | pass |
| prep | 82482 | 81148 | 0.980 | 0.987 | 0.988 | pass |

| B, ms | T_cols | T_ctor | T_first | T_warm |
|---|---|---|---|---|
| median | 3886 | 1159 | 2721 | 2442 |

- Gate 2: `T_cols - T_warm` = 1444 ms; the floor is 1393 ms.  Pass.
- Gate 3: `(T_ctor + T_first) / T_cols` = 0.999.  Pass.
- Three calls: `3 x T_cols` = 11659 ms, and `T_ctor + T_first + 2 x T_warm`
  = 8766 ms, 25% less.

The unstratified columns path's wall ratio, 1.048, is the closest to the
limit.  Its eight pairs range from 0.90 to 1.15.  The code path's work is
unchanged: the call validates, finds the pairs and builds the networks as
before, now inside a temporary Pedigree.  Its memory ratio is 0.951.

Stratified with unknown strata (gate 1b) is the case the plan flagged: a
temporary Pedigree keeps the full pair arrays while the call's filtered
copy is in use.  It passes at both sizes, with memory at 0.979 and 1.008.
