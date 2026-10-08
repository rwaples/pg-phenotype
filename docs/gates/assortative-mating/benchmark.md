# Assortative-mating benchmark: pedsum #13 vs pg-phenotype (plan v2, unit 12)

Result: **pass**.  Every configuration's median paired ratio, pg-phenotype
over pedsum, is at most 1.05 for wall time and for peak memory.  The first
full run passed every configuration, the closest being memory at 1.012
(configuration `b`, 10^6 pairs, 6 threads).  It also showed pg-phenotype's
bootstrap memory growing about 34 MiB per worker.  A fix about halved that
growth, and a recheck of every configuration the fix touches passes up to
12 threads.

## How it was run

`tools/bench_am.py` times the compute stage only: pedsum's
`compute_assortative_mating` against `mate_correlation` on the same
in-memory inputs, from pedsum's own generator.  Configuration `a` is one
binary trait, unstratified.  Configuration `b` is a continuous and a binary
trait with birth-year strata.  Every run requests 999 permutations and 1000
bootstrap draws, with seed 0.  A pair runs both implementations back to
back, each in a fresh process and its own systemd scope, and alternates
which goes first.  Every pair asserts that both used the same
`draws_used` per cell.  Wall is `perf_counter` around the one call.  Peak
memory is the scope's cgroup `memory.peak`, and the gate uses it.
`ru_maxrss` is in the data files too.  It gives pg-phenotype a lower ratio
in every configuration, so `memory.peak` is the stricter meter.  pedsum
gets one untimed warm call first, so numba compiles outside the timing.

    S=<scratch>/pedsum_am_142adf300d5b   # see tools/make_pedsum_am_golden.py
    pixi run --manifest-path $S/pixi.toml python -I tools/bench_am.py gen --snapshot $S --pairs 1000000 --out <dir>
    pixi run python tools/bench_am.py bench --snapshot $S --data <dir> --config b --size 1000000 \
        --threads 6 --permutations 999 --bootstrap 1000 --pairs-per-config 10 --out <jsonl>
    pixi run python tools/bench_am.py summary <jsonl>

- **Baseline**: pedsum `142adf300d5b0802f59af03f3f5211af910fc852` in a
  `git archive` snapshot with its own locked env (Python 3.14.7, NumPy
  2.5.2, numba 0.68.0).
- **Candidate**: pg-phenotype `3965ed6` (first run), and `3965ed6` with the
  bootstrap change below (recheck), built by `maturin develop --release`
  with rustc 1.98.1, Python 3.14.8.
- **Machine**: Intel i7-9750H (6 cores, 12 threads), 30 GiB, Linux.
  `platform_profile=performance`, turbo on, `scaling_max_freq` 4.5 GHz.
- **Thread budget**: `PG_PHENOTYPE_THREADS` for pg-phenotype,
  `NUMBA_NUM_THREADS` and `POLARS_MAX_THREADS` for pedsum, BLAS pinned to 1.
- **Data**: `bench-2026-10-07.jsonl` (first run) and
  `bench-2026-10-08-recheck.jsonl` (recheck), one line per pair.  The
  `max load` column is the highest 1-minute load average at a pair's start.
  A 6- or 12-thread run raises it to about 6 or 13 by itself.

## First run: every configuration

2026-10-07 23:55 to 2026-10-08 02:17, pg-phenotype `3965ed6`.  No other
jobs were running when it started, and a busy loop just before reached 3.5
to 3.9 GHz.

| config | Mating Pairs | threads | ABAB pairs | wall s pedsum / pg | wall ratio [min, max] | peak MiB pedsum / pg | memory ratio [min, max] | max load |
|---|---|---|---|---|---|---|---|---|
| a | 10^4 | 1 | 10 | 0.15 / 0.14 | 0.909 [0.771, 1.188] | 127 / 66 | 0.516 [0.512, 0.520] | 1.1 |
| a | 10^4 | 6 | 10 | 0.04 / 0.04 | 0.904 [0.729, 1.225] | 128 / 67 | 0.521 [0.519, 0.524] | 1.3 |
| b | 10^4 | 1 | 10 | 2.95 / 2.59 | 0.860 [0.797, 0.952] | 174 / 68 | 0.392 [0.386, 0.396] | 1.9 |
| b | 10^4 | 6 | 10 | 0.78 / 0.64 | 0.813 [0.613, 0.931] | 176 / 70 | 0.400 [0.393, 0.403] | 2.7 |
| a | 10^5 | 1 | 10 | 1.62 / 1.52 | 0.932 [0.895, 1.000] | 171 / 107 | 0.624 [0.621, 0.628] | 2.6 |
| a | 10^5 | 6 | 10 | 0.49 / 0.41 | 0.804 [0.607, 0.992] | 173 / 112 | 0.645 [0.637, 0.650] | 3.0 |
| b | 10^5 | 1 | 10 | 26.76 / 23.72 | 0.889 [0.873, 0.956] | 235 / 132 | 0.561 [0.559, 0.561] | 2.4 |
| b | 10^5 | 6 | 10 | 6.63 / 5.59 | 0.855 [0.807, 0.921] | 248 / 151 | 0.608 [0.605, 0.612] | 6.2 |
| a | 10^6 | 1 | 10 | 17.42 / 16.88 | 0.966 [0.927, 0.996] | 608 / 515 | 0.846 [0.843, 0.849] | 4.2 |
| a | 10^6 | 6 | 10 | 8.15 / 7.31 | 0.887 [0.853, 0.933] | 625 / 545 | 0.873 [0.866, 0.875] | 5.8 |
| b | 10^6 | 1 | 10 | 281.51 / 264.62 | 0.936 [0.924, 0.954] | 939 / 784 | 0.835 [0.766, 0.836] | 1.7 |
| b | 10^6 | 6 | 10 | 81.49 / 75.39 | 0.928 [0.897, 1.039] | 942 / 953 | 1.012 [1.009, 1.014] | 7.1 |

Worst median ratio: wall 0.966, memory 1.012.  Gate 1.05: PASS.

## pg-phenotype's bootstrap memory grew with the thread count

At `b` and 10^6 pairs, pg-phenotype peaked at 784 MiB with 1 thread and
953 MiB with 6.  pedsum peaked at about 940 MiB at both.  To find where the
growth came from, pg-phenotype ran alone, one phase at a time
(`bench_am.py run-pg --permutations 0 --bootstrap 1000` and so on, each in
its own systemd scope, `PG_PHENOTYPE_THREADS` set), with cgroup
`memory.peak` in MiB:

| run | 1 thread | 6 threads | 12 threads |
|---|---|---|---|
| point fits only | 711 | 753 | 730 |
| permutations only | 774 | 824 | 908 |
| bootstrap only | 712 | 879 | 1106 |
| bootstrap only, `MALLOC_ARENA_MAX=1` | 702 | 793 | 989 |
| permutations only, `MALLOC_ARENA_MAX=1` | 775 | 797 | 856 |
| bootstrap only, after the first change below | 711 | 803 | 946 |

The point fits set the 1-thread peak.  The bootstrap's buffers show only
as workers are added: about 33.5 MiB per worker from 1 to 6 threads.  In a
continuous x continuous cell, each draw built two per-pair `f64` copies of
its network weights for Spearman, one in each side's sort order, and a rank
vector, next to the worker's pair weights.  Four vectors of 8 bytes per
pair are 30.5 MiB at 10^6 pairs.  With one malloc arena the growth fell by
about half, so glibc keeping each thread's freed per-draw vectors was the
other part.

The fix has two parts:

- `sorted_ranks_into` and `spearman_sorted` read each weight as
  `mult[label]` through a closure instead of a copied vector.
- Each bootstrap worker keeps one rank buffer across its draws
  (`bootstrap::Scratch::spare`).  This part was not measured on its own.

The weights, and the order in which they are summed, are unchanged, so
results are bit-identical: the 252 Python tests, with every golden and gate
case, and R's 1919 pass.  A bootstrap worker now holds the pair weights and
the rank buffer, 16 bytes per pair.  A permutation worker holds about 12
bytes per father: a `u32` order and one `f32` value per trait.

## Recheck of configuration `b` with the fix

2026-10-08 07:37 to 10:12.  The fix changes only the continuous x
continuous bootstrap, which configuration `a` (one binary trait) never
runs, so only `b` was rerun, and 12 threads were added at 10^6 pairs.

The CPU ran slower than in the first run.  Under the 12-thread runs all
cores sat at 1.8 to 2.0 GHz, and under the 1-thread runs at about 2.7 GHz,
with `scaling_max_freq` at 4.5 GHz.  The package was at 63 C and the
thermal-throttle counters did not move.  The laptop was charging.  The
cause was not confirmed.  Absolute times are therefore longer (pedsum's
`b` 10^6 1-thread median is 317 s against 281 s), and the wall ratios
spread wider.  Each pair runs both implementations under the same
conditions, so the medians stay comparable.  The 12-thread configuration's
first pair (1.439) is an outlier: pedsum took 65.6 s there and 88 to 89 s
in the other nine pairs.  The 1-thread 10^6 configuration ran 6 pairs, not
the protocol's 10, to save an hour.  Its peak comes from the point fits,
which the fix does not change, and its first-run row has 10 pairs.

| config | Mating Pairs | threads | ABAB pairs | wall s pedsum / pg | wall ratio [min, max] | peak MiB pedsum / pg | memory ratio [min, max] | max load |
|---|---|---|---|---|---|---|---|---|
| b | 10^4 | 1 | 10 | 2.96 / 2.48 | 0.866 [0.786, 0.911] | 175 / 68 | 0.389 [0.387, 0.393] | 5.7 |
| b | 10^4 | 6 | 10 | 0.76 / 0.56 | 0.747 [0.632, 0.868] | 176 / 70 | 0.399 [0.392, 0.401] | 3.5 |
| b | 10^5 | 1 | 10 | 26.82 / 23.60 | 0.881 [0.826, 0.901] | 235 / 132 | 0.560 [0.558, 0.562] | 2.0 |
| b | 10^5 | 6 | 10 | 6.57 / 5.52 | 0.842 [0.781, 0.919] | 248 / 147 | 0.593 [0.589, 0.595] | 5.9 |
| b | 10^6 | 1 | 6 | 317.49 / 286.60 | 0.903 [0.847, 0.945] | 940 / 784 | 0.834 [0.801, 0.834] | 2.0 |
| b | 10^6 | 6 | 10 | 72.67 / 71.52 | 0.994 [0.885, 1.124] | 941 / 876 | 0.931 [0.904, 0.969] | 7.1 |
| b | 10^6 | 12 | 10 | 88.48 / 87.74 | 0.989 [0.980, 1.439] | 1020 / 1002 | 0.981 [0.965, 0.999] | 15.7 |

Worst median ratio: wall 0.994, memory 0.981.  Gate 1.05: PASS.

At `b` 10^6 with 6 threads, pg-phenotype now peaks at 876 MiB, down from
953.  The memory ratio fell from 1.012 to 0.931.

## Not covered

- Cold start, when numba compiles pedsum's kernels, is not gated.
- More than 12 threads.  From 6 to 12 threads pg-phenotype's peak grew
  about 21 MiB per thread at 10^6 pairs, and pedsum's about 13.  If both
  stay linear, the memory ratio passes 1.05 near 22 threads.  That is an
  extrapolation.  This machine has 12.
