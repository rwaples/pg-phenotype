# Changelog

## v0.1.1

Assortative mating's permutation test runs faster on one thread, and
`mate_correlation` lowers a host's peak memory.  Results are unchanged, bit
for bit.

- The permutation pass stores each father's traits interleaved, so a draw
  gathers every trait of a donor in one pass, and walks each cell's fathers
  as one array of records.  On pedsum's benchmark pedigree (two traits,
  birth-year strata, 10^6 Mating Pairs, 999 permutations, 1 thread) pedsum's
  CLI ran 0.84 of pedsum #13's wall time, where 0.1.0 ran 1.09.
- `mate_correlation` returns the process's free heap pages to the system
  (`malloc_trim`, glibc only) before it computes.  The computation runs on
  pool threads, and glibc keeps memory freed on the caller's thread in that
  thread's arena, where the pool cannot reuse it.  In pedsum's CLI at 10^5
  pairs and 6 threads the peak fell from 1.05 to 0.98 of pedsum #13's.

[The benchmark report](https://github.com/rwaples/pg-phenotype/blob/v0.1.1/docs/gates/assortative-mating/benchmark.md#pg-phenotype-011-in-pedsums-cli)
has every configuration.

## v0.1.0

First release.  The shared `Trait` input and PA-FGRS (`pafgrs.prepare`,
`pafgrs.score_univariate`, `pafgrs.score_bivariate`) for Python and R, on
pg-phenotype-core linked to pedigree-graph-core v0.12.2 (`62c82fa`).

Assortative mating (`pg_phenotype.assortative.mate_correlation`, R
`assortative_mate_correlation()`): the Mate Correlation of one or two traits
with sandwich SEs, the one-step Mate Network bootstrap and the sequential
father-permutation test, ported from pedsum (rwaples/pedsum#13, `142adf3`).
The Rust `Trait` carries `n_levels`, the declared level count.

Wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64) and Windows
(x86_64), abi3 for Python 3.13 and later.  The R source tarball is attached
to the GitHub release.
