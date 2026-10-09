# Changelog

## Unreleased

- An ordinal trait with no declared levels and a code far above the number of
  rows (say `1e12`) raised `sparse_ordinal_codes` only after allocating one
  flag per possible level, which could abort the process; the check now
  needs at most one flag per present value.
- Python rejects pedigree ids and stratum labels outside `[-2^63, 2^63)` with
  `invalid_integer_value`, as R does, where it cast them to wrong int64
  values.  A non-numeric object entry raises `invalid_integer_value` too,
  not a bare `ValueError`.  An object column (a list holding `None`) takes
  its integers exactly: an id past 2^53 was rounded to a neighbour's id,
  and `2^63 - 1` was refused.  `None`, NaN and `pd.NA` are all missing.
- Rust: `BivParams`'s fields are private, so every value has passed
  `BivParams::new`; read them through `h2()`, `rg()` and `rho_within()`.
- Error `reason` fields are stable slugs instead of English prose:
  `invalid_cip` gives `empty`, `length_mismatch`, `age_not_finite`,
  `ages_not_increasing`, `cip_out_of_range`, `cip_decreases` or
  `prevalence_not_positive`, and `inconsistent_parameters` gives
  `non_genetic_covariance_not_psd`.  The messages are unchanged.
- PA-FGRS's `invalid_trait_value` (a binary status not 0 or 1) carries the
  `kind` field, as assortative mating's does.  Rust: `InvalidTraitValue` is
  merged into `InvalidTraitCode`, and error `kind`/`expected`/`actual` fields
  are `TraitKind`s.
- Python `pafgrs.prepare(ndegree=256)` (or any int outside 1..5) raises
  `degree_out_of_range`, not `OverflowError`; an int past int64 reports the
  int64 bound as its `value`.  R's `pafgrs_prepare()` does the same for a
  whole `ndegree` past 2^63, where it raised `parameter_out_of_range`.
  Rust: `prepare` takes `ndegree: i64` and does the one range check.
- R: a count or seed outside int64 is a `parameter_out_of_range` error, as
  in Python; whole numbers up to 2^63 are accepted (the limit was 2^53).  A
  bad stratum label's error carries its `value`, and
  `assortative_mate_correlation()` has `pafgrs_prepare()`'s
  `too_many_rows` check.
- The thread budget's rules live once, in the Rust core, for both hosts.
  Python now caps a budget at 2^31 - 1, as R does, and its
  `PG_PHENOTYPE_THREADS` error names that range.

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
