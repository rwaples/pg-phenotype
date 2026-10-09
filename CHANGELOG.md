# Changelog

## Unreleased

A `Pedigree` (R: `pedigree()`) validates a pedigree once for several
methods or calls: `mate_correlation` and `pafgrs.prepare` take one wherever
they take columns, and columns work as before.  A Pedigree keeps the Mating
Pairs and Mate Networks its first `mate_correlation` finds, so a later call
skips validation and finding them.  It keeps its input's row order: traits,
ages, strata and probands are matched to it by position, and `ped.ids`
(R: `ped$ids`) aligns values from another table.  A Prep shares the
Pedigree's ids and does not hold it.  A Pedigree cannot be pickled or saved.
With columns, the order of errors is unchanged: a column that cannot be
read (`missing_field`, `invalid_integer_value`), then a bad parameter, then
pedigree-graph's checks of the pedigree.

- Rust: `mate_correlation` and `prepare` take a `PedigreeArg`
  (`Columns(PedigreeInput)` or `Built(&Pedigree)`), and
  `PedigreeInput::validate` is gone; use `Pedigree::new`.
- Python: the private natives `_native.mate_correlation` and
  `_native.prepare` take one positional `pedigree`, a `_native.Pedigree` or
  the column tuple.

`mate_correlation` computes Spearman only when asked: pass `spearman=True`
(R: `spearman = TRUE`) to keep a continuous x continuous cell's `spearman`
record.  The result's `settings` echo `spearman`.  Every other value is
unchanged.

- `mate_correlation` takes about half the time and less memory: on a
  1M-row pedigree (401,523 Mating Pairs) with two continuous traits and
  `permutations=0`, the median call fell from 1.2 s to 0.6 s, and from
  16 s to 5.6 s at 5.4M rows, where the peak memory fell from +1.1 GB to
  +0.65 GB.  With `permutations=0` the per-cell pairs are no longer kept
  for the permutation test, the Mate Networks of a cell with no missing
  values are reused, and the father lookups use a row index.

- A stratified biserial or polyserial cell whose continuous side had more
  strata than its other side panicked (`index out of bounds` or
  `mid > len`) while computing its SE; it now gets one.  With two traits
  and strata, about 1 call in 6 at `min_stratum_networks=2` and 1 in 25 at
  the default hit it in the test fixtures.  v0.2.0 has the bug too.

- A public tetrachoric correlation of two binary variables, from a 2 x 2
  table or paired values: Rust `correlation::tetrachoric` and
  `tetrachoric_pairs`, Python `pg_phenotype.correlation.tetrachoric`, R
  `correlation_tetrachoric()`.  It is the Mate Correlation's fit and
  `boundary` flag, with the two-step sandwich SE (each pair its own
  cluster) and a Wald CI on the Fisher-z scale.  Pairs with a missing
  value are dropped and counted.  New error codes: `pair_length_mismatch`,
  and `invalid_table` from the hosts.  `docs/gates/tetrachoric` compares it
  with simACE's `tetrachoric_from_table` (ADR 0006).
- Tetrachoric and polychoric fits no longer stop far from the maximum after
  a Newton overshoot.  Newton could step to where a populated cell's
  probability is rounding noise or floored at 1e-300.  The score and
  Hessian there are huge or infinite, the next step is about 0, and Newton
  stopped as if converged, flagged as a boundary: `[[9891, 104], [1, 4]]`
  gave rho = 0.9915 where the maximum is 0.7969.  Newton now hands over to
  bounded Brent when a populated cell's probability falls below 1e-12.  On
  about 140,000 random tables with near-empty cells, none now misses its
  maximum, against about 1 in 1,000 before.  This departs from pedsum,
  which has the same flaw (ADR 0005).

## v0.2.0

Error details are stable and the same in Python and R: `reason` fields are
slugs, out-of-range degrees, counts and seeds raise the package's own errors,
and the thread budget's rules live in the Rust core.  Code that matched the
old English `reason` text or caught `OverflowError` needs updating.

- pedigree-graph-core is pinned at pedigree-graph's v0.12.3 tag.

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
  `too_many_rows` check.  A count in its result past R's integer range
  (`permutations = 3e9`) comes back as a double, not wrapped negative.
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
