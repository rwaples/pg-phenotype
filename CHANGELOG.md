# Changelog

## Unreleased

- PA-FGRS scoring stores each proband's conditioning covariance as packed
  upper-triangle columns instead of a square matrix, which roughly halves
  that buffer, and scores faster.  Scores are unchanged bit for bit.  The
  base and new builds were timed in alternating processes in 48
  configurations: two pedigrees, degree 2 and 3, 1 and 6 threads, dense,
  asymmetric and sparse observation, univariate and bivariate.  On the
  799,992-row pedigree, a dense bivariate score at degree 3 fell from
  16.6 s to 14.9 s with 6 threads and from 67.5 s to 61.8 s with one.
  Median time changed by -15% to +3% for dense and asymmetric scores on
  that pedigree, -19% to -0.5% on the 112,358-row one, and -7% to +2% for
  sparse scores on both.  Peak process memory is unchanged within 0.4%.
  The cause of the gain on the small pedigree is not established.
- Every Python call that computes in the thread pool, and the `Pedigree`
  constructor, hands the process's free heap pages back to the system
  first (`malloc_trim`, glibc only): `pafgrs.prepare` and both score
  functions now do, as `Pedigree`, `mate_correlation`, `parent_offspring`
  and `relative_pairs.correlation` did.  The release moved from each
  wrapper into the native binding's one crossing module, which every
  method's call goes through.  Its memory effect is measured only for
  `mate_correlation` (v0.1.1).
- `mate_correlation`, `pafgrs.score_univariate` and
  `pafgrs.score_bivariate` raise `TypeError` naming the argument
  (`traits[1] must be a pg_phenotype.Trait, got ndarray`) for an argument
  that is not a `Trait`, where they raised `AttributeError`, as
  `parent_offspring` and `relative_pairs.correlation` did.
- PA-FGRS scores' `metadata` is a read-only mapping, as every other
  method's is, in R's order: the parameters, `prevalence`, `threshold`,
  `controls_without_age`, `n_probands`, `ndegree`, then the versions.  Code
  that needs updating: code that changed a score's `metadata` in place.
- The `pg_phenotype_version` in a result's metadata is the native
  binding's version, as in R; `tests/test_version.py` keeps it equal to the
  distribution's.
- Private `_native` signatures: methods take a `Trait` whole and strata as
  one `(labels, known)` pair, `score_bivariate` takes its traits, ages and
  CIP tables as pairs, and none takes `threads`; the binding reads the
  committed thread budget, as R's does.
- In R, `pedigree()`, `pafgrs_prepare()`, both score functions,
  `assortative_mate_correlation()`, `regression_parent_offspring()` and
  `relative_pairs_correlation()` hand free heap pages back to the system
  before they compute, through the same crossing module as Python; R
  released none.  Its memory effect in R is not measured.
- PA-FGRS score metadata comes with the scores, its keys written once in
  the core for both hosts: the parameters, `prevalence`, `threshold`,
  `controls_without_age`, `n_probands` and `ndegree`.  R no longer makes a
  second native call for `n_probands` and `ndegree`, and the private
  native score functions no longer return `threshold`,
  `controls_without_age`, `rho_within` or `prevalence` beside the
  metadata.  Values, types and order are unchanged, except that R's
  `controls_without_age` follows the rule for counts in R metadata: a
  double, not an overflowed integer, past R's integer range.
- The R binding's private native routines take the pedigree as one
  argument (a `pedigree()` pointer or the column list) and each `trait()`
  and `pafgrs_cip()` whole.
- The Polychoric Fit, the tetrachoric and polychoric fit the Mate
  Correlation and the public tetrachoric share, has its own module,
  `polychoric`, with pedsum's primitives (ADR 0005, 0011 amended).  Results
  are bit-identical.  Code that needs updating, in Rust only:
  `correlation::Tetrachoric` and `correlation::Estimate` carry
  `polychoric::Reason` (boundary, constant_margin, empty_category,
  no_complete_pairs, sandwich_undefined), and `relative_pairs::Reason::Fit`
  wraps it, in place of `assortative::Reason`; `assortative::Reason`
  converts from it.  `Ci`, `CiMethod` and `CiScale` live in `wald`, and
  `Point` in `polychoric`; `assortative` re-exports all four.  The reason
  names hosts see are unchanged.
- The tetrachoric ADR is renumbered 0006 -> 0011, because the Pedigree ADR
  also had 0006 (#11).  The Relative-Pair Correlation's `method` entry
  `estimators` now cites ADR 0011.  A test fails on a duplicate ADR number.
- In Rust, `Error`'s positional fields hold a `Position`, and
  `Error::fields` and the new `Error::message` take an `IndexBase` (`Zero`
  for Python and Rust, `One` for R); `Display` is `message(IndexBase::Zero)`.
  Each host now gets its message and fields from one rule, so a new
  positional pg-phenotype error cannot report one base in R's message and
  another in its fields.  A pedigree error's message is still
  pedigree-graph's, 0-based in R as well.  Python and R output is unchanged.
- Every method checks a trait in one order, its Admission: the length,
  the kind, each present value, then (the Mate Correlation alone) present,
  not constant, every level taken.  PA-FGRS scoring now follows it and
  checks the age after the trait, so a call with two faults reports a
  different one: a trait of the wrong length and kind raises
  `trait_length_mismatch` (was `trait_kind_mismatch`); a bad status value
  with an age of the wrong length, or with a bad age in any row, raises
  `invalid_trait_value` (was the age's `trait_length_mismatch` or
  `invalid_age` when the bad age's row came first).  A call with one
  fault raises what it did.  In Rust, PA-FGRS also bounds a binary
  trait's codes by `Trait::n_levels`, as the other methods do; a Python or
  R trait never carries a code beyond its levels, so their output is
  unchanged.

## v0.3.0

New methods: the Relative-Pair Correlation, parent-offspring regression, a
public tetrachoric correlation, and an Aalen-Johansen PA-FGRS CIP.  A
`Pedigree` validates a pedigree once for several calls and looks up rows by
id.  Code that needs updating: `mate_correlation` computes Spearman only
with `spearman=True`, and `permutations` defaults to `None`; a numeric
input given as text is a `TypeError`; in Rust, `mate_correlation` and
`prepare` take a `PedigreeArg`.

- pedigree-graph-core stays pinned at pedigree-graph's v0.12.3 tag.

Relative-Pair Correlation (#1, unit 1):
`pg_phenotype.relative_pairs.correlation` (R:
`relative_pairs_correlation()`, Rust: `relative_pairs::correlation`)
correlates one trait between the members of each Relationship Category's
Relative Pairs, or a named Category Group's (`{"PO": ["MO", "FO"]}`), one
cell each; the default is every category up to degree 3.  A continuous
trait gets Pearson's r, a binary trait the tetrachoric (primary, ADR
0011's fit, numbered 0006 in this release) and phi.  A symmetric
category's pairs enter in both orders, so its correlation is intraclass and
row-order free; a directed one's enter once, in its roles' order.  Each
cell reports `n`, `n_dropped`, and a
binary cell's Concordance (symmetric) or 2 x 2 table (directed).  A value
is `None` only when undefined, with a reason; there is no pair floor and no
SE yet ([ADR 0009](docs/adr/0009-relative-pair-correlation.md)).  Pairs
come from pedigree-graph-core's `relationship_moments` and are never
stored; `memory_budget_bytes=` (default 1 GiB) bounds its accumulators.  A
continuous trait is centred on its midrange before quantization, and
`insufficient_precision` is raised when the quantization step exceeds
2^-20 of its interquartile range.  New error codes: `unknown_category`,
`duplicate_category`, `mixed_category_group`, `duplicate_entry_name`,
`reserved_group_name` (a group named like a category code),
`empty_category_group`, `insufficient_precision`.  The gate against simACE
`6355ffa` passes ([report](docs/gates/relative-pairs/README.md)).

`Cip.from_aalen_johansen(age, status, entry=None)` (R:
`pafgrs_cip_aalen_johansen()`, Rust: `Cip::aalen_johansen`) estimates a
PA-FGRS CIP from data (#4): status 0 censored, 1 case, 2 death as a
competing risk, with optional late entry.  It returns the estimate at each
case age as a step table, and K is its last value.  An opt-in smooth CIP is
#12.

`Cip` (R: `pafgrs_cip()`) takes `interpolation`: `"linear"`, the default
and the behaviour so far, or `"step"`, which holds the value at the last
age at or below.  Scoring reads it from the table.  The native
`check_cip`, `score_univariate` and `score_bivariate` take the table as
one `(ages, cip, interpolation)` tuple.  A person is at risk at `t` when
`entry < t <= age`.  New validation codes: `invalid_status`,
`entry_not_before_age`, and `cip_not_estimable` (`reason`
`zero_prevalence` or `prevalence_one`)
([ADR 0010](docs/adr/0010-aalen-johansen-cip.md)).  The gate
(`docs/gates/aalen-johansen`) matches simACE's `_aalen_johansen` bit for
bit on continuous ages and R survival's `survfit` within 4.5e-14 on all
cases.  With late entry, cases over rows (fitACE's `compute_empirical_cip`)
gives a K about 30% lower.  The development `r` environment gains
`r-survival`.

`Pedigree.align(ids, values)` (R: `pedigree_align()`, Rust:
`Pedigree::align`) moves values given per id onto a Pedigree's rows, NaN
(R: `NA`) where no id names a row, and `Pedigree.rows(ids)` (R:
`pedigree_rows()`, 1-based) gives the row of each id (#9).  An id that is
not a row's is an `unknown_id` error and one listed twice a
`duplicate_lookup_id` error, each with the id and its positions; `align`
raises `pair_length_mismatch` when `values` is not `ids`'s length.  The
first call builds an id index the Pedigree keeps, 12 bytes per row
([ADR 0008](docs/adr/0008-a-pedigree-looks-up-rows-by-id.md)).
`pafgrs.prepare` on such a Pedigree reuses it for the probands; otherwise
it builds one for the call, as before.

- `pair_length_mismatch` names its two columns in a `columns` field and in
  its message (`x`, `y` from `tetrachoric`; `ids`, `values` from `align`).
- A numeric input (a trait, ages, strata, `align`'s values, `tetrachoric`'s
  `x` and `y`) given as text is a `TypeError`.  Before, numpy parsed
  `["1", "2"]` as numbers.  `pd.NA` in a list or object array is missing,
  where it raised `TypeError`.
- `regression.parent_offspring` returns free heap to the system before its
  native call, as `mate_correlation` and `Pedigree` do.
- Native calls read their inputs with the GIL released, so an input that
  another Python thread could write mid-call is now copied first: a
  writeable numpy array passed directly or in a mapping.  polars and
  copy-on-write pandas columns are read-only views and are still read in
  place.  A Pedigree built from a mapping of writeable int64 arrays holds a
  copy of each column (8 bytes per row) while it validates.
- `Trait.values` is read-only, and a Trait no longer shares a writeable
  array with its caller.
- `pafgrs.prepare` reads `probands` as it reads pedigree ids: a missing or
  non-integer id is an `invalid_integer_value` error, where numpy's cast
  used to truncate it.

Parent-offspring regression (#3): `pg_phenotype.regression.parent_offspring`
(R: `regression_parent_offspring()`, Rust: `regression::parent_offspring`)
fits the OLS line of each child's trait on its midparent's, its mother's and
its father's value, pooled and, with `stratum=`, within strata of the child.
Each fit reports the slope, intercept, r, r2, the plain OLS SE, and the Mate
Network sandwich SE with a Wald p-value and CI.  Every pedigree row is a
candidate child, and rows left out are counted by reason.  A trait that is
all missing or constant gives fits with a reason, not an error
([ADR 0007](docs/adr/0007-parent-offspring-regression.md)).  The gate
against simACE is in `docs/gates/po-regression`.

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

`mate_correlation(..., inference=False)` (R: `inference = FALSE`) returns
the cells' point estimates and the counts alone (#10).  It builds no Mate
Networks, SEs, CIs, permutation records or Within-Person Cross-Trait
Correlation: the network counts, `within_person` and `permutation` are
`None`, and `se` and `ci` are `None` with the reason `not_requested`.
Every cell estimate and count equals the default's; the default still
gives `within_person`.  It takes no draws, so a nonzero `permutations` or
`bootstrap` is a `parameter_out_of_range` error.  `permutations` now
defaults to `None`: 999 with inference, 0 without.  The result's
`settings` echo `inference`, and a `Pedigree` builds no Mate Networks for
such a call.  A stratified call still builds each cell's networks for the
thin-stratum rule, so it saves less than an unstratified one.

- Rust: `Settings::permutations` is an `Option<i64>`, `None` by default,
  resolved by `DEFAULT_PERMUTATIONS` (999) with inference and 0 without.

- On a 5M-row pedigree (2.0M Mating Pairs), a `Pedigree`, one two-trait
  call and two one-trait calls (all continuous and unstratified,
  `permutations=0`) peaked
  491 MB above the loaded columns before this change.  They now peak at
  397 MB with inference and 324 MB without.  The two-trait call without
  inference takes 0.7-1.0 s, the one-trait calls 0.15-0.26 s.  Every run
  gave the same values.
- With inference, a cell no longer copies its pairs to pool them when they
  already form one stratum (every unstratified cell).  The numbers are
  unchanged bit for bit.
- `Pedigree(...)` returns free heap memory to the system before it builds
  (glibc), as `mate_correlation` does.  On the same pedigree, a second
  Pedigree built after the first was freed peaked 62 MB lower.

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
  with simACE's `tetrachoric_from_table` (ADR 0011, numbered 0006 in this
  release).
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
- A tetrachoric or polychoric fit with an interior maximum is no longer
  flagged as a boundary because the NLL at the bound is rounding (#8).  The
  plateau test compares the NLL at the bound with the optimum's.  When a
  populated cell's probability there is below 1e-12 it is rounding noise,
  so the bound looked far better than it is: `[[14045855, 3530593],
  [1, 6012126]]` came out 18.8 nats above the optimum, within the 22-nat
  tolerance at n = 2.4e7, and its SE was withheld.  At the bound, a
  populated cell below 1e-12 is now recomputed by quadrature without
  cancellation, and the test runs on that NLL: the table is 431 nats from
  its bound.  This departs from pedsum (ADR 0005).

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
