# Changelog

## v0.5.0

- pedigree-graph-core is pinned at pedigree-graph's v0.13.0 tag, with
  the shared `KinshipRows` and family-ordered pair reduction (issue #24),
  and the Python dependency takes `pedigree-graph<0.14`.  PA-FGRS
  `prepare` builds one `KinshipRows` beside its `AncestorSignatures` and
  every worker's walker reads it.  Relative-Pair Correlations without the SE get the
  engine's sib reuse: on simACE's 2M-row `bench_scale` pedigree, median of
  5, one thread, 8.15 s to 7.24 s, and 20.4 s to 13.4 s with the rows
  shuffled.  Results are identical.

- PA-FGRS `prepare` is 11 to 18% faster at one thread (at six threads 12
  to 13%, except `ndegree=2` on simACE row order, unchanged) and needs 6 to
  13% less memory at six threads.  The kinship among each sibship's relatives reads a parent
  in the sibship's union straight from its row of the local matrix and
  each member's twin once, rather than looking both up in pedigree-wide
  columns for every pair; the triangle dictionary hashes with FxHash; and
  the candidate relatives come from the engine in family order, so each
  full sib reuses its predecessor's relationship sets.  A refused
  candidate buffer raises `allocation_failed` with operation
  `pafgrs_candidates`.  Values are bit-identical.  On the 2M-row pedigree,
  median of 5, one thread: `ndegree=2` 18.5 s to 16.5 s, `ndegree=3`
  64.1 s to 55.0 s, `ndegree=2` with the rows shuffled 29.1 s to 23.8 s;
  six threads, `ndegree=3` 20.4 s to 17.7 s, peak 3491 MiB to 3287 MiB.

- `relative_pairs.correlation(..., se=True)` walks the pedigree in family
  order, 21 to 28% faster when sibs are not stored together and 2 to 6%
  faster otherwise; results are identical.  On the 2M-row pedigree with
  the rows shuffled, median of 5, one thread: one trait 31.9 s to 23.1 s,
  two traits with generation strata and `covariance=True` 47.1 s to
  37.2 s.  The smaller per-lane cluster store leaves more of
  `memory_budget_bytes` to the sweeps, which then run more requests at
  once: that call at six threads, shuffled, budget 8 GiB, peaks at
  6134 MiB rather than 5547 MiB (four sweeps at once instead of three) and
  takes 13.7 s rather than 17.4 s, still within the budget.

- `relative_pairs.correlation(..., se=True)` (R: `se = TRUE`) gives every
  Relative-Pair Correlation estimate a cluster sandwich SE (issue #18;
  ADR 0013): pooled, stratum and stratified cells, Pearson, tetrachoric
  and phi, same-trait and cross-trait.  Two pairs are in one cluster when
  they share a member or their most recent common ancestor (MRCA).  Each
  `Estimate` gains `se` and `se_reason`; `se` is `None` with
  `no_estimate`, `boundary`, `sandwich_undefined`, `degenerate_meat` (the
  meat is zero to within rounding, as in a cell whose pairs share one
  MRCA), `negative_meat` or `within_person` (a within-person estimate
  under `se=True` alone; see the covariance below).  The SE comes from the same pass over the pedigree as
  the cells and does not depend on the thread count.  On a 120k-row
  pedigree it costs 1.5-1.6 times the call's time with one trait and
  2.4-3.7 times with two traits and generation strata at one thread
  (1.6-1.8 and 2.1-3.3 times at eight), and memory held to
  `memory_budget_bytes`, whose new error is `memory_budget_exceeded` with
  operation `relative_pairs_se` (or `allocation_failed`).  `se` defaults
  to `False`, and without it the estimates are unchanged; the result's
  `settings` gain `se`.  On 600 simACE reps the SE calibrates, the
  stratified fit included: median SD(r̂)/SE 0.94-1.09 and Wald coverage
  0.89-0.97 per scenario, category type, estimator and fit
  (`docs/gates/relative-pairs-se/`).

- `relative_pairs.correlation(..., se=True, covariance=True)` (R:
  `covariance = TRUE`) adds `covariance`, a `CellCovariance` (R: a list of
  matrices with dimnames) with the covariance between the estimates the
  ACE Fit reads (issue #19; ADR 0014).  One symmetric matrix per form:
  `pooled` (the primary estimate), `phi` (binary traits) and `stratified`
  (with `stratum`), over `cells` in order and then `within_person`.  Two
  entries count together when they share an MRCA or a member, whichever
  cells they are in.  The diagonal is each SE squared, bit for bit, and the
  row and column of an estimate with no SE are NaN.  The matrices do not
  depend on the thread count.  `covariance` defaults to `False`;
  `covariance=True` without `se=True` raises `ValueError`.  On a 120k-row
  pedigree it adds 3-6% to the `se=True` call with one trait, 20-35% with
  two traits and generation strata, and 1.4-2.3 times with two traits and
  `categories=["MZ"]`.  Its sums count against `memory_budget_bytes`
  (`memory_budget_exceeded` with operation `relative_pairs_covariance`).
  `settings` gains `covariance`, and `method` gains `covariance`.
- With two traits and `covariance=True`, every within-person estimate gets
  an SE: each row is an entry, and the meat is a kinship kernel, its rows'
  squared influences plus twice their products over every relative pair of
  degree at most 2 (MZ, MO, FO, FS, MHS, PHS, GP, Av).  The pass traverses
  those categories whether `categories` names them or not, so the SE does
  not depend on the request, and the requested cells do not change.  A
  negative kernel meat gives `negative_meat`.  On 600 simACE reps SD/SE is
  0.88-1.07.  Under `se=True` alone, `se_reason="within_person"` now means
  "the Within-Person SE comes with the covariance".
- `relative_pairs.ace` reports an SE for every value (`A_se`,
  `A_se_reason`, ..., `rA_se`, `rA_se_reason`) from the result's
  covariance: `sqrt(g Sigma g')` over each solve's linear map, with E and
  rA, rC, rE by the delta method, weights treated as fixed.  Without the
  covariance every SE is `None` with `no_covariance`.  SE reasons:
  `no_covariance`, `row_without_se`, `negative_variance`,
  `covariance_not_positive_definite`.  `AceRow` gains `variance`,
  `TraitShares` and `CrossShares` gain `weights` (the weighting each solve
  used), and `settings` gains `weights` (the request, `"default"` for
  `None`).  Every field added since v0.4.0, `RelativePairCorrelation`'s
  `covariance` included, comes last with a `None` default, so positional
  construction from v0.4.0 still works.
- `relative_pairs.ace` takes `weights=`: `"pairs"` (the count weights of
  v0.4.0), `"inverse_variance"` or `"gls"`.  The last two need the
  result's covariance of the estimate read (`missing_covariance`
  otherwise), and leave out a row with no SE (row reason `no_se`).
  `weights="gls"` asked for refuses a covariance block with no Cholesky
  factor: the solve's values and SEs are
  `covariance_not_positive_definite`.  The default, `weights=None` (R:
  `NULL`), is `"pairs"` for phi and without the covariance of the estimate
  read; otherwise each trait's solve and the cross solve take `"gls"` when
  their entered cells' block has a Cholesky factor and `"pairs"` when it
  has none, so the default always returns values.  A result without
  `covariance=True` fits as in v0.4.0, bit for bit.  Under `"pairs"` a cell
  with no SE (a boundary MZ tetrachoric) leaves every value that reads it
  `row_without_se`; leave the category out, or use `"inverse_variance"` or
  `"gls"`.  Code that needs updating: an `ace` call on a result computed
  with `covariance=True` now fits GLS where it can for the primary
  estimator; pass `weights="pairs"` for the v0.4.0 values.  On 600
  simACE reps GLS's spread is 0.78-0.85 of the count weights' for
  continuous A, E, sA, rA, sE and rE.  Continuous A, E, sA and rA spread 7-21% more than their SE
  under every weighting; against each rep's realized truth the SE holds
  (`docs/gates/relative-pairs-ace-se/`).
- In Rust, `relative_pairs::correlation` takes `se: Se` (`Off`,
  `Estimates`, `Covariance`) where #18 took `se: bool`, and
  `RelativePairCorrelation` gains `covariance: Option<Covariance>`.
  `ace::fit` takes `covariance: Option<&Covariance>`.  `TraitFit` and
  `CrossFit` hold `Result<Solved, Reason>`, a value and its SE, where they
  held `Result<f64, Reason>`.  `AceRequest` gains `weights:
  Option<Weights>`, `None` the default, which the core resolves per fit;
  `TraitFit` and `CrossFit` gain `weights: Weights`; `Row` gains
  `variance`; `Covariance::new` returns `Result` with
  `covariance_shape_mismatch`, and `Se::from_flags` maps a host's `se` and
  `covariance` flags; `Reason` gains
  `CovarianceNotPositiveDefinite`, `NoCovariance`, `RowWithoutSe` and
  `NegativeVariance`; `Error` gains `MissingCovariance` and
  `CovarianceShapeMismatch` (`covariance_shape_mismatch`).

- pedigree-graph-core is pinned at pedigree-graph's v0.12.4 tag, whose
  errors render their rows and positions in the host's index base.  A
  pedigree error's message in R now counts from 1, like its fields and like
  every other pg-phenotype error (it counted from 0 in v0.4.0 and before).
  Python output is unchanged.

- The glossary is `GLOSSARY.md`, renamed from `CONTEXT.md` (simACE #35).
  Update any link to the old file.

## v0.4.0

The Relative-Pair Correlation takes strata and two traits, and the new ACE
Fit (`relative_pairs.ace`) solves A, C, E and rA, rC, rE from its cells
(#1, #6, #17; ADR 0012).  `parent_offspring` takes `lines=` and
`inference=`, and the table fits no longer depend on the table's layout.
Code that needs updating: `relative_pairs.correlation` takes `traits=`
where it took `trait=`, and its `settings` reports
`quantization_exponents` and `quantization_steps`.

- `relative_pairs.correlation` takes `traits=`, where it took `trait=`: a
  Trait, or a sequence of one or two of one kind (issue #17).  R's
  `relative_pairs_correlation` takes `traits`, a `trait()` or a list of
  them.  Two traits give each entry a cell per trait combination, read as
  trait `first_trait` in a pair's first member and `second_trait` in the
  second: a symmetric entry's (0, 0), (1, 1) and (0, 1), a directed
  entry's (0, 0), (0, 1), (1, 0) and (1, 1).  The symmetric cross-trait
  cell enters each pair in both orientations, with a threshold, or a mean
  and SD, per trait.  `n` counts the pairs with at least one complete
  orientation, and a symmetric cross-trait cell adds `n_entries`, its
  complete orientations.  A binary cross-trait cell reports its 2 x 2
  `table`.  The result's `within_person` is the Within-Person Cross-Trait
  Correlation, each row's trait 0 against its trait 1.
  `res.cell(code)` is the (0, 0) cell, and `first_trait=` and
  `second_trait=` pick another.  None of the pairs are stored, and one
  pedigree traversal gives every trait combination.  New error causes:
  `trait_count` for no trait or more than two, and
  `unsupported_trait_kind` for two traits of different kinds.  In R, cells
  count traits from 1.  One-trait results are unchanged apart from the
  new fields (ADR 0012).  Code that needs updating: a call that passes
  `trait=` (R: `trait =`) by name, and code that builds a `Cell` or a
  `RelativePairCorrelation` by position.  In Rust,
  `relative_pairs::correlation` takes a slice of traits and an
  `Option<Strata>`, and `RelativePairCorrelation` gains `within_person`.
- `relative_pairs.correlation`'s `settings` reports
  `quantization_exponents` and `quantization_steps`, one per continuous
  trait, where it reported `quantization_exponent` and
  `quantization_step`.  Both are `None` for binary traits, as before.
  Code that needs updating: code that reads either old key.  In Rust,
  `RelativePairCorrelation::quantization_exponent` is
  `quantization_exponents`, an `Option<Vec<i64>>`.
- `relative_pairs.correlation` takes `stratum=` (R: `stratum`), one
  integer label per row, NA where unknown (issue #17).  A directed pair is
  in its first role's stratum, and a symmetric pair in a stratum only when
  both members are.  Each cell then gets `strata`, a `StratumCell` per
  known label with the pooled fields, and `stratified`, the primary
  estimator fitted once over every pair of two known strata with a
  threshold, or a mean and SD, per stratum and side, and one correlation.
  New counts: `n_cross_stratum` (symmetric pairs in two known strata),
  `n_unknown_stratum`, and the stratified fit's `n`, `n_entries` (a
  symmetric cross-trait cell's complete orientations fitted) and
  `n_degenerate_stratum` (pairs in a stratum left out for a constant
  side).  A cell's `n` is its strata's `n` plus `n_cross_stratum` plus
  `n_unknown_stratum`.  Without `stratum` the pooled fields are as
  before.  `method` gains `traits`, `within_person` and `strata`.  Code
  that needs updating: R's `relative_pairs_correlation` takes `stratum`
  before `memory_budget_bytes`, so a budget passed by position must be
  passed by name.
- The ACE Fit: `relative_pairs.ace(result, categories=None,
  components=("A", "C"), shared_environment=None, estimator="primary",
  estimate=None)` (R: `relative_pairs_ace`, Rust: `ace::fit`) solves each
  trait's A, C and E shares from a Relative-Pair Correlation result by
  weighted least squares, `r = 2 phi A + s C` per cell, and with two
  traits the cross-trait correlations rA, rC and rE (issues #6 and #17).
  `phi` is the nominal kinship `2^-(degree+1)`, `s` is 1 for a category
  in `shared_environment`, and a row's weight is the pairs behind the
  value it reads.  A cross-trait row weighs complete orientations, each
  once: a symmetric cell's `n_entries`, a directed cell's `n`.
  `shared_environment` is required to fit C; it has no default.  Values
  are unconstrained: `rA`, `rC` and `rE` are `None` with
  `nonpositive_component` when a share is at most 0, an `|r|` above 1 is
  flagged `out_of_range`, and an underdetermined fit gives
  `not_identified`.  #6's closed forms are calls: Falconer is
  `ace(res, categories=["MZ", "FS"], shared_environment=["MZ", "FS"])`.
  The tetrachoric is on the liability scale; `estimator="phi",
  estimate="pooled"` gives the observed scale.  No SE.  New error codes:
  `unknown_entry`, `mixed_kinship`, `mixed_shared_environment`,
  `missing_shared_environment`, `no_stratified_form` and
  `missing_estimator`; `unknown_category` also covers a
  `shared_environment` code.  ADR 0012 records the design, and
  [docs/relative-pairs.md](docs/relative-pairs.md) is the reference.
- `regression.parent_offspring` takes `lines=` (any of `"midparent"`,
  `"mother"`, `"father"`; default all three) and `inference=` (default
  `True`), and R's `regression_parent_offspring` takes `lines` and
  `inference` (issue #15).  A line not requested is `None` (`NULL`) in
  every subset.  Without inference no Mate Networks are built:
  `n_mate_networks` is `None`, and `se`, `p` and `ci` are `None` with
  `not_requested`.  Slopes, intercepts, r, `se_ols`, counts and drops are
  the full call's bit for bit.  `settings` now also holds `lines` and
  `inference`.  On a synthetic 800,000-row, four-generation pedigree with
  a built `Pedigree`, through Python (median of 5 alternating processes,
  9 calls each), the midparent line without inference took 0.017 s pooled
  and 0.055 s by generation, where the full call took 0.097 s and 0.181 s
  before this change.  Each subset's arrays are now sized before they are
  filled, so the full call itself fell to 0.081 s and 0.170 s.
  Code that needs updating: code that reads a `Subset` field as always a
  `Fit`, or `n_mate_networks` as always an int, when it passes these
  arguments.  Rust callers of `regression::parent_offspring` pass a
  `Settings` (`Settings::default()` for the old behaviour), and
  `Subset::fit` returns an `Option`.
- The table fits, tetrachoric and polychoric (`correlation.tetrachoric`,
  the Mate Correlation's binary and ordinal cells, and the Relative-Pair
  Correlation's tetrachoric), no longer depend on the table's layout
  (issue #16).  Swapping the two variables, or recoding both as `1 - x`,
  leaves rho unchanged, and recoding one negates it, at a boundary too.  Every table
  is fitted in the greatest of its eight layouts, and rho's sign is
  restored.  A table already in that layout is fitted exactly as before.
  On the 2457 tetrachoric gate tables, interior fits moved by at most
  1.6e-11 and boundary fits by up to 0.036 (400 of 1210 by more than
  1e-6).  No boundary flag changed, and no pedsum golden moved beyond its
  tolerance.  ADR 0005 lists this as a fourth departure from pedsum.
  A table equal to a sign-reversing layout of itself (equal rows or equal
  columns) gives 0 without a fit.  The biserial and polyserial fits are
  not covered: at a boundary, recoding their discrete side can still move
  rho by more than its sign.
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
