# ADR 0014: The ACE Fit's SE reads the cells' covariance, and GLS weights are the default per solve where the result carries it

**Status:** accepted
**Date:** 2026-10-17
**Context:** issue #19; ADR 0012 (the ACE Fit, count weights), ADR 0013
(the per-estimate SE); `docs/gates/relative-pairs-ace-se/`; amends ADR
0012 and ADR 0013

## Context

ADR 0012's ACE Fit weighs each cell by what entered it and reports no SE.
It deferred inverse-variance weights until the cells had an SE (#19).
ADR 0013 gives every Relative-Pair Correlation estimate a cluster sandwich
SE, one estimate at a time.  The ACE Fit pools many cells, and those cells
share people and MRCAs across categories.  Its SE needs the covariance
between the cells' estimates, not only their variances.  ADR 0013 left
the Within-Person Cross-Trait Correlation without an SE, because a row
has no cluster design of its own.

Three constraints from ADR 0013 hold.  No result may depend on the thread
count.  Memory counts against `memory_budget_bytes`.  `se=False` and
`se=True` may not slow down by more than 5%.  ADR 0012 keeps `ace` a pure
function over the cells a host holds, so the covariance must travel in
the correlation result.

## Decision

### The covariance travels in the result, opt-in

- `correlation(..., se=True, covariance=True)` (R: `covariance = TRUE`)
  adds `covariance` to the result: per form the ACE Fit reads, one
  symmetric n x n matrix.  The forms are the pooled primary estimate, the
  pooled phi (binary traits only) and the stratified fit (with strata
  only).  Stratum cells and cross-form blocks are left out: `ace` never
  reads a stratum cell, and one fit reads one form.
- A cell's slot is its position: `cells` in order, then `within_person`
  last.  n is the number of cells, plus 1 with two traits.  This is the
  order the hosts already pass to `ace`, so no index is stored.
- The diagonal is each estimate's meat, written from the sweep, so
  `sqrt(diag)` equals `se` bit for bit.  The row and column of an estimate
  with no SE are NaN, its diagonal included.  The variance is defined
  exactly where the SE is.
- `covariance=True` without `se=True` raises at the host boundary.  The
  core cannot represent it: `correlation` takes `se: Se`, one of `Off`,
  `Estimates` and `Covariance`.  `settings` gains `covariance`.
- Collection.  `Store::sweep` hands each sum it forms to a `Sink` before
  squaring it: each (cluster, person) group's, each cluster's, and the
  per-person totals.  `se=True` passes `Discard`, whose empty methods
  inline away.  `covariance=True` passes an `Emission`, which keeps the
  covered estimates' sums in three key-sorted tallies.  The combine forms
  `Σ_jk = Σ_m Φ_jm Φ_km + Σ_i S_ji S_ki - Σ_(i,m) S_j(i,m) S_k(i,m)` as
  pairwise merge-joins of two sources' tallies.  One task owns each
  (grouping, source s <= source t) block, and the blocks are added in a
  fixed order.  Two entries count once when they share an MRCA or a
  member, whichever requests they are in; with j = k this is ADR 0013's
  meat.
- Determinism.  Every emitted value is a sum the sweep already formed in
  key order from exact integers.  Each block is summed in key order by one
  task.  The matrices are bit-identical at 1, 3 and 8 threads and with or
  without batched sweeps (`the_covariance_does_not_depend_on_threads_or_batching`
  in `relative_pairs.rs`;
  `test_the_covariance_does_not_depend_on_the_thread_count`).
- Memory.  Before the pass, the plan reserves the matrices and the
  combine's blocks (four matrices per form), the kernel and the
  within-person source.  After the pass, before any sweep, a bound on
  every request's tallies, from its store rows, is checked against what
  remains; over it, the call raises `memory_budget_exceeded` with
  operation `relative_pairs_covariance`.  The store and the influences
  are dropped before the combine.

Why opt-in: the covariance costs more than the 5% rule allows.  Measured
off/on/off on `se_cal_lam3_K10/rep1` (120k rows), 15 warm calls over 3
processes per case, at 1 and 8 threads, `covariance=True` against
`se=True`:

| case | relative cells alone | with the within-person kernel |
|---|---|---|
| one continuous trait | 1.03-1.06 | 1.04-1.05 |
| two binary traits, generation strata | 1.22-1.24 | 1.24-1.35 |
| two continuous traits, generation strata | 1.14-1.26 | 1.21-1.30 |
| two traits, strata, `categories=["MZ"]` | | 1.36-2.31 |

Peak RSS at one thread, measured before the kernel was added: 272 MiB with
`se=True` and 273 MiB with the covariance for one continuous trait; 339
and 442 MiB for two binary traits with strata; 338 and 405 MiB for two
continuous traits with strata.  The narrow call is cheap with `se=True`
alone (0.07-0.09 s) and pays for the kernel's degree-2 traversal; that
attribution is UNVERIFIED.  With `covariance=False` the relative cells'
SEs are bit-identical to the base build's, and `se=True` stays within the
5% gate: after/before 0.95-1.01 with the `Sink` seam and 1.00-1.02 with
the kernel, except one case at 1.12 that a rerun pinned to one core put
at 1.00.  The base build is 4a6bd19 by the plan; the bench files do not
record its revision (UNVERIFIED).  Only `ace` reads the covariance.

### The Within-Person SE is a kinship kernel, under the covariance only

- Each row is one entry, its own pair.  Its influence ψ_i is its stride on
  its own engine cell under the existing `Weights`.  The meat is
  `Σ_i ψ_i² + 2 Σ_(a<b) ψ_a ψ_b` over every relative pair of degree at
  most 2 whose members are both entries: MZ, MO, FO, FS, MHS, PHS, GP and
  Av.  It covers the pooled, phi, stratum and stratified estimates.
- Why not clustering by person or by mother.  Relatives' within-person
  entries are correlated (about ρ₁ρ₂), and a sibship cannot hold
  parent-offspring or cross-generation pairs.  On fixed simACE pedigrees
  (1000 liability draws each; `wp_proto` prototype, Pearson), SD(r̂)/SE
  and coverage were:
  - person alone: 1.18-1.42, coverage 0.84-0.90;
  - shared mother: 1.09-1.23, coverage 0.90-0.92;
  - kernel, degree <= 2: 1.00-1.07, coverage 0.94-0.95;
  - kernel, degree <= 4: 0.98-1.03, coverage 0.94-0.95.
  For binary traits person and mother differ by 7% or less.  Degree 2 was the
  user's choice (2026-10-17): it is inside the band, and every degree-2
  category is in the default traversal.
- The kernel does not depend on the request.  A degree-2 category the
  call did not ask for is traversed for the kernel alone: it adds its
  kernel term and leaves every cell's estimate unchanged
  (`test_a_narrow_call_traverses_the_kernel_and_leaves_its_cells`).
- Exact sums.  A binary ψ is a constant per engine label, so the table's
  counts give the requested categories' terms.  A continuous ψ is
  quantized to integers of at most 2^40 before the pass; each pair's
  product is summed exactly in i128 lanes, merged by addition, and becomes
  one float after the pass.  The sums are exact up to 2^46 terms.  Past
  that a sum may have wrapped, so the call raises `arithmetic_overflow`;
  the pass's sums wrap rather than panic, and the table's terms are counted
  before they are added.
- The link to the relative cells is K0: a within-person row and a
  relative pair are dependent when the row is one of the pair's members.
  The within-person source is a persons tally alone.  K1 (each person's
  degree-2 neighbourhood) was rejected: it counts a relative of both
  members twice and needs per-person accumulators in the pass.
- I + A (A the kernel's 0/1 adjacency) is not positive semidefinite: a
  grandmother with m grandchildren in different sibships gives a smallest
  eigenvalue of -√m.  A negative meat gives `negative_meat`, and the
  degenerate test uses the scale `(n + 2P) / n²` over n rows and P kernel
  pairs (`test_a_grandmother_star_gives_a_negative_kernel_meat`).  On the
  600 reps no within-person estimate lost its SE.
- Only under `covariance=True`.  Under `se=True` alone the within-person
  estimates keep `se_reason="within_person"`.  The kernel's per-pair work
  and a narrow call's degree-2 traversal would put `se=True` over the 5%
  rule.  This way `se=True` changes by the `Sink` seam alone.

### The ACE SE is `G Σ Gᵀ`

- With weights W fixed, each solve is linear in the cells it reads:
  `θ = G r` with `G = X̃⁺ T`, `T` the whitening (`TᵀT = W`).  The pivoted
  Householder QR takes T's columns as extra right-hand sides after y.
  Reflections run per right-hand side, so y's operations are today's and
  the values under `"pairs"` are bit-identical to v0.4.0's.
- A, C and the cross covariances take their rows of G.  E = 1 - A - C,
  rA, rC and rE take the delta method through the stacked rows, so the
  trait and cross solves' covariance enters with no special case.
  `Var(q) = g Σ gᵀ`.
- The sum runs over g's support: the entries above the solve's rounding
  bound, `m ε max|g|`, the bound the QR puts on its rank (`Linear::noise`
  in `ace.rs`).  A NaN row whose coefficient is a rounding zero, such as
  the within-person row against sA when sE is free, would otherwise take
  every SE with it.
- Weights are treated as fixed (first-order feasible GLS).  The calibration
  below is the check.
- Reasons.  Without the covariance of the form read every SE is
  `no_covariance`.  Under `"pairs"`, a value that reads a row with no SE is
  `row_without_se`.  `g Σ gᵀ < 0` is `negative_variance`.
- A row with no SE under `"pairs"` (user decision, 2026-10-17: kept).  The
  row still enters, since pairs weights need no variance, so its unknown
  variance is part of every value that reads it, and those SEs are
  `row_without_se`.  A tiny cell does this, and so does a tetrachoric at
  its boundary: the MZ cell in 219 of the 600 calibration reps.  The
  caller can leave the category out with `categories`, or use
  `"inverse_variance"` or `"gls"`, which leave the row out (`no_se`) and
  report the SE from the rows that remain.  Phi meets this most, since its
  default is `"pairs"` (next section).
- The rest of `AceFit` is as in ADR 0012.  Rows gain `variance`; each trait
  fit and the cross fit gain `weights`, the weighting their solve used;
  `settings` gains `weights`, the request.

### Weights, and the default per solve

- `weights=` is `"pairs"` (ADR 0012's count weights), `"inverse_variance"`
  (each row's `1 / Σ_kk`) or `"gls"` (the inverse of the entered rows'
  block of Σ, through its Cholesky factor).
- `"inverse_variance"` and `"gls"` need the covariance of the form read
  (`missing_covariance` otherwise).  They leave out a row with no SE (row
  reason `no_se`).  `weights="gls"` asked for refuses a block with no
  Cholesky factor: the solve's values and SEs are
  `covariance_not_positive_definite`, with no fallback to other weights.
- **The default (user decision, 2026-10-17, revised the same day after
  review).**  `weights=None` (R: `NULL`) is resolved in the core, where
  `AceRequest.weights` is `Option<Weights>`.  Without the covariance of the
  form read, and for phi, every solve is `"pairs"`; a result without
  `covariance=True` fits exactly as v0.4.0, with every SE `no_covariance`.
  Otherwise each fit, each trait's solve and the cross solve, is resolved
  on its own: `"gls"` when its entered rows' block of Σ has a Cholesky
  factor, and `"pairs"` when it has none.  The fallback fit is that fit
  under `"pairs"` exactly: a row with no SE enters it.  Each trait fit and
  the cross fit record the weighting they used in `weights`;
  `settings["weights"]` records the request, the name given or
  `"default"`.
- Why per fit.  The first default refused whole: on small pedigrees the
  block is often not positive definite.  On
  `tests.pedigrees.random_pedigree(3, n_founders=40)` to `1000` with two
  continuous traits, a trait's block of the pooled Σ had a smallest
  eigenvalue of -0.007 to -0.042, so A, sA and the rest lost the values
  v0.4.0 returned.  A fit with a positive definite block keeps GLS's
  efficiency whatever the other fits' blocks are.

Calibration on 600 simACE reps, six scenarios
(`docs/gates/relative-pairs-ace-se/README.md`, `summary.tsv`,
`calibration.tsv`).  Medians over scenarios, stratified forms:

- Continuous (Pearson).  `"gls"` is the most efficient: its SD(θ̂) is
  0.78-0.85 of `"pairs"`'s for A, E, sA, rA, sE and rE, down to 0.63, and
  0.90 for C.  `"inverse_variance"` is 0.88-0.98.  C, sC, sE and rE
  calibrate under every weighting (SD/SE 0.89-1.03, coverage 0.96-0.98).
  A, E, sA and rA do not (next section).
- Tetrachoric.  SD/SE is 0.93-1.05 under `"pairs"` and 0.95-1.17 under
  `"inverse_variance"` and `"gls"`, coverage 0.90-0.98.  `"gls"` gains
  0.90-0.97 in SD for A, sA and rA and about 1 for C and E, and its ratio
  runs higher than `"pairs"`'s (A1 1.16 against 1.02, E2 1.17 against
  1.05).
- Boundary MZ tetrachorics.  A boundary fit has no SE.  Under `"pairs"`
  every value that reads it loses its SE (`row_without_se`): A1 in 118 of
  600 reps, A2 in 98, sA in 203 and rA in 219; at K 0.02, 96 and 98 of
  100 reps.  `"inverse_variance"` and `"gls"` leave the cell out (`no_se`,
  421 rows per form in 219 reps), so every value keeps an SE, from the
  cells that remain.
- Phi.  `"inverse_variance"` and `"gls"` are unsafe at low prevalence.  At
  K 0.02 the ratio for A, C and E is 1.54-6.06, coverage falls to 0.01,
  and SD(θ̂) is 1.51-3.65 times `"pairs"`'s (rA 13.86 under `"gls"`).
  Under `"pairs"`, phi calibrates: 0.88-1.14, coverage 0.92-0.97.  The
  mechanism, a sparse cell's variance moving with its estimate, is
  UNVERIFIED.  This is why phi defaults to `"pairs"`.
- The machinery.  None of the 9000 fits gave
  `covariance_not_positive_definite` or `negative_variance`, and no cell
  gave `negative_meat`.
- Within-person kernel SE on the reps: 0.88-1.07, coverage 0.89-0.98,
  within 0.0014 of the prototype's `kin2` on the same reps.  The λ 0.5
  Pearson reps sit at 0.88-0.91, so the SE is conservative there; the
  cause is UNVERIFIED.

The gate's README recommended keeping `"pairs"` as the one default.  The
user chose the default by the result instead: GLS for the primary estimator
gains 15-22% in SD for continuous traits at equal or better calibration,
while phi, where GLS fails, stays on pairs, and a call without the
covariance is unchanged.  A solve whose block has no Cholesky factor takes
pairs, so the default always returns values.

### Continuous A, E, sA and rA spread more than their SE

On the continuous reps, A, E, sA and rA spread 7-21% more than their SE
under every weighting.  Median SD/SE is 1.08-1.21 under `"pairs"`,
1.09-1.20 under `"inverse_variance"` and 1.07-1.17 under `"gls"`, with
coverage 0.89-0.94.  Each cell's SE calibrates (ADR 0013).

A probe fitted each rep's realized-truth cells (A x A plus C x C terms,
E x E within person, as in the ACE truth gate) with the same `ace` call
and split the error (gate README, "Where the continuous excess comes
from").  Around each rep's own realized truth the SE is not too small,
except for E2: SD(θ̂ - θ_realized)/SE is 0.48-1.14 per scenario and
weighting, medians 0.63-1.00.  The realized truth itself varies across reps by 0.4-1.1 SE,
and for A1, E1, A2, sA and rA the two parts add to more than the SE.

This fits a rep-level common shock that moves every cell's realized
correlation together, such as the realized genetic variance of the
phenotyped generations, inherited from the same unphenotyped ancestors.
The cluster covariance treats pairs with no shared member or MRCA as
independent and has no term for it; no within-sample SE would.  The
mechanism is UNVERIFIED: the probe does not separate a population-level
shock from other cross-family dependence.

E2 is the exception: SD(θ̂ - θ_realized)/SE is 0.97-1.59 under every
weighting, realized coverage 0.77-0.96.  Its cause is UNVERIFIED.

The SE is shipped as calibrated against the realized truth, with this
caveat documented.  The follow-up issue tracks the across-rep gap.

Rejected:

- `ace` re-running the pass.  It would need the pedigree and the traits,
  and ADR 0012 keeps `ace` pure over the cells.
- Per-estimate covariance rows on each `Estimate`.  Every value would be
  stored twice and the slot order leaked into each estimate; one matrix
  per form holds the order once.
- One global re-merge of every request's rows in (cluster, person) order.
  It forms every row's stride and weights a second time, about another
  full set of sweeps, and its person term still needs dense per-person
  totals for every covered estimate.
- Clustering the within-person entries by mother.  Calibrated 1.09-1.23
  for Pearson on the fixed pedigrees.
- Covariance always on with `se=True`.  It costs up to 1.35 times
  `se=True`, 2.31 times on a narrow call, and the tallies' memory, for a
  matrix only `ace` reads.
- Inverse-variance weights from the SEs alone, without the covariance.
  They would be a second rule for what a weighting needs, and their SE
  would still need the covariance.
- A fallback from `weights="gls"` asked for when Σ's block is not positive
  definite.  It would change the weights the caller named.  The default
  falls back, and each fit records the weighting it used.
- Resolving the default in the hosts.  Python and R each repeated the
  rule, and a host cannot see a block's Cholesky factor before the solve.
- `"pairs"` as the one default.  It leaves the continuous GLS efficiency
  on the table for every caller who asked for the covariance.

## Consequences

- `ace` on a result with the covariance now fits GLS by default for the
  primary estimator, so its values differ from v0.4.0's.  Without the
  covariance the values are v0.4.0's bit for bit, and `weights="pairs"`
  gives them with the covariance too.
- Under the default, a boundary MZ tetrachoric leaves the fit (`no_se`)
  rather than taking the SEs of what reads it.  The tetrachoric's GLS
  ratio runs up to 1.17 (median, stratified E2) against 1.05 under
  `"pairs"`.
- Under the default, a fit whose Σ block is not positive definite fits by
  `"pairs"` and says so in its `weights`; `weights="gls"` asked for leaves
  that fit's values undefined.  None occurred on the 600 reps.
- The within-person SE needs `covariance=True`, so its value does not
  depend on which categories the call requested, at the price of a
  degree-2 traversal on narrow calls.
- Continuous A, E, sA and rA SEs are short of the across-rep spread by
  7-21% on simACE reps.  They describe the error around the sample's
  realized truth.  A caller who needs the across-population spread should
  widen them; the cause is open.
- In Rust, `correlation` takes `Se`, which `Se::from_flags` makes from a
  host's two flags; `ace::fit` takes `Option<&Covariance>`;
  `Covariance::new` returns `covariance_shape_mismatch` for a matrix of
  another length; `TraitFit` and `CrossFit` hold `Solved` (value and SE)
  and `weights`; `AceRequest` gains `weights: Option<Weights>`; `Row` gains
  `variance`; `Error` gains `missing_covariance` and
  `covariance_shape_mismatch`.
