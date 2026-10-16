# ADR 0012: Relative-Pair cells take strata and two traits, and the ACE Fit solves A, C and E from them

**Status:** accepted
**Date:** 2026-10-15
**Context:** issues #17, #6 and #1; simACE
`plans/pg-phenotype-relative-pairs-v2.md` (D1-D11); amends ADR 0009

## Context

ADR 0009's first unit correlates one trait per Relationship Category,
pooled over every pair.  fitACE needs more.  `estimate_rg`
(`fitACE_pafgrs/fitace_pafgrs/pafgrs_bivariate.py`) computes
`rA = 2 (rho_within - rho_cross) / sqrt(h2_1 h2_2)` from tetrachorics
pooled over everyone, with full-sib Falconer h2
(`fitace/ltm/falconer.py`).  Issue #17 quotes the fitACE#5 investigation,
not rerun here, at a true rA of 0.5:

- Pooling generations gave 0.308 at C = 0 and rE = 0.  simACE's default
  `gen_censoring` gives each generation its own follow-up, so prevalence
  differs by generation, and full sibs share one.  Per-generation
  correlations pooled by pair count gave 0.517.
- rE = 0.2 gave 0.950: within-person and full-sib correlations cannot
  separate rA from rE.
- C = 0.2 with rC = 0.7 gave 0.392: full-sib Falconer estimates A + 2C.

One fix for all three needs per-generation cells, cross-trait cells, and
more categories than within-person and full sibs.  Issue #6 asks for
simACE's closed-form h2 estimators (`compute_observed_h2_estimators`: sibs
`2 r_FS`, Falconer `2 (r_MZ - r_FS)`, hs `4 mean(r_MHS, r_PHS)`, cousins
`8 r_1C`) and fitACE's `r_tet / (2k)` per category.  Each one solves
`r = 2 kinship A + C_shared` over a chosen set of categories.

ADR 0002 makes PA-FGRS's `h2`, `rg` and `rho_within` inputs, and leaves
estimating them to the caller.

## Decision

Strata (amends ADR 0009):

- `stratum=` is one integer label per row, NA where unknown, read through
  the same input as `mate_correlation` and `parent_offspring`
  (`stratum_length_mismatch`).
- A directed pair is in its first role's stratum (the offspring's, the
  descendant's), whatever the second member's, as ADR 0007 reads a child.
  A symmetric pair is in stratum s only when both members are in s.  A
  complete symmetric pair whose members are in two known strata counts in
  `n_cross_stratum`.  A complete pair in no stratum cell because a stratum
  is unknown (a directed pair's first member's, either member's of a
  symmetric pair) counts in `n_unknown_stratum`.  A cell's `n` is its
  strata's `n` plus those two counts.
- Strata nest in the Cell, as in `parent_offspring`.  A Cell keeps its
  pooled fields and adds `strata` (each known label of any row, ascending,
  to a StratumCell with the pooled fields; `n = 0` when it has no pairs),
  `n_cross_stratum`, `n_unknown_stratum` and `stratified`.  Without
  `stratum` these are `None`, 0, 0 and `None`.
- `stratified` is `mate_correlation`'s shared-rho form for the primary
  estimator.  One fit covers every block of two known strata, rows by the
  first member's stratum and columns by the second's, both orders for a
  symmetric cell.  A binary side gets a threshold per stratum and side, a
  continuous side a mean and 1/N SD, and one correlation is fitted.  A
  cross-stratum symmetric pair enters, each member at its own stratum's
  threshold or mean.  The binary fit is the Polychoric Fit on a `Table`
  with the strata as its row and column strata.  Phi has no stratified
  form.
- `Stratified.n` counts the pairs fitted: both strata known, and outside
  every stratum whose side is constant over the blocks kept.  Such a
  stratum carries no information on the correlation, so its blocks are left
  out, and the rule repeats until no side is constant.  Leaving blocks out
  never makes a constant side vary, so the result does not depend on the
  order strata are found in.  `n_degenerate_stratum` counts the pairs left
  out.  There is still no minimum pair count.

Two traits (amends ADR 0009):

- `traits=` takes a Trait or a sequence of one or two of one kind; the
  keyword was `trait=`.  Two traits of different kinds raise
  `unsupported_trait_kind` until a biserial cell exists.
- Each cell reads trait `first_trait` in a pair's first member and
  `second_trait` in the second.  A symmetric entry gives the cells (0, 0),
  (1, 1) and (0, 1).  The cross cell (0, 1) enters each pair in both
  orientations, trait 0 of one member against trait 1 of the other, with a
  threshold, or a mean and SD, per trait.  A directed entry gives (0, 0),
  (0, 1), (1, 0) and (1, 1), its roles read apart.  `cell(code)` is the
  (0, 0) cell.
- A binary cross-trait cell reports its 2 x 2 table, trait `first_trait`
  of one member by trait `second_trait` of the other, a symmetric cell's
  two orientations summed.  Only a symmetric same-trait cell reports a
  Concordance.
- `n` counts the unordered pairs with at least one complete orientation:
  trait `first_trait` of one member and `second_trait` of the other both
  observed.  For a same-trait cell this is ADR 0009's `n`.  A symmetric
  cross-trait cell adds `n_entries`, its complete orientations, between
  `n` and `2n`; its estimates read those entries.
- Each member's engine label is its stratum code and its joint (trait 0,
  trait 1) label.  One `relationship_moments` traversal then gives every
  trait combination's pooled cell, stratum cells and stratified fit, and
  counts `n` and `n_entries` exactly.
- `within_person` is the Within-Person Cross-Trait Correlation: each row's
  trait 0 against its trait 1, computed in the core from the traits, with
  the row's stratum read for both.  It has strata and a stratified fit as
  any cell has.  No Category Group may be named `within_person`
  (`reserved_group_name`), so the name means this cell alone.  Its
  accumulator spans every pair of strata though only a row's own stratum
  fills, so it is held to `memory_budget_bytes` as one engine lane is.
- Each continuous trait is centred and quantized on its own.  `settings`
  reports `quantization_exponents` and `quantization_steps`, one per
  trait, where it reported `quantization_exponent` and
  `quantization_step`.

The ACE Fit:

- `ace::fit` (Python `pg_phenotype.relative_pairs.ace`, R
  `relative_pairs_ace`) is a pure function over the cells of a
  Relative-Pair Correlation result.  It does no pedigree work.
- On the correlation scale (each trait's variance 1), a same-trait cell k
  of trait t expects `r_k = 2 phi_k A_t + s_k C_t`.  `phi_k` is
  `2^-(degree+1)`, which equals pedigree-graph's `nominal_kinship` for all
  23 categories.  `s_k` is 1 for a Shared-Environment Category and 0
  otherwise.  `E_t = 1 - A_t - C_t`.  A Category Group's categories must
  share a degree (`mixed_kinship`) and all be shared or none
  (`mixed_shared_environment`).
- With two traits, a cross solve over the cross-trait cells,
  `r12_k = 2 phi_k sA + s_k sC`, and the within-person cell,
  `r_wp = sA + sC + sE`, gives the cross-trait covariances.  Then
  `rA = sA / sqrt(A_1 A_2)`, `rC = sC / sqrt(C_1 C_2)` and
  `rE = sE / sqrt(E_1 E_2)`.  The within-person cell enters the cross
  solve whatever `categories=` names.
- `components=` is A, C or both; a component not requested is fixed at 0
  and reported `not_requested`.  `shared_environment=` lists the
  Shared-Environment Categories and is required when C is fitted
  (`missing_shared_environment`).  It has no default.
- `estimator=` is `"primary"` or `"phi"`.  `estimate=` is `"stratified"`
  or `"pooled"`, and defaults to stratified when any cell has a stratified
  form.  The binary primary, the tetrachoric, is on the liability scale,
  so there is no Dempster-Lerner step.  `estimator="phi"` with
  `estimate="pooled"` gives the observed scale, which simACE's
  observed-scale report reads.  Phi with the stratified form raises
  `no_stratified_form`.
- Each row's weight counts what is behind the estimate it reads, the
  pooled cell or `Stratified`.  A same-trait row weighs its unordered
  pairs, `n`, and the within-person row its rows, `n`.  A cross-trait row
  weighs complete orientations, each once: a symmetric cell's `n_entries`
  (or `Stratified.n_entries`), a directed cell's `n`, each of its (0, 1)
  and (1, 0) cells holding one orientation.  The solve is weighted least
  squares by Householder QR with column pivoting on the
  `sqrt(weight)`-scaled design.
- Values are unconstrained.  `rA`, `rC` or `rE` is `None` with
  `nonpositive_component` when a trait's component is at most 0.  An `|r|`
  above 1 is reported and flagged `out_of_range`.  A component zero in
  every row that entered is `not_identified` and left out, so the others
  are still solved: C with no Shared-Environment row leaves A to the
  same-trait rows.  In the cross solve, sC zero in every relative row
  leaves sC and sE to the within-person row alone; both are
  `not_identified`, that row is left out, and sA is solved from the
  relative rows.  Any other design with fewer independent rows than
  unknowns gives `not_identified` for that solve.  A cell whose
  value is undefined does not enter, and its row names the reason.
- #6's closed forms are calls that identify the model exactly: sibs
  (`categories=["FS"], components="A"`), Falconer
  (`categories=["MZ", "FS"], shared_environment=["MZ", "FS"]`),
  half-sibs (`["PHS"]`, A) and cousins (`["1C"]`, A).  The
  parent-offspring estimate stays `parent_offspring`'s slope.
- No SE, as ADR 0009.
- PA-FGRS's parameters stay inputs (ADR 0002).  The ACE Fit is a separate
  estimator.  A caller may pass its `A` and `rA` to PA-FGRS as `h2` and
  `rg`; nothing in PA-FGRS calls it.

Validation, in `docs/gates/relative-pairs-ace/README.md`:

- Truth recovery is gated only where the binary outcome is an exact
  threshold of the ACE liability: continuous `liability1` and
  `liability2` with Pearson, and simACE `simple_ltm` status before
  censoring (`t1 < 1e6`).
- Three cells, 12 reps each: C = 0 and rE = 0; C = 0 and rE = 0.2; C = 0.2
  and rC = 0.7.  A target passes only when every rep gives a defined value
  and the mean is within 2 Monte Carlo SE of the truth.  Failed reps are
  listed, never averaged away.  In the C = 0 cells rC is not tested, and
  the share of reps where it is undefined is reported.
- A stress test runs production frailty with `gen_censoring`.  The
  stratified rA must improve on `estimate_rg`'s 0.308.  A and C
  attenuation are reported, not gated.

Rejected:

- Pooling generations, ADR 0009's pooled cell alone.  Under
  `gen_censoring` it gives the 0.308 above.
- Averaging per-stratum correlations by pair count.  A stratum whose
  estimate is undefined (a sparse PHS stratum) would need its own rule, and
  cross-stratum symmetric pairs would be lost.  The shared-rho fit has
  neither problem and already exists in `mate_correlation`.
- Stratum cells as extra entries of `cells`.  Existing callers would see
  `cells` change; nested cells leave a call without `stratum` as it was.
- A Concordance for a symmetric cross-trait cell.  Its table is not
  mirrored: trait 0 affected with trait 1 unaffected is a different count
  from the reverse.
- A traversal per trait combination.  It walks the pedigree up to four
  times.  A symmetric pair's two entries would be tallied apart, so the
  pairs with at least one complete orientation (`n`) could not be counted
  exactly.
- Mixed trait kinds now.  They need biserial cells; polychoric and
  biserial cells are #20.
- The ACE Fit as a field of the result (#6's first question).  It needs
  the caller's components, Shared-Environment Categories and categories,
  which the correlation does not; as a function it refits without
  recomputing pairs.
- One function per closed form.  Each is an exactly identified case of the
  one weighted least-squares fit, which also covers #17's joint fit.
- Dempster-Lerner conversion.  The tetrachoric already estimates the
  liability correlation, and phi stays available for the observed scale.
- A default `shared_environment`.  Which categories share C depends on
  how the data's households form: under simACE's household by mother,
  MZ, FS and MHS share it and PHS does not, and other data need not match.
  A wrong default would move A without an error.
- A minimum pair count.  ADR 0009's rule stands: an undefined cell leaves
  the fit with its reason, and a sparse defined cell enters with its small
  weight.  On #17, PHS cells at 20000 people per generation were too
  sparse: 4 of 12 reps gave NaN.
- Inverse-variance weights.  They need an SE, which waits on ADR 0009's
  calibration study (#19).  Counts are the weights until then.
- Pair-count weights in the cross solve.  A directed pair enters two
  cross cells, (0, 1) and (1, 0), each weighted by its pairs, while a
  symmetric pair enters one double-entered cell weighted once, though its
  two orientations are distinct observations.  That counted directed pairs
  double relative to symmetric ones.
- Gating truth recovery on censored status.  Status after censoring
  loses cases at random, which attenuates the tetrachoric.

## Consequences

- A call without `stratum` and with one trait returns ADR 0009's cells
  with the new fields at their empty values.  Code that passed `trait=` by
  keyword, or read `quantization_exponent` or `quantization_step`, needs
  updating, so the next release is 0.4.0.
- Every trait combination's cells and stratified fits come from the one
  traversal, and the within-person cell needs none, so a two-trait
  stratified call costs one pedigree walk.
- `ace` with `estimate="stratified"` leaves out a cell without a
  stratified form, with the reason `no_stratified_form`, and raises when
  no cell has one.
- The half-sib recipe over both MHS and PHS cells gives 4 times their
  pair-weighted mean, where simACE's `hs` takes the plain mean.  Under
  household by mother MHS also carries C.
- With count weights a large cell dominates the fit.  The fit
  reports no residual and no goodness of fit.
- fitACE can replace `estimate_rg` and its Falconer `h2_source=estimated`
  with `ace` on stratified cells.  That switch is a fitACE follow-up.
- The SE unit is #19; polychoric and biserial cells are #20.
