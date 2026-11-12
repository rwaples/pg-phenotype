# ADR 0015: PA-FGRS refuses a proband whose conditioning meets a negative variance

**Status:** accepted
**Date:** 2026-11-12
**Context:** issue #32; ADR 0002 (the parameters), ADR 0003 (exact kinship),
ADR 0004 (the conditioning order)

## Context

A PA-FGRS score conditions a joint normal of the proband's genetic
liability and its relatives' liabilities.  Its covariance has `h2` on the
proband's diagonal, 1 on each relative's, and `2 phi h2` off the diagonal,
with `phi` the exact kinship (ADR 0003).  That is BioPsyk/PAFGRS's matrix:
`FGRS_wrapper` fills `2 * kinship * h2`, then sets `diag(covmat) <- 1` and
`covmat[1,1] <- h2` (`R/wrapper.R:106-114`).  The paper's matrix has the
same form, 1 on a relative's diagonal, `r h2` off it and `h2` for the
proband, and its text and supplement do not mention inbreeding (Dybdahl
Krebs et al. 2024, Material and methods; doi in the README).

The exact kinship carries inbreeding and the diagonal does not.  A person
with inbreeding coefficient `F` has additive variance `(1 + F) h2` (twice
their kinship with themself), so the matrix is the members' covariance
minus `h2 F` on each inbred member's diagonal.  With no inbred member it
is a covariance.  With one it can stop being positive semidefinite: an MZ
pair's kinship is `(1 + F) / 2`, so a twin's row holds `(1 + F) h2`
against a diagonal of `h2` (the proband) or 1 (a relative).

Issue #32's case is MZ twins with `F = 3/8`, both cases and the only
people observed.  Before this ADR:

| `h2` | A twin | The twins' parents and sib |
|---|---|---|
| 0.5 | finite | finite |
| 0.7 | `var` clamped from -0.070 to 0 | finite |
| 0.75 | `var` clamped from -0.134 to 0 | finite, from an indefinite matrix |
| 0.8, 1.0 | `var` clamped to 0 | `est` NaN |

R's `pa_fgrs` on the same matrices returns the same estimates, the
negative variances unclamped, and NaN for both `postM` and `postVar` where
pg-phenotype gave a NaN `est`.  In a bivariate score the NaN reached the
other trait even at `rg = 0`.

## Decision

- **The matrix stays R's.**  Putting each member's inbreeding on the
  diagonal would change every score in an inbred pedigree and leave
  BioPsyk/PAFGRS, the reference these scores are gated against.  That
  change is issue #36, not this ADR.
- **A proband whose conditioning meets a negative variance is not
  scored.**  Each conditioning step keeps a positive semidefinite
  covariance one, so a negative variance proves the matrix was not.  It
  shows in one of two ways: the posterior variance of a target is below
  `-1e-9` (below that is rounding, still clamped to 0), or an
  observation's variance given the ones before it is negative, which makes
  its truncated moments NaN and through them every estimate.
- **Such a proband's `est` and `var` (bivariate: both traits' and `cov12`)
  are NaN**, and the result's metadata counts these probands in
  `covariance_not_psd`.  `n_relatives` and `n_obs` are still the counts.
  The other probands of the call are scored as before.

A per-proband flag, not an error: one inbred twin pair in a register must
not cost every other proband its score, and the result already reports
what a score call left out as a metadata count (`controls_without_age`,
ADR 0002).

### Rejected: test the covariance itself

A Cholesky test of each proband's covariance refuses every proband whose
matrix is indefinite, whatever the conditioning order.  It was built and
dropped: on the stored BioPsyk/PAFGRS fixture `seed2_ndeg1_h2_0.75` it
refuses 4 of the 82 probands R scores, each with a finite estimate and a
positive variance (0.074 to 0.430).  Refusing them would leave the
reference where the reference gives an answer.  It also needs every row's
inbreeding at `prepare`, to keep the test off probands that cannot fail
it.

## Consequences

- A proband refused now scored NaN or a variance of 0 before.  Every other
  score is bit for bit what it was; `pafgrs_properties.rs` holds an
  outbred pedigree's scores from before.
- The refused probands are those for which R's `pa_fgrs` returns NaN or a
  negative `postVar`.  Where R returns a finite score with a positive
  variance from an indefinite matrix (the table's 0.75 row, the fixture's
  4 probands), so does pg-phenotype, uncounted.  Such a score is not a
  posterior of any model; issue #36 removes the cause.
- Which indefinite matrices are refused depends on the conditioning order,
  which ADR 0004 fixes, so the result is still a function of the inputs.
- A bivariate proband is refused whole, also when only one trait's block
  is indefinite and `rg = 0`.  A caller who wants the other trait's score
  scores it alone.
- A caller must expect NaN in `est` and `var`.  `covariance_not_psd > 0`
  says why.
