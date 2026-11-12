# ADR 0015: PA-FGRS refuses a proband whose covariance is not positive semidefinite

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
people observed.  A twin's covariance is indefinite above `h2 = 64/121`,
and that of the twins' parents and sib above `h2 = 8/11`.  Before this
ADR:

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
- **A proband whose covariance is not positive semidefinite is not
  scored.**  Its `est` and `var` (bivariate: both traits' and `cov12`) are
  NaN, and the result's metadata counts such probands in
  `covariance_not_psd`.  `n_relatives` and `n_obs` are still the counts.
  The other probands of the call are scored as before.
- **The test is a Cholesky factorization of the covariance** before the
  conditioning, with `1e-9` of slack below zero, so a matrix singular to
  rounding (an MZ pair at `h2 = 1`) passes.  It reads the matrix alone, so
  ADR 0004's conditioning order does not decide who is refused.
- **Only a proband with an inbred member is tested**: the proband itself
  or an observed relative with `F > 0`.  The others' covariance is one by
  construction.  `prepare` keeps one flag per row: whether its parents may
  share an ancestor, read off the terminal sets it already builds for the
  kinship (ADR 0003).  Every inbred row is flagged.  A row whose parents'
  sets overflowed is flagged too, which costs a factorization and changes
  no result.
- **The conditioning is checked too**, whoever the members are.  It stops
  at an observation whose variance given the ones before it is below
  `-1e-9`; a posterior that is not finite, or has a variance below `-1e-9`,
  is refused; so is a bivariate posterior whose two estimates' block, with
  `1e-9` added to its diagonal, is not a covariance.  Conditioning a
  positive semidefinite matrix keeps it one, so each of these proves the
  matrix was not, and after the Cholesky test none should fire.  The clamp
  of a posterior variance at 0 only absorbs rounding.

A per-proband flag, not an error: one inbred twin pair in a register must
not cost every other proband its score, and the result already reports
what a score call left out as a metadata count (`controls_without_age`,
ADR 0002).

### Rejected: refuse only where the conditioning meets a negative variance

This keeps every score R gives with a positive variance, and was the
first version of the fix.  It refuses the probands for which R's `pa_fgrs`
returns NaN or a negative `postVar`, and no others.

An indefinite matrix whose conditioning stays positive then passes,
uncounted, and its score can be far from anything a posterior allows.  On
`tests/pedigrees.py::random_pedigree(0, generations=5)` with
`random_trait(n, 0)`, `ndegree = 2` and `h2 = 1`, 13 of 84 probands were
refused and 41 more had a covariance with an eigenvalue below `-1e-9`.
One of those scored `est = 24.03` against a prior variance of 1.  Which
indefinite matrices were caught also depended on the conditioning order.

## Consequences

- **pg-phenotype leaves BioPsyk/PAFGRS on probands whose covariance is
  indefinite.**  R returns a finite score with a positive variance for
  some of them: 4 of the 82 probands of the stored fixture
  `seed2_ndeg1_h2_0.75` (R variances 0.074 to 0.430), and the table's 0.75
  row.  They are now NaN and counted; `tests/test_r_golden.py` names the
  fixture's count.
- A score that is not refused is bit for bit what it was: the test reads a
  copy of the matrix.  `pafgrs_properties.rs` holds an outbred pedigree's
  scores from before.
- A proband refused now scored NaN, a variance of 0, or a finite value
  from an indefinite matrix before.
- A bivariate proband is refused whole, also when only one trait's block
  is indefinite and `rg = 0`.  A caller who wants the other trait's score
  scores it alone.
- A proband with an inbred member pays one factorization per score, about
  the cost of its conditioning.  A pedigree with no inbreeding pays one
  terminal-set comparison per row in `prepare` and one flag read per
  observed relative in a score.
- A caller must expect NaN in `est` and `var`.  `covariance_not_psd > 0`
  says why, and `covariance_not_psd == 0` says every covariance was
  positive semidefinite.
