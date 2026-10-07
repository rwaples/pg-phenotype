# ADR 0004: The canonical PA-FGRS conditioning order

**Status:** accepted
**Date:** 2026-10-06
**Context:** simACE `plans/pg-phenotype-extraction-v3.md`, decision #16

## Context

Pearson-Aitken conditioning is order-dependent.  R PAFGRS sorts relatives
by `order(-w, -C[r, p], -rowSums(C[-1, ]))` (`est_liab.R:117`), relying on
R's stable `order` for ties.  fitACE's numba kernels used one scalar key,
`w*1e12 + 2 phi h2 * 1e6`, with an unstable argsort.  Nearly every proband
in simACE data has tied keys (siblings with equal `w` and kinship), so the
numba score depended on sort internals: on `test/small_test` the numba
score and the same computation with NumPy's argsort differ by up to 0.03,
and agree to 6e-16 where no key ties.

## Decision

Observations are sorted most informative first and conditioned from the
least informative end, by R's lexicographic key plus a unique final key.

- **Univariate**, relative `r` among the informative relatives: `w`
  descending, `C[r, p]` descending, `sum_j C[r, j]` descending (proband
  column and diagonal included, as R's `rowSums(covmat[-1, ])`), row
  ascending.
- **Bivariate**, observation `o = (r, t)`: `w` descending, `|C[o, p1]| +
  |C[o, p2]|` descending, `sum_j |C[o, j]|` descending, row ascending,
  trait ascending.

### Exact keys

Off the diagonal `C = 2 h2 phi` with `h2 > 0`, so the univariate keys are
compared as `phi(r, p)` and `phi(r, p) + sum_{j != r} phi(r, j)`.  Float32
kinship values are dyadic and their float64 sums are exact, so a tie in
exact arithmetic is a tie in the key and falls to the row.  The bivariate
row sum is `2 phi_rp (h2_t + |cov_g|) + 1 + |rho_within| [(r, t') observed]
+ 2 (h2_t A + |cov_g| B)`, with `A` and `B` the exact kinship sums over the
other people observed on `t` and on `t'`: one fixed expression over exact
parts.

## Consequences

- Scores do not depend on input order, sort stability or thread count.
- This is a deliberate change from numba's key.  The fitACE gate reports
  the order step separately and does not gate on it.
- Against R PAFGRS the order is the same whenever R's floating keys are
  exact; the golden fixtures use `h2` in {0.25, 0.5, 0.75} for that reason
  and agree to 1.1e-15.
