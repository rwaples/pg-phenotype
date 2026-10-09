# ADR 0007: Parent-offspring regression clusters by whole-pedigree Mate Network

**Status:** accepted
**Date:** 2026-10-10
**Context:** issue #3; simACE `plans/pg-phenotype-po-regression.md`

## Context

simACE regresses a child's trait on its midparent's twice:
`simace/analysis/stats/correlations.py` (`_po_regression`, liability by
generation and by sex x generation, affected status pooled) and
`simace/analysis/validate/heritability.py` (`_midparent_regression`).  Both
report the OLS SE.  Siblings share a midparent, and a parent who remates
links two sibships, so the OLS SE is too small.  pg-phenotype's Pedigree
already holds the Mating Pairs and Mate Networks that `mate_correlation`
clusters by.

## Decision

- `regression::parent_offspring` (Python
  `pg_phenotype.regression.parent_offspring`, R
  `regression_parent_offspring`) fits three OLS lines of the child's trait:
  on the midparent `(mother + father) / 2`, on the mother's value, and on the
  father's.  A single-parent line needs only that parent as a pedigree row.
  Slopes are raw; the library does not convert them to h2.
- One trait per call, continuous, binary or ordinal; codes are regressed as
  numbers.  Categorical raises.  An all-missing or constant trait gives
  fits with a reason, not an error.
- `stratum=` is one label per row, read at the child.  Each known label
  gets its own three fits; a child of unknown stratum is in the pooled
  fits only.
- Every row is a candidate child.  A row not analysed is counted once,
  under the first of `parent_not_in_pedigree`, `child_value_missing`,
  `parent_value_missing`, so `n + sum(dropped)` is the subset's row count.
- The slope reports two SEs.  `se_ols` is `sqrt(SSres / (n - 2) / Sxx)`,
  simACE's.  `se` is the cluster-robust sandwich
  `sqrt(G/(G-1) sum_g (sum_{i in g} (x_i - xbar) e_i / Sxx)^2)`, with the
  G/(G-1) factor only (no (n-1)/(n-2) factor, unlike statsmodels' default).
  The p-value and the CI are the two-sided Wald test and interval from `se`;
  there is no OLS p-value.
- Clusters are Mate Networks of the **whole pedigree**.  A parent in no
  Mating Pair is a network of its own, so its children share one cluster.
  Dropping children for a missing value or a stratum never splits a
  network; G counts the networks with an analysed child, and the labels
  are renumbered 0..G-1 per fit.  `mate_correlation` instead recomputes
  networks when strata drop pairs (ADR 0006).  Keeping whole-pedigree links
  is conservative: a parent missing the trait still joins two sibships.
- `se` is exactly 0 when every network's score sum is within its rounding
  bound, `n eps sum |x_i - xbar| (|y_i| + |slope x_i| + |intercept|) / Sxx`.
  A constant child value fits exactly (slope 0, both SEs 0).  A zero `se`
  has no Wald test: `p` and `ci` are `None` with `zero_se`.

Rejected:

- An OLS p-value.  simACE's comes from a normal approximation to the t
  tail (`_t_sf_python`) that nothing reads; an exact one needs the
  incomplete beta.
- A fixed minimum number of children.  A fit is undefined only when the
  estimate is (no children, a constant parent value); `se_ols` needs three
  children and `se` two networks.  Callers keep their own floors (simACE: 10).
- Networks recomputed per subset, as `mate_correlation` does for strata.
  That would split a sibship's network when the parent who links it is
  missing the trait or sits in another stratum.

## Consequences

- With two networks that each hold one parent value, which is common for
  a midparent fit in a small stratum, the line passes through both
  networks' means and `se` is 0.  Without the rounding bound it was about
  1e-16 with `p = 0`.
- The sandwich SE needs many networks.  In the gate it falls to 0.31 of
  the OLS SE with 8 networks.  With 139 to 1987 networks it is 0.97 to 1.10
  times the OLS SE for an additive trait alone and up to 1.79 times for a
  trait with a shared environment.  Mate Networks do not capture dependence through
  ancestry: pooled over generations a child is a parent elsewhere, and
  within one generation cousins' parents are sibs in different networks.
- Under random mating and no environment shared by parent and child, the
  midparent slope estimates h2 and a single-parent slope h2/2, on the
  observed scale for a 0/1 trait.  This is documented, not computed.
- The gate (`docs/gates/po-regression`) matches simACE bit for bit on its
  float64 paths.  simACE's validation forms its midparent in float32, and
  differs by up to 3.1e-7 relative.
