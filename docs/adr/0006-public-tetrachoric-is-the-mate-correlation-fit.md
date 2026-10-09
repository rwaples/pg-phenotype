# ADR 0006: The public tetrachoric is the Mate Correlation's fit

**Status:** accepted
**Date:** 2026-10-09
**Context:** issue #2; simACE issue #48

## Context

simACE and fitACE compute tetrachoric correlations with simACE's own numba
code (`simace/analysis/stats/tetrachoric.py`), and the relative correlation
method (#1) needs one here.  pg-phenotype already fits a tetrachoric
correlation inside the Mate Correlation: the polychoric fit on a 2 x 2
table, on pedsum's numerics (ADR 0005), with a boundary flag and a
two-step sandwich SE.

simACE's version differs in its numerics (Acklam's `ndtri`, a 20-point
Owen's T, bounded Brent alone on [-0.999, 0.999]) and in its SE, which
treats the thresholds as known and uses the wrong Fisher information
(simACE #48: 4x too small at rho = 0, and finite at the search edge).

## Decision

- `pg_phenotype_core::correlation::tetrachoric` (Python
  `pg_phenotype.correlation.tetrachoric`, R `correlation_tetrachoric`)
  runs the Mate Correlation's code on one 2 x 2 table: the same `Tables`,
  the same Newton and Brent fit, the same `boundary` flag and reasons.  It
  does not reproduce simACE's numerics.
- The SE is the two-step sandwich with every pair its own cluster,
  `sqrt(n/(n-1) * sum n_ij IF_ij^2)`: the Mate Correlation's sandwich when
  every Mate Network is one pair.  On a 2 x 2 table it equals the
  delta-method SE with the thresholds estimated, times `sqrt(n/(n-1))`.
  The SE and CI are withheld at a boundary.
- It takes a table or paired 0/1 values; pairs with a missing value are
  dropped and counted.  The module is `correlation` so the other
  estimators can join it.

## Consequences

- A binary Mate Correlation cell whose pairs are all lone couples gives
  the standalone result: the same rho bit for bit, the SE to 1e-12
  (`tests/test_correlation.py`).
- The gate (`docs/gates/tetrachoric`) compares rho with simACE over 2457
  tables.  Inside the bound the two agree to 1.1e-7.  Switching simACE or
  fitACE over changes their SEs, by design.
- The gate found a Newton flaw the Mate Correlation shares; ADR 0005 lists
  its fix as a second departure from pedsum.
