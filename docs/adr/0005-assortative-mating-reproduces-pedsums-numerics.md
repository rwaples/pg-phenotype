# ADR 0005: Assortative mating reproduces pedsum's numerics

**Status:** accepted
**Date:** 2026-10-07
**Context:** simACE `plans/pg-phenotype-assortative-v2.md`, decisions D1-D3, D6

## Context

The assortative-mating method moves from pedsum #13 into pg-phenotype
(D1), and pedsum will call pg-phenotype once it is released (D7).  pedsum's
numbers come from several numerical layers: numba kernels that call the C
library's `erfc`, SciPy's `ndtr`, `ndtri` and `owens_t`, SciPy's bounded
Brent (`minimize_scalar`), and NumPy reductions.  pg-phenotype already had a
standard-normal module (`normal.rs`) for PA-FGRS, written for R's `qnorm`.

A first port that reused `normal.rs` and the `libm` crate differed from
pedsum by one or two ulp in a few primitives.  That was enough to change a
discrete outcome: in the golden fixture `small_strat`, a zero-count cell's
corner probability came out as rounding noise on one side and not the
other, and one side withheld an SE that the other reported.  Any
difference in a primitive can move a threshold, a stopping draw or a
boundary flag, and those are part of the parity contract.

## Decision

The assortative module carries its own copies of the primitives pedsum
uses, ported for bit-identical results, in preference to sharing
`normal.rs`:

- `cephes.rs`: SciPy's cephes `ndtr`, `erf`, `erfc` and `expm1`, and xsf's
  `owens_t` (Patefield & Tandy), from xsf `f7b85f5` as vendored by SciPy
  1.18.1.  It also holds glibc 2.39's `erfc`
  (`sysdeps/ieee754/dbl-64/s_erf.c`), on the platform `exp`, which the
  kernels' normal CDF calls as numba does.
- `kernels::ndtri`: AS 241 in pedsum's operation order, which rounds
  differently from `normal::quantile`.
- `fit::minimize_bounded`: SciPy 1.18.1's `_minimize_scalar_bounded`.
- The Wald `z` is SciPy's `ndtri(0.975)`, one ulp above AS 241's.
- Sums keep pedsum's order where pedsum fixes it: fixed blocks of 16,384
  pairs in the kernels, pair-by-pair accumulation in the bootstrap and
  permutation passes, and NumPy's pairwise summation in the float64
  standardisation that precedes the permutation pass's float32 storage.
  Sums pedsum leaves to NumPy elsewhere run in index order.

The port is checked against stored pedsum results (`tests/golden/pedsum_am_<sha12>/`,
written by `tools/make_pedsum_am_golden.py` in a snapshot of the pinned
pedsum commit), not against a live pedsum.

## Consequences

- On a 3,110-value grid every primitive matches SciPy or numba bit for bit
  (`primitives_parity.rs`).  Over 74 golden cases every count, reason, flag
  and p-value matches pedsum (`docs/gates/assortative-mating`).
- Summing NumPy's BLAS `dot` the same way would tie the code to one CPU's
  OpenBLAS kernel, so the latent fits' sums stay in index order.  A fit
  that falls back to bounded Brent can stop up to 3.2e-10 away from
  pedsum's, inside Brent's 1e-7 tolerance.  The parity tests allow 1e-7 for
  latent estimates and 1e-12 (relative) for closed-form ones.
- Two normal-distribution modules now exist side by side.  `normal.rs`
  stays R's `qnorm` for PA-FGRS. `cephes.rs` and `kernels::ndtri` stay
  pedsum's for assortative mating.  Merging them would change one
  method's results to match the other's oracle.
- The ports match glibc and SciPy at the versions named above, on Linux.
  A platform with another C library, or a SciPy that changes `owens_t`,
  can differ from pedsum by an ulp. The goldens then show it.
- One deliberate departure: pedsum's polychoric sandwich sums over
  zero-count cells, where float noise can make a term NaN and withhold the
  SE.  An empty cell is no term of the estimating equations, so
  pg-phenotype skips it.  The one golden this changes matches pedsum with
  the same fix to 4.4e-16 (`DEVIATIONS` in `tests/test_assortative_golden.py`).
