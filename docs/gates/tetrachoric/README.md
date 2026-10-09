# Tetrachoric gate: simACE vs pg-phenotype (issue #2)

Result: **pass for rho; the SE differs by design**.  Run 2026-10-09 on simACE
`6355ffa6aebe37238cb2724dfcde288830fdcb6c` (simACE's pixi env: NumPy 2.5.3,
numba 0.67.0, SciPy 1.18.1, JIT on) against the pg-phenotype working tree on
`122d612` (Linux, SciPy 1.18.1 for the oracle):

    pixi run --manifest-path <simACE>/pixi.toml python -B tools/make_simace_tetrachoric_golden.py \
        --simace <simACE checkout> --out tests/golden/simace_tetrachoric_6355ffa6aebe
    pixi run python -B tools/tetrachoric_gate.py --out docs/gates/tetrachoric/gate.json

simACE's side is `tetrachoric_from_table(n11, n10, n01, n00)`, which
`tetrachoric_corr_se` and `simace/analysis/stats/correlations.py` call.  The
oracle is `tests/oracle/tetrachoric_reference.py`: on a 2 x 2 table the MLE
of rho solves `Phi2(tau_x, tau_y; rho) = n00 / n` exactly, so rho is a
`brentq` root to 1e-15, and the SEs are closed forms there.
`tests/test_correlation.py` asserts every statement below.  `gate.json`
holds the numbers.

## Tables

2457 tables, rows by the `x` level:

- `small`: every table with n <= 10 (1001), empty and constant ones included.
- `expected`: the rounded expected table for n in {50, 200, 10^3, 10^4,
  10^5, 10^6}, prevalences {0.001, 0.01, 0.05, 0.2, 0.5} on each side and
  rho in {-0.95, -0.5, 0, 0.3, 0.7, 0.95, 0.99, 0.999} (720).
- `sampled`: one multinomial draw of each `expected` setting (720).
- `one_empty`: one empty cell, each of the four, at n in {20, 100, 10^3, 10^4} (16).

## Discrete outcomes

| pg-phenotype | simACE | tables |
|---|---|---|
| `no_complete_pairs` | NaN | 1 |
| `constant_margin` | NaN | 440 |
| interior fit | r | 806 |
| `boundary` (SE withheld) | r | 1210 |

An undefined estimate is NaN in simACE exactly when pg-phenotype gives a
reason.  pg-phenotype flags a boundary on 1210 tables:

- 1180 have an empty cell.  Their MLE is at |rho| = 1, so rho is not
  identified.  Both stop somewhere on the flat likelihood: simACE's |r| runs
  from 0.68 to 0.999, and pg-phenotype returns its point with `boundary`.
  simACE still reports a finite SE on 448 of the 1210.
- 30 have no empty cell and |rho| >= 0.9989, inside the flag's 1e-3 margin
  of the bound.  simACE's r is within 7.4e-4.  simACE searches only to
  ±0.999, pg-phenotype to ±0.9999.

## rho

On the 806 interior fits:

- **pg-phenotype to simACE**: median 2.3e-9, 99th percentile 7.4e-8, max
  1.13e-7, at `[[4986, 5007], [2, 5]]` (x prevalence 7e-4).  There
  pg-phenotype matches the oracle to 2.5e-15.  So the gap is simACE's:
  Acklam's `ndtri` (relative error about 1e-9) moves an extreme threshold.
- **pg-phenotype to the oracle**: median 1.4e-15, max 2.4e-8.  The large
  gaps are fits that fall back to bounded Brent (xatol 1e-7) near the bound.

Acceptable: the largest gap is about 1/600 of the smallest SE in the grid
(6.8e-5).

## SE

- **pg-phenotype to the oracle's two-step SE** (times `sqrt(n/(n-1))`):
  median relative gap 2.9e-15, max 2.4e-5, at rho = 0.9989.  Near the bound
  the SE moves with rho by about `1 / (1 - |rho|)`, so a 2e-8 Brent gap
  becomes about 2e-5 in the SE.
- **pg-phenotype to the known-thresholds SE**: ratio 1.000 to 1.155,
  median 1.0096.  Estimating the thresholds widens the SE, most for
  unbalanced margins.
- **simACE to the known-thresholds SE**: ratio 0.0004 to 0.25, median 0.108.
  simACE's Fisher information is `n phi2^2 / prod(p)` where it should be
  `n phi2^2 sum(1/p)` (simACE #48).  The ratio is
  `sqrt(prod(p) sum(1/p))`, at most 0.25, when every cell is 1/4.

Switching simACE to pg-phenotype multiplies its tetrachoric SEs by 4.0 to
2832 (median 9.3) on these tables.  fitACE's Falconer `se_h2`
(`fitace/ltm/falconer.py`) scales with them.  This is the fix for simACE
#48, not a regression.

## Found and fixed

On `[[9891, 104], [1, 4]]` pg-phenotype returned rho = 0.9915, flagged as a
boundary, where the MLE is 0.7969 (the oracle and simACE agree).  Newton from
0 stepped to 0.9915.  There p10 underflows to the 1e-300 floor, the Hessian
is infinite, the next step is 0, and Newton stopped as if converged.  The
Mate Correlation shares this fit, and so does pedsum at its pin
(`pedsum/assortative_mating.py:474-499`).

A first fix, which handed a stop back to Brent when its NLL was above the
start's, was incomplete.  A code review found tables where the overshoot
lands below the start's NLL, and a sweep of skewed random tables found
others where the Hessian is huge but finite (1e225).  About 1 in 1,000
such tables missed its maximum.  The cause in every case is a populated
cell whose probability, `Phi(h) - Phi2(h, k)` near 1, is rounding noise,
so the score and Hessian are too.  `Tables::terms` now gives a NaN Hessian,
and Newton hands over to bounded Brent, while a populated cell's
probability is below 1e-12 (ADR 0005).  The sweep (10 seeds, about 140,000
tables with an interior maximum, n up to 3e7) then found none missed.
`tests/test_correlation.py` keeps six of these tables and a 4000-table
sweep.  Both fail without the fix.

The fix moved the gate's counts from 805 interior and 1211 boundary fits to
806 and 1210.  No assortative-mating golden moved.
