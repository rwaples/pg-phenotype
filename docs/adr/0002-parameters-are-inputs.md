# ADR 0002: PA-FGRS parameters are inputs; the CIP contract

**Status:** accepted
**Date:** 2026-10-06
**Context:** simACE `plans/pg-phenotype-extraction-v3.md`, decisions #5-#9, #18

## Decision

`h2`, `rg`, `rho_within` and one CIP table per trait are inputs.  The
library derives thresholds and `w`; estimating parameters stays with the
caller (fitACE_pafgrs keeps its estimators).

| Input | Contract |
|---|---|
| CIP | `ages` strictly increasing and finite; `cip` non-decreasing in `[0, 1)` with a positive last value.  **K is the last value**, not a separate argument. CIP at an age interpolates linearly: 0 below the first age, K at and above the last (`numpy.interp(age, ages, cip, left=0, right=K)`). |
| Threshold | `Phi^-1(1 - K)` by AS241, R's `qnorm` algorithm and coefficients. |
| `w` | Case: 1.  Control: `clip(CIP(age) / K, 0, 1)`.  Missing status: 0.  Control without an age: 0, counted in `controls_without_age`. A case needs no age: thresholds are lifetime-only. |
| `h2` | `(0, 1]` per trait. |
| `rg` | `[-1, 1]`. |
| `rho_within` | Optional; defaults to `rg * sqrt(h2_1 * h2_2)`.  `abs(rho_within) <= 1` and `(rho_within - rg sqrt(h2_1 h2_2))^2 <= (1 - h2_1)(1 - h2_2)`, so the non-genetic cross-trait covariance is positive semidefinite. |

A parameter outside its domain raises `parameter_out_of_range` (or
`inconsistent_parameters`) before any work.  A PA-FGRS trait is a binary
`Trait` (0, 1 or missing) on the pedigree's input rows, with its ages passed
beside it, each finite and `>= 0` or missing.

Making K the last CIP value removes an inconsistency in fitACE: its
`main()` took K from the config prevalence while its workflow took the CIP
endpoint.

## Consequences

- With two traits, a relative informs a score when observed on either.
- `n_relatives` counts people observed on at least one trait in this score
  call, not relatives in the prep.
- A proband with no informative relative gets `est = 0`, `var = h2`
  (bivariate: `cov12 = rg sqrt(h2_1 h2_2)`), `n_relatives = 0`.
