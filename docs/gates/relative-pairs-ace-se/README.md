# ACE Fit SE and weights: calibration on simACE reps (issue #19)

> Recorded on pedigree-graph 0.13 and not rerun on 0.14, whose path-count
> full and half categories (its ADR 0018) move a few 1C and Av-family pairs
> on these random-mating pedigrees (about 0.1-0.4% of 1C). The result is not
> expected to change; a rerun on 0.14 would restate it.

Result: **no weighting fully calibrates the continuous A, E, sA and rA.
The tetrachoric SEs are close to calibrated under every weighting. `gls` is the most
efficient for continuous traits. `inverse_variance` and `gls` fail on phi
at low prevalence. The default stays `"pairs"`.**

Decision (2026-10-17, after this run): the user chose a default by the
result instead. `weights=None` is `"gls"` for the primary estimator when
the result carries the covariance of the form read, and `"pairs"` for phi
or without that covariance
([ADR 0014](../../adr/0014-ace-fit-se-from-the-cells-covariance.md)).
The recommendation below is kept as the run gave it.

- **Continuous traits (Pearson).** C, sC, sE and rE calibrate: median
  SD(θ̂)/SE is 0.89-1.03 and coverage 0.96-0.98 under all three weightings.
  A, E, sA and rA spread more than their SE in every scenario. Their median
  ratio is 1.08-1.21 under `pairs`, 1.09-1.20 under `inverse_variance` and
  1.07-1.17 under `gls`, with coverage 0.89-0.94. The weighting changes
  this little.
- **Binary primary (tetrachoric), pooled and stratified.** The median
  ratio is 0.93-1.05 under `pairs` and 0.95-1.17 under `inverse_variance`
  and `gls`, with coverage 0.90-0.98. Under `pairs`, a boundary MZ
  tetrachoric (no SE) leaves every value that reads it without an SE
  (`row_without_se`). Over the 600 reps, A1 has no SE in 118, A2 in 98,
  sA in 203 and rA in 219. Of the 100 reps at each K 0.02 scenario, 96
  (λ 0.5) and 98 (λ 3) have such a cell. `inverse_variance` and `gls`
  leave that cell out (`no_se`), so every value keeps an SE.
- **Efficiency.** For continuous traits, `gls` has the smallest spread.
  Its median SD relative to `pairs` is 0.78-0.79 for A and E, 0.84-0.85
  for sA and rA, 0.80 for sE and rE, and 0.90 for C (lowest 0.63).
  `inverse_variance` is 0.88-0.98. For the tetrachoric the gain is
  0.90-0.97 for A, sA and rA and about 1 for C and E.
- **Phi.** `inverse_variance` and `gls` fail at K 0.02, at both λ. The
  ratio for A, C and E is 1.54-6.06, with coverage down to 0.01. SD(θ̂)
  is 1.51-3.65 times the SD under `pairs`, and 13.86 times for rA under
  `gls`. Under `pairs`, phi calibrates: 0.88-1.14, coverage
  0.92-0.97.
- **No failures of the machinery.** None of the 9000 fits returned
  `covariance_not_positive_definite` or `negative_variance`. No
  correlation cell returned `negative_meat`, and every within-person
  estimate has an SE.
- **Within-person kernel SE.** On these reps the ratio is 0.88-1.07 and
  coverage 0.89-0.98, matching the #19 prototype's `kin2` design. The λ 0.5 Pearson reps sit at 0.88-0.91. The
  [discussion](#within-person-kernel-se) has the detail.
- **Issue #21.** The trait-1 FO excess does not show up in A1. A1's ratio
  at λ 3 (1.02-1.17) matches λ 0.5 (1.06-1.15) and is no higher than A2's
  (1.14-1.22).

Run 2026-10-17 (01:14-01:22 UTC) on 600 simACE reps in pg-phenotype's
pixi env. The extension was built from the working tree with `pixi run
build-dev` (cargo reported it up to date). Wall time 478.3 s with 8
workers.

    PG_PHENOTYPE_THREADS=1 RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 pixi run python -B tools/relative_pairs_ace_se_calibration.py \
        /data/Documents/simACE/results/se_calibration --jobs 8 \
        --out docs/gates/relative-pairs-ace-se --reps <scratch>/ace_se_reps.tsv

Each worker runs on one thread (the tool refuses to start otherwise). The
host has 12 cores. Its 1-minute load average was 2.19 at launch, just
after a 12-rep trial run, and 7.79 at the end. Tree state,
from `run.json`:

| field | value |
|---|---|
| `git describe --always --dirty` | `v0.4.0-7-g4a6bd19-dirty` (HEAD `4a6bd19`, with U1-U3 of #19 uncommitted) |
| `tree_diff_sha256` | `53cfce2e83399c80450dad04bb33dfdc0a38fe045b3bde18e709b753e104e38e` |
| `tool_sha256` | `5a0569bc82371a304667765be32d9b9e855e891cdec4885083aa981eb08cc8df` |

`tree_diff_sha256` is the SHA-256 of `git diff HEAD --binary`, followed by
each untracked file's path and SHA-256 in path order. The tool and this
directory are left out of it.

Rerun on 2026-10-17, after the review fixes (the per-fit default weights)
and the simplification pass, with the same command into a scratch
directory: `tree_diff_sha256`
`82197e85b92dc82a5705c585749bc6899293e1a99e812102eb09f54493e69e1b`, the
same `tool_sha256`. Its `calibration.tsv`, `summary.tsv` and per-rep table
were byte-identical to the ones recorded here (`cmp`), so the committed
results hold for the committed tree. The tool passes `weights` explicitly
on every fit, so the default's change does not reach it.

Files:

- `calibration.tsv` has one row per scenario x kind x estimate x
  estimator x weights x quantity. Its columns are defined in `summarise`'s
  docstring.
- `summary.tsv` has the medians, minima and maxima over the six scenarios.
- `run.json` records the command, the threads, the tree, the wall time and
  the NumPy check.

`--from-reps <scratch>/ace_se_reps.tsv --out <dir>` rebuilds both tables
byte for byte (checked on this run). The per-rep rows are 111,000
lines (12.9 MB) and are not kept here.

## Inputs

simACE `config/se_calibration.yaml`: six scenarios, 100 reps each, the
same reps as the [Relative-Pair SE gate](../relative-pairs-se/README.md).
Each rep has N 20000 per generation and 6 generations. The last 3 are
phenotyped (about 60000 rows). There is no assortment and no C
transmission. Both checks are made against every rep's `params.yaml`; four
scenarios' files predate the `Ctm_` keys (simACE ADR 0025), which then
count as 0. The household is assigned by mother. rA = rC = rE = 0 (from
`config/_default.yaml`).

| scenario | `mating_lambda` | K | trait1 A, C, E | trait2 A, C, E |
|---|---|---|---|---|
| `se_cal_lam05_K10` | 0.5 | 0.10 | 0.5, 0, 0.5 | 0.4, 0.2, 0.4 |
| `se_cal_lam05_K02` | 0.5 | 0.02 | 0.5, 0, 0.5 | 0.4, 0.2, 0.4 |
| `se_cal_lam3_K10` | 3.0 | 0.10 | 0.5, 0, 0.5 | 0.4, 0.2, 0.4 |
| `se_cal_lam3_K02` | 3.0 | 0.02 | 0.5, 0, 0.5 | 0.4, 0.2, 0.4 |
| `se_cal_lam05_K10_hiA` | 0.5 | 0.10 | 0.8, 0, 0.2 | 0.5, 0.3, 0.2 |
| `se_cal_lam3_K10_hiA` | 3.0 | 0.10 | 0.8, 0, 0.2 | 0.5, 0.3, 0.2 |

## Method

Per rep and kind, the tool calls `relative_pairs.correlation` once. The
two kinds are continuous (`liability{t}`, Pearson) and binary
(`affected{t}`, tetrachoric and phi). Both are read over the rows whose
case status is observed (`relative_pairs_se_calibration.load_rep`). The
call uses both traits, the default categories (the 12 codes up to degree
3), `stratum` = generation, `se=True` and `covariance=True`.

It then calls `relative_pairs.ace(res,
shared_environment=["MZ", "FS", "MHS"], estimate=..., estimator=...,
weights=...)` for each form and each weighting:

- Forms: pooled primary, pooled phi (binary only) and stratified primary.
- Weightings: `pairs`, `inverse_variance` and `gls`.

That is 15 fits per rep. The quantities are A, C and E per trait, and sA,
sC, sE, rA, rC and rE. Each form's within-person estimate is a row too
(`weights` = `kernel`, quantity `r_wp`), with its kinship-kernel SE.

Each row records the value, the SE and both reasons. It also records the
fit's count of rows left out for want of an SE (`rows_no_se`). Per SE
reason, it records the cells of that form whose estimate has a value and
no SE (`cell_se_<reason>`).

Per scenario x kind x form x weights x quantity, `calibration.tsv` has
these columns:

- `ratio` = SD(θ̂) / mean SE over the reps with an SE. Its approximate MC
  SE is `ratio_mc_se` = ratio √(1/(2(n-1)) + CV(SE)²/n), about 0.07-0.09
  at 100 reps.
- `cover_mean` is the Wald 95% coverage of the across-rep mean θ̂ over the
  reps with an SE. This is the SE's calibration target; its MC SE is about
  0.022.
- `cover_truth` and `z_truth` are against the nominal truth.
- `efficiency` = SD(θ̂ | weights) / SD(θ̂ | `pairs`) over the reps both
  define.
- The reason counts.

The nominal truths are on the liability scale:

- A, C, E per trait, from `params.yaml`.
- s = r√(X1 X2) = 0.
- rA = rE = 0.
- rC is undefined, because trait 1's C is 0.
- r_wp = 0.

Phi's A, C and E are on the observed scale and have no nominal truth.

## Checks of the script

| check | result |
|---|---|
| Rep 1 of every scenario: every fit's values and SEs recomputed with NumPy (`check_fit`). The check runs WLS or GLS per solve from the result's cells and covariance, then `G Σ Gᵀ` with the delta method for E and the r's. It confirms each row's `used`, `variance` and `no_se`, and each `row_without_se` and `nonpositive_component` | 90 fits; 1029 SEs compared; largest relative gap 9.0e-13 for values and 4.1e-12 for SEs (`run.json`) |
| `--self-test`: the same NumPy check on a generated 2400-row two-trait pedigree, plus `summarise` on rows with known statistics | pass in 0.5 s; values 1.3e-15, SEs 2.1e-15 over 180 SEs |
| `--from-reps` against the direct run (12-rep trial and this run) | `calibration.tsv` and `summary.tsv` byte-identical |

The library's documented `no_se` behaviour holds as checked. Under
`inverse_variance` or `gls`, a cell with a value and no SE enters as
`value=None, reason="no_se"`.

## Results

Each table below gives the median over the six scenarios, then [min, max]
in brackets, then the median coverage in parentheses. `summary.tsv` has
every form, including pooled continuous, which is within 0.01 of
stratified everywhere. `calibration.tsv` has each scenario.

### Continuous, stratified Pearson

| quantity | `pairs` SD/SE (coverage) | `inverse_variance` | `gls` | efficiency IV | efficiency GLS | with SE (`pairs`, IV, GLS) |
|---|---|---|---|---|---|---|
| A1 | 1.10 [1.02, 1.17] (0.91) | 1.11 [1.05, 1.21] (0.92) | 1.09 [0.99, 1.17] (0.93) | 0.88 [0.74, 0.95] | 0.79 [0.63, 0.84] | 600, 600, 600 |
| C1 | 0.95 [0.93, 0.99] (0.97) | 0.94 [0.93, 0.98] (0.98) | 0.97 [0.92, 1.00] (0.97) | 0.95 [0.84, 0.97] | 0.90 [0.75, 0.93] | 600, 600, 600 |
| E1 | 1.08 [1.02, 1.17] (0.94) | 1.09 [1.02, 1.19] (0.93) | 1.07 [1.02, 1.20] (0.94) | 0.92 [0.82, 0.96] | 0.78 [0.66, 0.87] | 600, 600, 600 |
| A2 | 1.15 [1.14, 1.22] (0.92) | 1.15 [1.12, 1.20] (0.93) | 1.09 [1.06, 1.12] (0.94) | 0.88 [0.80, 0.92] | 0.78 [0.67, 0.82] | 600, 600, 600 |
| C2 | 0.91 [0.88, 0.98] (0.97) | 0.89 [0.84, 0.93] (0.98) | 0.90 [0.85, 0.96] (0.97) | 0.92 [0.87, 0.93] | 0.90 [0.86, 0.92] | 600, 600, 600 |
| E2 | 1.18 [1.16, 1.23] (0.90) | 1.17 [1.16, 1.24] (0.92) | 1.12 [1.10, 1.15] (0.93) | 0.89 [0.81, 0.91] | 0.79 [0.67, 0.83] | 600, 600, 600 |
| sA | 1.21 [1.07, 1.33] (0.91) | 1.20 [1.09, 1.33] (0.92) | 1.17 [1.06, 1.27] (0.89) | 0.93 [0.89, 0.95] | 0.84 [0.76, 0.86] | 600, 600, 600 |
| sC | 0.95 [0.82, 1.10] (0.96) | 0.94 [0.81, 1.13] (0.96) | 1.03 [0.85, 1.12] (0.96) | 0.98 [0.93, 1.01] | 0.94 [0.88, 0.96] | 600, 600, 600 |
| sE | 0.97 [0.96, 0.98] (0.96) | 0.96 [0.93, 0.98] (0.97) | 0.93 [0.90, 0.96] (0.97) | 0.93 [0.90, 0.95] | 0.80 [0.71, 0.83] | 600, 600, 600 |
| rA | 1.21 [1.07, 1.33] (0.90) | 1.20 [1.08, 1.33] (0.92) | 1.17 [1.06, 1.27] (0.89) | 0.93 [0.90, 0.95] | 0.85 [0.76, 0.86] | 600, 600, 600 |
| rC | 0.30 [0.28, 0.32] (1.00) | 0.21 [0.09, 0.68] (1.00) | 0.39 [0.14, 0.48] (1.00) | 1.09 [0.77, 1.56] | 1.10 [0.81, 1.52] | 262, 258, 251 |
| rE | 0.97 [0.97, 0.99] (0.96) | 0.96 [0.94, 0.98] (0.97) | 0.93 [0.90, 0.96] (0.97) | 0.93 [0.89, 0.95] | 0.80 [0.69, 0.83] | 600, 600, 600 |

rC cannot be interpreted. Trait 1's true C is 0, so rC = sC/√(C1 C2) is
defined only in the reps where both fitted C's are above 0 (about 44%).
It is then a ratio of two numbers near 0 (`nonpositive_component`
elsewhere). The same holds in every table below.

### Binary, stratified tetrachoric

| quantity | `pairs` SD/SE (coverage) | `inverse_variance` | `gls` | efficiency IV | efficiency GLS | with SE (`pairs`, IV, GLS) |
|---|---|---|---|---|---|---|
| A1 | 1.02 [0.91, 1.31] (0.94) | 1.11 [0.92, 1.19] (0.92) | 1.16 [0.93, 1.21] (0.90) | 0.91 [0.86, 0.97] | 0.90 [0.85, 0.98] | 482, 600, 600 |
| C1 | 1.01 [0.98, 1.08] (0.94) | 1.03 [0.92, 1.06] (0.94) | 1.09 [1.01, 1.13] (0.94) | 0.97 [0.86, 1.00] | 0.99 [0.86, 1.07] | 482, 600, 600 |
| E1 | 1.05 [0.94, 1.23] (0.94) | 1.07 [1.00, 1.22] (0.93) | 1.10 [1.03, 1.26] (0.93) | 0.96 [0.90, 0.99] | 0.96 [0.88, 0.98] | 482, 600, 600 |
| A2 | 0.95 [0.89, 1.07] (0.96) | 1.00 [0.91, 1.14] (0.95) | 1.02 [0.92, 1.25] (0.95) | 0.97 [0.95, 1.04] | 0.97 [0.95, 1.08] | 502, 600, 600 |
| C2 | 1.05 [0.95, 1.12] (0.94) | 1.08 [0.96, 1.09] (0.93) | 1.13 [0.99, 1.19] (0.91) | 0.99 [0.97, 1.03] | 1.01 [1.00, 1.04] | 502, 600, 600 |
| E2 | 1.05 [1.01, 1.13] (0.94) | 1.12 [1.08, 1.20] (0.93) | 1.17 [1.08, 1.30] (0.93) | 0.99 [0.95, 1.11] | 0.99 [0.93, 1.17] | 502, 600, 600 |
| sA | 1.04 [0.63, 1.13] (0.96) | 1.04 [0.94, 1.16] (0.94) | 1.06 [0.94, 1.14] (0.94) | 0.95 [0.92, 0.97] | 0.93 [0.90, 1.00] | 397, 600, 600 |
| sC | 0.94 [0.77, 1.10] (0.96) | 1.02 [0.77, 1.12] (0.94) | 1.06 [0.80, 1.14] (0.93) | 0.98 [0.94, 1.00] | 0.99 [0.93, 1.00] | 397, 600, 600 |
| sE | 0.96 [0.90, 1.00] (0.96) | 0.96 [0.90, 1.08] (0.96) | 0.97 [0.90, 1.19] (0.96) | 0.97 [0.94, 0.99] | 0.94 [0.89, 1.05] | 397, 600, 600 |
| rA | 0.98 [0.51, 1.13] (0.97) | 1.03 [0.94, 1.16] (0.94) | 1.07 [0.94, 1.14] (0.94) | 0.94 [0.88, 0.97] | 0.91 [0.89, 0.98] | 381, 600, 600 |
| rC | 0.15 [0.02, 0.44] (1.00) | 0.16 [0.01, 0.41] (1.00) | 0.23 [0.04, 0.48] (1.00) | 1.12 [0.81, 1.47] | 1.94 [0.52, 2.79] | 164, 272, 233 |
| rE | 0.93 [0.67, 1.00] (0.98) | 0.95 [0.90, 1.08] (0.96) | 0.96 [0.89, 1.19] (0.96) | 0.97 [0.94, 1.03] | 0.94 [0.89, 1.07] | 381, 600, 600 |

Under `pairs` at K 0.02, the binary ratio and coverage cover only the reps
that kept an SE: those without a boundary MZ cell in the rows the value
reads. That selection explains the extremes. Stratified rA at λ 3, K 0.02
is 0.51 over the reps that kept an SE, and sA there is 0.63. Pooled
tetrachoric is in `summary.tsv` (median ratio 0.94-1.04 under `pairs`,
0.96-1.12 under `inverse_variance`, 0.99-1.16 under `gls`).

### Binary, pooled phi

| quantity | `pairs` SD/SE (coverage) | `inverse_variance` | `gls` | efficiency IV | efficiency GLS | with SE (`pairs`, IV, GLS) |
|---|---|---|---|---|---|---|
| A1 | 1.04 [0.85, 1.14] (0.94) | 1.11 [0.87, 2.43] (0.93) | 1.25 [0.91, 2.83] (0.93) | 1.10 [1.03, 1.78] | 1.22 [1.04, 1.99] | 600, 600, 600 |
| C1 | 1.05 [0.91, 1.07] (0.95) | 1.19 [1.06, 2.48] (0.92) | 1.16 [1.03, 2.06] (0.93) | 1.13 [1.05, 2.09] | 1.08 [1.02, 1.65] | 600, 600, 600 |
| E1 | 1.02 [0.90, 1.20] (0.94) | 1.21 [0.91, 4.77] (0.94) | 1.36 [0.96, 4.87] (0.94) | 1.17 [1.00, 2.58] | 1.28 [1.03, 2.52] | 600, 600, 600 |
| A2 | 0.97 [0.89, 1.05] (0.95) | 1.01 [0.90, 2.19] (0.93) | 1.03 [0.94, 2.50] (0.94) | 1.02 [0.98, 1.80] | 1.03 [0.98, 1.98] | 600, 600, 600 |
| C2 | 1.04 [0.99, 1.10] (0.94) | 1.18 [1.03, 3.92] (0.91) | 1.20 [1.02, 3.65] (0.90) | 1.09 [1.04, 2.91] | 1.09 [0.97, 2.59] | 600, 600, 600 |
| E2 | 1.05 [0.94, 1.17] (0.92) | 1.16 [1.05, 5.98] (0.90) | 1.23 [1.04, 6.06] (0.90) | 1.04 [1.01, 3.65] | 1.07 [0.99, 3.58] | 600, 600, 600 |
| sA | 1.04 [0.91, 1.14] (0.95) | 1.18 [1.05, 1.34] (0.90) | 1.27 [1.11, 1.34] (0.90) | 1.02 [0.93, 1.19] | 1.06 [0.92, 1.34] | 600, 600, 600 |
| sC | 1.03 [0.76, 1.10] (0.97) | 1.04 [0.82, 1.23] (0.94) | 1.08 [0.83, 1.23] (0.93) | 1.07 [0.83, 1.15] | 1.09 [0.80, 1.14] | 600, 600, 600 |
| sE | 0.95 [0.88, 1.02] (0.97) | 1.11 [1.00, 1.28] (0.94) | 1.14 [0.98, 1.32] (0.94) | 1.08 [0.95, 1.27] | 1.08 [0.89, 1.28] | 600, 600, 600 |
| rA | 1.04 [0.92, 1.14] (0.95) | 1.11 [0.81, 1.23] (0.90) | 1.13 [0.12, 1.34] (0.88) | 1.16 [1.00, 3.98] | 1.23 [1.05, 13.86] | 600, 598, 575 |
| rC | 0.88 [0.36, 1.19] (0.95) | 1.04 [0.67, 1.21] (0.97) | 0.98 [0.20, 1.19] (0.95) | 1.13 [0.88, 2.19] | 1.53 [1.07, 2.86] | 558, 442, 450 |
| rE | 0.95 [0.88, 1.02] (0.97) | 1.11 [0.99, 1.28] (0.94) | 1.14 [0.97, 1.32] (0.94) | 1.07 [0.90, 1.27] | 1.08 [0.84, 1.28] | 600, 600, 600 |

The maxima are K 0.02, at both λ. A1, C2 and E2 under
`inverse_variance` or `gls` reach ratio 2.35-6.06, coverage 0.01-0.55
and efficiency 1.69-3.65. Over A, C and E at K 0.02, the ratio is
1.54-6.06 and the efficiency 1.51-3.65. A phi cell with few concordant pairs has a
small estimated variance that moves with its estimate, so weights by
1/Var(r̂) favour the cells that happen to be low. The cause is
UNVERIFIED; only the outcome above is measured.

### SE reasons

Summed over every rep, form and quantity:

| reason | where | count |
|---|---|---|
| `row_without_se` (SE) | binary `pairs`: pooled tetrachoric 1860, stratified 1798 | 3658 |
| `no_se` (row left out) | binary tetrachoric under `inverse_variance` and `gls`: 421 rows per form and weighting, in 219 reps (λ 0.5: 96 K 0.02, 2 K 0.10, 4 high A; λ 3: 98, 11, 8) | 421 per fit set |
| `cell_se_boundary` | the same 421 cells (MZ tetrachorics at the bound) in each tetrachoric form | 421 per form |
| `nonpositive_component` (value) | mostly rC (true C1 = 0): continuous 334-349 per form and weighting, binary 51-367, phi 42-175 | |
| `covariance_not_positive_definite` | none | 0 |
| `negative_variance` | none | 0 |
| `negative_meat`, `sandwich_undefined`, `degenerate_meat` (cells) | none | 0 |

### Within-person kernel SE

| kind | form | scenario | mean r̂ | SD(r̂) | mean SE | SD/SE (MC SE) | coverage | truth coverage |
|---|---|---|---|---|---|---|---|---|
| continuous | stratified primary | lam05_K02 | 0.0006 | 0.0044 | 0.0049 | 0.91 (0.06) | 0.98 | 0.97 |
| continuous | stratified primary | lam05_K10 | 0.0002 | 0.0044 | 0.0049 | 0.90 (0.06) | 0.97 | 0.97 |
| continuous | stratified primary | lam05_K10_hiA | 0.0003 | 0.0049 | 0.0056 | 0.88 (0.06) | 0.98 | 0.98 |
| continuous | stratified primary | lam3_K02 | -0.0009 | 0.0048 | 0.0047 | 1.02 (0.07) | 0.94 | 0.94 |
| continuous | stratified primary | lam3_K10 | -0.0009 | 0.0048 | 0.0047 | 1.02 (0.07) | 0.94 | 0.94 |
| continuous | stratified primary | lam3_K10_hiA | -0.0012 | 0.0057 | 0.0054 | 1.07 (0.08) | 0.95 | 0.91 |
| binary | pooled primary | lam05_K02 | 0.0294 | 0.0458 | 0.0432 | 1.06 (0.08) | 0.96 | 0.88 |
| binary | pooled primary | lam05_K10 | 0.0578 | 0.0155 | 0.0147 | 1.05 (0.07) | 0.95 | 0.03 |
| binary | pooled primary | lam05_K10_hiA | 0.0586 | 0.0162 | 0.0152 | 1.07 (0.08) | 0.95 | 0.03 |
| binary | pooled primary | lam3_K02 | 0.0354 | 0.0453 | 0.0426 | 1.06 (0.08) | 0.94 | 0.82 |
| binary | pooled primary | lam3_K10 | 0.0591 | 0.0152 | 0.0146 | 1.04 (0.07) | 0.92 | 0.04 |
| binary | pooled primary | lam3_K10_hiA | 0.0613 | 0.0145 | 0.0150 | 0.97 (0.07) | 0.95 | 0.03 |
| binary | pooled phi | lam05_K02 | 0.0034 | 0.0046 | 0.0045 | 1.02 (0.07) | 0.93 | 0.92 |
| binary | pooled phi | lam05_K10 | 0.0169 | 0.0048 | 0.0046 | 1.05 (0.07) | 0.94 | 0.03 |
| binary | pooled phi | lam05_K10_hiA | 0.0172 | 0.0050 | 0.0047 | 1.06 (0.08) | 0.95 | 0.03 |
| binary | pooled phi | lam3_K02 | 0.0040 | 0.0048 | 0.0046 | 1.04 (0.08) | 0.89 | 0.91 |
| binary | pooled phi | lam3_K10 | 0.0173 | 0.0047 | 0.0045 | 1.03 (0.07) | 0.92 | 0.05 |
| binary | pooled phi | lam3_K10_hiA | 0.0180 | 0.0045 | 0.0047 | 0.97 (0.07) | 0.93 | 0.03 |
| binary | stratified primary | lam05_K02 | 0.0017 | 0.0465 | 0.0438 | 1.06 (0.08) | 0.95 | 0.95 |
| binary | stratified primary | lam05_K10 | 0.0078 | 0.0157 | 0.0151 | 1.04 (0.07) | 0.96 | 0.90 |
| binary | stratified primary | lam05_K10_hiA | 0.0093 | 0.0163 | 0.0156 | 1.05 (0.07) | 0.95 | 0.88 |
| binary | stratified primary | lam3_K02 | 0.0075 | 0.0449 | 0.0432 | 1.04 (0.07) | 0.95 | 0.94 |
| binary | stratified primary | lam3_K10 | 0.0096 | 0.0159 | 0.0150 | 1.06 (0.08) | 0.91 | 0.92 |
| binary | stratified primary | lam3_K10_hiA | 0.0117 | 0.0148 | 0.0154 | 0.96 (0.07) | 0.95 | 0.93 |

Pooled continuous equals stratified to two decimals (`calibration.tsv`).

- **The shipped kernel reproduces the prototype.** All 30 rows above are
  within 0.0014 of the prototype's `kin2` ratio on the same reps
  (`wp_summary.tsv`), with identical coverage. For example, Pearson at
  λ 0.5, K 0.10 is 0.9009 in both.
- **It calibrates on the reps, except Pearson at λ 0.5.** The binary forms
  are 0.96-1.07, and the λ 3 Pearson reps are 1.02-1.07. The λ 0.5 Pearson
  reps are 0.88-0.91, 1.4-1.9 MC SE below 1, so the SE is
  conservative there (coverage 0.97-0.98). On the fixed pedigrees, the
  prototype's kin2 was 1.00-1.07 (`wp_fixedped_summary.tsv`). This is the
  prototype's open point 1, now seen through the public API: across reps,
  λ 0.5 r̂ varies less than the model predicts. The cause is UNVERIFIED.
- **The pooled binary within-person r̂ is not the liability's 0.** Pooled
  tetrachoric 0.058-0.061 and pooled phi 0.017-0.018 at K 0.10 give truth
  coverage 0.03-0.05. Stratified, the mean is 0.002-0.012 and truth
  coverage 0.88-0.95. As the prototype found, case status pooled across
  generations with different follow-up windows makes the two traits look
  correlated. The SE is calibrated around the across-rep mean either way.

## Where the continuous excess comes from

Each cell's SE calibrates in the #18 gate. Yet A, E, sA and rA, which pool
many cells, spread 7-21% more than their SE, while C, sC, sE and rE do
not. The binary ratios are near 1 because the binary SEs are 2.2-5.8
times the continuous ones (A1, `pairs`), so a fixed extra variance shows
less there.

A scratch probe split each continuous fit's error into two parts, using
`decomposition` and `realized_result` from
`tools/relative_pairs_ace_validation.py` (all 600 reps, 7 min, not kept).
It fitted the same `ace` call to each rep's realized-truth cells (A x A
plus C x C terms, E x E within person, as in the
[ACE truth gate](../relative-pairs-ace/README.md)). The two parts are
d = θ̂ - θ_realized and the realized truth's own spread across reps.

Medians over scenarios, stratified (pooled agrees to 0.01):

| quantity | SD(θ̂)/SE | SD(d)/SE: `pairs`, IV, GLS | SD(θ_realized)/SE (`pairs`) | coverage of θ_realized (`pairs`) |
|---|---|---|---|---|
| A1 | 1.10 | 0.76, 0.83, 0.95 | 0.95 | 0.99 |
| A2 | 1.15 | 0.84, 0.94, 1.00 | 0.85 | 0.97 |
| C1 | 0.95 | 0.80, 0.83, 0.87 | 0.63 | 0.995 |
| C2 | 0.91 | 0.95, 0.95, 0.96 | 0.70 | 0.96 |
| E1 | 1.08 | 0.74, 0.78, 0.89 | 0.95 | 0.99 |
| E2 | 1.18 | 1.14, 1.21, 1.26 | 0.83 | 0.92 |
| sA, rA | 1.21 | 0.71, 0.72, 0.77 | 0.96 | 0.99 |
| sE, rE | 0.97 | 0.64, 0.63, 0.67 | 0.70 | 1.00 |

What this shows:

1. **The SE is not too small for the error around a rep's own realized
   truth**, except for E2. Per scenario and weighting, SD(d)/SE is
   0.48-1.14, with medians 0.63-1.00. E2 is 0.97-1.59, its realized
   coverage 0.77-0.96.
2. **The realized truth itself varies across reps** by 0.4-1.1 SE. Its
   rep-level variation adds to the total spread. For A1, E1, A2, sA and
   rA, the sum of the two parts exceeds the SE.

This fits a rep-level common shock that moves every cell's realized
correlation together. One candidate is the realized genetic variance of
the phenotyped generations, which all families inherit from the same
unphenotyped ancestors. The cluster covariance treats pairs without a
shared member or MRCA as independent, so it has no term for such a shock.
That would leave each cell's SE nearly right but understate the
covariance between cells, and so the SE of whatever pools them. It also
explains why `gls` narrows the gap only a little. The mechanism is
UNVERIFIED: this probe does not separate a population-level shock from
any other cross-family dependence.

E2 is the one value whose SD(d) exceeds its SE, under every weighting.
E2 = 1 - A2 - C2 needs Cov(Â2, Ĉ2) through the shared (MZ, FS, MHS) and
unshared trait-2 cells. Its cause is UNVERIFIED.

## Issue #21 (trait-1 FO excess at λ 3)

Continuous, stratified SD/SE (MC SE):

| scenario | A1 `pairs` | A1 `gls` | A2 `pairs` | A2 `gls` |
|---|---|---|---|---|
| lam05_K02 | 1.09 (0.08) | 1.07 (0.08) | 1.14 (0.08) | 1.06 (0.08) |
| lam05_K10 | 1.15 (0.08) | 1.12 (0.08) | 1.14 (0.08) | 1.09 (0.08) |
| lam05_K10_hiA | 1.06 (0.08) | 1.07 (0.08) | 1.22 (0.09) | 1.08 (0.08) |
| lam3_K02 | 1.11 (0.08) | 1.11 (0.08) | 1.16 (0.08) | 1.11 (0.08) |
| lam3_K10 | 1.17 (0.08) | 1.17 (0.08) | 1.15 (0.08) | 1.09 (0.08) |
| lam3_K10_hiA | 1.02 (0.07) | 0.99 (0.07) | 1.21 (0.09) | 1.12 (0.08) |

The excess does not show up in A1. If #21's FO excess reached the ACE
Fit, it would raise trait 1 at λ 3 above λ 0.5 and above trait 2. A1 at
λ 3 (1.02-1.17) matches λ 0.5 (1.06-1.15), and A2 (1.14-1.22) is no
lower. The FO generation-5 stratum is a small share of A1's weight. The
general excess above swamps it.

## Recommendation: the default `weights`

**Keep `"pairs"` as the default.** Recommend `"gls"` in the docs for
continuous traits, and warn against `"inverse_variance"` and `"gls"` with
phi.

- **Calibration does not pick a weighting.** For continuous traits, every
  weighting leaves A, E, sA and rA 7-21% above their SE. `gls` narrows A2
  from 1.15 to 1.09, E2 from 1.18 to 1.12 and sA from 1.21 to 1.17. The
  gap is in Σ (see above), not in the weights.
- **`gls` is the most efficient for continuous traits.** Its SD relative
  to `pairs` is 0.78-0.85 for A, E, sA, rA, sE and rE, as low as 0.63, at
  equal or better calibration. It never failed: no Cholesky failure, no
  `negative_variance`. For continuous traits it dominates.
- **For the tetrachoric the gain is small** (0.90-0.97 for A, sA and rA;
  about 1 for C and E). Its median ratio is 0.01-0.14 higher than under
  `pairs` (stratified): A1 is 1.16 under `gls` against 1.02, and E2 1.17
  against 1.05.
  On the other side, `pairs` loses the SE of every value that reads a
  boundary MZ tetrachoric, which happened in 219 of 600 reps and almost
  every K 0.02 rep. `gls` and `inverse_variance` keep the SE by leaving
  the cell out.
- **For phi, `inverse_variance` and `gls` are unsafe at low prevalence.**
  At K 0.02, A, C and E reach ratio 1.54-6.06, coverage as low as 0.01,
  and SD 1.51-3.65 times that of `pairs` (rA 13.86 under `gls`).
- **One default for every estimator** must therefore be `"pairs"`. It is
  also the only weighting whose values equal the pre-#19 fit bit for bit
  (ADR 0012 D7).

An estimator-dependent default (`"gls"` for the primary estimator,
`"pairs"` for phi) would gain efficiency for continuous traits. It would
make the default depend on another argument, and it would change the
tetrachoric's values and calibration for a small gain. It is not
recommended without a further look at the tetrachoric case.

## Why the continuous A, E, sA and rA SEs are short (issue #22)

Result: **most of the gap is dependence between related families that
ADR 0013's clusters treat as independent.** Entries whose MRCAs are close
relatives are correlated. Counting MRCA relatives up to degree 3 together
closes about two thirds of the gap on these reps. The kernel acts on the
ACE Fit almost entirely through the covariance between cells: the median
cell's own SE stays calibrated, and the ACE Fit pools many cells. Some gap
remains, largest when every recorded generation is phenotyped. This is a
diagnosis, not a fix: no SE has changed.

`tools/relative_pairs_ace_se_probe.py` (its docstring has the method) runs
on each rep. It takes `relative_pairs.correlation(..., se=True,
covariance=True, influence=True)` and the per-entry influences (issue
#25), and rebuilds the cells' stratified covariance at several *levels*.
It refits `relative_pairs.ace` on each, stratified Pearson, under each
weighting. Levels:

- `base` is ADR 0013's design. On every rep it reproduces the result's own
  covariance to within 1e-9 of its largest entry.
- `deg<d>` also counts two entries together when their MRCAs are distinct
  relatives of degree at most d (MZ at degree 0). It subtracts the pairs
  that already count through a shared member, so each pair of entries
  counts once.
- `kin3` weights the degree-at-most-3 MRCA pairs by 2φ: a pair joined
  through a member keeps weight 1, and one joined only through related
  MRCAs gets 2φ.
- `_diag` and `_off` add a level's extra term to the diagonal alone or to
  the off-diagonal alone.

On small simulated pedigrees, every level's covariance matches a
brute-force sum over every two entries to within 9e-16 of its largest
entry, kin3's weights included.

A coarser partition is no alternative. The pedigree is one connected
component, so joining each MRCA with its parents leaves about 200 of 44k
clusters, and two generations up leaves one or two.

Outputs: `probe/se_calibration.tsv` (the 600 reps above) and
`probe/depth.tsv` (simACE's `config/ace_se_depth.yaml`), with their
`run.json`. The depth grid holds se_cal_lam3_K10 and varies the recorded
pedigree. ped3, ped6 and ped9 (G_ped 3, 6, 9; G_pheno 3) share G_sim 11
and their seeds, so rep k of each has the same phenotyped rows and only
its recorded ancestry is truncated. pheno6 (G_ped 6, G_pheno 6) has its
own seeds.

Every fit gave a finite value and SE at every level, except rC, which has
no value when a trait's C is 0 (`nonpositive_component`). The smallest
eigenvalue of the stratified covariance over the cells with an SE,
relative to the largest, was at least 0.0014 in every se_calibration rep
and 0.0003 in every depth rep (median over scenarios 0.0018-0.0031 by
level).

Median over the six se_calibration scenarios of SD(θ̂)/mean SE, `pairs`
(the estimator is fixed, so only the SE moves):

| quantity | base | deg1 | deg3 | kin3 | deg1 off-diagonal only | deg1 diagonal only |
|---|---|---|---|---|---|---|
| A1 | 1.10 | 1.00 | 0.98 | 1.04 | 1.01 | 1.08 |
| A2 | 1.15 | 1.05 | 1.03 | 1.09 | 1.06 | 1.14 |
| E1 | 1.08 | 1.00 | 0.97 | 1.03 | 1.00 | 1.06 |
| E2 | 1.18 | 1.09 | 1.07 | 1.13 | 1.10 | 1.17 |
| sA | 1.21 | 1.10 | 1.06 | 1.14 | 1.10 | 1.20 |
| rA | 1.21 | 1.09 | 1.06 | 1.14 | 1.10 | 1.20 |
| C1 | 0.95 | 0.93 | 0.93 | 0.94 | 0.96 | 0.91 |
| C2 | 0.91 | 0.89 | 0.89 | 0.90 | 0.90 | 0.90 |
| sE | 0.97 | 0.88 | 0.85 | 0.91 | 0.88 | 0.96 |
| rE | 0.97 | 0.88 | 0.86 | 0.91 | 0.89 | 0.96 |
| median cell's own SE | 1.01 | 1.00 | 0.99 | 1.00 | 1.01 | 1.00 |

Per scenario at deg3, A1 is 0.88-1.05, A2 1.00-1.05, E2 1.06-1.10 and sA
0.93-1.14 (base: 1.02-1.17, 1.14-1.22, 1.16-1.23 and 1.07-1.34). MZ MRCA
pairs (deg0) change nothing at this precision. Under `gls`, deg3 narrows
A1 from 1.09 to 1.04 and sA from 1.17 to 1.04. The cells' own ratios
spread around their median: 0.84-1.23 over every cell and scenario at
base, 0.79-1.21 at deg3.

The depth grid, base → deg1 → deg3, 100 reps a scenario:

| quantity | ped3 | ped6 | ped9 | pheno6 |
|---|---|---|---|---|
| A1 | 1.11 → 1.05 → 1.05 | 1.16 → 1.06 → 1.03 | 1.16 → 1.06 → 1.03 | 1.20 → 1.06 → 0.99 |
| A2 | 1.08 → 1.02 → 1.02 | 1.09 → 0.99 → 0.97 | 1.09 → 0.99 → 0.98 | 1.40 → 1.24 → 1.17 |
| E2 | 0.98 → 0.93 → 0.93 | 1.01 → 0.94 → 0.93 | 1.01 → 0.94 → 0.93 | 1.31 → 1.20 → 1.16 |
| sA | 1.11 → 1.04 → 1.03 | 1.16 → 1.05 → 1.02 | 1.16 → 1.05 → 1.02 | 1.16 → 1.02 → 0.95 |
| sE | 0.93 → 0.87 → 0.87 | 0.93 → 0.84 → 0.82 | 0.93 → 0.84 → 0.82 | 0.93 → 0.83 → 0.78 |
| median cell's own SE | 1.02 → 1.01 → 1.01 | 1.02 → 1.00 → 0.99 | 1.02 → 1.00 → 0.99 | 1.06 → 1.01 → 1.00 |

What the numbers show:

- **The kernel acts on the ACE Fit almost entirely off the diagonal.** The
  off-diagonal terms alone give almost all of a level's effect, and the
  diagonal alone almost none. The median cell's own SD/SE stays at about
  1, as ADR 0013 found, though single cells range from 0.84 to 1.23 at
  base (up to 1.28 for pheno6's GGP). The kernel is one model of the added
  covariance, so this does not show that every missing term lies between
  cells.
- **The gap is there without recorded ancestry above the phenotyped rows**
  (ped3). The related MRCAs that matter sit in and just above the
  phenotyped generations. For these estimators and kernels, recording more
  than three generations above changes nothing: ped6 and ped9 differ in
  about 450 of 487k related MRCA pairs and agree to within 0.01. The
  paired design holds the simulated ancestry fixed, so this says nothing
  about deeper ancestry's own contribution.
- **Phenotyping more generations widens the gap**, most for trait 2 (A2
  1.40 and E2 1.31 in pheno6). There, the degree-3 kernel leaves A2 and E2
  at 1.16-1.17. Its source is open: relationships beyond degree 3, or
  structure the kernel does not model (trait 2 has C 0.2).
- **sE and rE over-correct** under every kernel (0.78-0.91), so no level
  is a calibrated SE as it stands.
- **Recorded generation 0's genetic moments predict nothing.** Its realized
  variances of A per trait and their covariance predict none of the spread
  out of sample: 5-fold cross-validated R², median over scenarios −0.08 to
  −0.02 for A, E, sA and rA (single scenarios −0.16 to 0.06). That rules
  out these three summaries, not every founder-level effect: effects
  through particular lineages or unrecorded ancestors are untested. The
  phenotyped generations' realized variances and cross-trait covariances
  of A, C and E predict, as medians over scenarios, 28-33% of A and E and
  44% of sA and rA out of sample (41-53% in sample). Those covariates are
  themselves a product of the same family dependence, so they do not
  separate a population shock from it.
- **Around the realized truth** (base, medians over scenarios),
  SD(θ̂ − θ_realized)/SE is 0.71-0.84 for A1, A2, E1, sA and rA, and 1.14
  for E2. The realized truth's own spread is 0.83-0.96 SE. The two parts
  covary negatively (2 Cov/SE² −0.55 to −0.18; about 0 for sA and rA), so
  they do not simply add.

Next is a decision between two directions. One is an MRCA-kinship term in
ADR 0014's covariance; the degree and weighting would first need to
calibrate sE and rE and the deeper-phenotyped case. The other is for ADR
0014 to record that the SE understates the across-rep spread of the
A-side quantities by about 5-20% in multi-generation pedigrees, with a
documented way to widen it.
