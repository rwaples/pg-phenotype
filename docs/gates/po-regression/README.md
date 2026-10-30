# Parent-offspring regression gate: simACE vs pg-phenotype (issue #3)

Result: **pass**.  Run 2026-10-10 on simACE
`6355ffa6aebe37238cb2724dfcde288830fdcb6c` (simACE's pixi env) against the
pg-phenotype working tree on `e2bded7`:

    git -C <simACE> worktree add --detach <dir> 6355ffa6aebe
    pixi run --manifest-path <simACE>/pixi.toml python -B tools/make_simace_po_regression_golden.py \
        --simace <dir> --out tests/golden/simace_po_regression_6355ffa6aebe
    pixi run python -B tools/po_regression_gate.py --out docs/gates/po-regression/gate.json

`tests/test_regression.py` asserts the outcomes and the agreement below.
`gate.json` holds the numbers.

## Inputs

Six `run_simulation` pedigrees (`n2000_g4`, `n2000_g4_am` with
`assort1 = 0.3`, `n500_g3_remating` with `mating_lambda = 1.5`,
`n200_g3_noc` with `C1 = 0`, `n30_g3`, `n12_g3`) with affected status at 10%
prevalence, as simACE's stats tests build them.  Each simACE result maps to
one pg-phenotype midparent fit (`tests/po_regression_golden.py`):

| simACE | pg-phenotype |
|---|---|
| `compute_parent_offspring_corr`, gen g | stratum `generation`, label g, `liability1` |
| `compute_parent_offspring_corr_by_sex`, sex s, gen g | stratum `2 generation + sex`, `liability1` |
| `compute_parent_offspring_affected_corr` | pooled, `affected1` (binary) |
| the same, `affected1` all 0 | pooled, zeros |
| `_validate_parent_offspring`, A1 and A1 + C1 + E1 | pooled |
| the same, A1 x 1e-11, C1 = E1 = 0 | pooled |

## Outcomes

| outcome | comparisons |
|---|---|
| both defined | 52 |
| both undefined (`constant_parent_value`; simACE: zero midparent variance) | 6 |
| simACE None, fewer than 10 trios (`n12_g3` by sex) | 4 |
| simACE None, under the validation's 100-offspring floor (`n30_g3`, `n12_g3`) | 4 |
| simACE None, parent SD under `safe_linregress`'s 1e-10 (the 1e-11 scale) | 6 |

pg-phenotype estimates every case simACE withholds by its floors, and its
1e-11-scaled slope equals the unscaled one to 1e-6.  No case is defined in
simACE and undefined in pg-phenotype.

## Agreement

- **float64 paths** (by generation, by sex, affected; 44 comparisons):
  slope, intercept, r, r2 and the OLS SE are bit-identical, and n equals
  simACE's `n_pairs`.  Both use two-pass centred sums.
- **simACE's validation** (8 comparisons): up to 3.1e-7 relative
  (r^2 on `n2000_g4_am` A1 + C1 + E1).  simACE stores A, C and E as float32
  and forms the midparent in float32.  Redoing that in NumPy and fitting
  with `scipy.stats.linregress` reproduces simACE's numbers
  (`simace_validate_float32`): exactly on the AVX2 machine that wrote the
  golden, and within 16 float32 ulps where AVX-512 or ARM orders NumPy's
  float32 sums differently (CI saw up to 2 on x86_64 and 6 on arm64),
  counting the intercept's ulps at the scale of `ybar - slope * xbar`
  (`test_gate_validation_gaps_are_simaces_float32_midparent`).  Every value
  simACE stored is a float32.  pg-phenotype works in float64.  The gate's
  tolerance there is 1e-6 relative.

## SE

pg-phenotype's Mate Network sandwich SE over its OLS SE, on the 52 defined
fits: min 0.31, 10th percentile 0.74, median 1.13, 90th 1.39, max 1.79.

- Below 0.75 only with 9 networks or fewer (the min is 8 networks, 24
  children), where a cluster-robust SE is noisy and biased low.
- On the 30 fits with 139 to 1987 networks: 0.97 to 1.10 for A1 alone,
  whose sibling residuals (Mendelian sampling) are independent, and 1.02
  to 1.79 for traits with a shared environment (liability, affected,
  A1 + C1 + E1), whose sibling residuals are correlated.  The largest is
  the pooled affected fit of `n2000_g4_am`.

Switching simACE to pg-phenotype's `se` widens the SEs of traits with a
shared environment; its point estimates do not change.
