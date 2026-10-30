# Relative-Pair Correlation SE: cluster-design calibration (issue #18)

> Recorded on pedigree-graph 0.13 and not rerun on 0.14, whose path-count
> full and half categories (its ADR 0018) move a few 1C and Av-family pairs
> on these random-mating pedigrees (about 0.1-0.4% of 1C). The result is not
> expected to change; a rerun on 0.14 would restate it.

Result: **the MRCA designs calibrate; the Mate Network designs do not.**
Clustering by the pair's most recent common ancestor (`mrca`), alone or
joined with shared members (`mrca_person`), gives SD(r̂)/SE within about 5%
of 1 in every scenario and category type, and median Wald coverage of 0.95
over all cells (0.93-0.96 per scenario and type, lowest at K 0.02).  The Mate
Network designs (`mn`, `mn_person`) match them while networks stay small.
When serial mating joins most of a generation into one network, the
spread of r̂ is 1.5-2.3 times their SE, and coverage falls to 0.61-0.78.

The shipped SE (ADR 0013's `mrca_person`, `relative_pairs.correlation(...,
se=True)`) reproduces this script's per rep and calibrates the same way,
the stratified shared-rho fit included: median SD(r̂)/SE 0.94-1.09 and
coverage 0.89-0.97 per scenario, category type, estimator and fit (see
[The shipped SE](#the-shipped-se-through-the-public-api---rust)).

Run 2026-10-16 on 600 simACE reps, against the pg-phenotype working tree
on `012a0f2`:

    pixi run --manifest-path <simACE>/pixi.toml simace run se_calibration --jobs 10 --no-plots
    POLARS_MAX_THREADS=1 pixi run python -B tools/relative_pairs_se_calibration.py \
        <simACE>/results/se_calibration --jobs 11 --out docs/gates/relative-pairs-se \
        --reps <scratch>/reps.tsv

The tool now refuses to compute reps unless all five thread variables
(`PG_PHENOTYPE_THREADS`, `RAYON_NUM_THREADS`, `POLARS_MAX_THREADS`,
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) are 1
(`tools/_calibration_runner.py`), so a rerun sets them all.

`calibration.tsv` has one row per scenario x trait combination x estimator
x cell x stratum: mean `n`, mean and SD of r̂ across reps, and for each
design the mean SE, `ratio_` = SD(r̂)/mean SE, `cover_` = Wald 95%
coverage of the across-rep mean r̂, `cv_se_` = the SE's own coefficient of
variation, and `undefined_` = reps whose meat was negative.
`summary.tsv` holds the medians below.  `--from-reps` rebuilds both from the
per-rep rows, which are not kept here (about 40 MB).

## Inputs

simACE `config/se_calibration.yaml`: six scenarios, 100 reps each, N 20000
per generation, 6 pedigree generations of which the last 3 are phenotyped
(the cohort's `affected{t}` is null before generation 3).  The reps'
`run.yaml` records simace `0.1.1.dev38+gec3d0a396`.

| scenario | `mating_lambda` | K | trait1 A, C | trait2 A, C |
|---|---|---|---|---|
| `se_cal_lam05_K10` | 0.5 | 0.10 | 0.5, 0 | 0.4, 0.2 |
| `se_cal_lam05_K02` | 0.5 | 0.02 | 0.5, 0 | 0.4, 0.2 |
| `se_cal_lam3_K10` | 3.0 | 0.10 | 0.5, 0 | 0.4, 0.2 |
| `se_cal_lam3_K02` | 3.0 | 0.02 | 0.5, 0 | 0.4, 0.2 |
| `se_cal_lam05_K10_hiA` | 0.5 | 0.10 | 0.8, 0 | 0.5, 0.3 |
| `se_cal_lam3_K10_hiA` | 3.0 | 0.10 | 0.8, 0 | 0.5, 0.3 |

Cells: the 12 categories up to degree 3, plus the groups PO (MO, FO) and HS
(MHS, PHS); each pooled and in each generation's stratum (ADR 0012's rule);
trait1, trait2 and their cross-trait combinations (both orientations of a
symmetric pair); Pearson on liability and the tetrachoric and phi on case
status, over the rows whose case status is observed.  The stratified
shared-rho fit is calibrated through the public API below.

## Method

Per cell, each unordered pair's influence φ_d on r̂ (r̂ - r ≈ Σ φ_d / n):
Pearson's `u v - r/2 (u² + v²)`; the two-step tetrachoric's
`[(ij - p11) - Qa (i - pa) - Qb (j - pb)] / φ2(h, k; ρ)`; a symmetric
pair's φ is the mean over its two entries.  The meat of each design
(`docs/relative-pairs-se-research.md`, Q2, Q3, Q5) is a sum of squares of
per-key sums of φ, and SE = √meat / n, with no small-sample factor.

| design | clusters |
|---|---|
| `pair` | each unordered pair (the current lower bound) |
| `person` | pairs sharing a member (dyadic) |
| `mn` | the Mate Network of the pair's top mating |
| `mn_person` | same Mate Network or a shared member (inclusion-exclusion) |
| `mrca` | the pair's MRCA: the ancestor member of a lineal pair, the shared parent of a half relationship, the shared mother of a full one (the first shared parent in (mother, father) path order under a loop or a double half relationship) |
| `mrca_person` | same MRCA or a shared member (inclusion-exclusion) |

## Checks of the script

| check | result |
|---|---|
| tetrachoric and Pearson influence (directed, symmetric, symmetric cross-trait) against n ∂r̂/∂w_d by central differences | pass (`--self-test`) |
| the scipy two-step tetrachoric against `correlation.tetrachoric` | max 2e-8 (at ρ = -0.94); 1e-15 at moderate ρ |
| each design's meat against brute force over pairs of pairs | pass (`--self-test`) |
| every pooled same-trait r̂ against `relative_pairs.correlation` on `012a0f2`, rep 1 of each scenario | Pearson, phi 1e-10; tetrachoric 1e-8 |
| the MRCA key is an ancestor of both members at the category's depths, by an independent per-pair walk (`check_mrca`, rep 1 of each scenario) | 300 sampled pairs per category: pass |
| one-off, not in the script: every pooled, stratum and cross-trait r̂ and `n` on one rep against the uncommitted strata and two-trait build (`pgp-relative-pairs` worktree, 440 cells) | `n` equal; max gap 4e-14 |

## Result

Medians over cells estimated in every rep with at least 1000 pairs on
average (2381 cells).  Ratio is SD(r̂)/mean SE; 1 is calibrated.  With 100
reps one cell's ratio is good to about ±7% (one SD of its log), so a
median over 120+ cells is good to about ±1%.

| scenario | type | `pair` | `person` | `mn` | `mn_person` | `mrca` | `mrca_person` |
|---|---|---|---|---|---|---|---|
| λ 0.5, K 0.10 | sibling | 1.27 | 1.02 | 0.99 | 0.99 | 0.99 | 0.99 |
| | lineal | 1.14 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| | collateral | 1.33 | 1.05 | 0.99 | 0.99 | 0.99 | 0.99 |
| λ 0.5, K 0.02 | sibling | 1.31 | 1.07 | 1.05 | 1.05 | 1.04 | 1.05 |
| | lineal | 1.18 | 1.04 | 1.05 | 1.03 | 1.05 | 1.04 |
| | collateral | 1.21 | 1.08 | 1.03 | 1.03 | 1.03 | 1.03 |
| λ 0.5, K 0.10, high A | sibling | 1.44 | 1.06 | 0.99 | 0.99 | 0.98 | 0.99 |
| | lineal | 1.23 | 1.00 | 1.01 | 1.00 | 1.01 | 1.00 |
| | collateral | 1.47 | 1.10 | 1.00 | 1.00 | 1.01 | 1.00 |
| λ 3, K 0.10 | sibling | 1.20 | 1.03 | 2.16 | 2.16 | 1.01 | 1.01 |
| | lineal | 1.08 | 0.98 | 1.53 | 1.52 | 0.98 | 0.98 |
| | collateral | 1.26 | 1.04 | 1.61 | 1.61 | 0.99 | 0.99 |
| λ 3, K 0.02 | sibling | 1.17 | 1.04 | 2.21 | 2.21 | 1.02 | 1.02 |
| | lineal | 1.16 | 1.05 | 1.65 | 1.66 | 1.05 | 1.05 |
| | collateral | 1.16 | 1.05 | 1.71 | 1.71 | 1.02 | 1.03 |
| λ 3, K 0.10, high A | sibling | 1.34 | 1.07 | 2.26 | 2.26 | 1.03 | 1.03 |
| | lineal | 1.17 | 1.00 | 1.61 | 1.61 | 1.00 | 1.00 |
| | collateral | 1.34 | 1.06 | 1.57 | 1.57 | 0.98 | 0.98 |

Median coverage over all cells: `pair` 0.86-0.91, `person` 0.94-0.95,
`mn` 0.89-0.91 (0.61-0.78 at λ 3), `mrca` and `mrca_person` 0.95 in every
category type; per scenario and type 0.93-0.96, with 0.93-0.94 at K 0.02.  Across all 2381 cells the log ratio's SD is 0.087 for
`mrca` and 0.086 for `mrca_person`, against 0.071 from 100 reps alone; for
`person` 0.093, for `mn` 0.358.

Findings:

1. **The naive pair SE is too small** by 8-47% at the median, more for
   collateral and sibling cells and at high A, as the research note
   predicted.
2. **Dyadic clustering is not enough** for symmetric categories: 1.02-1.10
   at the median, most in collateral cells at high A (1.10).  It misses
   disjoint pairs within a sibship and cousins sharing grandparents.
3. **Mate Networks percolate under serial mating.**  At λ 3 one network
   holds 76-89% of a cell's pairs (median
   `largest_network_share`; 0.2-0.7% at λ 0.5).  The meat then has a few
   dominant terms, the SE varies across reps (median CV 0.30-0.34 against
   0.02-0.07 for the other designs), and it is too small.  ADR 0009's
   candidate fails exactly where half-sibs are common.
4. **MRCA and MRCA ∪ person agree.**  Their SEs differ by at most 7% in any
   cell (median 0%, 95th percentile 1.7%).  The first-order link the
   research note expected the union to need, (c, m) with (m, gm) in one
   lineal cell, is rare with three phenotyped generations; it could matter
   with more.
5. **Low prevalence.**  At K 0.02 every design, `pair` included, sits about
   4% above 1.  The worst cells are phi in generation 5 at K 0.02, about
   0.5% cumulative incidence and under one expected concordant pair per
   cell: ratio near 2 and coverage near 0.3 under all six designs.  The
   Wald SE fails there whatever the clusters; it is not a design effect.

## The shipped SE through the public API (`--rust`)

Run 2026-10-16 on the same 600 reps, against a release wheel of the
pg-phenotype working tree on `f161268` (the Relative-Pair SE, uncommitted):

    PG_PHENOTYPE_THREADS=1 POLARS_MAX_THREADS=1 pixi run python -B \
        tools/relative_pairs_se_calibration.py <simACE>/results/se_calibration \
        --rust --jobs 11 --out docs/gates/relative-pairs-se --reps <scratch>/rust_reps.tsv

As above, a rerun sets all five thread variables to 1.

Per rep, `relative_pairs.correlation(..., stratum=generation, se=True)`
with both traits, once for the 12 codes and once for the PO and HS groups,
on liability (Pearson) and case status (tetrachoric, phi), each over the
rows whose case status is observed: the cells above, plus each cell's
stratified fit.  Beside each pooled and stratum estimate, this script's r̂
and `mrca_person` SE for the same rep.  `rust_calibration.tsv` has the
columns of `calibration.tsv` for the one design `rust`, plus `mean_n11`
(a binary cell's mean count of pairs with both members affected; the
stratified fit reads its pooled cell's) and `no_se_<reason>` (reps whose
estimate has a value and no SE).  A boundary tetrachoric is left out of
r̂'s statistics, as above, and counted in `no_se_boundary`.
`rust_summary.tsv` has the medians per scenario, category type, estimator
and fit (`cells`: pooled and stratum; `stratified`), over the cells
estimated in every rep with at least 1000 pairs on average.
`--rust --from-reps` rebuilds both tables, byte for byte.

**Against this script.**  Over the 261,864 pooled and stratum estimates
with both SEs, the largest relative gap is 3.5e-6, a tetrachoric at
ρ̂ 0.997 in an MZ stratum of 86 pairs; Pearson and phi agree to the per-rep
file's 8 decimals, and r̂ to 1.1e-13.  In 29 estimates, phi = 1 in an MZ
stratum of at most 120 pairs, the Rust SE is `degenerate_meat` where this
script gives a rounding-sized SE (below 5e-9).  So in all 2381 cells of the
medians, SD(r̂)/SE and coverage equal `calibration.tsv`'s `mrca_person`
columns to 4 decimals.

**Result.**  Median SD(r̂)/SE, with median Wald coverage in parentheses:

| scenario | type | Pearson | phi | tetrachoric | stratified Pearson | stratified tetrachoric |
|---|---|---|---|---|---|---|
| λ 0.5, K 0.10 | sibling | 0.98 (0.95) | 0.99 (0.94) | 0.99 (0.95) | 0.96 (0.95) | 0.99 (0.95) |
| | lineal | 1.01 (0.95) | 0.99 (0.96) | 0.99 (0.96) | 1.02 (0.96) | 0.99 (0.96) |
| | collateral | 1.00 (0.95) | 0.99 (0.95) | 0.99 (0.96) | 1.00 (0.95) | 0.99 (0.96) |
| λ 0.5, K 0.02 | sibling | 0.97 (0.95) | 1.07 (0.89) | 1.07 (0.93) | 0.96 (0.95) | 1.08 (0.94) |
| | lineal | 0.99 (0.96) | 1.07 (0.91) | 1.08 (0.94) | 1.00 (0.96) | 1.09 (0.94) |
| | collateral | 1.00 (0.95) | 1.05 (0.92) | 1.05 (0.94) | 1.01 (0.95) | 1.05 (0.94) |
| λ 0.5, K 0.10, high A | sibling | 0.96 (0.96) | 1.00 (0.95) | 1.01 (0.95) | 0.94 (0.96) | 1.00 (0.95) |
| | lineal | 0.99 (0.96) | 1.01 (0.96) | 1.01 (0.95) | 1.00 (0.96) | 1.01 (0.95) |
| | collateral | 1.02 (0.95) | 0.99 (0.95) | 0.99 (0.95) | 1.02 (0.94) | 0.99 (0.96) |
| λ 3, K 0.10 | sibling | 1.00 (0.95) | 1.01 (0.95) | 1.01 (0.95) | 1.03 (0.95) | 0.98 (0.95) |
| | lineal | 1.00 (0.95) | 0.94 (0.96) | 0.94 (0.96) | 1.01 (0.95) | 0.96 (0.96) |
| | collateral | 1.00 (0.95) | 0.96 (0.96) | 0.96 (0.96) | 1.02 (0.94) | 0.98 (0.96) |
| λ 3, K 0.02 | sibling | 0.99 (0.96) | 1.04 (0.92) | 1.03 (0.95) | 1.03 (0.95) | 0.99 (0.94) |
| | lineal | 0.99 (0.95) | 1.07 (0.92) | 1.07 (0.94) | 1.01 (0.95) | 1.09 (0.94) |
| | collateral | 0.99 (0.95) | 1.05 (0.91) | 1.04 (0.95) | 0.99 (0.94) | 1.00 (0.95) |
| λ 3, K 0.10, high A | sibling | 0.99 (0.96) | 1.04 (0.94) | 1.04 (0.94) | 1.03 (0.95) | 1.06 (0.94) |
| | lineal | 1.02 (0.94) | 1.00 (0.95) | 0.99 (0.95) | 1.04 (0.94) | 0.97 (0.96) |
| | collateral | 1.02 (0.95) | 0.98 (0.96) | 0.98 (0.95) | 1.04 (0.95) | 0.95 (0.96) |
| all | sibling | 0.98 (0.96) | 1.02 (0.93) | 1.02 (0.95) | 1.00 (0.95) | 1.01 (0.95) |
| | lineal | 1.00 (0.95) | 1.01 (0.94) | 1.02 (0.95) | 1.01 (0.95) | 1.02 (0.95) |
| | collateral | 1.00 (0.95) | 1.00 (0.95) | 1.00 (0.95) | 1.02 (0.95) | 0.99 (0.95) |

1. **The stratified fit calibrates.**  Over its 514 cells the median ratio
   is 1.01 (5th-95th percentile 0.89-1.15), the log ratio's SD 0.078
   against 0.071 from 100 reps alone, and median coverage 0.95 (5th
   percentile 0.91).  Its SE varies less across reps than a stratum's
   (median CV 0.036 against 0.058 over the pooled and stratum cells).
2. **Pooled and stratum cells** match the script's `mrca_person` (2381
   cells: median ratio 1.01, 5th-95th percentile 0.88-1.15; median
   coverage 0.95).  By estimator the log ratio's SD is 0.071 for Pearson,
   0.082 for the tetrachoric and 0.101 for phi.
3. **Low prevalence.**  At K 0.02 the binary estimators' medians are
   0.99-1.09, most 3-9% above 1, and phi's coverage 0.89-0.92, as finding
   5 above.  By the mean count of pairs with both members affected
   (`mean_n11`), over every binary pooled and stratum cell:

   | `mean_n11` | cells | median ratio | median coverage | lowest coverage |
   |---|---|---|---|---|
   | under 1 | 43 | 2.54 | 0.31 | 0.00 |
   | 1-5 | 192 | 0.97 | 0.93 | 0.55 |
   | 5-20 | 315 | 1.06 | 0.92 | 0.84 |
   | 20 or more | 1224 | 1.00 | 0.95 | 0.85 |

   Under about one expected concordant pair the Wald SE fails, as ADR 0013
   records; it is not a cluster effect.
4. **No SE.**  4252 estimates stop at the tetrachoric's bound
   (`boundary`; 3083 of them in MZ cells, the rest with no concordant
   pair), 421 of them stratified fits.  33 are `degenerate_meat` (phi = 1
   in an MZ stratum).  None is `sandwich_undefined` or `negative_meat`.
5. **One cell type sits above the band.**  The father-offspring Pearson of
   trait1 at λ 3 in generation 5: 1.28 at K 0.10 and at K 0.02 (independent
   seeds) and 1.18 at high A, coverage 0.86-0.92; pooled 1.11-1.23.  Its
   mother-offspring counterpart is 0.98-1.03, and trait2's FO 1.02-1.04.
   It is the same in `calibration.tsv` under `person`, `mrca` and
   `mrca_person`, which coincide for a lineal stratum cell, so it is not
   the implementation.  A probe that also joins pairs through the child's
   mother, or through the father's mother or father, leaves the ratio at
   1.27-1.29: the extra spread does not come from those links.  Its source
   is open.
