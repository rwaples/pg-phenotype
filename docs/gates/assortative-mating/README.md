# Assortative-mating gate: pedsum #13 vs pg-phenotype (plan v2, unit 9)

Result: **pass**.  Run 2026-10-07 on pedsum `142adf300d5b0802f59af03f3f5211af910fc852`
(a `git archive` snapshot in its own locked pixi env: Python 3.14, NumPy 2.5.2,
SciPy 1.18.1, numba 0.68.0, 12 numba threads) against the pg-phenotype working
tree (pedigree-graph-core `62c82fa`, Linux, glibc 2.39, AVX2 without AVX-512):

    S=<scratch>/pedsum_am_142adf300d5b   # see tools/make_pedsum_am_golden.py
    pixi run --manifest-path $S/pixi.toml python -I tools/make_pedsum_am_golden.py --snapshot $S \
        --set gate --out tests/golden/pedsum_am_142adf300d5b/gate
    pixi run python tools/pedsum_am_gate.py --out docs/gates/assortative-mating/gate.json

Cases: the 28 parity fixtures and 46 gate cases.  Six gate cases are
pedsum's own benchmark pedigrees (`benchmarks/generate_assortative_mating.py`)
at 10^4 Mating Pairs, in configurations `a` (one binary trait, unstratified)
and `b` (continuous and binary, birth-year strata), 3 seeds each, with 999
permutations and 1000 bootstrap draws.  The other 40 are random
multi-generation pedigrees (60 to 500 founders) with random trait kinds,
strata, missingness, thin-stratum rule, draw counts and seeds.  `gate.json`
holds the numbers.

## Discrete outcomes

Every count, reason, estimator, order, `boundary` flag, CI method, presence of
`p` and `ci`, failure histogram, `draws_used` and `stopped_early` matches in
every case but one, and every p-value is identical.  The exception is a
deliberate fix of a pedsum fragility:

- `small_strat`, cell (1, 1), stratified polychoric SE: pedsum withholds it
  (`sandwich_undefined`), pg-phenotype reports 0.1054.  The sandwich's `A_rr`
  sums `count * (d2/pi - score^2)` over every cell of a populated stratum
  table, zero counts included.  Father stratum 9 has two equal thresholds (an
  empty level), so one zero-count cell's phi2 corner difference is pure
  cancellation noise: 6.9e-18 at pedsum's rho-hat, a different residue at ours
  (3.6e-16 away).  `noise / 1e-300` squared overflows, `0 * inf` is NaN, and
  pedsum withholds the SE.  An empty cell is no term of the estimating
  equations, so pg-phenotype skips empty cells in `A_rr` and in the
  threshold cross terms.  pedsum patched the same way, in the pinned
  snapshot, gives 0.10541470792145655 for this SE; pg-phenotype gives
  0.10541470792145699.  Every other SE in the fixture already matched.

## Floats (D3)

| class | tolerance | largest gap |
|---|---|---|
| p-values | exact | 0 |
| closed form (Pearson, Spearman, phi, point-biserial, odds ratio), within-person, network share | 1e-12 x max(1, \|pedsum\|) | 4.5e-15 (odds-ratio SE) |
| latent (tetrachoric, polychoric, biserial, polyserial) value, SE, CI | 1e-7, Brent's `xatol` | 3.2e-10 (stratified polychoric CI) |

The primitives are bit-identical to SciPy and numba on a 3110-value grid
(`crates/core/src/assortative/primitives_parity.rs`): SciPy's cephes `ndtr`
and xsf `owens_t`, glibc's `erfc` under the kernels' normal CDF, AS 241,
Phi2, phi2 and its rho-derivative, and SciPy's bounded Brent.  What remains
is summation order: pedsum sums the latent NLL and score with NumPy's BLAS
`dot`, pg-phenotype in index order.  Newton absorbs that.  A fit that falls
back to bounded Brent (here `near_perfect_ord_cont_depth`, two bracket exits
in pedsum) stops at a point that depends on exact NLL comparisons, within its
own tolerance.  Emulating one CPU's OpenBLAS `ddot` kernel to close the gap was
rejected: pedsum's own results move with the BLAS kernel its CPU selects.

Paths exercised in pedsum (its `FIT_OUTCOMES`): Newton, the Hessian and
bracket hand-overs to Brent (the iteration limit never fired), and 400 full
refits of nonconcave one-step bootstrap draws (`perfect_2x2_boot`,
`separated_biserial_boot`).  Thread count does not change any result
(`test_results_do_not_depend_on_the_thread_count`, 1 vs 4 threads).

## Per estimator

| estimator (form) | quantity | n | max abs gap | max rel gap | where |
|---|---|---|---|---|---|
| biserial (crude) | ci | 52 | 5.6e-17 | 2.9e-15 | degenerate_stratum:mate_correlation[1].crude.biserial.ci[1] |
| biserial (crude) | p_perm | 26 | 0.0e+00 | 0.0e+00 | - |
| biserial (crude) | se | 25 | 3.5e-17 | 2.0e-15 | gate_b_1e4_s2:mate_correlation[1].crude.biserial.se |
| biserial (crude) | value | 26 | 0.0e+00 | 0.0e+00 | - |
| biserial (stratified) | ci | 28 | 9.7e-17 | 1.5e-15 | thin_birthyear:mate_correlation[1].stratified.biserial.ci[0] |
| biserial (stratified) | p_perm | 14 | 0.0e+00 | 0.0e+00 | - |
| biserial (stratified) | se | 14 | 4.2e-17 | 2.3e-15 | thin_birthyear:mate_correlation[1].stratified.biserial.se |
| biserial (stratified) | value | 14 | 0.0e+00 | 0.0e+00 | - |
| odds_ratio (crude) | ci | 95 | 3.1e-15 | 1.2e-15 | bin_only:mate_correlation[0].crude.odds_ratio.ci[1] |
| odds_ratio (crude) | se | 48 | 4.5e-15 | 5.1e-14 | gate_a_1e4_s2:mate_correlation[0].crude.odds_ratio.se |
| odds_ratio (crude) | value | 48 | 0.0e+00 | 0.0e+00 | - |
| pearson (crude) | ci | 90 | 2.2e-16 | 2.9e-15 | gate_random_34:mate_correlation[0].crude.pearson.ci[1] |
| pearson (crude) | p_perm | 41 | 0.0e+00 | 0.0e+00 | - |
| pearson (crude) | se | 45 | 2.8e-17 | 1.9e-15 | dangling_and_phantoms:mate_correlation[0].crude.pearson.se |
| pearson (crude) | value | 47 | 0.0e+00 | 0.0e+00 | - |
| pearson (stratified) | ci | 70 | 6.2e-17 | 1.5e-14 | gate_random_09:mate_correlation[3].stratified.pearson.ci[1] |
| pearson (stratified) | p_perm | 30 | 0.0e+00 | 0.0e+00 | - |
| pearson (stratified) | se | 35 | 3.5e-17 | 1.5e-15 | gate_random_09:mate_correlation[3].stratified.pearson.se |
| pearson (stratified) | value | 35 | 0.0e+00 | 0.0e+00 | - |
| phi (crude) | ci | 100 | 5.6e-16 | 6.8e-14 | gate_random_11:mate_correlation[3].crude.phi.ci[1] |
| phi (crude) | se | 49 | 5.4e-16 | 4.0e-14 | gate_a_1e4_s0:mate_correlation[0].crude.phi.se |
| phi (crude) | value | 51 | 0.0e+00 | 0.0e+00 | - |
| point_biserial (crude) | ci | 52 | 5.6e-17 | 4.1e-15 | gate_random_03:mate_correlation[1].crude.point_biserial.ci[1] |
| point_biserial (crude) | se | 26 | 4.2e-17 | 1.6e-15 | gate_random_14:mate_correlation[2].crude.point_biserial.se |
| point_biserial (crude) | value | 26 | 0.0e+00 | 0.0e+00 | - |
| polychoric (crude) | ci | 104 | 3.1e-11 | 3.8e-11 | near_perfect_ord_cont_depth:mate_correlation[0].crude.polychoric.ci[0] |
| polychoric (crude) | p_perm | 42 | 0.0e+00 | 0.0e+00 | - |
| polychoric (crude) | se | 52 | 1.3e-11 | 3.7e-10 | near_perfect_ord_cont_depth:mate_correlation[0].crude.polychoric.se |
| polychoric (crude) | value | 52 | 3.0e-11 | 3.4e-11 | near_perfect_ord_cont_depth:mate_correlation[0].crude.polychoric.rho |
| polychoric (stratified) | ci | 56 | 3.2e-10 | 3.9e-10 | near_perfect_ord_cont_depth:mate_correlation[0].stratified.polychoric.ci[0] |
| polychoric (stratified) | p_perm | 29 | 0.0e+00 | 0.0e+00 | - |
| polychoric (stratified) | se | 31 | 1.4e-10 | 4.0e-09 | near_perfect_ord_cont_depth:mate_correlation[0].stratified.polychoric.se |
| polychoric (stratified) | value | 32 | 3.1e-10 | 3.5e-10 | near_perfect_ord_cont_depth:mate_correlation[0].stratified.polychoric.rho |
| polyserial (crude) | ci | 44 | 5.6e-17 | 3.2e-15 | cont_ord_birthyear:mate_correlation[1].crude.polyserial.ci[1] |
| polyserial (crude) | p_perm | 18 | 0.0e+00 | 0.0e+00 | - |
| polyserial (crude) | se | 22 | 3.5e-17 | 7.5e-16 | no_permutations:mate_correlation[1].crude.polyserial.se |
| polyserial (crude) | value | 22 | 0.0e+00 | 0.0e+00 | - |
| polyserial (stratified) | ci | 32 | 8.3e-17 | 2.1e-15 | gate_random_07:mate_correlation[2].stratified.polyserial.ci[1] |
| polyserial (stratified) | p_perm | 16 | 0.0e+00 | 0.0e+00 | - |
| polyserial (stratified) | se | 18 | 2.8e-17 | 6.2e-16 | near_perfect_ord_cont_depth:mate_correlation[2].stratified.polyserial.se |
| polyserial (stratified) | value | 18 | 0.0e+00 | 0.0e+00 | - |
| spearman (crude) | ci | 62 | 0.0e+00 | 0.0e+00 | - |
| spearman (crude) | value | 47 | 0.0e+00 | 0.0e+00 | - |
| tetrachoric (crude) | ci | 100 | 6.0e-16 | 2.8e-14 | gate_random_11:mate_correlation[0].crude.tetrachoric.ci[0] |
| tetrachoric (crude) | p_perm | 45 | 0.0e+00 | 0.0e+00 | - |
| tetrachoric (crude) | se | 48 | 9.5e-16 | 3.6e-14 | gate_a_1e4_s0:mate_correlation[0].crude.tetrachoric.se |
| tetrachoric (crude) | value | 51 | 0.0e+00 | 0.0e+00 | - |
| tetrachoric (stratified) | ci | 46 | 2.8e-16 | 2.2e-14 | gate_random_13:mate_correlation[0].stratified.tetrachoric.ci[1] |
| tetrachoric (stratified) | p_perm | 19 | 0.0e+00 | 0.0e+00 | - |
| tetrachoric (stratified) | se | 24 | 1.1e-16 | 4.3e-15 | gate_b_1e4_s1:mate_correlation[3].stratified.tetrachoric.se |
| tetrachoric (stratified) | value | 24 | 3.3e-16 | 8.6e-14 | small_strat:mate_correlation[0].stratified.tetrachoric.rho |
| within_person | value | 86 | 2.2e-16 | 6.5e-16 | gate_random_04:within_person.fathers.rho |
