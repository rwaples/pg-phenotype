# Assortative mating reference

`pg_phenotype.assortative.mate_correlation` (R: `assortative_mate_correlation`)
computes the **Mate Correlation** of one or two traits: the correlation
between the mother's value and the father's value of a trait, one
observation per **Mating Pair**.  This page describes what it computes and
what it returns.  The reasons behind the design are in
[the assortative-mating design notes](assortative-mating-design.md), and the
comparison with pedsum #13, the implementation this one is ported from, is
in [the gate report](gates/assortative-mating/README.md) and
[the benchmark report](gates/assortative-mating/benchmark.md).

## Call

```python
from pg_phenotype import Pedigree, Trait
from pg_phenotype.assortative import mate_correlation

ped = Pedigree(pedigree)  # or pass the columns
res = mate_correlation(
    ped,
    [Trait(liability), Trait(dx, kind="binary")],
    stratum=birth_decade,  # None by default
    permutations=999,
    bootstrap=0,
    seed=0,
    min_stratum_networks=10,
    spearman=False,
)
```

```r
ped <- pedigree(ped_df)
res <- assortative_mate_correlation(ped, list(trait(ped_df$liability), trait(ped_df$dx, kind = "binary")),
                                    stratum = ped_df$birth_decade, permutations = 999, bootstrap = 0,
                                    seed = 0, min_stratum_networks = 10, spearman = FALSE)
```

| Argument | Meaning | Default |
|---|---|---|
| `pedigree` | A `Pedigree`, or columns `id`, `mother`, `father` (`-1` or NA when missing) and optional `twin`, `sex`, which pedigree-graph's rules validate for this call alone | required |
| `traits` | One or two `Trait`s on the pedigree's rows, continuous, binary or ordinal | required |
| `stratum` | One integer label per row (a Depth, a birth-year bin), NA where unknown | `None` (unstratified) |
| `permutations` | The most father permutations a primary estimate runs. 0 turns the p-value off | 999 |
| `bootstrap` | Mate Network bootstrap draws. 0 gives sandwich (Wald) CIs | 0 |
| `seed` | Keys every permutation and bootstrap draw. Any int64 (R: a whole number below 2^53 in magnitude) | 0 |
| `min_stratum_networks` | With `stratum`, the fewest Mate Networks a sex x stratum may span. At least 1 | 10 |
| `spearman` | Add `spearman` to a continuous x continuous cell. It ranks every pair, which costs more than the rest of such a cell | `False` |

The thread count is the package budget (`configure_threads`, or
`PG_PHENOTYPE_THREADS`).  The same input and `seed` give the same result
under every budget.

A `Pedigree` keeps the Mating Pairs and Mate Networks that its first call
finds.  A later call on it skips validating the pedigree and finding the
pairs, and, when its strata drop no pair, building the networks.  The
result is the same as with the columns.  Traits and strata are matched to
the pedigree's rows by position; see
[Rows and alignment](../README.md#rows-and-alignment).

## Traits

A binary trait holds the codes 0 and 1.  An ordinal trait holds the codes
`0..k-1`, where `k` is the number of declared levels (`Trait.levels` in
Python, a factor's levels in R) or, without declared levels, the largest
code plus 1.  Every level must be taken by some row.  A continuous trait
holds finite numbers.  NA, `None` and NaN are missing.  A categorical trait
is refused.  [Errors](#errors) lists the codes for each refusal.

## Sample

A Mating Pair is a distinct mother and father, both pedigree rows, of at
least one child.  A child whose parent is missing, or whose parent id names
no row, makes no pair.  A parent who has a row but no trait values (a
phantom parent, for example) makes pairs whose values are missing.  Pairs
are ordered by (mother id, father id), so results do not depend on row
order.

With `stratum`, every pair with a mate of unknown stratum is dropped first.
`sample.n_dropped_unknown_stratum` counts them.

A **Mate Network** is a connected set of Mating Pairs through shared mates:
a mother with two mates joins their pairs, and so does a father.  The Mate
Network is the cluster of the sandwich SE and the unit the bootstrap
resamples.  `sample` reports the networks of every pair, and each cell
reports the networks of its own pairs.

### Cells

One trait gives one cell.  Two traits give four cells, in the order
(mother trait 0, father trait 0), (0, 1), (1, 0), (1, 1). `res.cell(i, j)`
returns cell `(i, j)`.  The two off-diagonal cells are separate estimates
and need not be equal.

Each cell uses the pairs where both of its values are present.
`n_dropped.mother_missing`, `father_missing` and `both_missing` count the
rest.  Means, standard deviations, thresholds and tables are computed over
those pairs, so a father with three mates counts three times.

With two traits, `within_person["mothers"]` and `within_person["fathers"]`
hold the **Within-Person Cross-Trait Correlation**: trait 0 against trait 1
over the distinct people of that sex who have both traits and an analysed
Mating Pair, each person once.  It uses the primary estimator of the cell
with the same trait kinds and has no SE, CI or p-value.

### Strata

`stratum` labels are ranked into codes, unknown first.  A mother and her
mate can be in different strata.  A cell then drops, before any fit, every
sex x stratum whose pairs span fewer than `min_stratum_networks` Mate
Networks, and every sex x stratum whose values are constant.  Dropping one
stratum's pairs can thin or flatten another, so both rules repeat, with the
networks recounted on the kept pairs, until neither drops a pair.
`n_dropped.small_stratum` and `n_dropped.degenerate_stratum` count the
dropped pairs.  The crude and the stratified estimates use the same kept
pairs.

## Estimators

Each cell reports the estimators of its mother x father trait kinds.  The
first, the primary estimator, is the only one with a stratified form and a
permutation p-value.

| Mother x father | Primary | Also reported |
|---|---|---|
| continuous x continuous | `pearson` | `spearman`, with `spearman=True` |
| binary x binary | `tetrachoric` | `odds_ratio`, `phi`, and `table` (the 2 x 2 counts) |
| binary x ordinal, ordinal x binary, ordinal x ordinal | `polychoric` | |
| continuous x binary, binary x continuous | `biserial` | `point_biserial` |
| continuous x ordinal, ordinal x continuous | `polyserial` | |

- `phi` and `point_biserial` are Pearson's r with the binary side coded 0
  and 1.  `spearman` is Pearson's r of the average ranks.  `odds_ratio` is
  `ad / bc` of the 2 x 2 table, and infinite when `bc = 0`.
- `tetrachoric` and `polychoric` are two-step maximum likelihood (Olsson
  1979).  Thresholds come from each side's margins in closed form, then one
  rho maximises the likelihood of the pair table.
- `biserial` and `polyserial` are the two-step estimator of Olsson, Drasgow
  & Dorans (1982).  The continuous side is standardised with the 1/N
  variance, the discrete side gets thresholds from its margins, then rho is
  fitted.
- rho is found by Newton's method on the analytic score, from 0 for the
  table estimators and from the ad hoc estimate of Olsson, Drasgow &
  Dorans (1982, eq 38) for the serial ones.  If the curvature is not
  positive, a step leaves (-0.9999, 0.9999), 12 steps do not converge to
  1e-8, or (table estimators) a populated cell's probability falls below
  1e-12, where it is mostly rounding, a bounded Brent search over that
  interval takes over.
- `boundary` is set on a latent estimate when rho is within 1e-3 of
  ±0.9999, or when the negative log-likelihood at the nearer bound is
  within 1e-6 (relative) of the optimum.  It comes from the fit, never from
  a zero count.

The latent estimators (`tetrachoric`, `polychoric`, `biserial`,
`polyserial`) estimate the correlation of a liability that is assumed
bivariate normal and cut by thresholds into the observed levels.  The other
estimators describe the observed values.  For a binary trait, read
`tetrachoric` unless you want a statement about the 0 and 1 codes
themselves: phi depends on prevalence.  Under a bivariate-normal liability
with correlation 0.3 and the same prevalence in both sexes, phi is 0.194 at
50% prevalence, 0.129 at 10%, and 0.046 at 1%.  The same holds for
`point_biserial` against `biserial`.  Every estimate is a phenotypic
correlation. None is a genetic correlation.

With `stratum`, each cell's `stratified` record holds the primary estimator
with continuous sides standardised within each sex x stratum and discrete
sides given thresholds per sex x stratum, sharing one rho.  It also holds
`n_strata_mothers` and `n_strata_fathers`, the strata the cell's pairs span.

## Inference

Each defined estimate gets an SE and a CI.  Each defined primary estimate
also gets a permutation p-value.  `res.method` states the design in words.

### Standard error

`se` is a cluster-robust sandwich over the cell's Mate Networks, of the
stacked two-step estimating equations: thresholds, stratum means and 1/N
variances first, then the rho score.  The first step's uncertainty
therefore reaches the estimate.  Per-pair influences are summed within each
network, squared, summed over the `G` networks, and scaled by `G/(G-1)`.
Networks are assumed independent. Dependence between them through shared
ancestry or siblings is not modelled.  `se` is on the scale of the estimate,
except for `odds_ratio`, where it is the SE of `log OR`.  `spearman` has no
sandwich.

### Confidence interval

Without a bootstrap, `ci` is the 95% Wald interval from `se`, with
`ci_method` `"sandwich"`.  For a correlation it is
`tanh(atanh(r) ± z se / (1 - r²))`, on the Fisher-z scale. For the odds ratio
it is `exp(log OR ± z se)`.  `z` is 1.959963984540054.  Both forms keep the
bounds in range.

With `bootstrap=N`, each of the N draws resamples the cell's Mate Networks
with replacement, weighting each pair by how often its network was drawn.
Pearson, Spearman, phi, point-biserial and the odds ratio are recomputed
exactly on the draw.  A latent estimate refits its thresholds and stratum
moments on the draw, then takes one Newton step from the estimate with the
draw's score and the full-sample Hessian.  A draw whose full-sample Hessian
is not positive is refit in full.  `ci` runs from the 2.5th to the 97.5th
percentile of the valid draws, as order statistics (NumPy's
`inverted_cdf`), with `ci_method` `"bootstrap"`.  `se` is still the
sandwich.  `bootstrap` counts the draws: `requested`, `valid`, `failed`,
and `failure_reasons`.

### Permutation p-value

Each permutation shuffles the fathers' trait vectors, both traits together,
among distinct fathers in the same block, and leaves the mothers in place.
A block is father stratum x which traits the father has, so a father keeps
one vector across all his mates and each cell keeps its pairs.  The null
hypothesis is that, given the Mating Pair structure, father trait vectors
are exchangeable within each block.  That is stronger than zero
correlation: it also excludes non-linear association and any link between a
father's values and his number of mates.

With `stratum`, a father swaps only within his own stratum, for the crude
estimate as well as the stratified one.  Every p-value then tests
within-stratum exchangeability, while the crude estimate and its CI
describe the pooled correlation.  A cohort trend shared by both mates, with
no assortment within a stratum, can give a crude CI that excludes 0 and a
crude p-value that does not reject.

The statistic of a latent primary (`permutation.statistic`
`"score_at_zero"`) replaces each side's value by its latent score, the
conditional mean of the liability given the level, or the standardised
value of a continuous side.  The statistic is the Pearson r of the mother
and father scores over the cell's pairs: the likelihood's score at rho = 0,
scaled to a correlation.  The father side's thresholds, means and standard
deviations are refit on every permuted sample.  For two continuous traits
the statistic is Pearson's r itself (`"pearson"`).

The p-value is two-sided by magnitude and sequential (Besag & Clifford
1991, closed scheme, `h = 20`).  Draws are read in order.  At the 20th
valid draw with `|T*| >= |T_obs|`, ties included, the test stops with
`p = 20 / l`, where `l` counts the valid draws so far.  A test that never
reaches 20 runs all `permutations` draws and reports `p = (b + 1) / (B + 1)`,
where `B` counts the valid draws and `b` those at least as extreme.  The
sequential p is exact under the null.  A null cell stops after about 98
draws on average at 999 permutations. A cell with a strong signal runs
every draw.

## Result

`mate_correlation` returns a frozen `MateCorrelation`.  In R the same
values come as nested lists with the same names, `NULL` where Python has
`None`. Trait indices count from 0 in both.

| `MateCorrelation` field | Contents |
|---|---|
| `sample` | `n_total` (Mating Pairs), `n_dropped_unknown_stratum`, `n_mate_networks`, `largest_mate_network_share`, `n_mothers_multiple_mates`, `n_fathers_multiple_mates` |
| `cells` | One `Cell` per cell, in the order above |
| `within_person` | Two traits only: `{"mothers": WithinPerson, "fathers": WithinPerson}` |
| `settings` | `permutations`, `bootstrap`, `seed`, `threads`, `ci_level` (0.95), `min_stratum_networks` (`None` without `stratum`), `spearman` |
| `method` | `se_method`, `ci_scale`, `bootstrap_unit`, `bootstrap_assumption`, `bootstrap_method`, `permutation_null`, `permutation_blocks`, `permutation_statistic`, `permutation_stopping`, `permutation_stop_h`: the method in words |
| `metadata` | `pg_phenotype_version`, `pedigree_graph_core_rev` |

| `Cell` field | Contents |
|---|---|
| `mother_trait`, `father_trait` | Indices into `traits` |
| `n` | Analysed Mating Pairs |
| `n_dropped` | `mother_missing`, `father_missing`, `both_missing`, `small_stratum`, `degenerate_stratum` |
| `n_mate_networks`, `largest_mate_network_share` | Over the cell's pairs. The share is `None` without pairs |
| `table` | Binary x binary only: the pair counts, rows by mother level, columns by father level |
| `crude` | One `EstimatorResult` per estimator, the primary first. `cell["odds_ratio"]` looks one up by name, `cell.primary` is the first |
| `stratified` | With `stratum`: `Stratified(result, n_strata_mothers, n_strata_fathers)` |

| `EstimatorResult` field | Contents |
|---|---|
| `estimator`, `primary` | The estimator's name. Whether it is the primary one |
| `value` or `reason` | The estimate, or `None` with the reason it is undefined |
| `boundary` | Latent estimators only: the boundary flag |
| `se`, `se_unavailable_reason` | The sandwich SE, or `None` with a reason |
| `ci`, `ci_method`, `ci_unavailable_reason` | The 95% CI and how it was built, or `None` with a reason |
| `bootstrap` | With `bootstrap > 0`: `Draws(requested, valid, failed, failure_reasons)` |
| `permutation` | Primary estimates only: `Permutation(statistic, p, p_unavailable_reason, draws, seed, n_fixed_fathers, stopped_early, draws_used, sequential_h)` |

In `permutation`, `draws` counts the draws read (`draws_used`, which
includes failed draws), and `stopped_early` says whether that is fewer than
requested.  `n_fixed_fathers` counts the cell's fathers who are alone in
their block, so no permutation moves them.

`WithinPerson` holds `estimator`, `n`, and `value` with `boundary`, or
`reason`.

## Reasons

A value that is `None` comes with one of these reasons.

| Reason | Where | Meaning |
|---|---|---|
| `no_complete_pairs` | estimate, draw | No pair has both values |
| `constant_margin` | estimate, draw | One side takes one value |
| `degenerate_stratum` | estimate, draw | A sex x stratum is constant |
| `empty_category` | estimate, draw | A level a stratum shows in the cell is absent in the draw. Levels are never merged |
| `single_mate_network` | SE, CI | The cell's pairs form one Mate Network |
| `boundary` | SE, CI | A latent fit at a bound, or a correlation of exactly ±1 |
| `infinite_odds_ratio` | SE, CI | A zero cell in the 2 x 2 table |
| `sandwich_undefined` | SE, CI | The rho equation is not concave at the estimate |
| `bootstrap_not_requested` | SE, CI | Spearman without a bootstrap |
| `too_many_failed_draws` | CI | Fewer than 95% of the bootstrap draws are valid |
| `not_requested` | p | `permutations=0` |
| `no_informative_permutations` | p | Every father of the cell is alone in his block |
| `no_valid_permutations` | p | No permuted sample gave a statistic |

Without a bootstrap, `ci_unavailable_reason` is the SE's reason.  With one,
it is `single_mate_network` or `too_many_failed_draws`. A latent estimate
at a bound and an infinite odds ratio are valid draws, so an odds-ratio CI
can reach infinity.

## Errors

The draw counts and `min_stratum_networks` are checked first, then the
pedigree by pedigree-graph's rules, then the traits in order, then
`stratum`.  Errors carry a stable `code` and `fields` (R: `$code` and
`$fields`, positions counted from 1).

| Code | Class | Cause |
|---|---|---|
| `trait_count` | `ValidationError` | Not one or two traits |
| `trait_length_mismatch` | `ValidationError` | A trait is not one value per pedigree row |
| `unsupported_trait_kind` | `ValidationError` | A categorical trait |
| `invalid_trait_value` | `ValidationError` | A binary value not 0 or 1, an ordinal value not a level code, a continuous value not finite |
| `all_missing_trait` | `ValidationError` | A trait missing in every row |
| `constant_trait` | `ValidationError` | A trait with one value in every non-missing row |
| `sparse_ordinal_codes` | `ValidationError` | An ordinal trait without declared levels skips a code below its largest |
| `unused_level` | `ValidationError` | A declared level no row takes |
| `stratum_length_mismatch` | `ValidationError` | `stratum` is not one label per row |
| `invalid_integer_value` | `ValidationError` | A `stratum` label is not a whole number |
| `parameter_out_of_range` | `ParameterError` | A negative `permutations` or `bootstrap`, `min_stratum_networks` below 1, or an integer outside int64 |

A cell whose own pairs make an estimate impossible is not an error: the
estimate is `None` with a [reason](#reasons).

## Caveats

- Mating Pairs are seen only through children.  A partnership with no
  recorded child is absent, so the Mate Correlation describes reproducing
  pairs.
- The latent estimators assume a bivariate-normal liability.
- Censored age-dependent diagnoses are not corrected.  A mate who has not
  yet been diagnosed counts as unaffected.

## References

- Besag, J. & Clifford, P. (1991). Sequential Monte Carlo p-values.
  *Biometrika*, 78(2), 301-304. <https://doi.org/10.1093/biomet/78.2.301>
- Olsson, U. (1979). Maximum likelihood estimation of the polychoric
  correlation coefficient. *Psychometrika*, 44(4), 443-460.
  <https://doi.org/10.1007/BF02296207>
- Olsson, U., Drasgow, F. & Dorans, N. J. (1982). The polyserial
  correlation coefficient. *Psychometrika*, 47(3), 337-347.
  <https://doi.org/10.1007/BF02294164>
- Owen, D. B. (1956). Tables for computing bivariate normal probabilities.
  *The Annals of Mathematical Statistics*, 27(4), 1075-1090.
  <https://doi.org/10.1214/aoms/1177728074>
- Patefield, M. & Tandy, D. (2000). Fast and accurate calculation of
  Owen's T function. *Journal of Statistical Software*, 5(5), 1-25.
  <https://doi.org/10.18637/jss.v005.i05>
- Wichura, M. J. (1988). Algorithm AS 241: The percentage points of the
  normal distribution. *Applied Statistics*, 37(3), 477.
  <https://doi.org/10.2307/2347330>
