# Relative-Pair Correlation and ACE Fit reference

`pg_phenotype.relative_pairs.correlation` (R: `relative_pairs_correlation`)
computes the **Relative-Pair Correlation**: the correlation of one or two
traits between the members of each Relationship Category's Relative Pairs,
or a Category Group's.  `pg_phenotype.relative_pairs.ace` (R:
`relative_pairs_ace`) solves the **ACE Fit** from those cells: each trait's
additive genetic (A), common environment (C) and unique environment (E)
shares and, with two traits, the cross-trait correlations rA, rC and rE.
This page describes what both compute and what they return.  The reasons
behind the design are in [ADR 0009](adr/0009-relative-pair-correlation.md),
[ADR 0012](adr/0012-relative-pair-strata-two-traits-and-the-ace-fit.md) and,
for the SE, [ADR 0013](adr/0013-relative-pair-se-clusters-by-common-ancestor.md).
The comparison with simACE is in
[the gate report](gates/relative-pairs/README.md).

## Call

```python
from pg_phenotype import Pedigree, Trait
from pg_phenotype import relative_pairs

ped = Pedigree(pedigree)  # or pass the columns
res = relative_pairs.correlation(
    ped,
    [Trait(liability1), Trait(liability2)],  # or one Trait
    categories=["MZ", "FS", "MHS", "PHS", {"PO": ["MO", "FO"]}, "1C"],
    stratum=generation,  # None by default
    se=True,  # False by default
)
fs = res.cell("FS")  # trait 0 in both members
fs.primary.value, fs.primary.se, fs.n, fs.stratified.estimate.value, fs.strata[3].n
res.cell("FS", first_trait=0, second_trait=1).n_entries
res.within_person.primary.value

fit = relative_pairs.ace(res, shared_environment=["MZ", "FS", "MHS"])
fit.traits[0].A, fit.traits[0].C, fit.traits[0].E
fit.cross.rA, fit.cross.rC, fit.cross.rE
```

```r
ped <- pedigree(ped_df)
res <- relative_pairs_correlation(ped, list(trait(ped_df$liability1), trait(ped_df$liability2)),
                                  categories = list("MZ", "FS", "MHS", "PHS", PO = c("MO", "FO"), "1C"),
                                  stratum = ped_df$generation, se = TRUE)
fit <- relative_pairs_ace(res, shared_environment = c("MZ", "FS", "MHS"))
```

| Argument | Meaning | Default |
|---|---|---|
| `pedigree` | A `Pedigree`, or columns `id`, `mother`, `father` (`-1` or NA when missing) and optional `twin`, `sex`, which pedigree-graph's rules validate for this call alone | required |
| `traits` | A continuous or binary `Trait` on the pedigree's rows, or a sequence of one or two of one kind | required |
| `categories` | The entries, in order: Relationship Category codes (`"FS"`, `"MO"`, `"1C"`) and mappings of a Category Group's name to its codes (`{"PO": ["MO", "FO"]}`; R: a named list element) | `None`: every category up to degree 3 |
| `stratum` | One integer label per row (a generation, a birth cohort), NA where unknown | `None` (no strata) |
| `memory_budget_bytes` | The bound on the engine's accumulators, with two traits the within-person accumulator, and with `se` the per-cluster sums | 2^30 (1 GiB) |
| `se` | Whether every estimate gets its cluster sandwich SE; see [SE](#se) | `False` |

The thread count is the package budget (`configure_threads`, or
`PG_PHENOTYPE_THREADS`).  Traits and strata are matched to the pedigree's
rows by position; see [Rows and alignment](../README.md#rows-and-alignment).

## Pairs

A Relative Pair is two rows in one Relationship Category, as
pedigree-graph's `relationship_moments` finds them.  The pairs are reduced
to exact sums as they are found and never stored.

A symmetric category's pairs (MZ, FS, MHS, PHS, 1C, ...) enter in both
orders.  A same-trait cell is then intraclass: one mean and SD, or one
threshold, for both members, and the result does not depend on row order.
For the tetrachoric this two-step estimate equals the equal-threshold fit
on the pairs counted once.

A directed category's pairs (MO, FO, GP, Av, ...) enter once, in its roles'
order, junior first: offspring then parent.  A Category Group's categories
are all symmetric, or all directed with the same roles once mother and
father both read as parent, so PO's roles are offspring and parent.  A
pair that fits a directed category in both orientations, possible in a
pedigree with loops, enters once in the lower row's orientation.

Each entry's cells use the pairs where the values they read are present.
There is no minimum number of pairs.  A requested category with no pairs
still gets its cells, with `n = 0`.

## Traits and cells

A binary trait holds 0 and 1, and a continuous trait finite numbers.  NA,
`None` and NaN are missing.  An all-missing or constant trait gives cells
with a [reason](#reasons), not an error.

| Trait kind | Primary | Also reported |
|---|---|---|
| continuous | `pearson` | |
| binary | `tetrachoric` | `phi` |

`tetrachoric` is the Polychoric Fit of a 2 x 2 table (ADR 0011).  It
estimates the correlation of a bivariate-normal liability cut at a
threshold, so it reads the liability scale.  `phi` is Pearson's r of the 0
and 1 codes, the observed scale, and depends on prevalence.

One trait gives one cell per entry.  Two traits give one cell per trait
combination, `first_trait` read in a pair's first member and
`second_trait` in the second:

| Entry | Cells, as `(first_trait, second_trait)` |
|---|---|
| symmetric | (0, 0), (1, 1), (0, 1) |
| directed | (0, 0), (0, 1), (1, 0), (1, 1) |

A symmetric cross-trait cell (0, 1) enters each pair in both orientations:
trait 0 of one member against trait 1 of the other.  Each trait gets its
own threshold, or mean and SD.  A directed entry's (0, 1) and (1, 0) read
its roles apart: (0, 1) is the offspring's trait 0 against the parent's
trait 1.

`res.cell(code)` returns the (0, 0) cell, and
`res.cell(code, first_trait=i, second_trait=j)` any other.

With two traits, `res.within_person` holds the **Within-Person Cross-Trait
Correlation**: each row's trait 0 against its trait 1, over every row with
both observed.  It is a `Cell` with category `"within_person"`, no members
or roles, and the same fields.

Continuous values are centred on their midrange and quantized, each trait
on its own, to 43 bits of their largest magnitude.  `settings` reports
each trait's `quantization_exponents` e and `quantization_steps` 2^-e.  A
trait whose step exceeds 2^-20 of its interquartile range (its SD when the
IQR is 0) raises `insufficient_precision`.  `constant_value` means
constant at the step.

## Strata

With `stratum`, each cell also gets a cell per known stratum label and a
stratified fit.  The pooled fields do not change.

A directed pair is in its first role's stratum, the offspring's or the
descendant's, whatever the second member's stratum.  A symmetric pair is in
stratum s only when both members are in s.  A symmetric pair whose members
are in two different known strata is **cross-stratum** and is in no stratum
cell.  The within-person cell puts each row in its own stratum.

`cell.strata` maps each known label of any row, ascending, to a
`StratumCell` with the pooled fields (`n`, `n_dropped`, `n_entries`,
`concordance`, `table`, `crude`) over that stratum's pairs.  A stratum
with no pairs has `n = 0`.

`cell.stratified` is the primary estimator fitted once over every block of
two known strata, rows by the first member's stratum and columns by the
second's (both orders for a symmetric cell).  A binary side gets a
threshold per stratum and side, a continuous side is standardised within
each stratum and side (mean and 1/N SD), and one correlation is fitted.
Cross-stratum pairs enter it, each member at its own stratum's threshold
or mean.  A pair with either stratum unknown does not.  A stratum whose
side is constant over the blocks kept carries no information on the
correlation, so its blocks are left out, and the rule repeats until no
side is constant.  Phi has no stratified form.

## Counts

Every count is of unordered pairs, except `n_entries`.  A pair is complete
in an orientation when trait `first_trait` of one member and
`second_trait` of the other are both observed.

| Count | Where | Meaning |
|---|---|---|
| `n` | `Cell`, `StratumCell` | Pairs with a complete orientation |
| `n_dropped` | `Cell`, `StratumCell` | The entry's other pairs |
| `n_entries` | symmetric cross-trait `Cell`, `StratumCell` | Complete orientations, between `n` and `2n`: what the estimates read. `None` elsewhere |
| `n_cross_stratum` | `Cell` with strata | Complete symmetric pairs whose members are in two known strata |
| `n_unknown_stratum` | `Cell` with strata | Complete pairs in no stratum cell for an unknown stratum: a directed pair's first member's, either member's of a symmetric pair |
| `Stratified.n` | `Cell.stratified` | Complete pairs fitted: both strata known, outside every stratum left out |
| `Stratified.n_entries` | symmetric cross-trait `Cell.stratified` | Complete orientations fitted, those in the blocks kept. `None` elsewhere |
| `n_degenerate_stratum` | `Cell.stratified` | Complete pairs with both strata known that a constant stratum left out |

A cell's `n` is its strata's `n` plus `n_cross_stratum` plus
`n_unknown_stratum`.  Without `stratum`, `strata` and `stratified` are
`None` and the two stratum counts are 0.

A binary symmetric same-trait cell reports its **Concordance**,
`concordance = (both 0, discordant, both 1)`.  Every other binary cell
reports `table`, its 2 x 2 counts: rows by trait `first_trait` of one
member (a directed cell's first role), columns by trait `second_trait` of
the other, a symmetric cell's two orientations summed.  A continuous cell
reports neither.

## SE

With `se=True` every estimate of a Relationship Category's cells, pooled,
stratum and `stratified`, gets a cluster sandwich SE.  Relative Pairs share
people, so an SE that treats pairs as independent is too small.  Two pairs
are in one cluster when they share a member or share their most recent
common ancestor (MRCA), by inclusion-exclusion:

```text
meat = sum over MRCAs m of Phi_m^2 + sum over members i of S_i^2
       - sum over (member, MRCA) of S_(i,m)^2
SE   = sqrt(meat)
```

`Phi` and `S` sum the influences on the estimate of the entries (complete
orientations) in each cluster, each influence divided by the fit's
entries.  With both orientations of every pair this is the pair-level
formula of ADR 0013; a symmetric cross-trait cell whose pairs are complete
in one orientation only counts each entry once, as its estimate does.
There is no small-sample factor.

A pair's MRCA is the ancestor member of a lineal pair (MO, GP, ...), the
shared parent of a half relationship, and the shared mother of a full one
(MZ by the twins' mother), found up the category's meioses in (mother,
father) order.  The shared parent can be an external parent, an id with no
row.  A pair with no common ancestor, MZ twins without parents, is its own
cluster.

The influence is Pearson's (phi's) `u v - r/2 (u^2 + v^2)`, the two-step
tetrachoric's threshold-corrected cell influence, and for `stratified`
the influence of its stratum-standardised Pearson or its Polychoric Fit.

`se` is `None` exactly when `se_reason` says why:

| `se_reason` | Cause |
|---|---|
| `no_estimate` | The estimate has no value; its `reason` says why |
| `boundary` | The tetrachoric fit stopped at a bound |
| `sandwich_undefined` | The two-step fit's Hessian is not negative definite, or the meat is not a number |
| `degenerate_meat` | The meat is zero to within rounding: over n entries, at most max(1024, 4 sqrt(n)) eps of its scale, its three sums of squares added as if every entry had the influence 1/n.  A cell whose pairs share one MRCA, or whose estimate is ±1, has a meat of zero |
| `negative_meat` | The inclusion-exclusion meat is negative |
| `within_person` | The within-person cell, which has no cluster design |

The SE is computed in the same pass over the pedigree as the cells, from
exact per-cluster sums kept beside the engine's.  These cost memory, held
to `memory_budget_bytes`, and time.  On a 120k-row pedigree the call
takes 1.5-1.6 times as long with one trait and 2.4-3.7 times with two
traits and generation strata at one thread, and 1.6-1.8 and 2.1-3.3 times
at eight.  Without `se` the call does neither.  The SEs do not depend on the thread count.  Near the budget
edge, whether the call raises `memory_budget_exceeded` can.

## Result

`correlation` returns a frozen `RelativePairCorrelation`.  In R the same
values come as nested lists with the same names, `NULL` where Python has
`None`, and trait indices count from 1.

| `RelativePairCorrelation` field | Contents |
|---|---|
| `cells` | Each entry's cells in request order, each entry's in the order above |
| `within_person` | With two traits, the within-person `Cell`; `None` with one |
| `settings` | `categories` (the entries' names), `trait_kind`, `memory_budget_bytes`, `threads`, `se` (whether the SE was asked for), `quantization_exponents`, `quantization_steps` (both `None` for binary traits) |
| `method` | `entry`, `traits`, `within_person`, `estimators`, `quantization`, `strata`, `se`: the method in words |
| `metadata` | `pg_phenotype_version`, `pedigree_graph_core_rev` |

| `Cell` field | Contents |
|---|---|
| `category`, `members` | The code or group name, and its pedigree-graph codes |
| `symmetric`, `roles` | Whether pairs enter in both orders; a directed cell's roles, first then second |
| `first_trait`, `second_trait` | The trait read in each member |
| `n`, `n_dropped`, `n_entries` | See [Counts](#counts) |
| `concordance`, `table` | Binary cells only; see [Counts](#counts) |
| `crude` | One `Estimate` per estimator, the primary first. `cell["phi"]` looks one up by name, `cell.primary` is the first |
| `strata` | With `stratum`: label to `StratumCell` |
| `n_cross_stratum`, `n_unknown_stratum` | With `stratum`; else 0 |
| `stratified` | With `stratum`: `Stratified(n, n_entries, n_degenerate_stratum, estimate)` |

An `Estimate` holds `estimator`, `primary`, `value` or `reason`,
`boundary`, the tetrachoric fit's bound flag (`None` for other
estimators), and `se` or `se_reason` (see [SE](#se); both `None`
without `se`, and R leaves both out).

## ACE Fit

### Model

On the correlation scale, each trait's variance 1, a same-trait cell k of
trait t expects

```text
r_k = 2 phi_k A_t + s_k C_t        E_t = 1 - A_t - C_t
```

`phi_k = 2^-(degree+1)` is the nominal kinship of the cell's categories
(0.5 for MZ, 0.25 for FS, MO and FO, 0.125 for MHS and PHS, 0.0625 for 1C).
`s_k` is 1 for a **Shared-Environment Category**, else 0.  A Category
Group's categories must share a degree and be all shared or none.

With two traits, the cross-trait cells and the within-person cell give the
cross-trait covariances:

```text
r12_k = 2 phi_k sA + s_k sC        r_wp = sA + sC + sE
rA = sA / sqrt(A_1 A_2)    rC = sC / sqrt(C_1 C_2)    rE = sE / sqrt(E_1 E_2)
```

Each trait's unknowns are the weighted least-squares solution over its
same-trait cells.  The cross-trait unknowns are solved over the cross-trait
cells (a directed entry's (0, 1) and (1, 0) both) and the within-person
cell.  A row's weight counts what is behind the value it reads, the pooled
cell or `Stratified`.  A same-trait row weighs its pairs, `n`, and the
within-person row its rows, `n`.  A cross-trait row weighs its complete
orientations, so each counts once: a symmetric cell's `n_entries`, a
directed cell's `n`, since each of its (0, 1) and (1, 0) cells holds one
orientation of a pair.  A directed pair complete both ways then weighs as
much in the cross solve as a symmetric one.  Values are unconstrained.
The ACE Fit has no SE yet (issue #19).

A component that no row informs does not take the others with it.  When
no same-trait row that entered is a Shared-Environment Category's, C is
`not_identified` and A is solved from those rows alone.  When no
cross-trait row that entered is a Shared-Environment Category's, sC and
sE appear only in the within-person row, so both are `not_identified`,
that row is left out, and sA is solved from the cross-trait rows.  Any
other design with fewer independent rows than unknowns gives
`not_identified` for every unknown of that solve.

### Arguments

| Argument | Meaning | Default |
|---|---|---|
| `result` | A `RelativePairCorrelation` | required |
| `categories` | The entries to fit, by code or group name. The within-person cell enters the cross solve either way | `None`: every entry |
| `components` | `"A"`, `"C"` or both. A component left out is fixed at 0 | `("A", "C")` |
| `shared_environment` | The Shared-Environment Categories' codes; required with C | `None` |
| `estimator` | `"primary"` (Pearson or tetrachoric) or `"phi"` | `"primary"` |
| `estimate` | `"stratified"` or `"pooled"` | `None`: stratified when the result has strata, else pooled |

The library never assumes which categories share C.  Under simACE's rule,
where the household follows the mother, they are MZ, FS and MHS, and PHS
is not.  A cell without a stratified form leaves a stratified fit with the
reason `no_stratified_form`.

### Recipes

A call that identifies the model exactly is a closed form.  `r` is the
value the fit reads: by default the stratified fit's when the result has
strata.

| Estimator | Call | Gives |
|---|---|---|
| sibs | `ace(res, categories=["FS"], components="A")` | `A = 2 r_FS` |
| Falconer | `ace(res, categories=["MZ", "FS"], shared_environment=["MZ", "FS"])` | `A = 2 (r_MZ - r_FS)`, `C = 2 r_FS - r_MZ` |
| half-sibs | `ace(res, categories=["PHS"], components="A")` | `A = 4 r_PHS` |
| cousins | `ace(res, categories=["1C"], components="A")` | `A = 8 r_1C` |
| one category | `ace(res, categories=[code], components="A")` | `A = r / (2 phi)` |

Two half-sib cells, `categories=["MHS", "PHS"], components="A"`, give 4
times their pair-weighted mean.  Under household by mother MHS also carries
C.  The parent-offspring regression slope is
`pg_phenotype.regression.parent_offspring`'s.

### Liability and observed scale

For binary traits the primary estimate is the tetrachoric, so the shares
are on the liability scale with no Dempster-Lerner step.
`estimator="phi", estimate="pooled"` reads phi and gives observed-scale
shares.  Phi with the stratified form raises `no_stratified_form`.

### Fit result

`ace` returns a frozen `AceFit`.

| `AceFit` field | Contents |
|---|---|
| `traits` | One `TraitShares` per trait |
| `cross` | With two traits, `CrossShares`; `None` with one |
| `rows` | One `AceRow` per cell considered, in the result's order, the within-person cell last |
| `settings` | `categories`, `components`, `shared_environment`, `estimator`, `estimate` (resolved) |
| `method` | `model`, `weights`, `solve`, `E`, `cross`, `se`: the method in words |
| `metadata` | `pg_phenotype_version`, `pedigree_graph_core_rev` |

| Type | Fields |
|---|---|
| `TraitShares` | `trait`, `A`, `C`, `E`, each with its `_reason`, and `rows` (the entries that entered) |
| `CrossShares` | `sA`, `sC`, `sE`, `rA`, `rC`, `rE`, each with its `_reason`; `rA_out_of_range`, `rC_out_of_range`, `rE_out_of_range` (`abs(r) > 1`); `rows` |
| `AceRow` | `entry`, `first_trait`, `second_trait`, `within_person`, `kinship`, `shared`, `value`, `weight`, `used`, `reason` |

## Reasons

A value that is `None` comes with one of these reasons.

| Reason | Where | Meaning |
|---|---|---|
| `no_complete_pairs` | estimate, `AceRow` | No pair has a complete orientation. A stratified fit: no complete pair has both strata known |
| `constant_margin` | estimate | A binary side shows one level. A stratified fit: every stratum was left out |
| `constant_value` | estimate | A continuous side is constant at the quantization step. A stratified fit: every stratum was left out |
| `no_stratified_form` | `AceRow` | A stratified fit was asked of a cell without one |
| `not_requested` | share | The component is not in `components` |
| `not_identified` | share | The rows that entered do not determine it: no row informs it (C without a shared row; sC and sE without a shared cross-trait row), or the solve has fewer independent rows than unknowns. `sE` and `rE` need the within-person cell |
| `nonpositive_component` | `rA`, `rC`, `rE` | The component is at most 0 in a trait |

An `AceRow` whose value is undefined carries the estimate's own reason and
does not enter the fit.  An estimate's `se_reason` is listed under
[SE](#se).

## Errors

Errors carry a stable `code` and `fields` (R: `$code` and `$fields`,
positions counted from 1).  `correlation` checks the pedigree, then the
traits in order, then `stratum`, then `categories`, then the budget.

| Code | Class | Cause |
|---|---|---|
| `trait_count` | `ValidationError` | Not one or two traits |
| `trait_length_mismatch` | `ValidationError` | A trait is not one value per pedigree row |
| `unsupported_trait_kind` | `ValidationError` | An ordinal or categorical trait, or two traits of different kinds |
| `invalid_trait_value` | `ValidationError` | A binary value not 0 or 1, a continuous value not finite |
| `stratum_length_mismatch` | `ValidationError` | `stratum` is not one label per row |
| `unknown_category` | `ValidationError` | A code that is not a Relationship Category's, in `categories` or in `ace`'s `shared_environment` |
| `duplicate_category` | `ValidationError` | A category in two entries, or twice in one group |
| `duplicate_entry_name` | `ValidationError` | Two entries with one name |
| `reserved_group_name` | `ValidationError` | A group named like a category code or `within_person` |
| `empty_category_group` | `ValidationError` | A group with no codes |
| `mixed_category_group` | `ValidationError` | A group mixing symmetric and directed categories, or directed roles |
| `insufficient_precision` | `ValidationError` | A continuous trait's quantization step above 2^-20 of its spread |
| `parameter_out_of_range` | `ParameterError` | `memory_budget_bytes` below 1 |
| `memory_budget_exceeded` | `ResourceError` | The accumulators, or with `se` the per-cluster sums (operation `relative_pairs_se`), do not fit the budget |
| `allocation_failed` | `ResourceError` | With `se`, the allocator refused the per-cluster sums |
| `unknown_entry` | `ValidationError` | An `ace` `categories` name that is no entry of the result |
| `mixed_kinship` | `ValidationError` | A group fitted whose categories differ in degree |
| `mixed_shared_environment` | `ValidationError` | A group fitted with categories both in and out of `shared_environment` |
| `missing_shared_environment` | `ValueError` | C fitted without `shared_environment` |
| `no_stratified_form` | `ValueError` | The stratified form with phi, or of a result no cell of which has one |
| `missing_estimator` | `ValueError` | Phi asked of a continuous cell |

A Python `ValueError` carries the message alone.  R raises these as
`pgphenotype_usage_error` with the code.  `ace` also raises `ValueError`
for a `components`, `estimator` or `estimate` that is not one of its
values.

## Caveats

- The SE is a Wald SE: no CI is reported, and in a sparse binary cell (under
  about one expected concordant pair) the Wald interval fails under every
  cluster design ([ADR 0013](adr/0013-relative-pair-se-clusters-by-common-ancestor.md)).
  The ACE Fit has no SE yet (#19).
- The ACE model assumes no assortative mating, no dominance, and C shared
  in full by the declared categories and not at all by the others.
- Count weights let a large cell dominate the fit, and the fit
  reports no residual or goodness of fit.
- The latent estimator assumes a bivariate-normal liability.  A censored
  age-dependent diagnosis counts a relative not yet diagnosed as
  unaffected, which attenuates the tetrachoric.
