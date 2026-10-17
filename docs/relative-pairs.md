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
[ADR 0012](adr/0012-relative-pair-strata-two-traits-and-the-ace-fit.md),
for the SE [ADR 0013](adr/0013-relative-pair-se-clusters-by-common-ancestor.md),
and for the covariance, the Within-Person SE and the ACE Fit's SE and
weights [ADR 0014](adr/0014-ace-fit-se-from-the-cells-covariance.md).
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
    covariance=True,  # False by default; needs se=True
)
fs = res.cell("FS")  # trait 0 in both members
fs.primary.value, fs.primary.se, fs.n, fs.stratified.estimate.value, fs.strata[3].n
res.cell("FS", first_trait=0, second_trait=1).n_entries
res.within_person.primary.value, res.within_person.primary.se
res.covariance.stratified  # (n, n): the cells in order, then within_person

fit = relative_pairs.ace(res, shared_environment=["MZ", "FS", "MHS"])  # gls per solve: the result has the covariance
fit.traits[0].A, fit.traits[0].A_se, fit.traits[0].C, fit.traits[0].E
fit.cross.rA, fit.cross.rA_se, fit.cross.rC, fit.cross.rE
fit.traits[0].weights, fit.cross.weights  # "gls", or "pairs" where gls has no Cholesky factor
```

```r
ped <- pedigree(ped_df)
res <- relative_pairs_correlation(ped, list(trait(ped_df$liability1), trait(ped_df$liability2)),
                                  categories = list("MZ", "FS", "MHS", "PHS", PO = c("MO", "FO"), "1C"),
                                  stratum = ped_df$generation, se = TRUE, covariance = TRUE)
fit <- relative_pairs_ace(res, shared_environment = c("MZ", "FS", "MHS"))
```

| Argument | Meaning | Default |
|---|---|---|
| `pedigree` | A `Pedigree`, or columns `id`, `mother`, `father` (`-1` or NA when missing) and optional `twin`, `sex`, which pedigree-graph's rules validate for this call alone | required |
| `traits` | A continuous or binary `Trait` on the pedigree's rows, or a sequence of one or two of one kind | required |
| `categories` | The entries, in order: Relationship Category codes (`"FS"`, `"MO"`, `"1C"`) and mappings of a Category Group's name to its codes (`{"PO": ["MO", "FO"]}`; R: a named list element) | `None`: every category up to degree 3 |
| `stratum` | One integer label per row (a generation, a birth cohort), NA where unknown | `None` (no strata) |
| `memory_budget_bytes` | The bound on the engine's accumulators, with two traits the within-person accumulator, with `se` the per-cluster sums, and with `covariance` the per-key sums, the matrices and the kinship kernel | 2^30 (1 GiB) |
| `se` | Whether every estimate gets its cluster sandwich SE; see [SE](#se) | `False` |
| `covariance` | With `se`, whether the result carries the covariance of the estimates `ace` reads, and with two traits the Within-Person SE; see [Covariance](#covariance) | `False` |

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
| `negative_meat` | The inclusion-exclusion meat is negative, or a within-person estimate's Kinship Kernel is (see [Within-Person SE](#within-person-se)) |
| `within_person` | A within-person estimate under `se=True` alone.  Its SE is the kinship kernel, which only `covariance=True` computes |

The SE is computed in the same pass over the pedigree as the cells, from
exact per-cluster sums kept beside the engine's.  These cost memory, held
to `memory_budget_bytes`, and time.  On a 120k-row pedigree the call
takes 1.5-1.6 times as long with one trait and 2.4-3.7 times with two
traits and generation strata at one thread, and 1.6-1.8 and 2.1-3.3 times
at eight.  Without `se` the call does neither.  The SEs do not depend on
the thread count.  Near the budget edge, whether the call raises
`memory_budget_exceeded` can.

## Covariance

The ACE Fit pools many cells, and those cells share people and MRCAs
across categories, so its SE needs the covariance between the cells'
estimates.  `covariance=True` (with `se=True`; alone it raises
`ValueError`) adds `res.covariance`, a frozen `CellCovariance` with one
matrix per form `ace` reads:

| Field | Estimates | Present |
|---|---|---|
| `pooled` | Each cell's pooled primary estimate, `crude[0]` | always |
| `phi` | Each cell's pooled phi | binary traits |
| `stratified` | Each cell's `stratified` estimate | with `stratum` |

Each matrix is a read-only `(n, n)` NumPy array (R: a matrix with
dimnames such as `"FS:1:2"` and `"within_person:1:2"`).  Rows and columns
are `res.cells` in order, then `within_person` with two traits, so n is
`len(res.cells)`, plus 1 with two traits.  Entry `(j, k)` is
`Cov(r_j, r_k)`: the SE's three sums joined across the two estimates, so
two entries count once when they share an MRCA or a member, whichever
cells they are in.  A within-person row and a pair count together when
the row is one of the pair's members.  Stratum cells and blocks across
forms are left out, because one fit reads one form.

The diagonal is each estimate's meat: `sqrt(cov[j, j])` equals that
estimate's `se` bit for bit.  The row and column of an estimate with no SE
are NaN, its diagonal included.

The matrices do not depend on the thread count or on how the sweeps were
batched.  Their sums come from the SE's sweeps and are held until the end
of the call.  They count against `memory_budget_bytes`: an estimate of
them that does not fit raises `memory_budget_exceeded` with operation
`relative_pairs_covariance`, after the pass and before the sweeps.  On the
120k-row pedigree above, `covariance=True` adds 3-6% to the `se=True`
call with one trait and 20-35% with two traits and generation strata, the
Within-Person SE included.  With two traits and `categories=["MZ"]` it
takes 1.4-2.3 times the `se=True` call, since the kernel traverses every
degree-2 category.  `settings` reports `covariance`.

### Within-Person SE

With two traits and `covariance=True`, every within-person estimate,
pooled, stratum and `stratified`, gets an SE.  Each row is one entry, its
own pair.  Rows that are relatives are not independent, so the meat is a
**Kinship Kernel**:

```text
meat = sum over rows i of psi_i^2 + 2 x sum over relative pairs (a, b) of psi_a psi_b
```

The pairs are every relative pair of degree at most 2 whose members are
both rows of the estimate: MZ, MO, FO, FS, MHS, PHS, GP and Av.  The pass
traverses these categories whether `categories` asks for them or not, and
the requested cells do not change.  So the SE does not depend on the
request.  Continuous influences are quantized to integers before the pass
and their products summed exactly, so the SE does not depend on the
thread count.

The kernel is not positive semidefinite: a grandmother whose grandchildren
are in different sibships can give a negative meat, reported as
`negative_meat`.  Its degenerate test uses the scale (n + 2P) / n^2 over
the estimate's n rows and P kernel pairs.  On 600 simACE reps
(`docs/gates/relative-pairs-ace-se/`) SD(r̂)/SE is 0.88-1.07 and coverage
0.89-0.98, and no within-person estimate lost its SE.  Under `se=True`
alone the within-person estimates have no SE (`within_person`): the
kernel would cost the `se=True` call more than 5%.

## Result

`correlation` returns a frozen `RelativePairCorrelation`.  In R the same
values come as nested lists with the same names, `NULL` where Python has
`None`, and trait indices count from 1.

| `RelativePairCorrelation` field | Contents |
|---|---|
| `cells` | Each entry's cells in request order, each entry's in the order above |
| `within_person` | With two traits, the within-person `Cell`; `None` with one |
| `covariance` | With `covariance=True`, the `CellCovariance`; see [Covariance](#covariance). `None` without |
| `settings` | `categories` (the entries' names), `trait_kind`, `memory_budget_bytes`, `threads`, `se` (whether the SE was asked for), `covariance` (whether the covariance was), `quantization_exponents`, `quantization_steps` (both `None` for binary traits) |
| `method` | `entry`, `traits`, `within_person`, `estimators`, `quantization`, `strata`, `se`, `covariance`: the method in words |
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
cell.  Values are unconstrained.  The weights are one of three; see
[Weights](#weights).

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
| `weights` | `"pairs"`, `"inverse_variance"` or `"gls"`; see [Weights](#weights) | `None`: `"pairs"` for phi or without the covariance of the estimate read; else `"gls"` per solve, `"pairs"` for a solve whose covariance has no Cholesky factor |

The library never assumes which categories share C.  Under simACE's rule,
where the household follows the mother, they are MZ, FS and MHS, and PHS
is not.  A cell without a stratified form leaves a stratified fit with the
reason `no_stratified_form`.

### Weights

| `weights` | A row's weight | Needs |
|---|---|---|
| `"pairs"` | What is behind the value it reads, the pooled cell or `Stratified`: a same-trait row's pairs, `n`; the within-person row's rows, `n`; a cross-trait row's complete orientations, each once (a symmetric cell's `n_entries`, a directed cell's `n`) | nothing |
| `"inverse_variance"` | `1 / Var(r_k)`, the covariance's diagonal | the covariance of the estimate read |
| `"gls"` | The inverse of the covariance of the rows that entered, through its Cholesky factor | the covariance of the estimate read |

Under `"pairs"` a directed pair complete both ways weighs as much in the
cross solve as a symmetric one, since each of its (0, 1) and (1, 0) cells
holds one orientation.  Under `"inverse_variance"` and `"gls"` a row whose
estimate has no SE cannot be weighted, so it leaves the fit with the row
reason `no_se`.  `weights="gls"` asked for refuses a covariance block with
no Cholesky factor: the solve's values and SEs are
`covariance_not_positive_definite`, with no fallback.
`"inverse_variance"` or `"gls"` without the covariance of the estimate
read raises `missing_covariance`.

Under `"pairs"` a row whose estimate has no SE still enters, so every value
that reads it gets the SE reason `row_without_se`.  The row's variance is
part of the value's, and it is unknown.  This happens with a tiny cell and
with a tetrachoric fit at its boundary, as a small MZ cell's often is.  To
get those SEs, leave the category out with `categories=`, or use
`"inverse_variance"` or `"gls"`, which leave the row out (`no_se`).  Phi
meets this most, since its default is `"pairs"`.

The default, `weights=None`, is resolved in the core, one solve at a time.
It is `"pairs"` for phi and for a result without the covariance of the
estimate read; a result without `covariance=True` fits exactly as in
v0.4.0.  Otherwise each trait's solve and the cross solve take `"gls"`
when the covariance of the rows that entered has a Cholesky factor, and
`"pairs"` when it has none, that solve alone.  Small pedigrees often lack
one: on `tests.pedigrees.random_pedigree(3, n_founders=40)` to `1000` with
two continuous traits, a trait's block has a smallest eigenvalue of -0.007
to -0.042.  Each `TraitShares` and the `CrossShares` record the weighting
their solve used in `weights`.  `settings["weights"]` records the request:
the name given, or `"default"`.

Guidance, from 600 simACE reps
([gate report](gates/relative-pairs-ace-se/README.md)):

- Continuous traits: `"gls"`.  Its spread is 0.78-0.85 of `"pairs"`'s for
  A, E, sA, rA, sE and rE at equal or better calibration.
- Tetrachoric: `"gls"` gains little (0.90-0.97 for A, sA and rA, about 1
  for C and E), and its SD/SE runs higher than under `"pairs"` (median
  up to 1.17 against 1.05).  It keeps an SE where `"pairs"` loses it: a
  boundary MZ tetrachoric has no SE, so under `"pairs"` every value that
  reads it gets `row_without_se`, while `"gls"` leaves the cell out.
- Phi: `"pairs"`.  At prevalence 0.02 `"inverse_variance"` and `"gls"`
  gave SD/SE of 1.54-6.06 for A, C and E and coverage down to 0.01.
- A covariance block that is not positive definite: `"pairs"`, which the
  default takes for that solve.

### The fit's SE

With the result's covariance, every value has an SE.  With the weights
fixed, each solve is linear in the cells it reads, `theta = G r`, so a
value's SE is `sqrt(g Sigma g')`: g its row of G and Sigma the cells'
covariance of the estimate read.  E = 1 - A - C and the cross-trait
correlations take the delta method through the trait and cross solves
together, so their covariance enters rA, rC and rE.  The sum runs over
g's support, the entries above the solve's rounding bound, so a row whose
coefficient is zero does not reach the SE.  The weights are treated as
fixed: GLS's SE does not count the noise in its estimated weights.

Without the covariance of the estimate read, every SE is `None` with
`no_covariance` and the values stand.  `AceRow.variance` is the row's
`Sigma_kk`.

On 600 simACE reps C, sC, sE and rE calibrate under every weighting:
median SD/SE 0.89-1.03, coverage 0.96-0.98.  Continuous A, E, sA and rA
spread 7-21% more than their SE under every weighting (coverage
0.89-0.94).  Against each rep's realized truth their SE is not too small;
the realized truth itself varies across reps by 0.4-1.1 SE.  That fits a
rep-level common shock, such as the realized genetic variance, which no
within-sample SE covers.  The mechanism is unverified, and E2 is short
against its realized truth too.  The SE describes the error around the
sample's realized truth; widen it for the spread across populations.

### Recipes

A call that identifies the model exactly is a closed form, under any
weights.  `r` is the value the fit reads: by default the stratified fit's
when the result has strata.

| Estimator | Call | Gives |
|---|---|---|
| sibs | `ace(res, categories=["FS"], components="A")` | `A = 2 r_FS` |
| Falconer | `ace(res, categories=["MZ", "FS"], shared_environment=["MZ", "FS"])` | `A = 2 (r_MZ - r_FS)`, `C = 2 r_FS - r_MZ` |
| half-sibs | `ace(res, categories=["PHS"], components="A")` | `A = 4 r_PHS` |
| cousins | `ace(res, categories=["1C"], components="A")` | `A = 8 r_1C` |
| one category | `ace(res, categories=[code], components="A")` | `A = r / (2 phi)` |

Two half-sib cells, `categories=["MHS", "PHS"], components="A"`, give 4
times their pair-weighted mean under `"pairs"` weights.  Under household by mother MHS also carries
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
| `settings` | `categories`, `components`, `shared_environment`, `estimator`, `estimate` as resolved, and `weights` as asked (`"default"` for `None`) |
| `method` | `model`, `weights`, `solve`, `E`, `cross`, `se`: the method in words |
| `metadata` | `pg_phenotype_version`, `pedigree_graph_core_rev` |

| Type | Fields |
|---|---|
| `TraitShares` | `trait`, `A`, `C`, `E`, each with its `_reason`, `_se` and `_se_reason`; `rows` (the entries that entered); `weights` (the weighting its solve used) |
| `CrossShares` | `sA`, `sC`, `sE`, `rA`, `rC`, `rE`, each with its `_reason`, `_se` and `_se_reason`; `rA_out_of_range`, `rC_out_of_range`, `rE_out_of_range` (`abs(r) > 1`); `rows`; `weights` |
| `AceRow` | `entry`, `first_trait`, `second_trait`, `within_person`, `kinship`, `shared`, `value`, `weight`, `used`, `reason`, `variance` (`Sigma_kk`; `None` without the covariance or an SE) |

Fields added since v0.4.0 (`covariance` on the result, every `_se` and
`_se_reason`, `weights` and `variance`) come last with a `None` default,
so positional construction from v0.4.0 still works.

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
| `no_se` | `AceRow` | In a solve under `"inverse_variance"` or `"gls"`, the estimate has no SE, so the row cannot be weighted |
| `covariance_not_positive_definite` | share, SE | `weights="gls"` asked for, and the covariance of the rows that entered the solve has no Cholesky factor |
| `no_covariance` | SE | The result carries no covariance of the estimate read |
| `row_without_se` | SE | Under `"pairs"`, the value reads a row whose estimate has no SE; see [Weights](#weights) |
| `negative_variance` | SE | `g Sigma g'` is negative: the covariance is not positive semidefinite in the value's direction |

An `AceRow` whose value is undefined carries the estimate's own reason and
does not enter the fit.  An SE whose value is undefined repeats the
value's reason.  An estimate's `se_reason` is listed under [SE](#se).

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
| `memory_budget_exceeded` | `ResourceError` | The accumulators, or with `se` the per-cluster sums (operation `relative_pairs_se`), or with `covariance` the per-key sums, the matrices or the kinship kernel (operation `relative_pairs_covariance`), do not fit the budget |
| `allocation_failed` | `ResourceError` | With `se` or `covariance`, the allocator refused the sums |
| `unknown_entry` | `ValidationError` | An `ace` `categories` name that is no entry of the result |
| `mixed_kinship` | `ValidationError` | A group fitted whose categories differ in degree |
| `mixed_shared_environment` | `ValidationError` | A group fitted with categories both in and out of `shared_environment` |
| `missing_shared_environment` | `ValueError` | C fitted without `shared_environment` |
| `no_stratified_form` | `ValueError` | The stratified form with phi, or of a result no cell of which has one |
| `missing_estimator` | `ValueError` | Phi asked of a continuous cell |
| `missing_covariance` | `ValueError` | `weights` `"inverse_variance"` or `"gls"` without the result's covariance of the estimate read |
| `covariance_shape_mismatch` | `ValidationError` | The result's covariance is not over its cells |

A Python `ValueError` carries the message alone.  R raises these as
`pgphenotype_usage_error` with the code.  `ace` also raises `ValueError`
for a `components`, `estimator`, `estimate` or `weights` that is not one
of its values, and `correlation` for `covariance=True` without
`se=True`.

## Caveats

- The SE is a Wald SE: no CI is reported, and in a sparse binary cell (under
  about one expected concordant pair) the Wald interval fails under every
  cluster design ([ADR 0013](adr/0013-relative-pair-se-clusters-by-common-ancestor.md)).
- The ACE Fit's continuous A, E, sA and rA SEs are 7-21% short of the
  spread across simACE reps, though calibrated against each rep's realized
  truth ([ADR 0014](adr/0014-ace-fit-se-from-the-cells-covariance.md)).
  Its SEs treat the weights as fixed.
- The ACE model assumes no assortative mating, no dominance, and C shared
  in full by the declared categories and not at all by the others.
- Under `"pairs"` a large cell dominates the fit.  The fit reports no
  residual or goodness of fit.
- The latent estimator assumes a bivariate-normal liability.  A censored
  age-dependent diagnosis counts a relative not yet diagnosed as
  unaffected, which attenuates the tetrachoric.
