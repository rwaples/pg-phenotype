# pg-phenotype

Phenotypes in the context of a pedigree, as an extension of
[pedigree-graph](https://github.com/rwaples/pedigree-graph): a Rust core with
Python and R bindings.  Relationships and kinship come from pedigree-graph's
core; each method is a module on top.

| Method | Module | Status |
|---|---|---|
| PA-FGRS family genetic risk scores (Dybdahl Krebs et al., Am J Hum Genet 2024, doi:10.1016/j.ajhg.2024.09.009) | `pg_phenotype.pafgrs` | available |
| Assortative mating: the Mate Correlation of one or two traits, ported from pedsum #13 | `pg_phenotype.assortative` | available |
| Correlation estimators on their own: the tetrachoric correlation of a 2 x 2 table | `pg_phenotype.correlation` | available |
| Parent-offspring regression: midparent, mother and father, with a Mate Network sandwich SE | `pg_phenotype.regression` | available |
| Relative-Pair Correlation: one or two traits' correlation between relatives per relationship category and stratum, and the ACE Fit of A, C, E and rA, rC, rE from it; no SE yet | `pg_phenotype.relative_pairs` | available |

## Python

```python
import pg_phenotype as pgp
from pg_phenotype import pafgrs

ped = pgp.Pedigree(pedigree)  # validated once, for every method
trait = pgp.Trait(dx, kind="binary")  # one value per pedigree row
prep = pafgrs.prepare(ped, ndegree=2)  # once per pedigree and degree
cip = pafgrs.Cip(ages, cip_values)  # K is the last value
cip = pafgrs.Cip.from_aalen_johansen(age, status, entry=entry_age)  # or estimate it
scores = pafgrs.score_univariate(prep, trait, age=dx_age, cip=cip, h2=0.4)
scores.est, scores.var, scores.n_relatives, scores.metadata

both = pafgrs.score_bivariate(prep, (t1, t2), age=(a1, a2), cip=(c1, c2), h2=(0.4, 0.6), rg=0.5)
```

- `pedigree`: columns `id`, `mother`, `father` (`-1` or NA when missing) and
  optional `twin`, `sex` (`0` female, `1` male), as a polars or pandas frame
  or a mapping of arrays.  It is validated by pedigree-graph's rules.
- `Pedigree(pedigree)` validates the columns once.  Every method takes a
  Pedigree or the columns, which it validates for that call alone.  A
  Pedigree helps when you prepare again (another `ndegree` or set of
  probands) or run several methods, or `mate_correlation` several times, on
  one pedigree.  Scoring many traits needs only the prep.
- `Trait(values, kind=None)`: one phenotype column on the pedigree's rows,
  NA where unknown.  The kind is continuous, binary, ordinal or categorical,
  inferred when unambiguous.
- PA-FGRS takes a binary trait.  `age` is the onset age for a case and the
  last observed age for a control; a control without an age is unobserved
  and counted.
- `probands=` restricts the scored rows; relatives are drawn from the whole
  pedigree.  Output is one record per proband in input-row order.
- `configure_threads(n)` or `PG_PHENOTYPE_THREADS` sets the thread budget
  (default 1).  Results are identical for every budget.

Errors are `pg_phenotype.ValidationError`, `ParameterError` and
`ResourceError`, each with a stable `.code` and `.fields`.

### CIP from data

`Cip.from_aalen_johansen(age, status, entry=None)` estimates the CIP with
the Aalen-Johansen estimator: `age` at onset, death or last observation,
`status` 0 censored, 1 case, 2 death (a competing risk), and `entry` the
age observation starts (late entry; `entry < t <= age` is at risk).  The
result is a `Cip` with `interpolation="step"`: between case ages it holds
the last estimate, where `Cip(ages, cip)` interpolates linearly.  K is its
last value.  For a CIP per sex, pass each sex's rows; scoring takes one CIP
per trait
([ADR 0010](docs/adr/0010-aalen-johansen-cip.md)).

### Assortative mating

```python
from pg_phenotype.assortative import mate_correlation

res = mate_correlation(ped, [pgp.Trait(liab), pgp.Trait(dx)], stratum=birth_decade, bootstrap=1000, seed=1)
cell = res.cell(0, 1)  # mothers' trait 0 x fathers' trait 1
cell.primary.value, cell.primary.ci, cell.primary.permutation.p
cell.stratified.result.value  # within birth-decade strata
res.within_person["mothers"]
```

The Mate Correlation is the correlation between the mother's and the
father's value of a trait over a pedigree's Mating Pairs.  Each cell uses
the estimators its trait kinds call for (Pearson, tetrachoric, polychoric,
biserial or polyserial, with their closed-form companions), a Mate Network
sandwich SE, an optional Mate Network bootstrap, and a sequential
father-permutation p-value.  [docs/assortative-mating.md](docs/assortative-mating.md)
describes the inputs, rules and every result field, and
[docs/assortative-mating-design.md](docs/assortative-mating-design.md)
explains the design.

### Tetrachoric correlation

```python
from pg_phenotype.correlation import tetrachoric

res = tetrachoric(x, y)  # paired 0/1 values, NA dropped
res = tetrachoric(table=[[n00, n01], [n10, n11]])  # rows by the x level
res.value, res.se, res.ci, res.boundary, res.reason
```

The Mate Correlation's tetrachoric fit on one table: two-step maximum
likelihood, a `boundary` flag, and the two-step sandwich SE with each pair
its own cluster ([ADR 0011](docs/adr/0011-public-tetrachoric-is-the-mate-correlation-fit.md)).

### Parent-offspring regression

```python
from pg_phenotype.regression import parent_offspring

res = parent_offspring(ped, pgp.Trait(liab), stratum=generation)
fit = res.pooled.midparent  # also .mother, .father
fit.slope, fit.se, fit.ci, fit.p, fit.se_ols, fit.n, fit.n_mate_networks
res.strata[2].father  # children with stratum label 2
parent_offspring(ped, pgp.Trait(liab), lines=["midparent"], inference=False)  # point estimates and se_ols
```

The OLS line of each child's trait on its midparent's, its mother's and its
father's value, pooled and within strata of the child.  `se` is the
cluster-robust sandwich over the whole pedigree's Mate Networks (a parent in
no Mating Pair is a network of its own), with a Wald `p` and `ci`; `se_ols`
is the plain OLS SE.  Slopes are raw: under random mating and no environment
shared by parent and child, the midparent slope estimates h2 and a
single-parent slope h2/2.  The sandwich SE needs many networks
([ADR 0007](docs/adr/0007-parent-offspring-regression.md)).  `lines=` fits
some of the three lines, and `inference=False` skips the networks and the
sandwich; neither moves a slope, `se_ols` or count.

### Relative-Pair Correlation

```python
from pg_phenotype import relative_pairs

res = relative_pairs.correlation(ped, pgp.Trait(dx, kind="binary"), categories=["MZ", "FS", {"PO": ["MO", "FO"]}, "1C"])
fs = res.cell("FS")
fs.primary.value, fs["phi"].value, fs.n, fs.n_dropped, fs.concordance
res.cell("PO").table, res.cell("PO").roles  # rows by offspring; ("offspring", "parent")

both = relative_pairs.correlation(ped, traits=[pgp.Trait(liab1), pgp.Trait(liab2)], stratum=generation)
both.cell("FS", first_trait=0, second_trait=1).stratified.estimate.value  # cross-trait, within generations
fit = relative_pairs.ace(both, shared_environment=["MZ", "FS", "MHS"])
fit.traits[0].A, fit.traits[0].C, fit.traits[0].E, fit.cross.rA
```

One cell per category code or named Category Group (default: every
pedigree-graph category up to degree 3).  A continuous trait gets Pearson's
r; a binary one the tetrachoric (primary) and phi.  A symmetric category's
pairs enter in both orders, so its correlation is intraclass and does not
depend on row order; a directed category's enter once, in its roles' order.
Pairs are reduced as pedigree-graph finds them and never stored
(`memory_budget_bytes=` bounds the accumulators).  There is no minimum pair
count and no SE yet: relative pairs share people, and the cluster design
waits on a calibration study
([ADR 0009](docs/adr/0009-relative-pair-correlation.md)).

`traits=` takes one Trait or two of one kind.  Two traits give each entry a
cell per trait combination, trait `first_trait` in one member against
`second_trait` in the other, and `within_person`, each row's trait 0
against its trait 1.  With `stratum=`, each cell also gets a cell per
stratum and a stratified fit: one correlation, with a threshold, or a mean
and SD, per stratum and side.  A directed pair is in its first role's
stratum, a symmetric pair in a stratum only when both members are.
`relative_pairs.ace` solves `r = 2 kinship A + C` by weighted least squares
over the cells, with C in the categories you name in `shared_environment`;
two traits add rA, rC and rE.  The closed forms, such as Falconer's
`2 (r_MZ - r_FS)`, are calls that identify the model exactly
([reference](docs/relative-pairs.md),
[ADR 0012](docs/adr/0012-relative-pair-strata-two-traits-and-the-ace-fit.md)).

## Rows and alignment

A Pedigree keeps its input's row order: row *i* of the Pedigree is row *i*
of the columns it was built from.  Traits, ages, strata and probands are
matched to it by position, and the caller aligns them.  Move values from
another table onto the Pedigree's rows with `align`:

```python
dx = ped.align(pheno["id"], pheno["dx"])  # NaN where a row has no value
trait = pgp.Trait(dx, kind="binary")
rows = ped.rows(pheno["id"])  # the row of each id, for values align does not take
```

```r
dx <- pedigree_align(ped, pheno$id, pheno$dx)  # NA where a row has no value
rows <- pedigree_rows(ped, pheno$id)  # 1-based
```

Each id must be a Pedigree row's (`unknown_id`) and appear once
(`duplicate_lookup_id`).  The first call builds an id index that the
Pedigree keeps: 12 bytes per row.

> **Warning:** Validation proves that the pedigree is well formed, not that
> a trait belongs to its rows.  A trait shuffled by a sort or a join passes
> every check and gives a plausible, wrong result.

## R

```r
library(pgphenotype)

ped <- pedigree(ped_df)
dx <- trait(ped_df$dx, kind = "binary")
prep <- pafgrs_prepare(ped, ndegree = 2L)
cip <- pafgrs_cip(ages, cip_values)
cip <- pafgrs_cip_aalen_johansen(age, status, entry = entry_age)  # or estimate it
scores <- pafgrs_score_univariate(prep, dx, age = ped_df$dx_age, cip = cip, h2 = 0.4)
both <- pafgrs_score_bivariate(prep, list(t1, t2), ages = list(a1, a2),
                               cips = list(c1, c2), h2 = c(0.4, 0.6), rg = 0.5)
am <- assortative_mate_correlation(ped, list(trait(ped_df$liab), trait(ped_df$dx)),
                                   stratum = ped_df$birth_decade, bootstrap = 1000, seed = 1)
tc <- correlation_tetrachoric(ped_df$dx1, ped_df$dx2)  # or table = matrix(...)
po <- regression_parent_offspring(ped, trait(ped_df$liab), stratum = ped_df$generation)
rp <- relative_pairs_correlation(ped, list(trait(ped_df$liab1), trait(ped_df$liab2)),
                                 categories = list("MZ", "FS", "MHS", PO = c("MO", "FO")),
                                 stratum = ped_df$generation)
fit <- relative_pairs_ace(rp, shared_environment = c("MZ", "FS", "MHS"))
```

The R package runs the same core as Python and returns the same scores bit
for bit.  `pedigree()` keeps the storage type of its ids (integer, double or
`bit64::integer64`) for every result; `length(ped)` and `ped$ids` give its
rows.  Errors are conditions of class `pgphenotype_error` (with
`pgphenotype_validation_error` and friends) carrying `$code` and `$fields`,
positions 1-based in both the fields and the message.  A pedigree error's
message is pedigree-graph's own and counts rows from 0; its fields count
from 1.

## Correctness (PA-FGRS)

- Rust against NumPy reference cores (fitACE's, verbatim) given the same
  covariance and order: within 1e-10.
- Against BioPsyk/PAFGRS's own `pa_fgrs` on stored fixtures: within 1e-10
  (1.1e-15 observed).
- Every kinship value bit-identical to pedigree-graph's `pair_kinship`.

## Correctness (Aalen-Johansen CIP)

- Against simACE's `_aalen_johansen` on 18 pipeline cases with continuous
  ages: bit-identical.  With whole-year ages simACE also counts a person at
  risk at the age they enter; moving entries one ulp earlier makes it
  bit-identical again (`docs/gates/aalen-johansen`).
- Against R survival's `survfit` on all 24 cases: within 4.5e-14 relative.

## Correctness (assortative mating)

- Against pedsum #13 at `142adf3` on 74 stored cases (28 fixtures, pedsum's
  own benchmark pedigrees at 10^4 pairs, and 40 random pedigrees), every
  count, reason, flag and p-value is identical.  Closed-form estimates agree
  within 4.5e-15, and latent ones within 3.2e-10, inside Brent's 1e-7
  tolerance.  One case differs on float noise in pedsum.  The
  [gate report](docs/gates/assortative-mating/README.md) has the details.
- On pedsum's benchmark pedigrees at 10^4 to 10^6 pairs and up to 12
  threads, pg-phenotype's median wall time and peak memory are at most
  0.994 and 0.981 of pedsum's
  ([benchmark report](docs/gates/assortative-mating/benchmark.md)).
- The SciPy and C-library functions pedsum calls are ported bit for bit
  ([ADR 0005](docs/adr/0005-assortative-mating-reproduces-pedsums-numerics.md)).

## Correctness (parent-offspring regression)

- Against simACE `6355ffa` on six simulated pedigrees, slope, intercept, r
  and the OLS SE are bit-identical on simACE's float64 paths.  Its
  validation, which forms the midparent in float32, differs by up to 3.1e-7
  relative.  Every case simACE leaves undefined is one of its floors (10
  trios, 100 offspring, parent SD 1e-10) or undefined here too
  ([gate report](docs/gates/po-regression/README.md)).
- Against a NumPy/SciPy reference with its own union-find networks and a
  matrix-form sandwich (`tests/oracle/po_regression_reference.py`), on
  random pedigrees with remating, missing and external parents: every field
  to 1e-10.

## Correctness (Relative-Pair Correlation)

- Against a reference on pedigree-graph's own pair list
  (`tests/oracle/relative_pairs_reference.py`: NumPy double-entry Pearson
  and phi, `correlation.tetrachoric` on the doubled table), on random
  inbred pedigrees with twins and on three simACE `small_test` reps: every
  count equal, Pearson and phi within 1e-12, the tetrachoric identical.
- Against simACE `6355ffa` on those reps: every pair count and table equal;
  on MO and FO, Pearson and phi within 1e-14 and the tetrachoric within
  1.2e-8.  Symmetric categories differ by design, by up to 3.4e-3 at a
  thousand pairs or more ([gate report](docs/gates/relative-pairs/README.md)).

Design decisions are in `docs/adr/`; vocabulary in `CONTEXT.md`.

## Development

```bash
pixi run test        # rebuilds the extension with test hooks, then pytest (no slow tests)
pixi run test-all    # the same with the slow tests, as CI runs it
pixi run test-rust
pixi run -e r r-test
pixi run test-against-pg ../pedigree-graph   # every suite against a pg checkout
```

To release a version, follow [How to release pg-phenotype](docs/releasing.md).
