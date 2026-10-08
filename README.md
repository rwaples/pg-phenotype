# pg-phenotype

Phenotypes in the context of a pedigree, as an extension of
[pedigree-graph](https://github.com/rwaples/pedigree-graph): a Rust core with
Python and R bindings.  Relationships and kinship come from pedigree-graph's
core; each method is a module on top.

| Method | Module | Status |
|---|---|---|
| PA-FGRS family genetic risk scores (Dybdahl Krebs et al., Am J Hum Genet 2024, doi:10.1016/j.ajhg.2024.09.009) | `pg_phenotype.pafgrs` | available |
| Assortative mating: the Mate Correlation of one or two traits, ported from pedsum #13 | `pg_phenotype.assortative` | available |

## Python

```python
import pg_phenotype as pgp
from pg_phenotype import pafgrs

trait = pgp.Trait(dx, kind="binary")  # one value per pedigree row
prep = pafgrs.prepare(pedigree, ndegree=2)  # once per pedigree
cip = pafgrs.Cip(ages, cip_values)  # K is the last value
scores = pafgrs.score_univariate(prep, trait, age=dx_age, cip=cip, h2=0.4)
scores.est, scores.var, scores.n_relatives, scores.metadata

both = pafgrs.score_bivariate(prep, (t1, t2), age=(a1, a2), cip=(c1, c2), h2=(0.4, 0.6), rg=0.5)
```

- `pedigree`: columns `id`, `mother`, `father` (`-1` or NA when missing) and
  optional `twin`, `sex` (`0` female, `1` male), as a polars or pandas frame
  or a mapping of arrays.  It is validated by pedigree-graph's rules.
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

### Assortative mating

```python
from pg_phenotype.assortative import mate_correlation

res = mate_correlation(pedigree, [pgp.Trait(liab), pgp.Trait(dx)], stratum=birth_decade, bootstrap=1000, seed=1)
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

## R

```r
library(pgphenotype)

dx <- trait(ped$dx, kind = "binary")
prep <- pafgrs_prepare(ped, ndegree = 2L)
cip <- pafgrs_cip(ages, cip_values)
scores <- pafgrs_score_univariate(prep, dx, age = ped$dx_age, cip = cip, h2 = 0.4)
both <- pafgrs_score_bivariate(prep, list(t1, t2), ages = list(a1, a2),
                               cips = list(c1, c2), h2 = c(0.4, 0.6), rg = 0.5)
am <- assortative_mate_correlation(ped, list(trait(ped$liab), trait(ped$dx)),
                                   stratum = ped$birth_decade, bootstrap = 1000, seed = 1)
```

The R package runs the same core as Python and returns the same scores bit
for bit.  Errors are conditions of class `pgphenotype_error` (with
`pgphenotype_validation_error` and friends) carrying `$code` and `$fields`,
positions 1-based.

## Correctness (PA-FGRS)

- Rust against NumPy reference cores (fitACE's, verbatim) given the same
  covariance and order: within 1e-10.
- Against BioPsyk/PAFGRS's own `pa_fgrs` on stored fixtures: within 1e-10
  (1.1e-15 observed).
- Every kinship value bit-identical to pedigree-graph's `pair_kinship`.

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

Design decisions are in `docs/adr/`; vocabulary in `CONTEXT.md`.

## Development

```bash
pixi run test        # rebuilds the extension with test hooks, then pytest
pixi run test-rust
pixi run -e r r-test
pixi run test-against-pg ../pedigree-graph   # every suite against a pg checkout
```

To release a version, follow [How to release pg-phenotype](docs/releasing.md).
