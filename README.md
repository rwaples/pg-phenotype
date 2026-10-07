# pg-phenotype

Phenotypes in the context of a pedigree, as an extension of
[pedigree-graph](https://github.com/rwaples/pedigree-graph): a Rust core with
Python and R bindings.  Relationships and kinship come from pedigree-graph's
core; each method is a module on top.

| Method | Module | Status |
|---|---|---|
| PA-FGRS family genetic risk scores (Dybdahl Krebs et al., Am J Hum Genet 2024, doi:10.1016/j.ajhg.2024.09.009) | `pg_phenotype.pafgrs` | available |
| Assortative mating (mate correlations of one or two traits) | `pg_phenotype.assortative` | planned |

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

## R

```r
library(pgphenotype)

dx <- trait(ped$dx, kind = "binary")
prep <- pafgrs_prepare(ped, ndegree = 2L)
cip <- pafgrs_cip(ages, cip_values)
scores <- pafgrs_score_univariate(prep, dx, age = ped$dx_age, cip = cip, h2 = 0.4)
both <- pafgrs_score_bivariate(prep, list(t1, t2), ages = list(a1, a2),
                               cips = list(c1, c2), h2 = c(0.4, 0.6), rg = 0.5)
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

Design decisions are in `docs/adr/`; vocabulary in `CONTEXT.md`.

## Development

```bash
pixi run test        # rebuilds the extension with test hooks, then pytest
pixi run test-rust
pixi run -e r r-test
pixi run test-against-pg ../pedigree-graph   # every suite against a pg checkout
```
