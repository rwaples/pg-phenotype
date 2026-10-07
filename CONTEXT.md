# pg-phenotype

Phenotypes in the context of a pedigree, as an extension of pedigree-graph:
PA-FGRS (Pearson-Aitken family genetic risk scores; Dybdahl Krebs et al.,
Am J Hum Genet 2024, doi:10.1016/j.ajhg.2024.09.009), assortative mating,
and further methods.  A caller passes a pedigree and traits; kinship and
relationships come from pedigree-graph-core.

## Language

**Trait**:
One phenotype column aligned to pedigree rows, missing where unknown, with a
kind: continuous, binary, ordinal or categorical.  A trait is values only;
what a method needs beyond them (an age, a CIP) it takes separately.
_Avoid_: phenotype column, variable

**Proband**:
A pedigree row that receives a score.  Every row by default; `probands=`
restricts the set.
_Avoid_: index person, subject

**Relative**:
A row whose closest relationship category to the proband is at most
`ndegree` and whose exact kinship to it is at least `0.5^(ndegree+1) - 1e-6`.
Unphenotyped rows are relatives too; they carry `w = 0`.
_Avoid_: neighbour, family member

**Observation**:
In PA-FGRS, one relative's status on one binary trait.  It informs a score
only when `w > 0`.  A bivariate score has up to two observations per
relative.

**CIP**:
Cumulative incidence proportion by age.  **K** is its last value, the
lifetime prevalence; the liability threshold is `Phi^-1(1 - K)`.
_Avoid_: incidence curve, AOO table

**w**:
The proportion of lifetime risk observed.  `1` for a case, `CIP(age) / K`
clipped to `[0, 1]` for a control, `0` for a missing status or a control
without an age.

**Prep**:
The trait-independent, in-memory relative structure of one pedigree at one
`ndegree`: per proband, its relatives, their kinship to it, and the kinship
among them.  Built once, scored many times.
_Avoid_: cache, index

**Canonical order**:
The order observations are conditioned in (ADR 0004): most informative
first, conditioned from the least informative end, with a unique final key.

**rho_within**:
A person's cross-trait liability correlation.  Defaults to the genetic
covariance `rg * sqrt(h2_1 * h2_2)`.

**Sibship group**:
Probands sharing both represented parents.  Prep walks one kinship triangle
per group because full sibs share nearly all their relatives.

**Terminal**:
A row with a missing parent.  Two rows share an ancestor exactly when they
reach a common terminal, which is how prep proves a kinship is zero.
