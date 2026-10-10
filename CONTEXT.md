# pg-phenotype

Phenotypes in the context of a pedigree, as an extension of pedigree-graph:
PA-FGRS (Pearson-Aitken family genetic risk scores; Dybdahl Krebs et al.,
Am J Hum Genet 2024, doi:10.1016/j.ajhg.2024.09.009), assortative mating,
and further methods.  A caller passes a pedigree and traits; kinship and
relationships come from pedigree-graph-core.

## Language

**Pedigree**:
The rows of one pedigree (id, mother, father, and optional twin and sex)
checked once by pedigree-graph's rules, in the order they were given.
Traits, ages, strata and probands are matched to its rows by position;
lining them up is the caller's job, since a valid Pedigree says nothing
about whether a trait belongs to it.  Its `align` and `rows` look ids up
and reject an id that is not a row's or is listed twice.  A method given pedigree columns
instead checks them into a Pedigree for that call alone.
_Avoid_: graph, pedigree frame

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
lifetime prevalence; the liability threshold is `Phi^-1(1 - K)`.  Given
as a table, or estimated from ages and survival status by Aalen-Johansen,
death competing, at the ages where a case occurs.  Between its ages it
reads linearly or, for an estimator's step function, as a step.
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

**Mating Pair**:
A distinct mother and father, both pedigree rows, of at least one child.
Pairs are ordered by (mother id, father id).
_Avoid_: couple, mates, union

**Mate Network**:
A connected set of Mating Pairs through shared mates: a remating mother or
father joins two pairs.  The sandwich SE clusters by Mate Network, and the
bootstrap resamples whole networks.  In Parent-Offspring Regression a
parent in no Mating Pair is a network of its own, and networks are always
the whole pedigree's.
_Avoid_: family, component

**Cell**:
One mother trait against one father trait.  One trait gives one cell; two
traits give four, in the order (0, 0), (0, 1), (1, 0), (1, 1).
_Avoid_: pair of traits, combination

**Mate Correlation**:
The correlation between the mother's trait and the father's trait over a
cell's Mating Pairs, by the estimator its trait kinds call for (Pearson,
tetrachoric, polychoric, biserial, polyserial and their closed-form
companions).
_Avoid_: spousal correlation, assortment coefficient

**Within-Person Cross-Trait Correlation**:
With two traits, the correlation of a person's two traits, per sex, once per
distinct person with an analysed Mating Pair.

**Parent-Offspring Regression**:
The ordinary least-squares line of a child's trait on a parent value: the
**midparent** `(mother + father) / 2`, the mother's value, or the father's.
A child is analysed when the parents its line needs are pedigree rows and
its own and their values are observed.  The slope is reported raw.
_Avoid_: PO correlation, heritability regression

**Relationship Category**:
pedigree-graph's class of a pair of rows by their closest relationship (MZ,
FS, MHS, PHS, MO, FO, 1C, GP, Av, ...).  A symmetric category's two members
play the same role; a directed one names its roles, junior first (offspring
then parent, niece or nephew then aunt or uncle).
_Avoid_: relationship type, relationship code

**Category Group**:
Several Relationship Categories given one name and analysed as one, such as
PO for MO and FO.  Its categories are all symmetric, or all directed with
the same roles once mother and father both count as parent: PO's roles are
offspring and parent.  Its name is never a Relationship Category's code.

**Relative Pair**:
Two rows in one Relationship Category.  A pair in a symmetric category
enters in both orders; a pair in a directed category enters once, in its
roles' order.
_Avoid_: relative (PA-FGRS's term), dyad, relationship pair

**Relative-Pair Correlation**:
The correlation of one trait between the members of a category's (or a
Category Group's) Relative Pairs, by the estimator the trait's kind calls
for: Pearson for a continuous trait, tetrachoric and phi for a binary one.
In both orders a symmetric category's correlation is intraclass: one mean
and spread, or one threshold, for both members.
_Avoid_: relative correlation, familial correlation

**Concordance**:
A symmetric category's binary Relative Pairs counted unordered: both
unaffected, discordant, both affected.  A directed category counts its
pairs in a 2 x 2 table in its roles' order instead.
