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

**Position**:
Where an entry sits in a list the caller passed (probands, ids, categories,
a CIP table) or among a Pedigree's rows.  Python and Rust count positions
from 0, R from 1, and an error reports them as its host counts, in its
message and its fields alike.  A level code is a value, not a Position.
_Avoid_: index (for a caller-facing place), offset

**Trait**:
One phenotype column aligned to pedigree rows, missing where unknown, with a
kind: continuous, binary, ordinal or categorical.  A trait is values only;
what a method needs beyond them (an age, a CIP) it takes separately.
_Avoid_: phenotype column, variable

**Admission**:
The checks a method makes of a Trait before reading it, in one order for
every method: its length, its kind, each present value, then, for a
method that needs the trait to vary (the Mate Correlation), a present
value, two distinct values, and every level taken.  Each method declares
the kinds it reads; one that reads a single kind reports a trait of
another as a kind mismatch.  A method checks its other columns (an age,
strata) after the trait.
_Avoid_: validation (of a trait), value rule

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
One trait of one member against one trait of the other: in the Mate
Correlation, the mother's against the father's; in the Relative-Pair
Correlation, a category's first member's against its second's.  One trait
gives one cell.  Two traits give four, in the order (0, 0), (0, 1), (1, 0),
(1, 1), except for a symmetric category, which gives three: (0, 0), (1, 1),
(0, 1).
_Avoid_: pair of traits, combination

**Polychoric Fit**:
The latent correlation of a table of two ordered sides, by two-step
maximum likelihood: each side's thresholds from its own margins, then
the correlation of a bivariate normal cut at them.  On a 2 x 2 table
it is the tetrachoric.  A fit is at a **boundary** when it lies at
the search bound or on a likelihood plateau reaching it; its sandwich
SE and Wald CI are then withheld.
_Avoid_: latent fit, tetrachoric fit (for tables larger than 2 x 2)

**Mate Correlation**:
The correlation between the mother's trait and the father's trait over a
cell's Mating Pairs, by the estimator its trait kinds call for (Pearson,
tetrachoric, polychoric, biserial, polyserial and their closed-form
companions).
_Avoid_: spousal correlation, assortment coefficient

**Within-Person Cross-Trait Correlation**:
With two traits, the correlation of a person's two traits, once per person.
The Mate Correlation takes it per sex over the people with an analysed
Mating Pair; the Relative-Pair Correlation over every row, and within each
stratum.

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
The correlation between the members of a category's (or a Category
Group's) Relative Pairs, of one trait in both or, with two traits, of one
trait in the first member against the other in the second, by the
estimator the traits' kind calls for: Pearson for continuous traits,
tetrachoric and phi for binary ones.  In both orders a symmetric category's
same-trait correlation is intraclass: one mean and spread, or one
threshold, for both members.  Its **stratified** form fits the primary
estimator once over all strata: each stratum gets its own mean and
spread, or threshold, on each side of the pair, and the strata share one
correlation.
_Avoid_: relative correlation, familial correlation

**MRCA** (of a Relative Pair):
The most recent common ancestor the pair's relationship runs through: the
ancestor member of a lineal pair, the shared parent of a half
relationship, the shared mother of a full one (MZ: the twins' mother).  It
can be an external parent, an id with no row.  The Relative-Pair
Correlation's SE clusters two pairs together when they share an MRCA or a
member.
_Avoid_: common ancestor (a pair can have several), founder

**Stratum** (of a Relative Pair):
The stratum label that places a Relative Pair in a stratum cell: a directed
pair's first role's label; a symmetric pair's shared label, when both
members carry the same one.  A symmetric pair whose members carry two
different known labels is **cross-stratum** and is in no stratum cell.

**Shared-Environment Category**:
A Relationship Category whose pairs the caller declares to share the common
environment C (under household by mother: MZ, FS, MHS).  The library never
assumes which categories these are.

**ACE Fit**:
Additive genetic (A), common environment (C) and unique environment (E)
variance shares solved from Relative-Pair Correlations as linear functions
of each category's nominal kinship and whether it is a Shared-Environment
Category; with two traits, also the cross-trait genetic, common-environment
and unique-environment correlations rA, rC, rE.
_Avoid_: Falconer estimate (a special case), heritability regression

**Cell Covariance**:
The covariance between the estimates of a Relative-Pair Correlation's
cells, one matrix per form the ACE Fit reads (pooled, phi, stratified).
Two pairs count together when they share an MRCA or a member, whichever
cells they are in; a person's within-person entry and a pair count together
when the person is one of the pair's members.  Its diagonal is each
estimate's SE squared.
_Avoid_: cross-cell SE, Sigma (in prose)

**Kinship Kernel**:
The Within-Person Cross-Trait Correlation's SE design.  Each person is one
entry, and every two people related at degree 2 or closer (MZ, MO, FO, FS,
MHS, PHS, GP, Av) add the product of their influences, whichever categories
were requested.  It is not a partition into clusters, and its meat can come
out negative.
_Avoid_: kinship cluster, relatedness cluster

**Weights** (of the ACE Fit):
How the ACE Fit weighs the cells it solves over: **pairs** (what entered
each cell, counted once), **inverse variance** (each cell by 1 over its
variance) or **GLS** (the inverse of the Cell Covariance of the cells that
entered).  The default is chosen per solve: GLS for the primary estimator
when the result carries the Cell Covariance of the form read and the
entered cells' block has a Cholesky factor, else pairs.  Each solve
records the weights it used.
_Avoid_: count weights (say pairs), precision weights

**Concordance**:
A symmetric category's binary Relative Pairs counted unordered: both
unaffected, discordant, both affected.  A directed category counts its
pairs in a 2 x 2 table in its roles' order instead.
