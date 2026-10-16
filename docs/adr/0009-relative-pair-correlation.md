# ADR 0009: The Relative-Pair Correlation double-enters symmetric pairs and reports no SE yet

**Status:** accepted; amended 2026-10-15 by ADR 0012 (strata, two traits
and `n_entries`)
**Date:** 2026-10-10
**Context:** issue #1; `docs/relative-pairs-se-research.md`

## Context

simACE and fitACE each compute correlations between relatives per
relationship category.  simACE uses `simace/analysis/stats/correlations.py`
(Pearson, phi and tetrachoric, pooled, by generation, by sex and across
traits) on pedigree-graph's `relationship_moments`, and fitACE uses
`fitace/ltm/falconer.py` (tetrachoric and SE per category group, through
simACE's `tetrachoric_corr_se`).  Both count each unordered pair once, the
lower row first, with a mean, variance or threshold for each side.  A
symmetric category's result therefore depends on row order.  Both drop a
category below a fixed number of pairs (simACE 10, fitACE 50), and simACE
reports a Pearson of 0.0 for a constant side.

Relative Pairs share people: a sibship of four holds six FS pairs, and a
mother is the parent in each of her children's MO pairs and the child in her
own.  An SE that treats pairs as independent is too small.  No published SE
fits one connected pedigree (`docs/relative-pairs-se-research.md`).

## Decision

- The method is the Relative-Pair Correlation:
  `relative_pairs::correlation` (Python
  `pg_phenotype.relative_pairs.correlation`, R
  `relative_pairs_correlation`).  It takes a Pedigree or its columns
  (ADR 0006) and one Trait.
- The first unit covers one trait, with no strata and no cross-trait cells.
  A continuous trait gets Pearson.  A binary trait gets the tetrachoric
  (primary) and phi.  Ordinal and categorical traits raise
  `unsupported_trait_kind`, so a later polychoric does not change what an
  existing call means.
- `categories=` takes Relationship Category codes and named Category Groups
  (such as PO for MO and FO).  Each entry gives one cell.  A group's
  categories are all symmetric or all directed with the same roles, and no
  category is in two entries.  A group is not named like any category code
  (`reserved_group_name`), so a cell named MO always holds MO's pairs.
  Errors give each entry's position in the caller's request.  Roles are
  compared with mother and father
  both read as parent, so pedigree-graph's (offspring, mother) and
  (offspring, father) make PO's (offspring, parent).  The default is every
  category up to degree 3.
- The trait is checked as `regression::parent_offspring` checks it: its
  length, then its kind (ordinal raises here, although regression reads
  it), then every present value (`invalid_trait_value` for a
  binary code other than 0 or 1, or a continuous value that is not finite).
- A symmetric category's pairs enter in both orders: Pearson is intraclass
  (one mean and variance) and the tetrachoric fits a symmetric table with
  one threshold.  Its two-step point estimate equals the equal-threshold fit
  on the pairs counted once.  A directed category's pairs enter once, in
  its roles' order (offspring then parent).
- `n` counts unordered pairs in every cell.  A binary symmetric cell reports
  its Concordance (both 0, discordant, both 1).  A binary directed cell
  reports its 2 x 2 table in role order.  The table the fit uses is never
  shown.  A pair with either member's value missing is dropped and counted.
- There is no minimum number of pairs.  A value is `None` only when it is
  undefined (`no_complete_pairs`, `constant_margin`, `constant_value`), with
  that reason.  Callers keep their own floors on `n`.  A requested category
  with no pairs still gets its cell, with `n = 0`.
- The tetrachoric is ADR 0011's fit (the same `Tables`, Newton and Brent,
  and boundary flag) on the fitted table.
- No SE or CI in the first unit.  The cluster design is chosen by a
  calibration study: naive-pair, person (dyadic), Mate Network and Mate
  Network or shared person, each against the empirical SD across simACE
  replicates.  A later ADR records the choice.
- The pairs come from pedigree-graph-core's `relationship_moments`, called
  from pg-phenotype's core with `Symmetric::Both`.  A binary value is a
  label (0, 1, missing) on each member.  A continuous value has a label for
  missing, and the present values are centred on their midrange
  (`min / 2 + max / 2`) before pedigree-graph quantizes them to 43 bits of
  their largest magnitude.  Pearson is shift-invariant, so centring changes
  nothing but the step, which becomes `(max - min) / 2^44` up to a power of
  two in place of `max |x| / 2^43`: values near 1e12 spaced 0.01 apart stay
  distinct.  The step is reported.  A trait whose step exceeds `2^-20` of
  its interquartile range (its standard deviation when the IQR is 0)
  raises `insufficient_precision`: one value of 1e12 among a million
  standard normals sets a step of 1/16, which the standard deviation (about
  1e9) would let through.  Pairs are never stored.
  `memory_budget_bytes=` (default 1 GiB, as in pedigree-graph's Python)
  bounds the engine's accumulators.

Rejected:

- Single entry, as simACE and fitACE do it.  Its result depends on row
  order, and it fits a side-specific threshold that has no meaning for a
  symmetric category.
- Single entry with pooled margins.  It gives the same point estimates as
  double entry, but it needs its own equal-threshold fit in place of ADR
  0011's.
- Quantizing the raw values.  Four sibling values near 1e12 spaced 0.01
  apart all round to one integer, and Pearson reports a constant trait.
- An SE with every pair its own cluster, labelled as such.  It is known to
  be too small wherever people share pairs, which is the problem simACE #48
  had with simACE's tetrachoric SE.
- An SE clustered by Mate Network or shared person in the first unit.  The
  design has no precedent and has not been calibrated.
- Codes only, without Category Groups.  fitACE's Falconer step reports PO
  and HS, and a tetrachoric cannot be merged after fitting.
- A `min_pairs` parameter.  Both callers would override its default.
- `pair_blocks`, which stores every pair: memory grows with the pair count.

## Consequences

- Results for symmetric categories differ from simACE's and fitACE's by
  design.  A gate compares against an independent path (pedigree-graph's
  pair list, NumPy's double-entry Pearson and `correlation.tetrachoric` on
  the doubled table).  Directed categories can also be compared with
  simACE.
- Symmetric cells do not depend on row order.  Directed cells do not
  either, except for a pair that fits a directed category in both
  orientations (possible in a pedigree with loops; a pair can be HAv
  either way).  pedigree-graph enters it once, in the lower row's
  orientation, so reordering the rows can move it between the 2 x 2
  table's off-diagonal cells.  This is pedigree-graph's rule, documented,
  and tested as such.
- A `constant_value` reason means constant at the reported quantization
  step.
- simACE can switch over after the first unit, because its plots read no
  per-category SE.  fitACE keeps simACE's tetrachoric until the SE unit.
- The SE unit reads its sums from the same pair stream, through a custom
  reducer, with no stored pairs: per-key sums suffice
  (`docs/relative-pairs-se-research.md`, Q5).  With double entry the
  unordered pair is the smallest cluster.
- Strata, cross-trait cells, polychoric and biserial come in later units on
  the same cell.
