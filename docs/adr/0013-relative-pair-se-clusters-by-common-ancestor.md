# ADR 0013: The Relative-Pair SE clusters pairs by their most recent common ancestor or a shared member

**Status:** accepted
**Date:** 2026-10-16
**Context:** issue #18 (cells), issue #19 (the ACE Fit); ADR 0009, ADR 0012;
`docs/relative-pairs-se-research.md`; `docs/gates/relative-pairs-se/`

## Context

ADR 0009 leaves the Relative-Pair Correlation without an SE until a
calibration study picks the cluster design among naive pair, person
(dyadic), Mate Network, and Mate Network or shared person.  simACE #55
measured its every-pair-its-own-cluster tetrachoric SE at a median
SD(r)/SE of 1.12-1.17, up to 1.53.

The study (`tools/relative_pairs_se_calibration.py`, results in
`docs/gates/relative-pairs-se/`) fits each candidate on 600 simACE reps in
six scenarios: serial mating (`mating_lambda` 0.5 and 3) crossed with
prevalence (0.10 and 0.02) at A 0.5/0.4, plus A 0.8/0.5 at prevalence 0.10
only, with C 0 and 0.2-0.3.  High A at low prevalence was not run.  It
covers every category up to degree 3, the PO and HS groups, pooled and
generation-stratum cells, same-trait and cross-trait cells, and Pearson,
tetrachoric and phi.  It adds a fifth candidate, the pair's most recent
common ancestor (MRCA), because the Mate Network failed:

- At `mating_lambda` 3 one Mate Network holds 76-89% of a cell's pairs.  The
  meat then has a few dominant terms: SD(r)/SE is 1.5-2.3 at the median and
  coverage 0.61-0.78.
- MRCA and MRCA or shared member give SD(r)/SE between 0.98 and 1.05 at the
  median in every scenario and category type.  Median coverage is 0.95 over
  all cells and 0.93-0.96 per scenario and type; the low end is prevalence
  0.02 (0.93-0.94), which runs about 4% high under every design.
- Dyadic clustering leaves 2-10% for symmetric categories, and the naive
  pair 8-47%.

## Decision

- Each Relative-Pair Correlation cell reports a sandwich SE that clusters
  two pairs together when they share a member **or** share an MRCA, computed
  by inclusion-exclusion:
  `meat = Σ_m Φ_m² + (Σ_i S_i² - Σ_d φ_d²) - (Σ_(i,m) S_(i,m)² - Σ_d φ_d²)`.
  SE = √meat / n.
- A pair's MRCA is one pedigree row: the ancestor member of a lineal pair
  (MO, FO, GP, GGP), the shared parent of a half relationship (MHS, PHS,
  HAv), and the shared mother of a full one (FS, Av, GAv, 1C; MZ by the
  twins' mother).  The reducer finds it from the category's up and down
  meiosis counts and the pedigree's parent rows, taking the first path in
  (mother, father) order when a loop gives several.  A double half
  relationship, such as double half first cousins classified as 1C, has
  no shared mother on one path and takes the first shared parent the same
  way.  A pair whose MRCA is a founder is clustered by that founder.  Only
  a pair with no common ancestor in the pedigree, MZ twins without parent
  rows, is its own cluster.
- φ_d is the pair's influence on r̂: Pearson's `u v - r/2 (u² + v²)` and the
  two-step tetrachoric's threshold-corrected cell influence, averaged over
  a symmetric pair's two entries.  The same design covers pooled, stratum
  and cross-trait cells.
- The stratified shared-rho fit (ADR 0012) uses the same clusters with its
  own cell influence (`polychoric::influence::CellInfluence`).  Its
  calibration runs on the Rust unit through the public API, as the gate's
  first follow-up.
- The meat streams: per-key sums of the per-pair statistics, keyed by MRCA
  row, person row, and (person, MRCA), through a custom
  `pedigree_graph_core::relationships::Reducer` after the point estimates.
  Pairs are never stored.
- A negative meat gives no SE, with a reason.  No small-sample factor.

Rejected:

- Mate Network (ADR 0009's candidate 3) and Mate Network or shared member
  (candidate 4).  They calibrate while networks stay small and fail when
  serial mating joins them.  The registries this serves have serial mating.
- MRCA alone.  It calibrates as well here: the SEs differ by at most 7% in
  any cell, median 0%.  But a person's pairs in one cell can have different
  MRCAs (the child and the parent in MO, maternal and paternal cousins in
  1C, MHS and PHS in HS).  Those are first-order links the union keeps.
  With only three phenotyped generations they are rare, so this calibration
  cannot rule them out for deeper data.
- Dyadic alone.  2-10% short for symmetric categories.
- Naive pair.  8-47% short.
- Connected component, which gives one cluster.

## Consequences

- SEs are 1.22 times the naive pair SE at the median over cells (5th-95th
  percentile 1.04-1.87, at most 2.36).  simACE's and fitACE's lower
  bounds can switch to this SE.
- The person and (person, MRCA) maps cost memory proportional to the
  phenotyped rows per cell, beyond the MRCA map alone.  `memory_budget_bytes`
  bounds them.
- Sparse cells: at under about one expected concordant pair, every design's
  Wald SE fails (phi at 0.5% cumulative incidence: SD(r)/SE near 2,
  coverage near 0.3).  This is a Wald-interval limit, not a cluster one.
  Reporting the concordance lets callers keep a floor.
- The ACE Fit's SE and inverse-variance weights (issue #19) need the
  covariance between cells, from the same per-key sums across cells.
