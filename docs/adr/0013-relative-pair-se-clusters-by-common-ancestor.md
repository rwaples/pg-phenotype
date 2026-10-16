# ADR 0013: The Relative-Pair SE clusters pairs by their most recent common ancestor or a shared member

**Status:** accepted; amended 2026-10-16 (what the implementation settled:
the MRCA key, the entry-level meat, one engine pass, opt-in, the
degenerate and negative meats, the calibration through the public API)
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

## Amendment (2026-10-16): what the implementation settled

- The MRCA key is the shared ancestor's original id, not its row.
  pedigree-graph defines sibs by the original parent ids
  (`SiblingIndex::build(ped.orig_mother, ped.orig_father)` in
  pedigree-graph-core's `relationships/engine.rs`, v0.12.4), so a sib or
  half-sib pair's shared parent can be an external parent with no row.
  The key is encoded as a dense cluster index: a row's ancestor is its row,
  an external parent `n + its rank` among the external ids, and a pair with
  no common ancestor `n + n_external + its first member's row`.  This is the
  same partition as the original ids.  A symmetric pair's MRCA is found from
  its (lower row, higher row) orientation, so both entries get one key.
- The meat is formed at the entry level, one oriented pair being one
  entry.  Each entry adds its influence, divided by the fit's entries, to
  its MRCA, to each member, and to each (member, MRCA), and SE = √meat:
  √(meat over raw influences) / n_entries.  With both orientations of every
  pair this is the pair-level formula above.  A symmetric cross-trait cell
  whose pairs are complete in one orientation only
  (n < n_entries < 2n) counts each entry once, as its point estimate does.
- One engine pass computes the table and the SE's statistics.  A
  `Reducer` wraps pedigree-graph-core's `CellReducer`, leaves its table
  byte for byte, and keeps, per (member, side, MRCA, engine cell), the
  member's partners' count and per value column their sum and sum of
  squares, as exact integers.  The three meat terms are groupings of that
  one store, read back in key order after the fits; the influences are
  applied there in f64.  No pair is stored.
- The SE is opt-in (`se`, default off).  Off, the call runs
  `relationship_moments` and the fits alone, with no measurable overhead:
  its medians are 0.94-1.04 times 8097f88's in the cases below, within the
  run-to-run spread.  On, the call takes, at one thread, 1.5-1.6 times
  the wall time with one trait and 2.4-3.7 times with two traits and
  generation strata; at eight threads 1.6-1.8 and 2.1-3.3 times.  After
  the pass the requests sweep in parallel, one thread a request; with two
  binary traits and strata at eight threads that took the call from 1.13 s
  to 0.39 s, its SEs bit for bit the same.  Measured off/on/off (and the
  parallel sweep before/after/before) on the simACE rep
  `se_cal_lam3_K10/rep1` (120k rows), 15 warm calls per case over 3
  processes.  An estimate's weights are built from the blocks it reads, so
  their cost grows with the strata squared, not cubed: with 80 strata and
  two binary traits the SE's share of the call fell from 49 s to 12 s at
  one thread.
- No SE, with a reason, for a meat that is zero to within rounding
  (`degenerate_meat`) or negative (`negative_meat`).  A meat over n
  entries is zero to within rounding when |meat| is at most
  max(2¹⁰, 4√n) ε of its scale, the meat's three sums of squares added as
  if each entry had the influence 1/n.  A cell whose pairs share one MRCA,
  or whose estimate is ±1, has a meat of zero in exact arithmetic, and its
  computed meat is rounding.  Over n entries the meat is at most 16n
  additions: each of at most 2n rows into its MRCA, member and
  (member, MRCA) sums, then at most 5n squares of those sums.  Each rounds
  by about ε of the scale, and the errors add as a random walk, to
  √(16n) ε = 4√n ε.  Below 2¹⁶ entries the floor 2¹⁰ ε holds: the
  rounding in the tests is below 0.05 ε of the scale, while the smallest
  true meat there is 10⁻⁵ of it.  The Within-Person cell has no cluster
  design and no SE (`within_person`).
- The SE's memory counts against `memory_budget_bytes`.  The engine's
  lanes are planned first, then the bytes kept back for after the pass:
  the MRCA ancestry (8 bytes a row), every request's influences (each
  holds only the blocks it reads), and one request's sweep, its weights
  and per-person totals.  Each lane may hold a share of the rest.  A plan
  whose one lane holds fewer rows than its first growth (1024) is refused
  before the pass.  A lane that reaches its share stops growing and the
  call raises `memory_budget_exceeded` (operation `relative_pairs_se`)
  after the pass.  After the pass the sweeps are planned again from what
  is held (the table, the store, the ancestry, the influences): as many
  requests sweep at once, each on one thread, as the rest of the budget
  holds, up to the thread count, else one at a time as planned.  Which
  rows a lane draws depends on the thread count, so near the budget edge
  whether that error is raised can too.  A result never depends on the
  thread count: the integer statistics are exact, one thread sweeps a
  request whole, and every float is summed in key order.
- Calibrated through the public API (`--rust` in
  `tools/relative_pairs_se_calibration.py`; `docs/gates/relative-pairs-se/`).
  On the same 600 reps, `relative_pairs.correlation(..., se=True)` gives
  the calibration's `mrca_person` SE for every pooled and stratum
  estimate, to a relative 3.5e-6, so those cells calibrate as above.  The
  stratified shared-rho fit calibrates too: over its 514 cells the median
  SD(r̂)/SE is 1.01 (5th-95th percentile 0.89-1.15) and the median Wald
  coverage 0.95, and per scenario, category type and estimator its medians
  are 0.94-1.09.  Cells with under about one expected concordant pair fail
  as above.  One cell type lies outside the band under every design tried,
  local or Mate Network: the father-offspring Pearson
  of trait1 at mating_lambda 3 in generation 5 (1.18-1.28 in three
  scenarios); its source is open.
