# ADR 0008: A Pedigree looks up rows by id

**Status:** accepted
**Date:** 2026-10-10
**Context:** issue #9; amends ADR 0006

## Context

ADR 0006 matches traits, ages, strata and probands to a Pedigree's rows by
position and leaves alignment to the caller.  A trait shuffled by a sort or
a join passes every check and gives a plausible, wrong result.  Callers
write their own alignment.  fitACE's PA-FGRS adapter (`Frames.of`,
`Frames.aligned` in `fitace_pafgrs/pg_phenotype_adapter.py`) sorts the
pedigree ids and `searchsorted`s the phenotype ids.  From reading the code,
a phenotype id above every pedigree id indexes past the end, and a
duplicated phenotype id is not detected.

`pafgrs.prepare` already looks up its probands: it builds pedigree-graph's
`IdIndex` on every call and raises `unknown_proband` and
`duplicate_proband`.  Building the index takes about 6 ms at 1M rows with
ids in row order and about 1.75 s at 5.36M rows out of order
(rwaples/pedigree-graph#43).

## Decision

- `Pedigree.rows(ids)` (R: `pedigree_rows()`, Rust: `Pedigree::rows`)
  gives the row of each id: 0-based in Python and Rust, 1-based in R.
- `Pedigree.align(ids, values)` (R: `pedigree_align()`, Rust:
  `Pedigree::align`) moves values given per id onto the rows.  A row that
  no id names gets NaN in Python and `NA` in R.  The core takes the fill
  value, so R's `NA` is not a NaN that the binding would have to tell
  apart from the caller's.
- An id that is not a row's is `unknown_id` and one listed twice is
  `duplicate_lookup_id`, each with the id and its position(s).
  `duplicate_id` stays pedigree-graph's code for a pedigree with two rows
  of one id.  `align` checks the lengths first, with `pair_length_mismatch`.
- The Pedigree builds the index on the first `rows` or `align` and keeps
  it.  The index is structure that no parameter changes, the kind ADR 0006
  lets a Pedigree keep.
- `prepare` uses that index for its probands when the Pedigree has one,
  and otherwise builds one for the call and drops it before the kinship
  walk, as before.  Its error codes are unchanged.

Rejected:

- Rebuilding the index on every call.  This keeps no state, but repeats
  up to 1.75 s per call on a large pedigree with ids out of order.
- Building the index when the Pedigree is built.  Every caller would pay
  12 bytes per row, including those that never align.
- `align` alone.  Strata are integer labels and probands are ids, so a
  float `align` does not cover them; `rows` does.
- Having `prepare` keep the index it builds.  A Pedigree made from columns
  for one call would hold it through the walk, which is the call's memory
  peak.

## Consequences

- A Pedigree that has aligned holds 12 more bytes per row: about 64 MB
  at 5.36M rows.  Each call also allocates 4 bytes per row and 8 per id
  while it runs.
- Callers can drop hand-written alignment.  fitACE's PA-FGRS adapter can
  switch once a release carries this.
- Alignment is checked only when the caller uses these methods.  A trait
  given by position is still not checked against the rows.
- pedigree-graph#43's faster index build would make the first call
  cheaper.
