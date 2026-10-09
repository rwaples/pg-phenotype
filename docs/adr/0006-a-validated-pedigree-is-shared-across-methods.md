# ADR 0006: A validated Pedigree is shared across methods

**Status:** accepted
**Date:** 2026-10-09
**Context:** simACE `plans/pg-phenotype-pedigree-v2.md`; rwaples/pg-phenotype#7; amends ADR 0001

## Context

ADR 0001 has each method take the pedigree columns and validate them with
pedigree-graph-core's `graph::build`.  A caller that runs several methods,
or one method on several traits, validates the same pedigree each time.
simACE's planned switch to `mate_correlation` (simACE #47) makes three
calls per pedigree, all in its one analyze process: liability for its
stats, then liability and the A component for its validation.  At 1M rows, validation
takes about 148 ms and finding the Mating Pairs and their Mate Networks
about 64 ms; at 5.4M rows, about 1.3 s and 0.45 s.  Both repeat on every
call.

## Decision

- `Pedigree(columns)` (R: `pedigree()`) runs `graph::build` once.  Every
  method (`mate_correlation`, `pafgrs.prepare`) takes a Pedigree or the
  columns.  Columns are checked into a Pedigree for that call alone, so
  existing callers keep working and pay what they paid before.
- With columns, a method checks its own parameters (draw counts,
  `ndegree`) before it validates the pedigree, as before.  The order is
  kept in the core, once for both hosts.
- Validation still goes only through pg-phenotype's pinned
  `graph::build`, as ADR 0001 requires.
- A Pedigree builds its Mating Pairs on first use and keeps them, and
  likewise the Mate Networks of all its pairs.  Inputs given per call
  (traits, strata) never enter it.  A call whose strata drop no pair
  reuses both.  A call whose strata drop pairs computes the networks of
  the pairs it keeps, for that call alone.
- A Pedigree keeps only structure that no parameter changes: its
  validated rows and, once used, its Mating Pairs and their Mate Networks.
  Anything that depends on a parameter (a PA-FGRS Prep for one `ndegree`
  and set of probands) stays an object the caller builds and holds, with
  its own lifetime and memory cost.
- A Pedigree keeps its input's row order: row *i* of the Pedigree is row
  *i* of the columns it was built from.  Traits, ages, strata and probands
  are matched to it by position, and the caller aligns them (with
  `ids` if they come from another table).  Validation proves the pedigree
  is well formed, not that a trait belongs to its rows.
- A Prep does not hold its Pedigree.  It shares the Pedigree's id buffer,
  which it would otherwise copy.
- A Pedigree exposes its length and its ids in row order, and nothing else.
  It cannot be pickled or saved.  In R it keeps the storage type of the
  ids it was given (integer, double, integer64) for every result.

Rejected:

- Accepting pedigree-graph's Python `PedigreeGraph` and reading its row
  arrays.  This would validate once per simACE rep, but its checks ran in
  another build of pedigree-graph-core, possibly at another version.  It
  would also tie pg-phenotype to that class's private storage.
- A Pedigree as the only input.  It would give one input type, but every
  caller (pedsum, fitACE's PA-FGRS adapter, R users) would have to change,
  and a caller that makes one call would gain nothing.
- Validating the columns in the host before the method's parameters.  The
  documented error order would change, and a bad setting would wait for
  validation of a large pedigree.

## Consequences

- A Pedigree holds 36 bytes per row: the ids, the parent ids and the
  parent and twin rows, which are what the methods read.  Once
  `mate_correlation` has run on it, it also holds 16 bytes per Mating Pair
  for the pairs, plus 8 for their networks.  At 5.36M rows (2.42M pairs)
  that is about 193 MB, plus 58 MB.
- A later `mate_correlation` call on one Pedigree skips validation, finding
  the pairs, and, without dropped strata, building the networks.  It still
  pays for the distinct fathers, the cells and the within-person
  correlation.  For simACE's three planned calls, summing the measured
  stage times projects about 1150 ms against about 1575 ms with columns at
  1M rows, and about 8.5 s against 12.0 s at 5.4M rows.
- A Prep outlives its Pedigree and keeps only the ids alive, as it did
  before.
- A Pedigree adds nothing to scoring many traits against one Prep, which
  already reuses its relatives.  It helps when a caller prepares again
  (another `ndegree` or set of probands) or runs several methods on one
  pedigree.
- A trait shuffled relative to the Pedigree's rows passes every check and
  gives a wrong result.  A checked align-by-id helper is left for later;
  fitACE's PA-FGRS adapter aligns phenotype ids to pedigree rows by hand
  today.
- Pedigree errors come from the constructor when a caller builds a
  Pedigree, and from the method, after its parameter checks, when it is
  given columns.  The codes are the same either way.
