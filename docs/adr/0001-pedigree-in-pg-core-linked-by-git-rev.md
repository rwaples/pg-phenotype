# ADR 0001: The pedigree goes in; pedigree-graph-core is linked by git rev

**Status:** accepted; amended by ADR 0006 (a validated Pedigree is shared across methods)
**Date:** 2026-10-06
**Context:** simACE `plans/pg-phenotype-extraction-v3.md`, decisions #1-#4, #10

## Context

PA-FGRS lived in fitACE_pafgrs as Numba kernels over a kinship matrix the
caller built.  The kinship came from a degree-capped sparse matrix, so a
relative-relative pair beyond the cap read as zero, and callers outside
simACE had to reproduce the matrix themselves.

## Decision

- A caller passes the pedigree (`id`, `mother`, `father`, optional `twin`,
  `sex`) and the traits.  pg-phenotype computes every relationship and
  kinship itself.
- `pg-phenotype-core` depends on `pedigree-graph-core` by git rev
  (`[workspace.dependencies]` in `Cargo.toml`), re-pinned on purpose only.
  pg-core is unpublished until its 1.0.
- Validation goes through `graph::build`, which checks lengths, ranges,
  duplicate ids, cycles, MZ codes and birth-year order.  The relationship
  and kinship inputs are built from the validated graph, never from
  `PedigreeColumns::try_borrow` alone, which only checks lengths and rows.
  Pedigree errors reach hosts with pedigree-graph's own codes.
- The layout follows pedigree-graph ADR 0007: `crates/core` (no host
  imports, `forbid(unsafe_code)`), `crates/python` (PyO3), `r/` (extendr).
  One Rayon pool per process, `configure_threads(n)` >
  `PG_PHENOTYPE_THREADS` > 1, and output identical across thread budgets.
- `prepare` returns an opaque handle held in memory; there is no save or
  load in v0.1.

### Consumer-gate routing

pedigree-graph's consumer gate (`tools/consumer_gate.py`) routes Python
imports through a candidate wheel.  A Rust consumer needs its build
redirected instead.  `tools/test_against_pg.sh <pg-checkout>` writes a
cargo `[patch."https://github.com/rwaples/pedigree-graph"]` path override
into a private `CARGO_HOME`, which cargo, maturin and `R CMD INSTALL` all
read, then runs the Rust, Python and (with `PG_PHENOTYPE_WITH_R=1`) R suites,
and restores the lock files.  Until pedigree-graph's gate calls it, CI runs
it weekly against pg `main`.

## Consequences

- A process that also imports the pedigree-graph wheel holds two copies of
  pg-core and two pools.  They run one after the other, so the cost is
  memory and a second `configure_threads`.
- pedsum validates a pedigree with its own `load_and_validate` before
  calling `graph::build`; the two must agree on pedsum's fixtures.
