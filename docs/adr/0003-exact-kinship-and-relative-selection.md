# ADR 0003: Exact kinship and relative selection in PA-FGRS

**Status:** accepted
**Date:** 2026-10-06
**Context:** simACE `plans/pg-phenotype-extraction-v3.md`, decisions #14, #15, #17

## Decision

- **Relatives.**  A relative of proband `p` is a row whose closest
  relationship category to `p` is at most `ndegree` (pedigree-graph-core's
  engine, `1 <= ndegree <= 5`) **and** whose exact kinship to `p` is at
  least `0.5^(ndegree+1) - 1e-6`.  With resolved parents exact kinship is
  at least nominal, so the threshold only removes zero-kinship pairs that
  share an external parent id.
- **Exact kinship everywhere.**  Proband-relative and relative-relative
  kinship are the pinned float32 recurrence of pedigree-graph ADR 0009, bit
  for bit.  The old 0-outside-support lookup (fitACE `pafgrs.py:396`) is
  gone: two relatives of `p` related beyond `ndegree` now covary.
- **Trait-independent prep.**  The prep is keyed on pedigree, `ndegree` and
  probands only.  Unphenotyped relatives stay in and get `w = 0` at score
  time.

## How prep computes it

Prep storage was chosen by measurement on simACE `cure_rA50_200k` (357k
rows, 200k probands, ndegree 3; 205M relative-relative entries):

| Option | Size | Note |
|---|---|---|
| (a) per-proband float32 triangles | 783 MiB | as fitACE, but with unphenotyped relatives |
| (b) unique pairs, shared | >= 605 MiB | 52.9M unique pairs as sorted keys and values, plus a lookup per read |
| (c) relative lists only | ~0 | every score call re-walks the kinship |
| **(a') triangles, per-chunk dictionary codes** | **271 MiB** | 82 distinct values; u8 codes, u16 or f32 when a chunk needs them |

The kinship is computed per *sibship group* (probands sharing both
parents): one triangle over the union of the group's candidate relatives,
its members, and their parents, filled by the recurrence in depth-major
order, so each operand is already in the triangle.  An operand outside it
goes to pedigree-graph-core's walker unless the two rows provably share no
ancestor.  The proof is exact: every upward path ends at a *terminal* (a
row with a missing parent), so two rows share an ancestor exactly when they
share a terminal.  pedigree-graph's 256-bit ancestor signatures answer the
same question approximately and, six generations deep, almost never prove
it.

On that pedigree prep fell from 28.8 s to 8.1 s on one thread (1.95 s on
eight), and every sampled value stayed bit-identical to pedigree-graph's
`pair_kinship`.

## Consequences

- Scores change wherever the capped kinship read zero.  The fitACE gate
  attributes every such change (plan decision #19).
- A row whose terminal set exceeds 64 labels is never declared unrelated;
  its pairs fall back to the walker.  That only costs time.
