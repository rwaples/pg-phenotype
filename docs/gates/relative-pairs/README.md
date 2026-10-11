# Relative-Pair Correlation gate: pg-phenotype vs a NumPy reference and simACE (ADR 0009)

Result: **pass**.  Run 2026-10-10 on simACE
`6355ffa6aebe37238cb2724dfcde288830fdcb6c` (simACE's pixi env) against the
pg-phenotype working tree on `6e11fd3`:

    pixi run --manifest-path <simACE>/pixi.toml python -B tools/make_simace_relative_pairs_golden.py \
        --simace <simACE checkout> --reps <simACE>/results/test/small_test/rep{1,2,3} \
        --out tests/golden/simace_relative_pairs_6355ffa6aebe
    pixi run python -B tools/relative_pairs_gate.py --out docs/gates/relative-pairs/gate.json

`tests/test_relative_pairs.py::test_the_simace_gate_agrees` asserts every
check below.  `gate.json` holds the numbers.

## Inputs

Three simACE `small_test` reps (seeds 12042-12044; built by simACE
`v0.1.0-32-g37e6b20-dirty`, as each rep's `run.yaml` records): the full
pedigree (6000 rows, `pedigree.parquet`) and the 3000 phenotyped rows'
`liability1` and `affected1` (`phenotype.parquet`), NaN on the other rows.
`inputs.npz` holds them, so the gate needs no simACE results directory.

simACE's side is what `simace stats` computes at `6355ffa` on
`graph.view(ids=phenotyped)`: `compute_liability_correlations`,
`compute_affected_correlations` (phi), `compute_tetrachoric` and
`Stratum.affection_table` for MZ, FS, MO, FO, MHS, PHS and 1C.  simACE
enters every pair once, the lower receiver row first for a symmetric
category.  pg-phenotype runs on the full pedigree with the trait missing
off the sample, so its pairs are the view's.

## Checks

| check | result |
|---|---|
| every default category (12) x rep, `n` and `n_dropped` against `tests/oracle/relative_pairs_reference.py` (pedigree-graph's `relationship_pairs`) | equal (36 cells per trait) |
| Pearson (`liability1`) against the reference's NumPy double-entry Pearson | max 7.2e-14 (rep2 MZ, 15 pairs); tolerance 1e-12 |
| phi (`affected1`) against the reference | max 0 |
| tetrachoric against `correlation.tetrachoric` on the reference's table | identical |
| `n` against simACE's `n_pairs`, all seven categories | equal |
| Concordance (symmetric) and 2 x 2 table (MO, FO) against simACE's table | equal |
| MO, FO Pearson against simACE | max 9.0e-15; tolerance 1e-12 |
| MO, FO phi against simACE | max 6.9e-18 |
| MO, FO tetrachoric against simACE | max 1.18e-8 (rep1 FO); tolerance 1.2e-7, ADR 0011's gate bound |
| a directed category under simACE's 10-pair floor: simACE withholds every estimate | none on these reps (MO, FO have 2000 pairs each) |

`tests/relative_pairs_golden.py::checks` holds these checks; the tool and
the test both call it.  Below simACE's floor pg-phenotype still estimates,
so the gate checks that simACE gave `None` there instead of a gap.

## Symmetric categories

A symmetric category is double-entered here and single-entered in simACE,
so the two differ by design and no tolerance applies.  Double minus single
entry, over the three reps:

| category | pairs | Pearson | tetrachoric | phi |
|---|---|---|---|---|
| FS | 2350-2378 | -4.8e-4 to -1.9e-5 | -3.0e-3 to -2.8e-5 | -9.3e-4 to -7.9e-6 |
| MHS | 1086-1456 | -2.2e-3 to -1.5e-4 | -3.4e-3 to -1.6e-4 | -1.0e-3 to -4.4e-5 |
| PHS | 1034-1226 | -1.9e-4 to -2.2e-5 | -2.0e-3 to -3.5e-4 | -5.7e-4 to -1.1e-4 |
| 1C | 9422-9729 | -1.0e-3 to -1.9e-4 | -1.8e-4 to -1.1e-5 | -5.0e-5 to -3.2e-6 |
| MZ | 15-18 | -3.9e-2 to +3.7e-2 | -1.8e-3 to +2.1e-3 | -6.3e-3 to -5.2e-3 |

Every non-MZ difference is negative on these reps.  The MZ cells (15 to 18 pairs) are where one pair
moves the estimate most; both tetrachoric fits stop at a bound on rep1 and
rep3, and both are undefined (a constant margin) on rep2.
