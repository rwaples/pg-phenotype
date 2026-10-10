# Aalen-Johansen CIP gate: simACE and survfit vs pg-phenotype (issue #4)

Result: **pass**.  Run 2026-10-10 on simACE
`6355ffa6aebe37238cb2724dfcde288830fdcb6c` (simACE's pixi env) and R 4.5.3
with survival 3.8.12 (pg-phenotype's `r` env), against the pg-phenotype
working tree on `46b7502`:

    pixi run --manifest-path <simACE>/pixi.toml python -B tools/make_simace_aj_golden.py \
        --simace <simACE checkout> --out tests/golden/simace_aj_6355ffa6aebe
    pixi run python -B tools/make_survival_aj_golden.py --golden tests/golden/simace_aj_6355ffa6aebe
    pixi run python -B tools/aj_gate.py --out docs/gates/aalen-johansen/gate.json

`tests/test_aalen_johansen.py` asserts the agreement below.  `gate.json`
holds the numbers per case.  The result is a `step` table of the case
ages, compared at those ages.

## Inputs

Four simACE pipeline runs (`run_simulation`, `run_phenotype` with the
default Weibull frailty traits, `run_censor` with the default Weibull death
as a competing risk), 6 generations of which the last 3 are phenotyped.
Entry, age and status are read as `compute_cumulative_incidence_aj` reads
them.  Each run gives 6 cases: two traits, pooled, female and male.

| scenario | per generation | windows (generations 3, 4, 5) |
|---|---|---|
| `late_entry` | 2000 | `[40, 80]`, `[0, 80]`, `[0, 45]` (simACE's default) |
| `no_entry` | 2000 | `[0, 80]` each; pg-phenotype gets no `entry` |
| `small` | 100 | as `late_entry` |
| `integer_ages` | 2000 | as `late_entry`, ages rounded up to whole years |

Rows simACE drops (entry after age: a death before the window, 0 to 52 per
run) are left out of the inputs.  Rows censored at their entry age (an
onset before the window) stay in.

## Agreement

- **simACE, continuous ages** (18 cases): the step ages and every value
  are bit-identical, and `K` equals `compute_cumulative_incidence_aj`'s
  last value on the pooled cases.
- **simACE, whole-year ages** (6 cases): the step ages match; values
  differ by up to 1.2e-3 absolute (`integer_ages/trait2/female`).  simACE
  counts a person at risk at the age they enter (`entry <= t`), and with
  whole years many case ages equal generation 3's entry at 40.  Moving
  every entry one ulp earlier reproduces simACE bit for bit, so the risk
  set is the only difference.
- **survfit** (24 cases, `survfit(Surv(entry, age, status) ~ 1, id,
  timefix = FALSE)`): the step ages match and values agree within 4.5e-14
  relative (`no_entry/trait2/male`).

## K against cases over rows

fitACE's `compute_empirical_cip` divides cases by all rows.  With late
entry that understates K by about 30%:

| case | Aalen-Johansen K | cases / rows |
|---|---|---|
| `late_entry/trait1/pooled` | 0.1059 | 0.0732 |
| `late_entry/trait2/pooled` | 0.2653 | 0.1817 |
| `small/trait2/pooled` | 0.2777 | 0.1900 |
| `integer_ages/trait1/pooled` | 0.1063 | 0.0733 |
| `no_entry/trait1/pooled` | 0.0967 | 0.0967 |

Without late entry everyone is followed to 80 or death, nobody is censored
before the last event, and the two agree.
