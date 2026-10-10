# ADR 0010: A CIP is estimated by Aalen-Johansen as a step function

**Status:** accepted
**Date:** 2026-10-10
**Context:** issue #4; simACE `simace/analysis/stats/incidence.py`
(`_aalen_johansen`); fitACE `fitace_pafgrs/pafgrs.py` (`compute_empirical_cip`);
amends ADR 0002

## Context

PA-FGRS takes a CIP table (ADR 0002), and pg-phenotype had no way to make
one from data.  fitACE's production CIP divides the cases with onset at or
before each age by all rows, with no correction for censoring or late
entry.  On simACE data with late entry that gives a K about 30% low (the
gate: 0.073 against 0.106).  simACE already has an Aalen-Johansen estimator
with late entry and death as a competing risk, used for its plots.

The estimator, as the R survival package documents it (`methods.Rnw`,
section "Aalen-Johansen"; `compete.Rnw`, equation `cuminc`): at each
distinct event age `t`, with `Y` at risk, `d1` cases, `d2` deaths and
all-cause survival `S` just before `t`, the CIP rises by `S d1 / Y` and `S`
falls by the factor `1 - (d1 + d2) / Y`.  This is `p(0) prod (I + A(t))`
for the states alive, case and dead.

## Decision

- `Cip::aalen_johansen` (Python `Cip.from_aalen_johansen`, R
  `pafgrs_cip_aalen_johansen`) takes per person `age`, `status` (0
  censored, 1 case, 2 death) and an optional `entry`, and returns a `Cip`.
  It reports no SE: simACE dropped its unused Greenwood SE (`6355ffa`), and
  PA-FGRS takes a point table.
- A `Cip` has an interpolation, `linear` (the default, as before) or
  `step`: between ages a step table holds the value at the last age at or
  below.  It is part of the table, so every scoring call in every binding
  reads it.
- The estimate is a step function that moves only at case ages, so the
  result is a `step` table of the distinct case ages and the estimate
  there.  Below the first case age `Cip` gives 0 and from the last on `K`,
  and a control's `w` is `F(age) / K` with `F` the estimate itself.
- A person is at risk at `t` when `entry < t <= age`, the counting-process
  convention survfit uses.  Without `entry` everyone is at risk from birth,
  `t <= age`, so a case at age 0 counts.  A case and a death at one age
  both use the survival just before it.  A person censored at an event age
  is at risk there.
- Input is checked, not cleaned: a missing or negative age or entry, a
  status outside {0, 1, 2}, `entry > age`, or a case or death at its entry
  age (never at risk) raises at the first offending person.  A censored
  person may leave at entry; simACE records an onset before the
  observation window that way.
- No strata.  Scoring takes one `Cip` per trait, so a per-sex table has no
  consumer; a caller wanting one passes that sex's rows.  Giving each
  person their stratum's CIP during scoring (by sex or birth cohort, say)
  is a separate change.
- An estimated `K` of 0 or 1 is not a CIP: `cip_not_estimable`
  (validation) with `reason` `zero_prevalence` or `prevalence_one`, not
  `invalid_cip`, whose position would name a table the caller never wrote.
  `K` is 1 when all-cause survival reaches exactly 0 with no death
  increment, whatever rounding gave the sum.
- The numerics are in the Rust core, grouped as simACE's
  `_aalen_johansen` groups them (`S d1 / Y`, `1 - (d1 + d2) / Y`, running
  sums and products), so the two agree bit for bit.

Rejected:

- simACE's risk set `entry <= t <= age`.  It differs from survfit's only
  when an entry age equals an event age, as with whole-year ages, and
  counts a person at risk at the age they enter.
- Linear interpolation between the case ages alone (the first draft).
  Every step goes up, so between two case ages the line lies above the
  estimate, by up to the next step's height; that raises the `w` of
  controls there, by nearly double in a sparse stratum.  An opt-in smooth
  CIP is #12.
- A linear table that copies each step's previous value to one ulp before
  the next case age (the second draft).  It reproduced the step function
  without a mode, but only while those ages stayed one ulp apart:
  `write.csv`, float32 or any rounding merged them into duplicate ages
  (`invalid_cip`), the table had up to `2m - 1` rows, and the slope across
  one ulp overflowed to infinity near 1e-300.
- A caller-supplied age grid.  The case ages are where the estimate moves;
  a coarser grid loses steps and a finer one adds nothing.
- Dropping invalid rows and counting them.  A `Cip` has nowhere to report
  a count, and which rows to drop (death before entry, say) is the
  caller's study design.
- A death column beside a binary status.  One status column with a
  competing level is how cmprsk and survfit take competing risks.

## Consequences

- A table saved and reloaded keeps its shape only if its interpolation
  travels with it; `Cip(ages, cip)` alone reads a step table as linear.
- A control's `w` jumps at each case age.  With few cases that is coarse;
  smoothing is the caller's choice (#12), not the default.
- With continuous ages the gate matches simACE bit for bit on 18 cases,
  and with whole-year ages differs by up to 1.2e-3 absolute; moving each
  entry one ulp earlier, which turns `entry < t` into simACE's
  `entry <= t`, reproduces simACE bit for bit.  survfit (R survival
  3.8-12, `timefix = FALSE`) agrees on all 24 cases within 4.5e-14
  relative.
- R's survival is a dependency of the development `r` environment and a
  Suggests of the R package, for the gate and one R test.
