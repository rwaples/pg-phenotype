# ACE Fit truth recovery on simACE reps (plan D10, #17)

> Recorded on pedigree-graph 0.13 and not rerun on 0.14, whose path-count
> full and half categories (its ADR 0018) move a few 1C and Av-family pairs
> on these random-mating pedigrees (about 0.1-0.4% of 1C). The result is not
> expected to change; a rerun on 0.14 would restate it.

Result: **pass**.  All 38 gated targets are within the Bonferroni bound
|z| <= 3.2125 (familywise 5%), and no rep is undefined.  The largest
|z| is 2.54.  Four targets fall outside 2 MC SE, the reported near-miss
flag:
- `ltm_c0_re0` liability A1, z -2.35.
- `ltm_c0_re02` uncensored-status C2, z -2.54.
- `ltm_c0_re02` uncensored-status rA, z -2.35.  Its z has grown with
  the reps at about a bias's rate ([12 versus 36 reps](#12-versus-36-reps)).
- `ltm_c02_rc07` liability rC, z -2.52.

The frailty stress criterion passes: stratified rA is 0.491 against
`estimate_rg`'s 0.308.

Run 2026-10-16 (05:41 UTC) in pg-phenotype's pixi env, 1124 s.  The
working tree was on `012a0f2` (`v0.3.0-22-g012a0f2-dirty`), with the
units of this plan uncommitted and D7's revised cross-trait weights
included.  The extension was rebuilt with `pixi run build-dev` before the
run.  `results.json` records the tree diff's SHA-256 and the tool's.

    pixi run python -B tools/relative_pairs_ace_validation.py \
        --results <simACE>/results/relative_pairs_ace \
        --config <simACE>/config/relative_pairs_ace.yaml \
        --out docs/gates/relative-pairs-ace/results.json

`<simACE>` is the simACE worktree on branch `pgp-relative-pairs`.  The
tool exits 1 when a gated target fails.  This run exited 0.
`results.json` holds every target's per-rep values, estimated and
realized, plus summaries of the diagnostics.

## Inputs

There are four scenarios in `config/relative_pairs_ace.yaml`.  The three
simple_ltm scenarios have 36 reps each and `frailty_gen_censoring` has
12.  Every rep has N 50000 per generation and 6 generations (300000
pedigree rows), and 150000 phenotyped rows in generations 3 to 5.  All
reps were built by simACE `v0.1.0-44-gb2181f9`, as each `run.yaml`
records.  The tool reads the nominal truths and `replicates` from the
config, and stops if any rep's `run.yaml` resolves a different value.

| scenario | A | C | E | rA | rC | rE | phenotype | reps | seeds |
|---|---|---|---|---|---|---|---|---|---|
| `ltm_c0_re0` | 0.5 | 0 | 0.5 | 0.5 | 0 | 0 | simple_ltm, prevalence 0.1 | 36 | 17000-17035 |
| `ltm_c0_re02` | 0.5 | 0 | 0.5 | 0.5 | 0 | 0.2 | simple_ltm, prevalence 0.1 | 36 | 17100-17135 |
| `ltm_c02_rc07` | 0.5 | 0.2 | 0.3 | 0.5 | 0.7 | 0 | simple_ltm, prevalence 0.1 | 36 | 17200-17235 |
| `frailty_gen_censoring` | 0.5 | 0 | 0.5 | 0.5 | 0 | 0 | default frailty, default `gen_censoring` | 12 | 17300-17311 |

Both traits have the same A, C and E.  Per rep, the tool calls
`relative_pairs.correlation` on the full pedigree with the default
categories (degree 3 and below).  The two traits are NaN off the
phenotyped rows (the `cohort.parquet` rows with a `t1`), and `stratum`
is the generation.  `relative_pairs.ace` then fits A and C with
`shared_environment=["MZ", "FS", "MHS"]`, because simACE assigns the
household by mother.  It runs once with `estimate="stratified"` and once
with `"pooled"`.

The ACE Fit weighs rows as D7 now specifies.  Same-trait and
within-person rows weigh the unordered pairs (or rows) that entered.
Cross-trait rows weigh complete orientations: a symmetric cross cell's
`n_entries`, and a directed cross cell's `n`.

| outcome | traits | scenarios | gated against |
|---|---|---|---|
| `liability` | `liability1`, `liability2`, Pearson | all | realized truth, simple_ltm scenarios |
| `uncensored_status` | `t1 < 1e6`, `t2 < 1e6`: simple_ltm status before censoring, tetrachoric | simple_ltm | nominal truth |
| `affected` | `affected1`, `affected2` after censoring, tetrachoric | frailty | stress criterion only |
| `genetic_value` | the simulated `A1`, `A2`, Pearson (truth A 1, C 0) | all | no (diagnostic) |

## Gate

The gate applies to the stratified fit.  Its targets are A1, A2, C1, C2,
rA and rE in every simple_ltm scenario, plus rC where both traits'
nominal C is above 0: G = 38 targets.  The pooled fit is reported, not
gated.

Each target has a z, with MC SE = rep SD / sqrt(reps):

- **Binary (`uncensored_status`)**: z = (mean - nominal truth) / MC SE.
- **Continuous (`liability`)**: z = mean(d) / MC SE(d), where d = fit -
  realized truth per rep ([Realized truth](#realized-truth)).

A target passes when every rep gives a value (a liability target needs
its realized truth too) and |z| <= Phi^-1(1 - 0.05 / (2 G)).  That is
Bonferroni at familywise 5%.  The tool computes the threshold from G,
which gives 3.2125 for G = 38.  Each target also reports
`within_2_mc_se` (|z| <= 2), so near-misses stay visible.

The old per-target rule was |z| <= 2.  Under it, an unbiased estimator
fails at least one of 38 independent targets with probability 0.83
under a normal z.  That rises to 0.94 under t with 11 degrees of freedom
(12 reps) and 0.88 with 35 (36 reps).

### Realized truth

A Pearson cell in pg-phenotype is `rho = (1/N) Σ z1 z2` over the cell's
entries (`Blocks::stratified_pearson` in `crates/core/src/relative_pairs.rs`).
Each side is standardised by its own stratum's mean and 1/N SD over those
entries.  A pooled cell is the same with one stratum.  simACE stores
`L = A + C + E` per row up to the float32 rounding R of the stored
components (at most 5e-7).  With `L = A + C + E + R`, every liability cell
is exactly

    rho_L = Σ_{X,Y in A,C,E,R} (1/N) Σ_entries (X_i - m_{X,s}) (Y_j - m_{Y,u}) / (sd_{L,s} sd_{L,u})

Here `m_{X,s}` is component X's mean over the side-s stratum's entries,
and `sd_{L,s}` is the liability's 1/N SD there.  The tool rebuilds each
cell's entries from pedigree-graph's `relationship_pairs` on the
phenotyped rows.  It stops if the 16-term sum misses pg-phenotype's cell
by more than 1e-9.  In this run the largest miss was 2.8e-14 over every
cell of every rep, pooled and stratified (`decomposition` in
`results.json`).

The realized truth cell keeps the A x A and C x C terms, and E x E only
in the within-person cell.  The dropped terms have expectation 0 under
the simulation: E between relatives, every cross-component term (A x E,
C x E, A x C), and R (at most 2.3e-9 here).  The tool puts the realized
cells in place of the estimates (`dataclasses.replace`) and runs the
same `relative_pairs.ace` call on them.  The rows, weights and design
are therefore the liability fit's own.  NumPy's weighted least squares
over the same-trait rows of every fit matches the ACE Fit's A and C
within 5.6e-16 (`numpy_wls_max_gap`).

The realized truth replaces two simpler choices, both of which were
measured and found wanting.
- **Nominal truth.**  It failed six liability targets.  The simulated
  pedigree departs from nominal kinship, so the realized truth itself
  differs from nominal ([Why realized truth](#why-realized-truth)).
- **Component fits scaled by population variance shares.**  The error
  of that shortcut was 0.02 to 2.8 times the gate's MC SE on reps 1 to 3:
  rA 0.02 to 0.04, rC 0.36, and A, C and rE 0.2 to 2.8.  A cell's entries
  are pair-weighted, so its share of each component differs from the
  population's.  On `ltm_c02_rc07` rep 1, FS entries carry A 0.4984 and
  C 0.1962 against population shares of 0.5016 and 0.1974.

## Results

Rep means, with MC SE = SD / sqrt(reps) in parentheses.  The pooled
fit's mean(d) is within 0.0001 of the stratified one on every gated
liability target, so only stratified values are shown.  `results.json`
has both.

### `ltm_c0_re0` (36 reps)

| target | nominal | liability stratified | z vs nominal | realized truth | d = fit - realized | z | within 2 MC SE | |
|---|---|---|---|---|---|---|---|---|
| A1 | 0.5 | 0.4997 (0.0013) | -0.2 | 0.5013 | -0.0016 (0.0007) | -2.3 | no | pass |
| A2 | 0.5 | 0.5001 (0.0008) | 0.1 | 0.5002 | -0.0001 (0.0007) | -0.2 | yes | pass |
| C1 | 0 | -0.0005 (0.0005) | -1.0 | -0.0007 | 0.0003 (0.0004) | 0.6 | yes | pass |
| C2 | 0 | 0.0008 (0.0005) | 1.5 | 0.0004 | 0.0004 (0.0004) | 1.0 | yes | pass |
| rA | 0.5 | 0.4993 (0.0014) | -0.5 | 0.4992 | 0.0001 (0.0010) | 0.1 | yes | pass |
| rE | 0 | 0.0000 (0.0015) | 0.0 | -0.0005 | 0.0005 (0.0010) | 0.5 | yes | pass |

| target | nominal | uncensored status stratified | z | within 2 MC SE | |
|---|---|---|---|---|---|
| A1 | 0.5 | 0.4970 (0.0020) | -1.5 | yes | pass |
| A2 | 0.5 | 0.4972 (0.0019) | -1.4 | yes | pass |
| C1 | 0 | -0.0017 (0.0012) | -1.4 | yes | pass |
| C2 | 0 | 0.0013 (0.0014) | 0.9 | yes | pass |
| rA | 0.5 | 0.5035 (0.0030) | 1.1 | yes | pass |
| rE | 0 | 0.0012 (0.0032) | 0.4 | yes | pass |

### `ltm_c0_re02` (36 reps)

| target | nominal | liability stratified | z vs nominal | realized truth | d = fit - realized | z | within 2 MC SE | |
|---|---|---|---|---|---|---|---|---|
| A1 | 0.5 | 0.5026 (0.0013) | 2.0 | 0.5017 | 0.0009 (0.0008) | 1.1 | yes | pass |
| A2 | 0.5 | 0.5010 (0.0013) | 0.8 | 0.5011 | -0.0000 (0.0009) | -0.0 | yes | pass |
| C1 | 0 | -0.0007 (0.0004) | -1.6 | -0.0004 | -0.0003 (0.0004) | -0.8 | yes | pass |
| C2 | 0 | -0.0010 (0.0006) | -1.6 | -0.0005 | -0.0005 (0.0005) | -1.0 | yes | pass |
| rA | 0.5 | 0.5006 (0.0014) | 0.4 | 0.5010 | -0.0004 (0.0010) | -0.4 | yes | pass |
| rE | 0.2 | 0.1995 (0.0012) | -0.4 | 0.1999 | -0.0004 (0.0006) | -0.6 | yes | pass |

| target | nominal | uncensored status stratified | z | within 2 MC SE | |
|---|---|---|---|---|---|
| A1 | 0.5 | 0.5056 (0.0030) | 1.9 | yes | pass |
| A2 | 0.5 | 0.5028 (0.0024) | 1.2 | yes | pass |
| C1 | 0 | -0.0012 (0.0015) | -0.8 | yes | pass |
| C2 | 0 | -0.0036 (0.0014) | -2.5 | no | pass |
| rA | 0.5 | 0.4921 (0.0034) | -2.3 | no | pass |
| rE | 0.2 | 0.2015 (0.0033) | 0.4 | yes | pass |

### `ltm_c02_rc07` (36 reps)

| target | nominal | liability stratified | z vs nominal | realized truth | d = fit - realized | z | within 2 MC SE | |
|---|---|---|---|---|---|---|---|---|
| A1 | 0.5 | 0.5001 (0.0014) | 0.1 | 0.5016 | -0.0015 (0.0011) | -1.3 | yes | pass |
| A2 | 0.5 | 0.4991 (0.0012) | -0.7 | 0.5000 | -0.0008 (0.0010) | -0.9 | yes | pass |
| C1 | 0.2 | 0.2000 (0.0006) | 0.1 | 0.1992 | 0.0008 (0.0007) | 1.2 | yes | pass |
| C2 | 0.2 | 0.2000 (0.0006) | 0.1 | 0.2000 | 0.0001 (0.0006) | 0.1 | yes | pass |
| rA | 0.5 | 0.5018 (0.0020) | 0.9 | 0.5012 | 0.0006 (0.0014) | 0.4 | yes | pass |
| rC | 0.7 | 0.6975 (0.0017) | -1.5 | 0.7010 | -0.0035 (0.0014) | -2.5 | no | pass |
| rE | 0 | -0.0010 (0.0034) | -0.3 | -0.0011 | 0.0001 (0.0026) | 0.0 | yes | pass |

| target | nominal | uncensored status stratified | z | within 2 MC SE | |
|---|---|---|---|---|---|
| A1 | 0.5 | 0.4994 (0.0029) | -0.2 | yes | pass |
| A2 | 0.5 | 0.5004 (0.0018) | 0.2 | yes | pass |
| C1 | 0.2 | 0.2004 (0.0015) | 0.3 | yes | pass |
| C2 | 0.2 | 0.1994 (0.0013) | -0.5 | yes | pass |
| rA | 0.5 | 0.5005 (0.0035) | 0.1 | yes | pass |
| rC | 0.7 | 0.7023 (0.0056) | 0.4 | yes | pass |
| rE | 0 | -0.0022 (0.0065) | -0.3 | yes | pass |

### `frailty_gen_censoring` liability (12 reps, reported)

| target | nominal | liability stratified | realized truth | d = fit - realized | z |
|---|---|---|---|---|---|
| A1 | 0.5 | 0.5007 (0.0020) | 0.4992 | 0.0015 (0.0011) | 1.3 |
| A2 | 0.5 | 0.5041 (0.0027) | 0.5026 | 0.0014 (0.0013) | 1.1 |
| C1 | 0 | -0.0014 (0.0006) | -0.0007 | -0.0007 (0.0006) | -1.3 |
| C2 | 0 | -0.0012 (0.0005) | -0.0009 | -0.0003 (0.0005) | -0.5 |
| rA | 0.5 | 0.5024 (0.0029) | 0.5011 | 0.0013 (0.0016) | 0.8 |
| rE | 0 | 0.0002 (0.0025) | 0.0008 | -0.0007 (0.0020) | -0.3 |

### rC where C is 0 (not tested)

rC = sC / sqrt(C1 C2) is undefined (`nonpositive_component`) whenever
either C is at most 0, in the fit or in the realized truth.  When it is
defined, it is the ratio of two numbers near 0.  `results.json` lists
the reps by number.

| scenario | outcome | fit undefined | defined with abs(rC) > 1 | realized truth undefined |
|---|---|---|---|---|
| `ltm_c0_re0` | liability | 22/36 | 7 reps | 27/36 |
| `ltm_c0_re0` | uncensored status | 27/36 | 3 reps | |
| `ltm_c0_re02` | liability | 29/36 | 3 reps | 28/36 |
| `ltm_c0_re02` | uncensored status | 30/36 | 2 reps | |
| `frailty_gen_censoring` | liability | 11/12 | rep 6 | 11/12 |
| `frailty_gen_censoring` | affected | 4/12 (reps 2, 8, 9, 11) | reps 1, 6, 7 | |

### `estimate_rg`'s formula on the same cells (reported)

fitACE's `estimate_rg` computes `2 (r_wp - r_FS,01) / sqrt(h2_1 h2_2)`
with `h2_t = 2 r_FS,tt`.  It is evaluated here on the cells each fit
read, unclamped (`estimate_rg` clamps to [-1, 1]).  The values match
fitACE#5's predictions: about 0.9 at rE 0.2 and biased low at C 0.2.
The ACE Fit's rA has neither bias.

| scenario | outcome | formula, stratified | ACE rA, stratified |
|---|---|---|---|
| `ltm_c0_re0` | liability | 0.5006 (0.0025) | 0.4993 (0.0014) |
| `ltm_c0_re02` | liability | 0.9016 (0.0032) | 0.5006 (0.0014) |
| `ltm_c0_re02` | uncensored status | 0.9004 (0.0060) | 0.4921 (0.0034) |
| `ltm_c02_rc07` | liability | 0.2779 (0.0012) | 0.5018 (0.0020) |
| `ltm_c02_rc07` | uncensored status | 0.2766 (0.0032) | 0.5005 (0.0035) |

## Failures

None.  No gated target exceeds |z| 3.2125, and no rep is undefined.
The four targets outside 2 MC SE are listed in the result line.  All
four were outside 2 MC SE in the 36-rep run on the earlier build as well
([12 versus 36 reps](#12-versus-36-reps)).

### 12 versus 36 reps

Three runs bear on whether a near-miss is noise or bias.
- **Earlier build, 12 reps** (2026-10-15).
- **Earlier build, 36 reps** (2026-10-15, 2123 s).  Same tool code and
  rule as the 12-rep run; it was built before D7's revision.
- **This run.**  36 reps, with D7's weights.

A and C, fitted and realized, come from same-trait rows, which D7's
revision leaves unchanged.  rA, rC and rE also read cross-trait rows.

A bias holds its mean as reps triple, so its |z| grows by about
sqrt(3) = 1.73.  Noise does not.  These are every gated target with
|z| > 2 in any of the three runs:

| scenario | outcome | target | z, 12 reps | z, 36 reps (earlier build) | ratio | z, this run |
|---|---|---|---|---|---|---|
| `ltm_c0_re0` | liability (realized) | A1 | -2.58 | -2.35 | 0.91 | -2.35 |
| `ltm_c0_re0` | liability (realized) | C1 | 2.77 | 0.65 | 0.23 | 0.65 |
| `ltm_c02_rc07` | liability (realized) | rC | -2.74 | -2.86 | 1.04 | -2.52 |
| `ltm_c0_re02` | uncensored status | C2 | -1.85 | -2.54 | 1.37 | -2.54 |
| `ltm_c0_re02` | uncensored status | rA | -1.17 | -2.20 | 1.88 | -2.35 |

The three targets that were outside 2 at 12 reps did not grow: A1 shrank,
C1 fell to 0.65, and rC held.  Two targets crossed 2 only at 36 reps.
`uncensored_status` C2 grew by 1.37.  `uncensored_status` rA grew by
1.88, a bias's rate, and sits at -2.35 here: a mean of 0.4921 (0.0034)
against 0.5.  The liability rA of the same scenario has z -0.4 against
its realized truth, so the binary outcome reads rA slightly low and the
pedigree does not.  The cause is UNVERIFIED.  It is inside the
Bonferroni bound, and more reps would settle whether it is a bias of the
tetrachoric cross-trait cells at rE 0.2.

**The rC offset is not ratio-estimator bias.**  In this run, the
liability rC of `ltm_c02_rc07` sits -0.0035 (0.0014) from its realized
truth.  rC = sC / sqrt(C1 C2) is a ratio, so its noise could have a
non-zero mean.  A per-rep split of d_rC tests this.  The first-order
part, `rC (dsC / sC - dC1 / (2 C1) - dC2 / (2 C2))` evaluated at the
realized values, accounts for -0.0036 (0.0014).  The remainder, the
nonlinear part, is +0.00004.  Three small shifts line up through rC's
gradient: dsC -0.00040 (z -0.8), dC1 +0.00079 (z +1.2) and dC2 +0.00008
(z +0.1).  Pooled gives the same split, and so did the earlier build's
36 reps.  This was a scratch check on `results.json`, not part of the
tool.

## Why realized truth

These numbers are from the 12-rep run of 2026-10-15.  They involve A
and C only, which D7's revision does not change.  The `results.json`
fields named below now hold this run's 36-rep values.

Against the nominal truth, six liability targets failed under the old
2 MC SE rule (`ltm_c0_re0` A1; `ltm_c0_re02` A1, C1, C2; `ltm_c02_rc07`
C1, C2), at z -4.8 to 3.1.  The cause is the simulated pedigree, not
the fit.

**The truths are what was simulated.**  Within each phenotyped
generation, the realized share of each simulated component in the
liability, averaged by generation size, is A 0.4992 to 0.5013, C 0 or
0.1998 to 0.2001, and E 0.4988 to 0.5006 or 0.3000 to 0.3011
(`realized_shares` in `results.json`).

**Part of it is kinship that the nominal value leaves out.**
pedigree-graph's `pair_kinship` counts every relationship path.  Over the
phenotyped pairs, its mean is 1.0078 to 1.0083 times nominal for Av, GAv
and 1C in all four scenarios, where 1.2% to 1.6% of the pairs are related
through a second path.  For every other category it is 1.0000 to 1.0005
times nominal (`exact_kinship`).  These are the liability cells that ran
high against `2 phi A + s C`.  In `ltm_c0_re02`, seven Av, GAv and 1C
cells had z 2.1 to 3.9.

**The rest is in the simulated genetic values themselves.**
`genetic_value` fits the simulated A1 and A2, which nominal kinship
predicts give A 1 and C 0 exactly:

| scenario | A1 | A2 | C1 | C2 | MHS (0,0) realized r_A | PHS (0,0) realized r_A |
|---|---|---|---|---|---|---|
| `ltm_c0_re0` | 1.0012 (0.0015) | 0.9998 (0.0021) | -0.0009 (0.0012) | 0.0001 (0.0009) | 0.2485 | 0.2507 |
| `ltm_c0_re02` | 1.0073 (0.0024) | 1.0065 (0.0030) | -0.0031 (0.0008) | -0.0025 (0.0007) | 0.2474 | 0.2505 |
| `ltm_c02_rc07` | 1.0027 (0.0033) | 1.0046 (0.0024) | -0.0018 (0.0009) | -0.0013 (0.0009) | 0.2477 | 0.2503 |
| `frailty_gen_censoring` | 0.9995 (0.0033) | 1.0046 (0.0033) | -0.0014 (0.0010) | -0.0021 (0.0011) | 0.2490 | 0.2490 |

C is below 0 in 7 of 8 genetic-value fits.  MHS's realized genetic
correlation is below 0.25 in all four scenarios, though its exact kinship
is 1.0002 to 1.0003 times nominal.  Why the realized relatedness departs
from even the exact kinship is UNVERIFIED.  simACE gives each offspring
`midparent + N(0, A/2)` and copies A to the second MZ twin
(`simace/simulation/simulate.py:1054-1079`), and nothing there treats
maternal and paternal sibships differently.  The realized truth carries
all of this, A x A and C x C as simulated, so the gate tests the fit
rather than the pedigree.

## Stress: production frailty with `gen_censoring` (reported)

`affected1`/`affected2` after the default `gen_censoring`.  Each
generation has its own follow-up window (generation 3 from age 40 to 80,
generation 4 from 0 to 80, generation 5 from 0 to 45), so prevalence
differs by generation (fitACE#5).  Nominal A 0.5, C 0, rA 0.5.

| target | stratified | pooled |
|---|---|---|
| A1 | 0.1106 (0.0041) | 0.0809 (0.0035) |
| A2 | 0.1997 (0.0025) | 0.1387 (0.0025) |
| C1 | 0.0025 (0.0030) | 0.0489 (0.0031) |
| C2 | 0.0112 (0.0016) | 0.0903 (0.0017) |
| rA | **0.4911 (0.0193)** | 0.3059 (0.0233) |
| rE | 0.0046 (0.0027) | 0.0325 (0.0025) |
| `estimate_rg` formula | 0.4529 (0.0417) | 0.2938 (0.0252) |

The criterion from D10 requires the stratified rA to be defined in every
rep and closer to 0.5 than `estimate_rg`'s 0.308 from fitACE#5.  It
**passes**: all 12 reps are defined (0.368 to 0.602) and the mean is
0.491.  The pooled ACE rA, 0.306, is close to the `estimate_rg` formula
on the same pooled cells (0.294) and to fitACE#5's 0.308.  The earlier
build's cross-trait weights gave a pooled rA of 0.115.

A is attenuated to 22% (trait 1) and 40% (trait 2) of the truth even
when stratified.  Frailty onset and censoring are not a threshold of the
liability, so the tetrachoric reads a weaker correlation, and the
attenuation cancels in rA, a ratio.  Pooling generations adds spurious
C (0.049, 0.090) and rE (0.033), and stratifying removes almost all of
it (C 0.003, 0.011; rE 0.005).

## Reproducing

The tool reads `pedigree.parquet`, `cohort.parquet` and `run.yaml` from
`<results>/<scenario>/rep<k>/`.  pg-phenotype's env has no YAML parser,
so the tool reads only the config's `pedigree` blocks and `replicates`,
and `run.yaml`'s flat scalars, with a line reader.  It checks each
against the other.  The decomposition and `pair_kinship` both read
pedigree-graph (a declared test extra).  Two results came from scratch
checks, not the tool: the population-share measurement and the rC offset
split.  The earlier build's 12- and 36-rep figures came from this tool's
earlier runs, whose `results.json` is not kept here.
