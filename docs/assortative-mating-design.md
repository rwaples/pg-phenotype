# About the assortative-mating design

These notes explain why `mate_correlation` computes what it does.  The
[reference](assortative-mating.md) says what it computes and returns.  The
method was designed in pedsum #13 and ported here unchanged, so every choice
below is pedsum's. [ADR 0005](adr/0005-assortative-mating-reproduces-pedsums-numerics.md)
records the one decision the port added, which is to reproduce pedsum's
numbers to the bit.

The code lives in `crates/core/src/assortative/`.  `sample.rs` builds the
Mating Pairs, Mate Networks and strata. `estimators.rs` holds the point
fits. `sandwich.rs`, `bootstrap.rs` and `permutation.rs` hold the
inference. `kernels.rs` holds every pass over a cell's pairs, and `mod.rs`
assembles a cell.

## The Mate Network is the cluster

Mating Pairs that share a parent are not independent.  A father with three
mates puts his value into three pairs, and treating those pairs as
independent would understate the variance.  A Mate Network is the smallest
unit that holds every pair sharing a parent, so the sandwich SE sums
influences within networks and the bootstrap resamples whole networks.
Each cell finds its networks over its own analysed pairs.

Clustering by pedigree component would also hold relatives together, but a
pedigree that is one component would leave one cluster and no SE.  The cost
of the Mate Network is an assumption: networks are taken as independent,
and dependence between them through ancestry or siblings is not modelled.
`method["bootstrap_assumption"]` states it, and each cell reports
`n_mate_networks` and `largest_mate_network_share` so a reader can see when
a few networks dominate.

## The default CI is a two-step sandwich

Refitting every estimator 1,000 times per cell does not scale to 10^7
pairs, so the bootstrap is opt-in.  The default `se` stacks the first-step
estimating equations (thresholds per sex x stratum, stratum means and 1/N
variances) with the rho score.  The first step's uncertainty reaches the
estimate through the cross term `A_rho_theta` (Olsson 1979 eqs 21-28;
Olsson, Drasgow & Dorans 1982 eq 37).  Each nuisance equation involves one
parameter, so `A_theta_theta` is diagonal and each estimator's influence is
one pass over its pairs.

The interval is Wald on the Fisher-z scale (log for the odds ratio), so the
bounds stay inside (-1, 1) or (0, inf).  At a boundary fit the sandwich is
not valid, so the CI is withheld with `boundary`.  Spearman has no influence
function and gets a CI only from the bootstrap.  pedsum measured the
sandwich SE within a ratio of 0.93 to 1.06 of the refit bootstrap SD on 400
remated pairs in 200 networks.

## A bootstrap draw refits the first step and takes one Newton step

Thresholds, stratum means and standard deviations are estimated from the
same sample as rho.  Holding them at their full-sample values in each draw
would treat them as known and narrow the CI, so every draw recomputes them
on its weighted sample.  rho then moves one Newton step from the estimate,
with the draw's score and the full-sample Hessian.

pedsum chose the full-sample Hessian over the draw's own by measurement:
its worst CI bound came within 0.0012 of a full refit at 2,500 remated
pairs and 0.0006 at 5,000, against 0.0048 and 0.0024 with the draw's own
Hessian.  A draw whose Hessian is not positive is refit in full, by the
same function the observed fit calls.

## The permutation null exchanges father vectors within blocks

A permutation moves trait vectors between distinct fathers, both traits
together, and leaves the mothers in place.  A father therefore carries one
vector to all his mates, and the Mating Pair structure (who mated with how
many) stays fixed.  Fathers swap only within their block, father stratum x
which traits the father has.  The stratum part keeps the null from mixing
cohorts.  The missingness part keeps each cell's eligible pairs, and so its
`n`, the same in every permutation.

The blocks are the same for the crude and the stratified form.  So with
`stratum`, the crude p-value also tests within-stratum exchangeability,
while the crude estimate and CI describe the pooled correlation.  A cohort
trend with no assortment within strata therefore moves the crude CI away
from 0 but not the crude p-value.  Pooling the crude donors across strata
would make the crude test reject on the trend itself, which is the
confound that stratifying is there to remove.  This is the right trade:
the p-value answers the assortment question, and the crude estimate stays
an honest description of the pooled sample.

## The statistic is the score at rho = 0

Refitting rho on every permutation needs an optimiser per draw and per
cell.  The score of the log-likelihood at rho = 0 needs none.  With `e` each
side's conditional latent mean given its value (`z` for a continuous side,
`(phi(tau_c) - phi(tau_{c+1})) / p_c` for level `c` of a discrete side),
the statistic is `sum e_m e_f / sqrt(sum e_m^2 sum e_f^2)`, the score scaled
to a correlation.  For two continuous traits it is Pearson's r.

Every father margin is refit on the permuted pairs, because pair-weighted
margins move when remating fathers swap.  The mothers' scores do not move,
so they are summed per father once, and each draw streams each cell's
distinct fathers once.

## The p-value stops sequentially

Most cells of a run are null or weak, and 999 draws of each are wasted
once it is clear there is no evidence against the null.  The p-value is
the closed sequential scheme of Besag & Clifford (1991): read the draws in
order and stop at the `h`-th valid draw at least as extreme as the observed
statistic, ties counted, with `h = 20` (the paper suggests 10 or 20).  Then
`p = h / l`, with `l` the valid draws so far. A test that never reaches `h`
falls back to the fixed-size `(g + 1) / (B + 1)`.  Under the null `p` is a
uniform rounded up to its support, so the test is exact at every size.  A
strongly significant cell never collects `h` exceedances and runs every
draw, so the scheme saves time only on null and weak cells.

The stopping draw must be a function of the draw sequence alone.  Draws
therefore run in batches whose sizes depend on the draws so far, never on
the thread count: 64 first, then at least double the draws so far and far
enough that the nearest open test would reach `h` at its rate so far.  A
strong-signal run takes two batches instead of sixteen.

## Threshold estimators fit rho on (-0.9999, 0.9999)

`polychoric` (tetrachoric at two levels) and `polyserial` (biserial at two
levels) take thresholds in closed form from the margins (Olsson 1979 eqs
15-18; Olsson, Drasgow & Dorans 1982 eq 36).  That leaves a one-dimensional
search for rho, kept inside (-0.9999, 0.9999) because the bivariate normal
is singular at ±1.  Newton's method on the analytic score finds rho, and
bounded Brent takes over when Newton cannot, so every fit reaches the
optimum Brent would.  Olsson's appendix A2 prints the rho-derivative of the
bivariate normal density with two errors. `bvn::pdf_and_drho` uses the
corrected form.

Phi2 is Owen's (1956) closed form in Owen's T function (Patefield & Tandy
2000), evaluated on the corner grids of a table, never per pair.

`boundary` comes from the fit, never from a zero count, because a zero cell
does not put rho at a bound: the 3 x 3 table with an empty centre fits rho
near 0.  The flag is set when rho is within 1e-3 of a bound, or when the
likelihood at the nearer bound is within 1e-6 (relative) of the optimum.
The second rule exists because a likelihood can be flat up to the bound.
On the table `[[30, 10], [0, 20]]` the negative log-likelihood is
60.684256 from 0.998 to 0.9999, the optimiser stops at 0.9986, and the
distance rule alone misses the boundary.

## Margins count each pair once

The Mate Correlation has one observation per Mating Pair, so every nuisance
parameter comes from the same observations.  Means, standard deviations,
thresholds and tables are computed over the cell's pairwise-complete pairs,
and a father with three mates counts three times.  Margins weighted per
person would standardise against a different sample from the one the
correlation runs over.  The Within-Person Cross-Trait Correlation is a
statistic about people, so it counts each person once.

## Thin strata are dropped by network count

A sex x stratum whose pairs come from a few networks gives thresholds and
standard deviations that swing between bootstrap draws.  It is often absent
or constant in a draw, which fails the draw and can withhold the CI.  The
alternatives were withholding the CI, or skipping the stratum inside a
draw, which changes the estimand from draw to draw.  So each cell drops,
before any fit, every sex x stratum whose pairs span fewer than
`min_stratum_networks` networks, and every constant one.  It counts
networks, not pairs, because the network is the cluster.  The default of
10 is an arbitrary choice.

A pair's mother and father strata can differ, so dropping one stratum's
pairs can remove another stratum's networks or leave it constant.  Both
rules therefore repeat, with networks recounted on the kept pairs, until
neither drops a pair.  The crude and stratified estimates share the kept
sample, so their difference shows only the adjustment.

## Results do not depend on the thread count

Three rules keep the result a function of the input and `seed` alone:

- Every draw takes its random numbers from SplitMix64 keyed by
  (seed, draw), with a separate stream domain for the bootstrap, so a draw
  is the same whichever worker runs it (`rng.rs`).
- Every pass over a cell's pairs adds fixed blocks of 16,384 pairs in block
  order (`kernels::per_block`), so a floating-point sum does not depend on
  how many workers share the blocks.
- Sequential stopping batches by draw count.

The permutation pass stores the fathers' trait values and summed mother
scores as float32 and accumulates in float64.  pedsum chose that because
the pass is memory-bound: it halved the pass's time at 12 threads on
3 x 10^6 pairs and moves a statistic by about 1e-7 relative.  A continuous
trait is standardised in float64 before the float32 cast, so values near
2e7 keep their variation.

## Calibration

`tests/test_assortative_calibration.py` simulates repeated datasets with a
known mate correlation and checks the inference against binomial Monte
Carlo bounds of 3 SD.  Under no assortment, with no remating, heavy
remating and a sparse binary trait, the permutation test rejects at most
alpha + 3 SD at alpha = 0.05 over 500 datasets, crude and stratified.
Under a birth-decade trend shared by both mates, the stratified test holds
the same bound.  At a mate correlation of 0.3 every sandwich CI covers the
truth at 95% within 3 SD over 500 datasets, and coverage above 0.995 fails
as too wide.  The one-step bootstrap (399 draws) is checked the same way
over 200 remated datasets.
