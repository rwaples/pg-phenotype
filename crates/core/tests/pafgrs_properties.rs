//! Properties of PA-FGRS: a proband's score reads its relatives' records,
//! never its own, and a case's age (its weight is 1 at any age) changes
//! nothing (issue #31 item 2).
//!
//! Then the bivariate score at `rg = 0` and under a trait swap (item 3),
//! within a tolerance, and the CIP table's interpolation.  Comparisons
//! are between two calls in one process: bit for bit where the issue
//! says the scores are equal, within [`close`] where the two calls
//! condition in different orders.

#![expect(
    clippy::expect_used,
    reason = "test helpers outside #[test], which clippy.toml does not exempt"
)]

mod common;

use std::sync::atomic::Ordering;

use common::{bits, check, close, pedigree, Ped};
use pg_phenotype_core::pafgrs::{
    prepare, score_bivariate, score_univariate, BivParams, BivScores, Cip, Interpolation, Prep,
    UniScores,
};
use pg_phenotype_core::{PedigreeArg, Trait, TraitKind};
use proptest::prelude::*;

/// A CIP table reaching prevalence 0.1 at 80, linear or a step.
fn cip(step: bool) -> Cip {
    let interpolation = if step {
        Interpolation::Step
    } else {
        Interpolation::Linear
    };
    Cip::new(
        vec![10.0, 30.0, 50.0, 80.0],
        vec![0.0, 0.03, 0.06, 0.1],
        interpolation,
    )
    .expect("a valid CIP table")
}

/// A status from a die: about a third cases, a few missing.
fn status(die: u8) -> f64 {
    match die % 10 {
        0 => f64::NAN,
        1..=3 => 1.0,
        _ => 0.0,
    }
}

/// An age from a die: 0 to 95 in steps of 5, or NaN.
fn age(die: u8) -> f64 {
    match die % 21 {
        20 => f64::NAN,
        k => 5.0 * f64::from(k),
    }
}

/// One trait's statuses and ages on `n` rows.
fn record(dice: &[u8], n: usize) -> (Vec<f64>, Vec<f64>) {
    let at = |r: usize, k: usize| dice[(2 * r + k) % dice.len()];
    (
        (0..n).map(|r| status(at(r, 0))).collect(),
        (0..n).map(|r| age(at(r, 1))).collect(),
    )
}

fn binary(values: &[f64]) -> Trait<'_> {
    Trait {
        values,
        kind: TraitKind::Binary,
        n_levels: None,
    }
}

fn prep(ped: &Ped, ndegree: i64) -> Prep {
    prepare(PedigreeArg::Columns(ped.input()), ndegree, None).expect("a valid pedigree")
}

/// The score position of `row` in a result's proband `ids`.
fn position(ids: &[i64], ped: &Ped, row: usize) -> usize {
    ids.iter()
        .position(|&id| id == ped.ids[row])
        .expect("every row is a proband")
}

/// Rows to perturb and what to: each row's new status and age dice.
fn perturbations() -> impl Strategy<Value = Vec<(usize, u8, u8)>> {
    prop::collection::vec((any::<usize>(), any::<u8>(), any::<u8>()), 1..=4)
}

/// The univariate inputs: a pedigree, its degree, one trait's dice, the
/// CIP table's kind, `h2`, and the perturbations.
type UniCase = (Ped, i64, Vec<u8>, bool, f64, Vec<(usize, u8, u8)>);

fn uni_case() -> impl Strategy<Value = UniCase> {
    (
        pedigree(),
        1i64..=3,
        prop::collection::vec(any::<u8>(), 256),
        any::<bool>(),
        prop_oneof![Just(0.2), Just(0.5), Just(0.8), Just(1.0)],
        perturbations(),
    )
}

/// A result's estimates as bits, whichever its kind.
fn est_bits<S: Ests>(s: &S) -> Vec<u64> {
    s.ests().iter().flat_map(|e| bits(e)).collect()
}

trait Ests {
    fn ests(&self) -> Vec<&[f64]>;
}

impl Ests for UniScores {
    fn ests(&self) -> Vec<&[f64]> {
        vec![&self.est]
    }
}

impl Ests for BivScores {
    fn ests(&self) -> Vec<&[f64]> {
        vec![&self.est[0], &self.est[1]]
    }
}

type UniBits<'a> = (&'a [i64], Vec<u64>, Vec<u64>, &'a [u32], [usize; 2]);

/// A univariate result's scores and counts, bit for bit.
fn uni_bits(s: &UniScores) -> UniBits<'_> {
    (
        &s.ids,
        bits(&s.est),
        bits(&s.var),
        &s.n_relatives,
        [s.controls_without_age, s.covariance_not_psd],
    )
}

/// A bivariate result's scores and counts, bit for bit.
type BivBits<'a> = (
    &'a [i64],
    [Vec<u64>; 4],
    Vec<u64>,
    &'a [u32],
    &'a [Vec<u32>; 2],
    [usize; 3],
);

fn biv_bits(s: &BivScores) -> BivBits<'_> {
    (
        &s.ids,
        [
            bits(&s.est[0]),
            bits(&s.est[1]),
            bits(&s.var[0]),
            bits(&s.var[1]),
        ],
        bits(&s.cov12),
        &s.n_relatives,
        &s.n_obs,
        [
            s.controls_without_age[0],
            s.controls_without_age[1],
            s.covariance_not_psd,
        ],
    )
}

fn uni(prep: &Prep, status: &[f64], age: &[f64], cip: &Cip, h2: f64) -> UniScores {
    score_univariate(prep, binary(status), age, cip, h2).expect("a valid trait")
}

/// Cases per property in an ordinary run: each prepares a pedigree and
/// scores it several times.
const CASES: u32 = 64;

/// Changing only row `i`'s status or age leaves `i`'s `est`, `var` and
/// `n_relatives` bit for bit.
#[test]
fn a_univariate_score_ignores_the_probands_own_record() {
    check(
        "uni own record",
        CASES,
        0.5,
        uni_case(),
        |(ped, ndegree, dice, step, h2, changes), checked| {
            let (prep, cip) = (prep(&ped, ndegree), cip(step));
            let (status, age) = record(&dice, ped.len());
            let base = uni(&prep, &status, &age, &cip, h2);
            for (row, s, a) in changes {
                let row = row % ped.len();
                let (mut status2, mut age2) = (status.clone(), age.clone());
                status2[row] = self::status(s);
                age2[row] = self::age(a);
                let got = uni(&prep, &status2, &age2, &cip, h2);
                prop_assert_eq!(&got.ids, &base.ids);
                let k = position(&base.ids, &ped, row);
                prop_assert_eq!(
                    got.est[k].to_bits(),
                    base.est[k].to_bits(),
                    "est of row {}",
                    row
                );
                prop_assert_eq!(
                    got.var[k].to_bits(),
                    base.var[k].to_bits(),
                    "var of row {}",
                    row
                );
                prop_assert_eq!(
                    got.n_relatives[k],
                    base.n_relatives[k],
                    "n_relatives of row {}",
                    row
                );
                // Counted where the change moved a relative's score.
                checked.fetch_add(
                    usize::from(est_bits(&got) != est_bits(&base)),
                    Ordering::Relaxed,
                );
            }
            Ok(())
        },
    );
}

/// Every case's age, replaced by another valid age or NaN, changes no
/// score: a case's weight is 1.
#[test]
fn a_cases_age_changes_no_univariate_score() {
    check(
        "uni case ages",
        CASES,
        0.3,
        (uni_case(), prop::collection::vec(any::<u8>(), 256)),
        |((ped, ndegree, dice, step, h2, _), new_ages), checked| {
            let (prep, cip) = (prep(&ped, ndegree), cip(step));
            let (status, age) = record(&dice, ped.len());
            let base = uni(&prep, &status, &age, &cip, h2);
            let rewritten: Vec<f64> = (0..ped.len())
                .map(|r| {
                    if status[r] == 1.0 {
                        self::age(new_ages[r % new_ages.len()])
                    } else {
                        age[r]
                    }
                })
                .collect();
            let got = uni(&prep, &status, &rewritten, &cip, h2);
            prop_assert_eq!(uni_bits(&got), uni_bits(&base));
            if bits(&rewritten) != bits(&age) && base.n_relatives.iter().any(|&n| n > 0) {
                checked.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        },
    );
}

/// The bivariate inputs: the univariate ones with a second trait, `h2`
/// per trait and `rg`.
type BivCase = (
    Ped,
    i64,
    [Vec<u8>; 2],
    [bool; 2],
    [f64; 2],
    f64,
    Vec<(usize, u8, u8, bool)>,
);

fn biv_case() -> impl Strategy<Value = BivCase> {
    let h2 = || prop_oneof![Just(0.2), Just(0.5), Just(0.8)];
    (
        pedigree(),
        1i64..=3,
        [
            prop::collection::vec(any::<u8>(), 256),
            prop::collection::vec(any::<u8>(), 256),
        ],
        [any::<bool>(), any::<bool>()],
        [h2(), h2()],
        prop_oneof![Just(-0.5), Just(0.0), Just(0.3), Just(0.9)],
        prop::collection::vec(
            (any::<usize>(), any::<u8>(), any::<u8>(), any::<bool>()),
            1..=4,
        ),
    )
}

fn biv(
    prep: &Prep,
    records: &[(Vec<f64>, Vec<f64>); 2],
    cips: &[Cip; 2],
    params: BivParams,
) -> BivScores {
    score_bivariate(
        prep,
        [binary(&records[0].0), binary(&records[1].0)],
        [&records[0].1, &records[1].1],
        [&cips[0], &cips[1]],
        params,
    )
    .expect("valid traits")
}

/// Changing only row `i`'s status or age on either trait leaves `i`'s
/// estimates, variances, `cov12`, `n_relatives` and `n_obs` bit for bit.
#[test]
fn a_bivariate_score_ignores_the_probands_own_record() {
    check(
        "biv own record",
        CASES / 2,
        0.4,
        biv_case(),
        |(ped, ndegree, dice, steps, h2, rg, changes), checked| {
            let prep = prep(&ped, ndegree);
            let cips = steps.map(cip);
            let params = BivParams::new(h2, rg, None).expect("valid parameters");
            let records = [record(&dice[0], ped.len()), record(&dice[1], ped.len())];
            let base = biv(&prep, &records, &cips, params);
            for (row, s, a, second) in changes {
                let row = row % ped.len();
                let mut changed = records.clone();
                let t = usize::from(second);
                changed[t].0[row] = status(s);
                changed[t].1[row] = age(a);
                let got = biv(&prep, &changed, &cips, params);
                prop_assert_eq!(&got.ids, &base.ids);
                let k = position(&base.ids, &ped, row);
                for t in 0..2 {
                    prop_assert_eq!(
                        got.est[t][k].to_bits(),
                        base.est[t][k].to_bits(),
                        "est{} of row {}",
                        t,
                        row
                    );
                    prop_assert_eq!(
                        got.var[t][k].to_bits(),
                        base.var[t][k].to_bits(),
                        "var{} of row {}",
                        t,
                        row
                    );
                    prop_assert_eq!(
                        got.n_obs[t][k],
                        base.n_obs[t][k],
                        "n_obs{} of row {}",
                        t,
                        row
                    );
                }
                prop_assert_eq!(
                    got.cov12[k].to_bits(),
                    base.cov12[k].to_bits(),
                    "cov12 of row {}",
                    row
                );
                prop_assert_eq!(
                    got.n_relatives[k],
                    base.n_relatives[k],
                    "n_relatives of row {}",
                    row
                );
                // Counted where the change moved a relative's score.
                checked.fetch_add(
                    usize::from(est_bits(&got) != est_bits(&base)),
                    Ordering::Relaxed,
                );
            }
            Ok(())
        },
    );
}

/// Every univariate score is finite with `var >= 0`, or `est` and `var`
/// are both NaN and the proband is counted in `covariance_not_psd` (issue
/// #32): the generator's inbred MZ twins give covariances that are not
/// positive semidefinite.
#[test]
fn a_univariate_score_is_finite_or_counted_as_not_psd() {
    check(
        "uni finite",
        CASES,
        1.0,
        uni_case(),
        |(ped, ndegree, dice, step, h2, _), checked| {
            let (prep, cip) = (prep(&ped, ndegree), cip(step));
            let (status, age) = record(&dice, ped.len());
            let base = uni(&prep, &status, &age, &cip, h2);
            let mut refused = 0;
            for (k, (e, v)) in base.est.iter().zip(&base.var).enumerate() {
                prop_assert!(
                    (e.is_finite() && v.is_finite() && *v >= 0.0) || (e.is_nan() && v.is_nan()),
                    "proband {}: est {} var {}",
                    base.ids[k],
                    e,
                    v
                );
                refused += usize::from(e.is_nan());
            }
            prop_assert_eq!(base.covariance_not_psd, refused);
            checked.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    );
}

/// Every case's age on either trait, rewritten, changes no bivariate
/// score.
#[test]
fn a_cases_age_changes_no_bivariate_score() {
    check(
        "biv case ages",
        CASES / 2,
        0.3,
        (biv_case(), prop::collection::vec(any::<u8>(), 256)),
        |((ped, ndegree, dice, steps, h2, rg, _), new_ages), checked| {
            let prep = prep(&ped, ndegree);
            let cips = steps.map(cip);
            let params = BivParams::new(h2, rg, None).expect("valid parameters");
            let records = [record(&dice[0], ped.len()), record(&dice[1], ped.len())];
            let base = biv(&prep, &records, &cips, params);
            let mut rewritten = records.clone();
            for (t, (status, age)) in rewritten.iter_mut().enumerate() {
                for r in 0..status.len() {
                    if status[r] == 1.0 {
                        age[r] = self::age(new_ages[(r + 32 * t) % new_ages.len()]);
                    }
                }
            }
            let got = biv(&prep, &rewritten, &cips, params);
            prop_assert_eq!(biv_bits(&got), biv_bits(&base));
            let ages = |r: &[(Vec<f64>, Vec<f64>); 2]| [bits(&r[0].1), bits(&r[1].1)];
            if ages(&rewritten) != ages(&records) && base.n_relatives.iter().any(|&n| n > 0) {
                checked.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        },
    );
}

/// With `rg = 0` (so `rho_within = 0`), each trait's bivariate `est`
/// and `var` are its univariate score, within a tolerance (the
/// bivariate conditions in another order), and `cov12` is 0.  Each
/// trait has its own missing statuses, so a relative may be observed on
/// one trait, both or neither.
#[test]
fn a_bivariate_score_with_rg_zero_is_two_univariate_scores() {
    check(
        "biv rg 0",
        CASES / 2,
        10.0,
        biv_case(),
        |(ped, ndegree, dice, steps, h2, _, _), checked| {
            let prep = prep(&ped, ndegree);
            let cips = steps.map(cip);
            let params = BivParams::new(h2, 0.0, None).expect("valid parameters");
            let records = [record(&dice[0], ped.len()), record(&dice[1], ped.len())];
            let joint = biv(&prep, &records, &cips, params);
            let unis = [0, 1].map(|t| uni(&prep, &records[t].0, &records[t].1, &cips[t], h2[t]));
            // The joint covariance is the two traits' blocks, so it is not
            // positive semidefinite exactly when one of them is not: the
            // joint score is then NaN on both traits and counted once.
            let refused = |k: usize| unis.iter().any(|u| u.est[k].is_nan());
            let mut n_refused = 0;
            for k in 0..joint.ids.len() {
                let all_nan = (0..2).all(|t| joint.est[t][k].is_nan() && joint.var[t][k].is_nan())
                    && joint.cov12[k].is_nan();
                prop_assert_eq!(all_nan, refused(k), "proband {}", joint.ids[k]);
                n_refused += usize::from(all_nan);
            }
            prop_assert_eq!(joint.covariance_not_psd, n_refused);
            for (t, one) in unis.iter().enumerate() {
                prop_assert_eq!(&joint.ids, &one.ids);
                for k in (0..one.est.len()).filter(|&k| !refused(k)) {
                    prop_assert!(
                        close(joint.est[t][k], one.est[k]) && close(joint.var[t][k], one.var[k]),
                        "trait {} proband {}: biv ({}, {}) vs uni ({}, {})",
                        t,
                        one.ids[k],
                        joint.est[t][k],
                        joint.var[t][k],
                        one.est[k],
                        one.var[k]
                    );
                    prop_assert_eq!(joint.n_obs[t][k], one.n_relatives[k]);
                    checked.fetch_add(usize::from(one.n_relatives[k] > 0), Ordering::Relaxed);
                }
            }
            prop_assert!(
                joint.cov12.iter().all(|&c| c == 0.0 || c.is_nan()),
                "{:?}",
                joint.cov12
            );
            Ok(())
        },
    );
}

/// Swapping the traits, with their ages, CIP tables and `h2`, swaps each
/// proband's `est`, `var` and `n_obs` and keeps `cov12` and `n_relatives`.
/// Equal heritabilities are left out: ties in the conditioning order then
/// break by trait index (`order.rs`), and the order changes the scores.
#[test]
fn swapping_the_traits_swaps_the_bivariate_scores() {
    check(
        "biv swap",
        CASES / 2,
        6.0,
        biv_case().prop_filter("distinct h2", |c| c.4[0] != c.4[1]),
        |(ped, ndegree, dice, steps, h2, rg, _), checked| {
            let prep = prep(&ped, ndegree);
            let cips = steps.map(cip);
            let records = [record(&dice[0], ped.len()), record(&dice[1], ped.len())];
            let a = biv(
                &prep,
                &records,
                &cips,
                BivParams::new(h2, rg, None).expect("valid"),
            );
            let [r0, r1] = records;
            let [c0, c1] = cips;
            let swapped = BivParams::new([h2[1], h2[0]], rg, None).expect("valid");
            let b = biv(&prep, &[r1, r0], &[c1, c0], swapped);
            for k in 0..a.ids.len() {
                for t in 0..2 {
                    prop_assert!(
                        close(a.est[t][k], b.est[1 - t][k]) && close(a.var[t][k], b.var[1 - t][k]),
                        "trait {} proband {}: ({}, {}) vs ({}, {})",
                        t,
                        a.ids[k],
                        a.est[t][k],
                        a.var[t][k],
                        b.est[1 - t][k],
                        b.var[1 - t][k]
                    );
                    prop_assert_eq!(a.n_obs[t][k], b.n_obs[1 - t][k]);
                }
                prop_assert!(
                    close(a.cov12[k], b.cov12[k]),
                    "cov12 of {}: {} vs {}",
                    a.ids[k],
                    a.cov12[k],
                    b.cov12[k]
                );
                prop_assert_eq!(a.n_relatives[k], b.n_relatives[k]);
                checked.fetch_add(usize::from(a.n_relatives[k] > 0), Ordering::Relaxed);
            }
            Ok(())
        },
    );
}

/// A valid CIP table of 1 to 8 knots, its interpolation, and two ages as
/// fractions of a knot interval (or below the first, or past the last).
fn cip_case() -> impl Strategy<Value = (Vec<f64>, Vec<f64>, bool, usize, [f64; 2])> {
    (1usize..=8)
        .prop_flat_map(|n| {
            (
                prop::collection::vec(0.01f64..20.0, n),
                prop::collection::vec(0.0f64..1.0, n),
                0.0f64..100.0,
                0.01f64..0.99,
                any::<bool>(),
                0..=n,
                [0.0f64..1.0, 0.0f64..1.0],
            )
        })
        .prop_map(|(gaps, rises, start, prevalence, step, interval, at)| {
            let ages: Vec<f64> = gaps
                .iter()
                .scan(start, |a, g| {
                    *a += g;
                    Some(*a)
                })
                .collect();
            let total: f64 = rises.iter().sum::<f64>().max(1e-3);
            let cip: Vec<f64> = rises
                .iter()
                .scan(0.0, |c, r| {
                    *c += r;
                    Some(*c)
                })
                .map(|c: f64| (prevalence * c / total).min(prevalence))
                .collect();
            let mut cip = cip;
            let last = cip.len() - 1;
            cip[last] = prevalence;
            (ages, cip, step, interval, at)
        })
}

/// `Cip::at` is the table's value at each knot and monotone within each
/// interval between knots (below the first and past the last included),
/// and `w` is in `[0, 1]`.  Across a knot it can fall by a few ulps, as a
/// NumPy mimic of the linear arm does (issue #31), so that is not asserted.
#[test]
fn the_cip_is_monotone_within_each_interval() {
    check(
        "cip intervals",
        256,
        1.0,
        cip_case(),
        |(ages, values, step, interval, at), checked| {
            let interpolation = if step {
                Interpolation::Step
            } else {
                Interpolation::Linear
            };
            let cip = Cip::new(ages.clone(), values.clone(), interpolation).expect("a valid table");
            for (a, v) in ages.iter().zip(&values) {
                prop_assert_eq!(cip.at(*a).to_bits(), v.to_bits(), "at the knot {}", a);
            }
            let n = ages.len();
            let (lo, hi) = match interval {
                0 => (ages[0] - 50.0, ages[0]),
                i if i == n => (ages[n - 1], ages[n - 1] + 50.0),
                i => (ages[i - 1], ages[i]),
            };
            // Ages in `[lo, hi)`: the interval's own knot, not the next.
            let [p, q] = at.map(|u| {
                (lo + u * (hi - lo))
                    .min(f64::from_bits(hi.to_bits() - 1))
                    .max(lo)
            });
            let (p, q) = (p.min(q), p.max(q));
            prop_assert!(
                cip.at(p) <= cip.at(q),
                "at({}) = {} > at({}) = {}",
                p,
                cip.at(p),
                q,
                cip.at(q)
            );
            for age in [p, q] {
                prop_assert!(
                    (0.0..=1.0).contains(&cip.w(age)),
                    "w({}) = {}",
                    age,
                    cip.w(age)
                );
            }
            checked.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    );
}

/// The issue #32 pedigree: founders 1, 2; their children 3, 4; 5, 6 the
/// children of 3 and 4; the MZ twins 7, 8 and their sib 9 the children of
/// 5 and 6.  The twins' `F` is 3/8, so their kinship is 11/16.
fn sib_mated() -> Ped {
    Ped {
        ids: (1..=9).collect(),
        mother: vec![-1, -1, 1, 1, 3, 3, 5, 5, 5],
        father: vec![-1, -1, 2, 2, 4, 4, 6, 6, 6],
        twin: vec![-1, -1, -1, -1, -1, -1, 8, 7, -1],
    }
}

/// With only the inbred MZ twins observed, both cases, the covariance of
/// each twin (`[[h2, 11/8 h2], [11/8 h2, 1]]`) is not positive
/// semidefinite above `h2 = 64/121`, and that of their parents and sib
/// (relatives' block `[[1, 11/8 h2], [11/8 h2, 1]]`) above `h2 = 8/11`.
/// A proband is refused, scored NaN and counted, exactly there: the twins
/// from 0.7, their parents and sib from 0.75.  At 0.75 conditioning alone
/// would not notice (the second twin's variance given the first is still
/// positive, and R PAFGRS's `pa_fgrs` returns 1.35655).  The scores not
/// refused equal `pa_fgrs`'s on the same covariance, to its printed
/// digits.
#[test]
fn inbred_mz_twins_are_refused_where_the_covariance_is_not_psd() {
    let ped = sib_mated();
    let prep = prep(&ped, 1);
    let cip = cip(false);
    let mut status = [NA; 9];
    status[6] = 1.0;
    status[7] = 1.0;
    let near = |x: f64, want: f64| (x - want).abs() < 5e-6;
    // h2, the est of 5, 6 and 9, the var of 7 and 8; NaN where refused.
    for (h2, sib_est, twin_var) in [
        (0.5, 1.11413, 0.107287),
        (0.7, 1.32979, NA),
        (0.75, NA, NA),
        (0.8, NA, NA),
        (1.0, NA, NA),
    ] {
        let s = uni(&prep, &status, &[NA; 9], &cip, h2);
        for k in 0..4 {
            assert_eq!((s.est[k], s.var[k], s.n_relatives[k]), (0.0, h2, 0), "{h2}");
        }
        for k in [4, 5, 8] {
            assert_eq!(s.n_relatives[k], 2);
            if sib_est.is_nan() {
                assert!(
                    s.est[k].is_nan() && s.var[k].is_nan(),
                    "h2 {h2} proband {k}"
                );
            } else {
                assert!(near(s.est[k], sib_est), "h2 {h2}: {}", s.est[k]);
                assert!(s.var[k] > 0.0);
            }
        }
        for k in [6, 7] {
            assert_eq!(s.n_relatives[k], 1);
            if twin_var.is_nan() {
                assert!(
                    s.est[k].is_nan() && s.var[k].is_nan(),
                    "h2 {h2} proband {k}"
                );
            } else {
                assert!(near(s.var[k], twin_var), "h2 {h2}: {}", s.var[k]);
            }
        }
        let refused = 3 * usize::from(sib_est.is_nan()) + 2 * usize::from(twin_var.is_nan());
        assert_eq!(s.covariance_not_psd, refused, "h2 {h2}");
    }
}

/// A second trait observed on rows the first is not: at `rg = 0` the
/// probands the first trait's covariance refuses score NaN on both traits
/// and are counted, and nobody else's second-trait score is touched.
#[test]
fn a_refused_bivariate_proband_is_nan_on_both_traits_and_counted() {
    let ped = sib_mated();
    let prep = prep(&ped, 1);
    let cips = [cip(false), cip(false)];
    let mut first = [NA; 9];
    first[6] = 1.0;
    first[7] = 1.0;
    let second = [1.0, 0.0, 1.0, 0.0, NA, NA, NA, NA, NA];
    let age = [60.0, 70.0, 40.0, 50.0, NA, NA, NA, NA, NA];
    let records = [
        (first.to_vec(), vec![NA; 9]),
        (second.to_vec(), age.to_vec()),
    ];
    let params = BivParams::new([0.8, 0.5], 0.0, None).expect("valid");
    let joint = biv(&prep, &records, &cips, params);
    let alone = uni(&prep, &second, &age, &cips[1], 0.5);
    assert_eq!(alone.covariance_not_psd, 0);
    assert_eq!(joint.covariance_not_psd, 5);
    for k in 0..9 {
        if k >= 4 {
            assert!(joint.est[0][k].is_nan() && joint.est[1][k].is_nan(), "{k}");
            assert!(joint.var[0][k].is_nan() && joint.var[1][k].is_nan(), "{k}");
            assert!(joint.cov12[k].is_nan(), "{k}");
        } else {
            assert!(
                close(joint.est[1][k], alone.est[k]) && close(joint.var[1][k], alone.var[k]),
                "{k}"
            );
            assert_eq!((joint.est[0][k], joint.var[0][k]), (0.0, 0.8), "{k}");
        }
    }
}

/// Twin 8 a case on both traits and twin 7 the proband, at an `h2` above
/// 64/121, where the joint covariance is not positive semidefinite:
/// refused.  Conditioning it would leave each posterior variance positive
/// and the two estimates' block with an eigenvalue near `-1e-8`, which
/// `pa.rs` tests on its own.
#[test]
fn a_bivariate_block_with_a_negative_eigenvalue_is_refused() {
    let ped = sib_mated();
    let prep = prep(&ped, 1);
    let cips = [cip(false), cip(false)];
    let mut status = [NA; 9];
    status[7] = 1.0;
    let records = [
        (status.to_vec(), vec![NA; 9]),
        (status.to_vec(), vec![NA; 9]),
    ];
    let params = BivParams::new([0.6114444314569172; 2], 0.1, None).expect("valid");
    let joint = biv(&prep, &records, &cips, params);
    assert!(joint.est[0][6].is_nan() && joint.cov12[6].is_nan());
    assert!(joint.covariance_not_psd >= 1);
}

/// Every bivariate proband has finite scores whose block is a covariance,
/// or is NaN throughout and counted; `rg` reaches both ends, where the
/// block is singular.
#[test]
fn a_bivariate_score_is_a_covariance_or_counted_as_not_psd() {
    check(
        "biv finite",
        CASES / 2,
        1.0,
        (
            biv_case(),
            prop_oneof![Just(-1.0), Just(-0.5), Just(0.3), Just(1.0)],
        ),
        |((ped, ndegree, dice, steps, h2, _, _), rg), checked| {
            let prep = prep(&ped, ndegree);
            let cips = steps.map(cip);
            let params = BivParams::new(h2, rg, None).expect("valid parameters");
            let records = [record(&dice[0], ped.len()), record(&dice[1], ped.len())];
            let s = biv(&prep, &records, &cips, params);
            let mut refused = 0;
            for k in 0..s.ids.len() {
                let all = [
                    s.est[0][k],
                    s.est[1][k],
                    s.var[0][k],
                    s.var[1][k],
                    s.cov12[k],
                ];
                if all.iter().all(|x| x.is_nan()) {
                    refused += 1;
                    continue;
                }
                let (v1, v2, c) = (s.var[0][k], s.var[1][k], s.cov12[k]);
                prop_assert!(
                    all.iter().all(|x| x.is_finite())
                        && v1 >= 0.0
                        && v2 >= 0.0
                        && c * c <= (v1 + 1e-9) * (v2 + 1e-9),
                    "proband {}: {:?}",
                    s.ids[k],
                    all
                );
            }
            prop_assert_eq!(s.covariance_not_psd, refused);
            checked.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    );
}

/// `got` against scores recorded as bits on x86_64 Linux: bit for bit
/// there, and within `1e-12` on a platform whose `exp` may round another
/// way.
fn assert_recorded(got: &[f64], want: &[u64], what: &str) {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        assert_eq!(bits(got), want, "{what}");
    }
    assert_eq!(got.len(), want.len(), "{what}");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        let w = f64::from_bits(*w);
        assert!((g - w).abs() <= 1e-12, "{what}[{k}]: {g} vs {w}");
    }
}

/// On a pedigree with no inbred row every score is, bit for bit, what the
/// scorer gave before it refused any proband (pg-phenotype `8d4bb5a`).
#[test]
fn scores_without_inbreeding_are_bit_for_bit_those_before_the_refusal() {
    let ped = outbred();
    let prep = prep(&ped, 2);
    let cip = cip(false);
    for (h2, est, var) in [
        (0.5, &UNI_EST_H2_50, &UNI_VAR_H2_50),
        (1.0, &UNI_EST_H2_100, &UNI_VAR_H2_100),
    ] {
        let s = uni(&prep, &OUTBRED_STATUS[0], &OUTBRED_AGE[0], &cip, h2);
        assert_recorded(&s.est, est, &format!("est at h2 {h2}"));
        assert_recorded(&s.var, var, &format!("var at h2 {h2}"));
        assert_eq!(s.n_relatives, [7, 6, 2, 3, 8, 8, 8, 2, 6, 7, 8, 7, 7, 2]);
        assert_eq!(s.covariance_not_psd, 0);
    }
    let records = [
        (OUTBRED_STATUS[0].to_vec(), OUTBRED_AGE[0].to_vec()),
        (OUTBRED_STATUS[1].to_vec(), OUTBRED_AGE[1].to_vec()),
    ];
    let params = BivParams::new([0.8, 0.5], -0.5, Some(-0.2)).expect("valid");
    let b = biv(&prep, &records, &[cip, self::cip(true)], params);
    assert_recorded(&b.est[0], &BIV_EST1, "est1");
    assert_recorded(&b.est[1], &BIV_EST2, "est2");
    assert_recorded(&b.var[0], &BIV_VAR1, "var1");
    assert_recorded(&b.var[1], &BIV_VAR2, "var2");
    assert_recorded(&b.cov12, &BIV_COV12, "cov12");
    assert_eq!(b.covariance_not_psd, 0);
}

const UNI_EST_H2_50: [u64; 14] = [
    0x3ff0f6ace259f991,
    0x3fe83db1709dab8b,
    0xbf80fc29901ea03c,
    0x3fdb7b20d1809829,
    0x3fe926291e49ceec,
    0x3fed238797b27df4,
    0x3ff6c703e8c62f62,
    0x3fcb901305714e17,
    0x3fe33fea07e8b8fb,
    0x3fe6016250999541,
    0x3fe5db8ee9959fcf,
    0x3fde9b2ccea12d19,
    0x3ff1eecf9d036e4b,
    0x3fdaf31114061d43,
];
const UNI_VAR_H2_50: [u64; 14] = [
    0x3fd80f63d453f6d9,
    0x3fda3e63628945ad,
    0x3fdfd7ff5da6db8e,
    0x3fdc8d27d29624d8,
    0x3fd97ded6d0220cc,
    0x3fd5d5636fe02ace,
    0x3fd0b24e702e0a02,
    0x3fdf1b6f8dec9759,
    0x3fdc4e2eb4f2eb14,
    0x3fdbc81d1f4b8043,
    0x3fdbbe06783c2431,
    0x3fda702ff2e27ab1,
    0x3fd18cff0af66b1a,
    0x3fdc8d510db9365f,
];
const UNI_EST_H2_100: [u64; 14] = [
    0x3ffb53d8da5ea85d,
    0x3ff3ac84af69d808,
    0xbf90e1b0c8829658,
    0x3feb73c93a3f0cdd,
    0x3ff29ccf75cc22ef,
    0x3ff55c15703d52c2,
    0x40004347bedbf784,
    0x3fdb7b679a71968b,
    0x3fee5d40ba33903f,
    0x3ff116f2127821c0,
    0x3ff0f22cf1228416,
    0x3fe76adc25a288b6,
    0x3ffd3496adcda8e5,
    0x3feb887683205a01,
];
const UNI_VAR_H2_100: [u64; 14] = [
    0x3fe36614f122391f,
    0x3fe6bff5a82972a2,
    0x3fefb04b627b9faf,
    0x3fe9276f9118341a,
    0x3fe5ef69db3f25d3,
    0x3fe04a51564e5f04,
    0x3fd19f1dc67aea4c,
    0x3fee3a0f3b4724d6,
    0x3fe9d57a40f35703,
    0x3fe8f8ddce7e8bc8,
    0x3fe8ea36d08fe328,
    0x3fe67c1fd892fe98,
    0x3fc8f342dd09b05c,
    0x3fe92e9bd439c4d3,
];
const BIV_EST1: [u64; 14] = [
    0x3ff517bf52660fa3,
    0x3febba44a6ac87fe,
    0xbfc2ce3837e75771,
    0x3fe30e254e502dc2,
    0x3fe338062d4d410d,
    0x3fe638e05658fd0e,
    0x3ff6e0c5b4a39c52,
    0x3fbccaf0129627dc,
    0x3fe7317d55ab1c8e,
    0x3fe7269a48267f36,
    0x3fec2338bccc4d0c,
    0x3fcc95dd9f61aa60,
    0x3ff5a803d21c0937,
    0x3fe5c3fdb8ab4f1b,
];
const BIV_EST2: [u64; 14] = [
    0x3fce0ce6109fff32,
    0x3fda370540b9e787,
    0x3fcc73e55100d141,
    0xbfb1548b4411bbd4,
    0x3fea1922195e69cc,
    0x3fea7758e0c4a54e,
    0x3fe2edfac49920a9,
    0x3fd0a16417432c7c,
    0x3fc2f9f0e32e12cd,
    0x3fd8e2a6ede648a4,
    0x3f9ec884da803ae0,
    0x3fec7b0f3e726786,
    0x3fe1880388cfc608,
    0xbfd1602f1986e744,
];
const BIV_VAR1: [u64; 14] = [
    0x3fe067c58c4a4560,
    0x3fe2cb0d1b32b569,
    0x3fe93ed6aa6d8c2e,
    0x3fe50eb1543305c3,
    0x3fe1b0395c8800f9,
    0x3fda9bc5148cc7fa,
    0x3fce090578a04d88,
    0x3fe7b4f1b3809e16,
    0x3fe53187449a549f,
    0x3fe44620e00c5cb0,
    0x3fe48c10d8aad01e,
    0x3fe2419cc15b0f6d,
    0x3fcda3b8b6428892,
    0x3fe53afef054ef2f,
];
const BIV_VAR2: [u64; 14] = [
    0x3fd944d6ecf3eecd,
    0x3fda13bacde3f4cc,
    0x3fdf1eda66e3dbb2,
    0x3fddb492d4b51e41,
    0x3fd65c5eed2f02bf,
    0x3fcf4eea78ba0147,
    0x3fccdefd3d964a49,
    0x3fdb6be26e0e32fa,
    0x3fdcfb636a160fa0,
    0x3fda9c63be6f086f,
    0x3fdcd24a7640b948,
    0x3fd680149611159f,
    0x3fd4ac4cd56e7786,
    0x3fde6f4c1c772fac,
];
const BIV_COV12: [u64; 14] = [
    0xbfc88a3924946154,
    0xbfcc2bec9baf2b1a,
    0xbfd393d9006e5685,
    0xbfd052e23050ed80,
    0xbfc77376c7d4c478,
    0xbfb8bce6f3af4c05,
    0xbfa3aa532fbead00,
    0xbfd0d0da0d0d7304,
    0xbfd05d73734f56bf,
    0xbfcdbee1aa2cf749,
    0xbfcfc94fc9c4a093,
    0xbfc7e89964bae31c,
    0xbfb3dbe6364bcb99,
    0xbfd0bd352e6c2291,
];

const NA: f64 = f64::NAN;

/// Two trait records on [`outbred`]'s rows.
const OUTBRED_STATUS: [[f64; 14]; 2] = [
    [
        1.0, 0.0, NA, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, NA, 1.0, 0.0, 0.0,
    ],
    [
        0.0, 1.0, 0.0, NA, 0.0, 1.0, 1.0, 0.0, 0.0, NA, 1.0, 0.0, 0.0, 1.0,
    ],
];
const OUTBRED_AGE: [[f64; 14]; 2] = [
    [
        50.0, 70.0, NA, NA, 40.0, 35.0, 60.0, 20.0, NA, 15.0, 10.0, 25.0, 30.0, 55.0,
    ],
    [
        65.0, 45.0, 80.0, NA, 30.0, 50.0, 25.0, NA, 35.0, 12.0, 18.0, NA, 40.0, 60.0,
    ],
];

/// A pedigree with no inbred row: founders 1 to 4 and 14; 5 and the MZ
/// twins 6, 7 are children of 1 and 2; 8 of 3 and 4; 9 of 1 and 4 (a half
/// sib of both families); 10, 11 of 5 and 8; the MZ twins 12, 13 of 14
/// and 6.
fn outbred() -> Ped {
    Ped {
        ids: (1..=14).collect(),
        mother: vec![-1, -1, -1, -1, 1, 1, 1, 3, 1, 5, 5, 14, 14, -1],
        father: vec![-1, -1, -1, -1, 2, 2, 2, 4, 4, 8, 8, 6, 6, -1],
        twin: vec![-1, -1, -1, -1, -1, 7, 6, -1, -1, -1, -1, 13, 12, -1],
    }
}
