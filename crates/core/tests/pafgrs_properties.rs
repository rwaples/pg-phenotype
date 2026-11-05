//! Properties of PA-FGRS: a proband's score reads its relatives' records,
//! never its own, and a case's age (its weight is 1 at any age) changes
//! nothing (issue #31 item 2).
//!
//! Every comparison is between two calls in one process, bit for bit.

#![expect(
    clippy::expect_used,
    reason = "test helpers outside #[test], which clippy.toml does not exempt"
)]

mod common;

use std::sync::atomic::Ordering;

use common::{check, pedigree, Ped};
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

/// Floats as bits, so NaN equals NaN.
fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A univariate result's scores and counts, bit for bit.
fn uni_bits(s: &UniScores) -> (&[i64], Vec<u64>, Vec<u64>, &[u32], usize) {
    (
        &s.ids,
        bits(&s.est),
        bits(&s.var),
        &s.n_relatives,
        s.controls_without_age,
    )
}

/// A bivariate result's scores and counts, bit for bit.
type BivBits<'a> = (
    &'a [i64],
    [Vec<u64>; 4],
    Vec<u64>,
    &'a [u32],
    &'a [Vec<u32>; 2],
    [usize; 2],
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
        s.controls_without_age,
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

/// Every univariate score is finite.  Fails: the PA covariance puts `h2`
/// on the proband's diagonal and 1 on each relative's whatever their
/// inbreeding (`uni.rs`), while its off-diagonal `2 phi h2` carries it, so
/// with inbred MZ twins (kinship `(1 + F) / 2`) the matrix is not positive
/// semidefinite.  After two generations of full-sib mating (`F = 3/8`) a
/// concordant MZ case pair's own variance clamps to 0 from `h2 = 0.7`, and
/// their parents and sib score NaN from `h2 = 0.8`.
#[test]
#[ignore = "finding: NaN scores with inbred MZ twins at high h2; notes/handoff-pgp-issue-31.md"]
fn univariate_scores_are_finite() {
    check(
        "uni finite",
        CASES,
        1.0,
        uni_case(),
        |(ped, ndegree, dice, step, h2, _), checked| {
            let (prep, cip) = (prep(&ped, ndegree), cip(step));
            let (status, age) = record(&dice, ped.len());
            let base = uni(&prep, &status, &age, &cip, h2);
            for (k, (e, v)) in base.est.iter().zip(&base.var).enumerate() {
                prop_assert!(
                    e.is_finite() && v.is_finite(),
                    "proband {}: est {} var {}",
                    base.ids[k],
                    e,
                    v
                );
            }
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

/// Whether two scores agree to `1e-10`, NaN matching NaN.
fn close(p: f64, q: f64) -> bool {
    (p.is_nan() && q.is_nan()) || (p - q).abs() <= 1e-10
}

/// With `rg = 0` (so `rho_within = 0`) and both traits observed, each
/// trait's bivariate `est` and `var` are its univariate score, within a
/// tolerance (the bivariate conditions in another order), and `cov12` is
/// 0.
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
            // Probands a trait scores NaN on alone (the finding of
            // `univariate_scores_are_finite`) are left out: the joint
            // score can be NaN on the other trait too.
            let finite = |k: usize| {
                unis.iter()
                    .all(|u| u.est[k].is_finite() && u.var[k].is_finite())
            };
            for (t, one) in unis.iter().enumerate() {
                prop_assert_eq!(&joint.ids, &one.ids);
                for k in (0..one.est.len()).filter(|&k| finite(k)) {
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
