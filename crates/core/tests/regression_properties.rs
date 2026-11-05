//! Symmetries of the parent-offspring regression (issue #31 item 5): the
//! parents' roles, an affine map of the trait, and the row order.

#![expect(
    clippy::expect_used,
    reason = "test helpers outside #[test], which clippy.toml does not exempt"
)]

mod common;

use std::sync::atomic::Ordering;

use common::{check, codes, pedigree, trait_of, values, Ped};
use pg_phenotype_core::regression::{
    parent_offspring, Fit, Parent, ParentOffspring, Reason, Settings,
};
use pg_phenotype_core::{PedigreeArg, TraitKind};
use proptest::prelude::*;

fn run(ped: &Ped, values: &[f64]) -> ParentOffspring {
    parent_offspring(
        PedigreeArg::Columns(ped.input()),
        trait_of(values, TraitKind::Continuous),
        None,
        Settings::default(),
    )
    .expect("a valid pedigree and trait")
}

fn fit(res: &ParentOffspring, parent: Parent) -> &Fit {
    res.pooled.fit(parent).expect("every line was asked for")
}

/// Cases per property in an ordinary run.
const CASES: u32 = 256;

/// Swapping the mother and father columns leaves the midparent line bit
/// for bit and exchanges the mother's and father's lines.
#[test]
fn swapping_the_parents_exchanges_their_lines() {
    check(
        "swapped parents",
        CASES,
        0.6,
        (pedigree(), codes()),
        |(ped, codes), checked| {
            let v = values(&codes, ped.len(), TraitKind::Continuous);
            let swapped = Ped {
                mother: ped.father.clone(),
                father: ped.mother.clone(),
                ..ped.clone()
            };
            let (a, b) = (run(&ped, &v), run(&swapped, &v));
            prop_assert_eq!(fit(&a, Parent::Midparent), fit(&b, Parent::Midparent));
            prop_assert_eq!(fit(&a, Parent::Mother), fit(&b, Parent::Father));
            prop_assert_eq!(fit(&a, Parent::Father), fit(&b, Parent::Mother));
            let defined = Parent::ALL
                .iter()
                .filter(|&&p| fit(&a, p).line.is_ok())
                .count();
            checked.fetch_add(defined, Ordering::Relaxed);
            Ok(())
        },
    );
}

/// Whether `p` and `q` agree to `tol` relative, or with one reason.
fn near(p: Result<f64, Reason>, q: Result<f64, Reason>, tol: f64) -> bool {
    match (p, q) {
        (Ok(s), Ok(t)) => (s - t).abs() <= tol * s.abs().max(t.abs()).max(1.0),
        (s, t) => s == t,
    }
}

/// `a y + b` on every row, with `a` a power of two of either sign, and a
/// shuffle of the rows leave each line's slope and r and the counts; the
/// SEs and the Wald p-value follow within a tolerance, and the intercept
/// becomes `a intercept + b (1 - slope)`.
#[test]
fn an_affine_map_and_a_row_shuffle_keep_the_slopes() {
    let strategy = (
        pedigree(),
        codes(),
        prop_oneof![Just(-4.0), Just(-0.5), Just(0.5), Just(2.0), Just(8.0)],
        -3i32..=3,
    )
        .prop_flat_map(|(ped, codes, a, b)| {
            let order = Just((0..ped.len()).collect::<Vec<_>>()).prop_shuffle();
            (Just(ped), Just(codes), Just(a), Just(b), order)
        });
    check(
        "affine",
        CASES,
        0.7,
        strategy,
        |(ped, codes, a, b, order), checked| {
            let v = values(&codes, ped.len(), TraitKind::Continuous);
            let mapped: Vec<f64> = order.iter().map(|&r| a * v[r] + f64::from(b)).collect();
            let (x, y) = (run(&ped, &v), run(&ped.permuted(&order), &mapped));
            for parent in Parent::ALL {
                let (p, q) = (fit(&x, parent), fit(&y, parent));
                prop_assert_eq!(
                    (p.n, p.n_mate_networks, p.dropped),
                    (q.n, q.n_mate_networks, q.dropped)
                );
                let (Ok(s), Ok(t)) = (p.line, q.line) else {
                    prop_assert_eq!(p.line.err(), q.line.err(), "{:?}", parent);
                    continue;
                };
                let tol = 1e-12;
                prop_assert!(
                    near(Ok(s.slope), Ok(t.slope), tol),
                    "{:?} slope {} vs {}",
                    parent,
                    s.slope,
                    t.slope
                );
                prop_assert!(near(s.r, t.r, tol), "{:?} r {:?} vs {:?}", parent, s.r, t.r);
                let intercept = a * s.intercept + f64::from(b) * (1.0 - s.slope);
                prop_assert!(
                    near(Ok(intercept), Ok(t.intercept), 1e-10),
                    "{:?} intercept {} vs {}",
                    parent,
                    intercept,
                    t.intercept
                );
                for (name, u, w) in [("se", s.se, t.se), ("se_ols", s.se_ols, t.se_ols)] {
                    prop_assert!(near(u, w, 1e-9), "{:?} {} {:?} vs {:?}", parent, name, u, w);
                }
                // An exact fit's SE is 0 or round-off, and which one
                // depends on the trait's location: `sandwich_se`'s bound
                // covers the scores' rounding, not the means'.  The SEs
                // agree within the tolerance above; the Wald test (none
                // for a zero SE, p = 0 for a round-off one) is compared
                // only past round-off.
                let past_round_off =
                    |se: Result<f64, Reason>| se.is_ok_and(|v| v > 1e-9 * s.slope.abs().max(1.0));
                if !(past_round_off(s.se) && past_round_off(t.se)) {
                    checked.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                prop_assert!(
                    near(s.wald.map(|w| w.p), t.wald.map(|w| w.p), 1e-8),
                    "{:?} p {:?} vs {:?}",
                    parent,
                    s.wald,
                    t.wald
                );
                checked.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        },
    );
}
