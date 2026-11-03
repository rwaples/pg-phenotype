//! Properties of the tetrachoric correlation of a 2 x 2 table.
//!
//! Bit-for-bit comparisons are between two calls in one process, never
//! against a stored value, so they hold on every platform.

#![expect(
    clippy::expect_used,
    reason = "test helpers outside #[test], which clippy.toml does not exempt"
)]

mod common;

use std::sync::atomic::Ordering;

use common::check;
use pg_phenotype_core::correlation::{tetrachoric, tetrachoric_pairs, Tetrachoric};
use pg_phenotype_core::polychoric::{Point, Reason};
use proptest::prelude::*;

type Table = [[u64; 2]; 2];

/// Cases per property in an ordinary run.
const CASES: u32 = 256;

/// Tables with every cell populated, or one cell near zero so that some
/// fits stop at a bound.
fn table() -> impl Strategy<Value = Table> {
    table_upto(2000)
}

/// [`table`] with cells up to `max`.
fn table_upto(max: u64) -> impl Strategy<Value = Table> {
    let cell = move || 0u64..=max;
    prop_oneof![
        (cell(), cell(), cell(), cell()).prop_map(|(a, b, c, d)| [[a, b], [c, d]]),
        (0u64..=3, cell(), cell(), cell(), 0usize..4).prop_map(|(small, b, c, d, at)| {
            let mut t = [small, b, c, d];
            t.swap(0, at);
            [[t[0], t[1]], [t[2], t[3]]]
        }),
    ]
}

fn run(t: Table) -> Tetrachoric {
    tetrachoric(t).expect("counts are below MAX_COUNT")
}

fn point(t: Table) -> Result<Point, Reason> {
    run(t).outcome.map(|e| e.point)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(1.0)
}

/// `-0.0` and `0.0` as one value: every zero rho is `+0` whatever the
/// layout, so `-1 * +0` is not what a reversed layout returns.
fn bits(x: f64) -> u64 {
    (x + 0.0).to_bits()
}

/// Issue #16 for every table: transposing or rotating the table keeps
/// rho, reversing one side's levels negates it, bit for bit.  The SE
/// stays and the interval flips with rho.
#[test]
fn every_layout_gives_one_rho_up_to_its_sign() {
    check("layouts", CASES, 2.3, table(), |t, checked| {
        let [[a, b], [c, d]] = t;
        let base = run(t).outcome;
        for (layout, sign) in [
            ([[a, c], [b, d]], 1.0),
            ([[d, c], [b, a]], 1.0),
            ([[d, b], [c, a]], 1.0),
            ([[c, d], [a, b]], -1.0),
            ([[c, a], [d, b]], -1.0),
            ([[b, a], [d, c]], -1.0),
            ([[b, d], [a, c]], -1.0),
        ] {
            match (&base, run(layout).outcome) {
                (Ok(p), Ok(q)) => {
                    let (pp, qp) = (p.point, q.point);
                    prop_assert_eq!(bits(qp.value), bits(sign * pp.value), "{:?}", layout);
                    prop_assert_eq!(qp.boundary, pp.boundary, "{:?}", layout);
                    match (p.se, q.se) {
                        (Ok(x), Ok(y)) => {
                            prop_assert!(close(x, y), "{:?}: SE {} vs {}", layout, x, y)
                        }
                        (x, y) => prop_assert_eq!(x.err(), y.err(), "{:?}", layout),
                    }
                    match (p.ci, q.ci) {
                        (Ok(x), Ok(y)) => {
                            let [lo, hi] = x.bounds;
                            let want = if sign > 0.0 { [lo, hi] } else { [-hi, -lo] };
                            let [ylo, yhi] = y.bounds;
                            prop_assert!(
                                close(ylo, want[0]) && close(yhi, want[1]),
                                "{:?}: CI {:?} vs {:?}",
                                layout,
                                y.bounds,
                                want
                            );
                        }
                        (x, y) => prop_assert_eq!(x.err(), y.err(), "{:?}", layout),
                    }
                    checked.fetch_add(1, Ordering::Relaxed);
                }
                (p, q) => prop_assert_eq!(p.map(|_| ()), q.map(|_| ()), "{:?}", layout),
            }
        }
        Ok(())
    });
}

/// A defined rho is in [-1, 1]; its SE is finite and positive, and its
/// interval contains it.
#[test]
fn estimates_are_in_range() {
    check("range", CASES, 0.3, table(), |t, checked| {
        let Ok(e) = run(t).outcome else {
            return Ok(());
        };
        prop_assert!((-1.0..=1.0).contains(&e.point.value), "{}", e.point.value);
        if let Ok(se) = e.se {
            prop_assert!(se.is_finite() && se > 0.0, "{}", se);
        }
        if let Ok(ci) = e.ci {
            let [lo, hi] = ci.bounds;
            let tol = 1e-12;
            prop_assert!(
                lo - tol <= e.point.value && e.point.value <= hi + tol,
                "{:?}",
                ci
            );
        }
        checked.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });
}

/// The likelihood reads the counts as proportions, so scaling them
/// moves rho only by the optimizer's stopping tolerance (`xatol` 1e-7).
/// Not at a bound: there the likelihood is flat and the fit stops
/// elsewhere (`[[177, 0], [1307, 33]]` gives 0.944, five times it
/// 0.910).
#[test]
fn scaling_the_counts_keeps_an_interior_rho() {
    check(
        "scaling",
        CASES,
        0.3,
        (table(), 2u64..=8),
        |(t, k), checked| {
            let scaled = t.map(|row| row.map(|c| c * k));
            let (Ok(p), Ok(q)) = (point(t), point(scaled)) else {
                return Ok(());
            };
            if p.boundary == Some(true) || q.boundary == Some(true) {
                return Ok(());
            }
            prop_assert!(
                (p.value - q.value).abs() < 1e-6,
                "{} vs {}",
                p.value,
                q.value
            );
            checked.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    );
}

/// A table and its pairs, with pairs missing a value, in any order.
fn table_and_pairs() -> impl Strategy<Value = (Table, Vec<(f64, f64)>)> {
    table_upto(750).prop_flat_map(|t| {
        let mut rows = Vec::new();
        for (i, row) in t.iter().enumerate() {
            for (j, &count) in row.iter().enumerate() {
                rows.extend((0..count).map(|_| (i as f64, j as f64)));
            }
        }
        let missing = prop::collection::vec(
            prop_oneof![
                (0u8..=1).prop_map(|y| (f64::NAN, f64::from(y))),
                (0u8..=1).prop_map(|x| (f64::from(x), f64::NAN)),
                Just((f64::NAN, f64::NAN)),
            ],
            0..20,
        );
        (Just(t), missing).prop_flat_map(move |(t, missing)| {
            let all: Vec<_> = rows.iter().copied().chain(missing).collect();
            (Just(t), Just(all).prop_shuffle())
        })
    })
}

/// Paired values count into the table, in any order; a pair with a
/// missing value is dropped.
#[test]
fn pairs_fit_as_their_table() {
    check(
        "pairs",
        CASES,
        0.3,
        table_and_pairs(),
        |(t, rows), checked| {
            let (x, y): (Vec<f64>, Vec<f64>) = rows.iter().copied().unzip();
            let pairs = tetrachoric_pairs(&x, &y).expect("binary values");
            let table = run(t);
            let dropped = rows
                .iter()
                .filter(|(a, b)| a.is_nan() || b.is_nan())
                .count() as u64;
            prop_assert_eq!(pairs.table, table.table);
            prop_assert_eq!(pairs.n, table.n);
            prop_assert_eq!(pairs.n_dropped, dropped);
            prop_assert_eq!(&pairs.outcome, &table.outcome);
            if table.outcome.is_ok() {
                checked.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        },
    );
}

/// The empty table has no pairs; a side with one level has a constant
/// margin.
#[test]
fn a_table_without_both_levels_has_a_reason() {
    let strategy = (0u64..=50, 0u64..=50, any::<bool>(), 0usize..2);
    check(
        "one level",
        CASES,
        0.3,
        strategy,
        |(a, b, empty_row, at), checked| {
            let mut t = if empty_row {
                [[a, b], [0, 0]]
            } else {
                [[a, 0], [b, 0]]
            };
            if at == 1 {
                t = if empty_row {
                    [t[1], t[0]]
                } else {
                    t.map(|[x, y]| [y, x])
                };
            }
            let expected = if a + b == 0 {
                Reason::NoCompletePairs
            } else {
                Reason::ConstantMargin
            };
            prop_assert_eq!(point(t), Err(expected));
            checked.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    );
}
