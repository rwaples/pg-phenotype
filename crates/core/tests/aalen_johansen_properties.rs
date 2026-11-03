//! Properties of the Aalen-Johansen CIP, against a recurrence written from
//! its documentation rather than from its sweep.
//!
//! Bit-for-bit comparisons are between two calls in one process, never
//! against a stored value, so they hold on every platform.

mod common;

use std::sync::atomic::Ordering;

use common::check;
use pg_phenotype_core::pafgrs::Cip;
use pg_phenotype_core::Error;
use proptest::prelude::*;

/// Cases per property in an ordinary run.
const CASES: u32 = 256;

/// One row: age, status, and an entry age before it.
#[derive(Clone, Copy, Debug)]
struct Row {
    age: f64,
    status: f64,
    entry: f64,
}

/// Ages in halves on `0..=40`, so ages tie.  An entry is below its age, or
/// at it for a censored row; a row at age 0 has entry 0, which only a
/// censored row can take, so events at age 0 come from the no-entry arm.
fn row() -> impl Strategy<Value = Row> {
    (0u32..=80, 0u8..=2, 0.0f64..1.0).prop_map(|(half, status, u)| {
        let age = f64::from(half) / 2.0;
        let status = if half == 0 { 0.0 } else { f64::from(status) };
        // At most `age - 0.5` for a positive age, since `2 age` is an integer.
        let entry = (2.0 * age * u).floor() / 2.0;
        Row { age, status, entry }
    })
}

/// Rows without entry: any age, including events at 0.
fn row_no_entry() -> impl Strategy<Value = Row> {
    (0u32..=80, 0u8..=2).prop_map(|(half, status)| Row {
        age: f64::from(half) / 2.0,
        status: f64::from(status),
        entry: 0.0,
    })
}

fn columns(rows: &[Row]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    (
        rows.iter().map(|r| r.age).collect(),
        rows.iter().map(|r| r.status).collect(),
        rows.iter().map(|r| r.entry).collect(),
    )
}

type Outcome = Result<(Vec<f64>, Vec<f64>), String>;

fn aj(rows: &[Row], with_entry: bool) -> Outcome {
    let (age, status, entry) = columns(rows);
    let entry = with_entry.then_some(entry.as_slice());
    outcome(Cip::aalen_johansen(&age, &status, entry))
}

/// An error as its code, and for `cip_not_estimable` its reason.
fn outcome(r: Result<Cip, Error>) -> Outcome {
    r.map(Cip::into_parts).map_err(|e| match e {
        Error::CipNotEstimable { reason } => reason.name().to_string(),
        e => e.code().to_string(),
    })
}

/// The oracle's case ages and CIP.
type Table = (Vec<f64>, Vec<f64>);

/// The documented estimator: at each distinct event age `t`, with `Y`
/// rows at risk (`entry < t <= age`), `d1` cases and `d2` deaths, and `S`
/// the all-cause survival before `t`, the CIP rises by `S d1 / Y` and `S`
/// falls by the factor `1 - (d1 + d2) / Y`.  Not estimable when the
/// lifetime CIP `K` is 0 (`zero_prevalence`) or 1 (`prevalence_one`).
///
/// Which of those holds is decided from counts, not from the rounded `K`:
/// `S` is exactly 0 from the first age where everyone at risk has an
/// event, so `K = 1` exactly when that happens with no death before it,
/// and `K = 0` exactly when no case comes before it.
fn oracle(rows: &[Row], with_entry: bool) -> Result<Table, &'static str> {
    let mut times: Vec<f64> = rows
        .iter()
        .filter(|r| r.status != 0.0)
        .map(|r| r.age)
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let (mut ages, mut cip) = (Vec::new(), Vec::new());
    let (mut s, mut cases) = (1.0, 0.0);
    let (mut exhausted, mut case_seen, mut death_seen) = (false, false, false);
    for t in times {
        let at_risk = rows
            .iter()
            .filter(|r| (!with_entry || r.entry < t) && t <= r.age)
            .count() as f64;
        let d1 = rows
            .iter()
            .filter(|r| r.age == t && r.status == 1.0)
            .count() as f64;
        let d2 = rows
            .iter()
            .filter(|r| r.age == t && r.status == 2.0)
            .count() as f64;
        if !exhausted {
            case_seen |= d1 > 0.0;
            death_seen |= d2 > 0.0;
            exhausted = d1 + d2 == at_risk;
        }
        cases += s * d1 / at_risk;
        s *= 1.0 - (d1 + d2) / at_risk;
        if d1 > 0.0 {
            ages.push(t);
            cip.push(cases);
        }
    }
    if exhausted && !death_seen {
        return Err("prevalence_one");
    }
    if !case_seen {
        return Err("zero_prevalence");
    }
    Ok((ages, cip))
}

fn rows_and_entry() -> impl Strategy<Value = (Vec<Row>, bool)> {
    prop_oneof![
        prop::collection::vec(row(), 1..60).prop_map(|r| (r, true)),
        prop::collection::vec(row_no_entry(), 1..60).prop_map(|r| (r, false)),
    ]
}

#[test]
fn row_order_does_not_matter() {
    let strategy =
        rows_and_entry().prop_flat_map(|(r, e)| (Just(r.clone()), Just(r).prop_shuffle(), Just(e)));
    check(
        "row order",
        CASES,
        0.3,
        strategy,
        |(rows, shuffled, with_entry), checked| {
            match (aj(&rows, with_entry), aj(&shuffled, with_entry)) {
                (Ok((ages_a, cip_a)), Ok((ages_b, cip_b))) => {
                    prop_assert_eq!(ages_a, ages_b);
                    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    prop_assert_eq!(bits(&cip_a), bits(&cip_b));
                    checked.fetch_add(1, Ordering::Relaxed);
                }
                (a, b) => prop_assert_eq!(a.err(), b.err()),
            }
            Ok(())
        },
    );
}

/// Also: valid rows never give a table `Cip::new` refuses (decreasing,
/// out of range, ages not increasing), only a not-estimable reason.
#[test]
fn matches_the_documented_recurrence() {
    check(
        "recurrence",
        CASES,
        0.3,
        rows_and_entry(),
        |(rows, with_entry), checked| {
            match (aj(&rows, with_entry), oracle(&rows, with_entry)) {
                (Ok((ages, cip)), Ok((want_ages, want_cip))) => {
                    prop_assert_eq!(ages, want_ages);
                    prop_assert_eq!(cip.len(), want_cip.len());
                    for (got, want) in cip.iter().zip(&want_cip) {
                        prop_assert!(
                            (got - want).abs() <= 1e-12 * want.abs(),
                            "{} vs {}",
                            got,
                            want
                        );
                    }
                    checked.fetch_add(1, Ordering::Relaxed);
                }
                (Err(reason), Err(want)) => prop_assert_eq!(reason, want),
                (got, want) => prop_assert!(false, "got {:?}, oracle {:?}", got, want),
            }
            Ok(())
        },
    );
}

/// Entry 0 removes no one from a risk set at a positive age.
#[test]
fn zero_entries_match_no_entry() {
    let strategy = prop::collection::vec(row_no_entry(), 1..60);
    check("zero entries", CASES, 0.3, strategy, |rows, checked| {
        let rows: Vec<Row> = rows.into_iter().filter(|r| r.age > 0.0).collect();
        prop_assume!(!rows.is_empty());
        let without = aj(&rows, false);
        prop_assert_eq!(&without, &aj(&rows, true));
        if without.is_ok() {
            checked.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    });
}

/// A row censored at its entry age is at risk nowhere.
#[test]
fn rows_censored_at_entry_change_nothing() {
    let strategy = (
        prop::collection::vec(row(), 1..60),
        prop::collection::vec((0u32..=80).prop_map(|h| f64::from(h) / 2.0), 1..10),
    );
    check(
        "censored at entry",
        CASES,
        0.3,
        strategy,
        |(rows, extra), checked| {
            let mut more = rows.clone();
            more.extend(extra.into_iter().map(|a| Row {
                age: a,
                status: 0.0,
                entry: a,
            }));
            let fewer = aj(&rows, true);
            prop_assert_eq!(&fewer, &aj(&more, true));
            if fewer.is_ok() {
                checked.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        },
    );
}

/// `K = 1 - 2^-40`, within any fixed tolerance of 1 but estimable: 40
/// cases that each halve survival, then one death.
#[test]
fn a_prevalence_just_below_one_is_estimable() {
    let mut rows: Vec<Row> = (1..=40)
        .map(|i| Row {
            age: f64::from(i) / 2.0,
            status: 1.0,
            entry: f64::from(i - 1) / 2.0,
        })
        .collect();
    rows.push(Row {
        age: 20.5,
        status: 2.0,
        entry: 0.0,
    });
    let (_, cip) = aj(&rows, true).expect("estimable");
    let k = cip[cip.len() - 1];
    assert!(k < 1.0 && 1.0 - k < 1e-12, "{k}");
    assert_eq!(oracle(&rows, true).map(|t| t.1.len()), Ok(40));
}
