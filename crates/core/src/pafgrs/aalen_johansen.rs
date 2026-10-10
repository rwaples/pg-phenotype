//! A CIP table estimated from ages and survival status by the Aalen-Johansen
//! estimator, with death as a competing risk and optional late entry
//! (ADR 0010).

use super::cip::Cip;
use crate::error::{CipEstimate, Error};

impl Cip {
    /// The Aalen-Johansen cumulative incidence of cause 1 (case), with
    /// cause 2 (death) competing, at each age where a case occurs.
    ///
    /// Per row: `age` is the age at onset, death or last observation;
    /// `status` is 0 (censored), 1 (case) or 2 (death); `entry`, when
    /// given, is the age the row enters the risk set, before `age` (or at it
    /// for a censored row, which is then at risk nowhere).  A row is at risk
    /// at age `t` when `entry < t <= age`, or `t <= age` without `entry`.  At
    /// each distinct event age `t`, with `Y` at risk, `d1` cases and `d2`
    /// deaths, and `S` the all-cause survival just before `t`, the CIP rises
    /// by `S d1 / Y` and `S` falls by the factor `1 - (d1 + d2) / Y`.
    ///
    /// # Errors
    ///
    /// [`Error::PairLength`] when `status` or `entry` is not `age`'s length.
    /// Then row by row, in this order: [`Error::InvalidAge`] for an age that
    /// is not finite and `>= 0`, [`Error::InvalidStatus`], [`Error::InvalidAge`]
    /// for such an entry, and [`Error::EntryNotBeforeAge`].  Last,
    /// [`Error::CipNotEstimable`] when the estimated `K` is 0 or 1.
    pub fn aalen_johansen(
        age: &[f64],
        status: &[f64],
        entry: Option<&[f64]>,
    ) -> Result<Cip, Error> {
        check(age, status, entry)?;

        let mut events: Vec<(f64, bool)> = age
            .iter()
            .zip(status)
            .filter(|&(_, &s)| s != 0.0)
            // `+ 0.0` turns an age of -0.0 into 0.0 before it becomes a table age.
            .map(|(&a, &s)| (a + 0.0, s == 1.0))
            .collect();
        events.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
        let exits = sorted(age);
        let entries = entry.map(sorted);
        let n = age.len();

        let (mut ages, mut cip) = (Vec::new(), Vec::new());
        let (mut survival, mut cases, mut deaths) = (1.0_f64, 0.0_f64, 0.0_f64);
        for group in events.chunk_by(|x, y| x.0 == y.0) {
            let t = group[0].0;
            let d1 = group.iter().filter(|e| e.1).count() as f64;
            let d2 = group.len() as f64 - d1;
            let entered = entries
                .as_ref()
                .map_or(n, |e| e.partition_point(|&x| x < t));
            let at_risk = (entered - exits.partition_point(|&x| x < t)) as f64;
            // Grouped as simACE's `_aalen_johansen` so the gate can match it
            // bit for bit.
            cases += survival * d1 / at_risk;
            deaths += survival * d2 / at_risk;
            survival *= 1.0 - (d1 + d2) / at_risk;
            if d1 > 0.0 {
                ages.push(t);
                cip.push(cases);
            }
        }

        // With no death increment, a survival of exactly 0 means every
        // row at risk became a case, so K is 1 whatever rounding gave.
        if cases >= 1.0 || (survival == 0.0 && deaths == 0.0) {
            return Err(Error::CipNotEstimable {
                reason: CipEstimate::PrevalenceOne,
            });
        }
        if cases <= 0.0 {
            return Err(Error::CipNotEstimable {
                reason: CipEstimate::ZeroPrevalence,
            });
        }
        Cip::new(ages, cip)
    }
}

fn check(age: &[f64], status: &[f64], entry: Option<&[f64]>) -> Result<(), Error> {
    let columns = [
        ("status", Some(status.len())),
        ("entry", entry.map(<[f64]>::len)),
    ];
    for (name, len) in columns {
        if let Some(len) = len.filter(|&len| len != age.len()) {
            return Err(Error::PairLength {
                columns: ["age", name],
                expected_length: age.len(),
                actual_length: len,
            });
        }
    }
    let valid_age = |v: f64| v.is_finite() && v >= 0.0;
    for (position, (&a, &s)) in age.iter().zip(status).enumerate() {
        if !valid_age(a) {
            return Err(Error::InvalidAge {
                field: "age",
                position,
                value: a,
            });
        }
        if !matches!(s, 0.0 | 1.0 | 2.0) {
            return Err(Error::InvalidStatus { position, value: s });
        }
        if let Some(&e) = entry.map(|e| &e[position]) {
            if !valid_age(e) {
                return Err(Error::InvalidAge {
                    field: "entry",
                    position,
                    value: e,
                });
            }
            // A censored row may leave at its entry age, at risk nowhere.
            if e > a || (e == a && s != 0.0) {
                return Err(Error::EntryNotBeforeAge {
                    position,
                    entry: e,
                    age: a,
                });
            }
        }
    }
    Ok(())
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut out = values.to_vec();
    out.sort_unstable_by(f64::total_cmp);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-15, "{actual:?} != {expected:?}");
        }
    }

    fn aj(age: &[f64], status: &[f64], entry: Option<&[f64]>) -> Cip {
        Cip::aalen_johansen(age, status, entry).unwrap()
    }

    #[test]
    fn steps_at_case_ages_with_death_competing() {
        // t=1: Y=5, a case: F=1/5, S=4/5.  t=2: a death, S=3/5.  t=3
        // censored.  t=4: Y=2, F+=3/5*1/2.  t=5: Y=1, F+=3/10*1.
        let cip = aj(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 0.0, 1.0, 1.0], None);
        assert_close(&[cip.at(1.0), cip.at(4.0), cip.at(5.0)], &[0.2, 0.5, 0.8]);
        assert_eq!(cip.at(0.5), 0.0);
        assert!((cip.prevalence() - 0.8).abs() < 1e-15);
    }

    #[test]
    fn a_case_and_a_death_at_one_age_share_the_survival_before_it() {
        // t=2: Y=4 (the row censored at 3 too), d1=d2=1: F=1/4, S=1/2.
        // t=5: Y=1, F+=1/2.
        let cip = aj(&[2.0, 2.0, 3.0, 5.0], &[1.0, 2.0, 0.0, 1.0], None);
        assert_close(&[cip.at(2.0), cip.at(5.0)], &[0.25, 0.75]);
    }

    #[test]
    fn a_row_censored_at_an_event_age_is_at_risk_there() {
        // t=2: Y=4, both rows censored at 2 included: F=1/4.  t=4: a death.
        let cip = aj(&[2.0, 2.0, 2.0, 4.0], &[1.0, 0.0, 0.0, 2.0], None);
        assert_close(&[cip.at(2.0), cip.prevalence()], &[0.25, 0.25]);
    }

    #[test]
    fn a_row_entering_at_an_event_age_is_not_at_risk_there() {
        // t=2: row 2 enters at 2, so Y=3: F=1/3, S=2/3.  simACE would count
        // it (Y=4).  t=4: Y=2, F+=2/3*1/2.  t=5: a death.
        let entry = [0.0, 0.0, 2.0, 0.0];
        let cip = aj(&[2.0, 3.0, 4.0, 5.0], &[1.0, 0.0, 1.0, 2.0], Some(&entry));
        assert_close(&[cip.at(2.0), cip.at(4.0)], &[1.0 / 3.0, 2.0 / 3.0]);
    }

    #[test]
    fn zero_entries_match_no_entry_when_no_age_is_zero() {
        let age = [1.0, 2.0, 3.0, 4.0, 5.0];
        let status = [1.0, 2.0, 0.0, 1.0, 1.0];
        assert_eq!(aj(&age, &status, None), aj(&age, &status, Some(&[0.0; 5])));
    }

    #[test]
    fn a_row_censored_at_its_entry_age_changes_nothing() {
        let age = [1.0, 2.0, 3.0, 4.0, 3.0];
        let status = [1.0, 0.0, 1.0, 0.0, 0.0];
        let entry = [0.0, 0.0, 0.0, 0.0, 3.0];
        assert_eq!(
            aj(&age, &status, Some(&entry)),
            aj(&age[..4], &status[..4], Some(&entry[..4]))
        );
    }

    #[test]
    fn a_negative_zero_age_is_tabled_as_zero() {
        let cip = aj(&[-0.0, 0.0, 1.0, 2.0], &[1.0, 1.0, 0.0, 0.0], None);
        assert!(cip.at(0.0) == 0.5 && cip.into_parts().0[0].is_sign_positive());
    }

    #[test]
    fn a_case_at_age_zero_counts_without_entry() {
        let cip = aj(&[0.0, 1.0, 2.0, 3.0], &[1.0, 0.0, 0.0, 0.0], None);
        assert_close(&[cip.at(0.0)], &[0.25]);
    }

    #[test]
    fn invalid_rows_name_the_first() {
        let err = |age: &[f64], status: &[f64], entry: Option<&[f64]>| {
            Cip::aalen_johansen(age, status, entry).unwrap_err()
        };
        assert_eq!(
            err(&[1.0], &[1.0, 0.0], None),
            Error::PairLength {
                columns: ["age", "status"],
                expected_length: 1,
                actual_length: 2
            }
        );
        assert_eq!(
            err(&[1.0], &[1.0], Some(&[])).code(),
            "pair_length_mismatch"
        );
        for bad in [f64::NAN, -1.0, f64::INFINITY] {
            assert_eq!(err(&[1.0, bad], &[1.0, 0.0], None).code(), "invalid_age");
        }
        for bad in [f64::NAN, 3.0, 0.5, -1.0] {
            let e = err(&[1.0, 2.0], &[1.0, bad], None);
            assert!(
                matches!(e, Error::InvalidStatus { position: 1, .. }),
                "{e:?}"
            );
        }
        let e = err(&[1.0, 2.0], &[1.0, 0.0], Some(&[0.0, f64::NAN]));
        assert!(
            matches!(
                e,
                Error::InvalidAge {
                    field: "entry",
                    position: 1,
                    ..
                }
            ),
            "{e:?}"
        );
        assert_eq!(
            err(&[1.0, 2.0], &[1.0, 0.0], Some(&[0.0, 2.5])),
            Error::EntryNotBeforeAge {
                position: 1,
                entry: 2.5,
                age: 2.0
            }
        );
        for event in [1.0, 2.0] {
            let e = err(&[1.0, 2.0], &[1.0, event], Some(&[0.0, 2.0]));
            assert!(
                matches!(e, Error::EntryNotBeforeAge { position: 1, .. }),
                "{e:?}"
            );
        }
    }

    #[test]
    fn a_k_of_zero_or_one_is_not_estimable() {
        let reason = |age: &[f64], status: &[f64], entry: Option<&[f64]>| match Cip::aalen_johansen(
            age, status, entry,
        )
        .unwrap_err()
        {
            Error::CipNotEstimable { reason } => reason,
            other => panic!("{other:?}"),
        };
        assert_eq!(reason(&[], &[], None), CipEstimate::ZeroPrevalence);
        assert_eq!(
            reason(&[1.0, 2.0], &[0.0, 2.0], None),
            CipEstimate::ZeroPrevalence
        );
        // Everyone at risk dies at 1; the case at 2 entered after: K = 0.
        let entry = [0.0, 1.5];
        assert_eq!(
            reason(&[1.0, 2.0], &[2.0, 1.0], Some(&entry)),
            CipEstimate::ZeroPrevalence
        );
        assert_eq!(
            reason(&[1.0, 2.0], &[1.0, 1.0], None),
            CipEstimate::PrevalenceOne
        );
        // Thirds: 1/3 + 2/3 * 1/2 + 1/3 * 1 may round below 1.
        assert_eq!(
            reason(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], None),
            CipEstimate::PrevalenceOne
        );
    }
}
