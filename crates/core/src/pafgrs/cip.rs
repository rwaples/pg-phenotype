//! The cumulative incidence proportion (CIP) table and what it gives a
//! trait: a lifetime threshold and each observation's weight `w` (ADR 0002).

use super::pa::Obs;
use crate::error::{CipProblem, Error};
use crate::input::{admit, Admits, Trait, TraitKind};
use crate::normal;

/// How a [`Cip`] reads between its ages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// A straight line between neighbouring ages: a smooth table.
    Linear,
    /// The value at the last age at or below: an estimator's step function.
    Step,
}

impl Interpolation {
    pub const ALL: [Interpolation; 2] = [Interpolation::Linear, Interpolation::Step];

    /// The name a host passes.
    pub fn name(self) -> &'static str {
        match self {
            Interpolation::Linear => "linear",
            Interpolation::Step => "step",
        }
    }

    pub fn from_name(name: &str) -> Option<Interpolation> {
        Interpolation::ALL.into_iter().find(|i| i.name() == name)
    }
}

/// CIP by age: `ages` strictly increasing and finite, `cip` non-decreasing
/// in `[0, 1)` with a positive last value, which is the lifetime
/// prevalence `K`, read between ages by `interpolation`.
#[derive(Clone, Debug, PartialEq)]
pub struct Cip {
    ages: Vec<f64>,
    cip: Vec<f64>,
    interpolation: Interpolation,
}

impl Cip {
    /// # Errors
    ///
    /// [`Error::InvalidCip`] at the first position that breaks the contract.
    pub fn new(ages: Vec<f64>, cip: Vec<f64>, interpolation: Interpolation) -> Result<Cip, Error> {
        let bad = |reason, position: usize| {
            Err(Error::InvalidCip {
                reason,
                position: position.into(),
            })
        };
        if ages.is_empty() {
            return bad(CipProblem::Empty, 0);
        }
        if ages.len() != cip.len() {
            return bad(CipProblem::LengthMismatch, ages.len().min(cip.len()));
        }
        for (i, (&age, &c)) in ages.iter().zip(&cip).enumerate() {
            if !age.is_finite() {
                return bad(CipProblem::AgeNotFinite, i);
            }
            if i > 0 && age <= ages[i - 1] {
                return bad(CipProblem::AgesNotIncreasing, i);
            }
            if !(0.0..1.0).contains(&c) {
                return bad(CipProblem::CipOutOfRange, i);
            }
            if i > 0 && c < cip[i - 1] {
                return bad(CipProblem::CipDecreases, i);
            }
        }
        if cip[cip.len() - 1] <= 0.0 {
            return bad(CipProblem::PrevalenceNotPositive, cip.len() - 1);
        }
        Ok(Cip {
            ages,
            cip,
            interpolation,
        })
    }

    /// The ages and the CIP at each.
    pub fn into_parts(self) -> (Vec<f64>, Vec<f64>) {
        (self.ages, self.cip)
    }

    /// The lifetime prevalence `K`, the last CIP value.
    pub fn prevalence(&self) -> f64 {
        self.cip[self.cip.len() - 1]
    }

    /// The liability threshold `Phi^-1(1 - K)`.
    pub fn threshold(&self) -> f64 {
        normal::quantile(1.0 - self.prevalence())
    }

    pub fn interpolation(&self) -> Interpolation {
        self.interpolation
    }

    /// CIP at `age`: 0 below the first age, `K` at and above the last.
    /// Between ages, linear as `numpy.interp(age, ages, cip, left=0,
    /// right=K)`, or the value at the last age at or below `age` for a step
    /// table.
    pub fn at(&self, age: f64) -> f64 {
        let (ages, cip) = (&self.ages, &self.cip);
        let last = ages.len() - 1;
        if age < ages[0] {
            return 0.0;
        }
        if age >= ages[last] {
            return cip[last];
        }
        let j = ages.partition_point(|&a| a <= age) - 1;
        match self.interpolation {
            Interpolation::Step => cip[j],
            Interpolation::Linear => {
                let slope = (cip[j + 1] - cip[j]) / (ages[j + 1] - ages[j]);
                slope * (age - ages[j]) + cip[j]
            }
        }
    }

    /// The proportion of lifetime risk a control observed by `age`.
    pub fn w(&self, age: f64) -> f64 {
        (self.at(age) / self.prevalence()).clamp(0.0, 1.0)
    }
}

/// A trait read through its CIP table: per row, whether it is a case and
/// the weight it carries (`0` when unobserved).
#[derive(Clone, Debug)]
pub(crate) struct Observed {
    pub(crate) threshold: f64,
    pub(crate) affected: Vec<bool>,
    pub(crate) w: Vec<f64>,
    pub(crate) controls_without_age: usize,
}

/// Binary alone: a status is a case or a control.
const ADMITS: Admits = Admits::new(&[TraitKind::Binary], false);

impl Observed {
    /// # Errors
    ///
    /// The trait's Admission: [`Error::TraitLength`],
    /// [`Error::TraitKindMismatch`] for a trait that is not binary, and
    /// [`Error::InvalidTraitCode`]; then the age's [`Error::TraitLength`]
    /// and [`Error::InvalidAge`], with `fields` naming the trait and age.
    ///
    /// `age` is the age at onset for a case and at last observation for a
    /// control, NaN when unknown.
    pub(crate) fn new(
        values: Trait<'_>,
        age: &[f64],
        cip: &Cip,
        n_rows: usize,
        fields: [&'static str; 2],
    ) -> Result<Observed, Error> {
        let [status_field, age_field] = fields;
        let status = admit(values, status_field, n_rows, ADMITS)?.values;
        if age.len() != n_rows {
            return Err(Error::TraitLength {
                field: age_field,
                expected_length: n_rows,
                actual_length: age.len(),
            });
        }
        let mut affected = vec![false; n_rows];
        let mut w = vec![0.0; n_rows];
        let mut controls_without_age = 0;
        for (row, (&s, &age)) in status.iter().zip(age).enumerate() {
            if !age.is_nan() && !(age.is_finite() && age >= 0.0) {
                return Err(Error::InvalidAge {
                    field: age_field,
                    position: row.into(),
                    value: age,
                });
            }
            // Admitted binary: a missing status (NaN) observes nothing.
            debug_assert!(s.is_nan() || s == 0.0 || s == 1.0);
            if s == 1.0 {
                affected[row] = true;
                w[row] = 1.0;
            } else if s == 0.0 {
                if age.is_nan() {
                    controls_without_age += 1;
                } else {
                    w[row] = cip.w(age);
                }
            }
        }
        Ok(Observed {
            threshold: cip.threshold(),
            affected,
            w,
            controls_without_age,
        })
    }

    /// Truncation bounds of row `row`: above the threshold for a case,
    /// below it for a control.
    #[inline]
    pub(crate) fn bounds(&self, row: usize) -> (f64, f64) {
        if self.affected[row] {
            (self.threshold, f64::INFINITY)
        } else {
            (f64::NEG_INFINITY, self.threshold)
        }
    }

    /// Row `row`'s observation: its truncation bounds and weight `w`.
    #[inline]
    pub(crate) fn obs(&self, row: usize) -> Obs {
        let (lower, upper) = self.bounds(row);
        Obs {
            lower,
            upper,
            w: self.w[row],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cip() -> Cip {
        Cip::new(
            vec![10.0, 20.0, 40.0],
            vec![0.0, 0.05, 0.1],
            Interpolation::Linear,
        )
        .unwrap()
    }

    #[test]
    fn a_step_table_holds_each_value_until_the_next_age() {
        let c = Cip::new(
            vec![10.0, 20.0, 40.0],
            vec![0.01, 0.05, 0.1],
            Interpolation::Step,
        )
        .unwrap();
        let at: Vec<f64> = [5.0, 10.0, 15.0, 19.999, 20.0, 39.0, 40.0, 99.0]
            .iter()
            .map(|&a| c.at(a))
            .collect();
        assert_eq!(at, [0.0, 0.01, 0.01, 0.01, 0.05, 0.05, 0.1, 0.1]);
        assert_eq!(c.w(30.0), 0.5);
    }

    #[test]
    fn interpolation_names_round_trip() {
        for i in Interpolation::ALL {
            assert_eq!(Interpolation::from_name(i.name()), Some(i));
        }
        assert_eq!(Interpolation::from_name("cubic"), None);
    }

    #[test]
    fn interpolation_follows_numpy_interp() {
        let c = cip();
        assert_eq!(c.at(5.0), 0.0);
        assert_eq!(c.at(10.0), 0.0);
        assert_eq!(c.at(15.0), 0.025);
        assert_eq!(c.at(20.0), 0.05);
        assert_eq!(c.at(40.0), 0.1);
        assert_eq!(c.at(99.0), 0.1);
        assert!((c.w(30.0) - 0.75).abs() < 1e-15);
        assert_eq!(c.prevalence(), 0.1);
    }

    #[test]
    fn invalid_tables_name_the_position() {
        let cases: [(Vec<f64>, Vec<f64>, CipProblem, usize); 6] = [
            (vec![], vec![], CipProblem::Empty, 0),
            (
                vec![1.0, 1.0],
                vec![0.1, 0.2],
                CipProblem::AgesNotIncreasing,
                1,
            ),
            (
                vec![1.0, f64::NAN],
                vec![0.1, 0.2],
                CipProblem::AgeNotFinite,
                1,
            ),
            (vec![1.0, 2.0], vec![0.2, 0.1], CipProblem::CipDecreases, 1),
            (vec![1.0, 2.0], vec![0.1, 1.0], CipProblem::CipOutOfRange, 1),
            (
                vec![1.0, 2.0],
                vec![0.0, 0.0],
                CipProblem::PrevalenceNotPositive,
                1,
            ),
        ];
        for (ages, values, reason, position) in cases {
            assert_eq!(
                Cip::new(ages, values, Interpolation::Linear).unwrap_err(),
                Error::InvalidCip {
                    reason,
                    position: position.into()
                }
            );
        }
    }

    fn binary(values: &[f64]) -> Trait<'_> {
        Trait {
            values,
            kind: TraitKind::Binary,
            n_levels: None,
        }
    }

    #[test]
    fn observation_rules() {
        let status = [1.0, 0.0, 0.0, f64::NAN, 1.0];
        let age = [f64::NAN, 30.0, f64::NAN, 50.0, 12.0];
        let obs = Observed::new(binary(&status), &age, &cip(), 5, ["s", "a"]).unwrap();
        assert_eq!(obs.affected, [true, false, false, false, true]);
        assert_eq!(obs.w[0], 1.0);
        assert!((obs.w[1] - 0.75).abs() < 1e-15);
        assert_eq!(&obs.w[2..], [0.0, 0.0, 1.0]);
        assert_eq!(obs.controls_without_age, 1);
        let err = Observed::new(binary(&[2.0]), &[1.0], &cip(), 1, ["s", "a"]).unwrap_err();
        assert_eq!(err.code(), "invalid_trait_value");
        let err = Observed::new(binary(&[1.0]), &[-1.0], &cip(), 1, ["s", "a"]).unwrap_err();
        assert_eq!(err.code(), "invalid_age");
        let err = Observed::new(binary(&[1.0]), &[1.0], &cip(), 2, ["s", "a"]).unwrap_err();
        assert_eq!(err.code(), "trait_length_mismatch");
        let continuous = Trait {
            values: &[1.0],
            kind: TraitKind::Continuous,
            n_levels: None,
        };
        let err = Observed::new(continuous, &[1.0], &cip(), 1, ["s", "a"]).unwrap_err();
        assert_eq!(err.code(), "trait_kind_mismatch");
    }

    /// The trait's Admission, then its age: a call with both wrong reports
    /// the trait's error, whichever row holds it.
    #[test]
    fn the_status_is_checked_before_the_age() {
        let code = |status: &[f64], age: &[f64]| {
            Observed::new(binary(status), age, &cip(), 3, ["s", "a"])
                .unwrap_err()
                .code()
        };
        assert_eq!(code(&[1.0, 0.0], &[1.0, 1.0, 1.0]), "trait_length_mismatch");
        assert_eq!(code(&[1.0, 2.0, 0.0], &[1.0, 1.0]), "invalid_trait_value");
        assert_eq!(
            code(&[1.0, 0.0, 2.0], &[-1.0, 1.0, 1.0]),
            "invalid_trait_value"
        );
        let continuous = Trait {
            values: &[1.5],
            kind: TraitKind::Continuous,
            n_levels: None,
        };
        let err = Observed::new(continuous, &[1.0], &cip(), 3, ["s", "a"]).unwrap_err();
        assert_eq!(err.code(), "trait_length_mismatch");
    }
}
