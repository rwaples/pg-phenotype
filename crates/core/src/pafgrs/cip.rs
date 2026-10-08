//! The cumulative incidence proportion (CIP) table and what it gives a
//! trait: a lifetime threshold and each observation's weight `w` (ADR 0002).

use super::pa::Obs;
use crate::error::{CipProblem, Error};
use crate::input::{Trait, TraitKind};
use crate::normal;

/// CIP by age: `ages` strictly increasing and finite, `cip` non-decreasing
/// in `[0, 1)` with a positive last value, which is the lifetime
/// prevalence `K`.
#[derive(Clone, Debug, PartialEq)]
pub struct Cip {
    ages: Vec<f64>,
    cip: Vec<f64>,
}

impl Cip {
    /// # Errors
    ///
    /// [`Error::InvalidCip`] at the first position that breaks the contract.
    pub fn new(ages: Vec<f64>, cip: Vec<f64>) -> Result<Cip, Error> {
        let bad = |reason, position| Err(Error::InvalidCip { reason, position });
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
        Ok(Cip { ages, cip })
    }

    /// The lifetime prevalence `K`, the last CIP value.
    pub fn prevalence(&self) -> f64 {
        self.cip[self.cip.len() - 1]
    }

    /// The liability threshold `Phi^-1(1 - K)`.
    pub fn threshold(&self) -> f64 {
        normal::quantile(1.0 - self.prevalence())
    }

    /// CIP at `age` by linear interpolation: 0 below the first age, `K` at
    /// and above the last, as `numpy.interp(age, ages, cip, left=0, right=K)`.
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
        let slope = (cip[j + 1] - cip[j]) / (ages[j + 1] - ages[j]);
        slope * (age - ages[j]) + cip[j]
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

impl Observed {
    /// # Errors
    ///
    /// [`Error::TraitKindMismatch`] for a trait that is not binary, then
    /// [`Error::TraitLength`], [`Error::InvalidTraitCode`], and
    /// [`Error::InvalidAge`], with `fields` naming the trait and age.
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
        if values.kind != TraitKind::Binary {
            return Err(Error::TraitKindMismatch {
                field: status_field,
                expected: TraitKind::Binary,
                actual: values.kind,
            });
        }
        for (field, len) in [(status_field, values.values.len()), (age_field, age.len())] {
            if len != n_rows {
                return Err(Error::TraitLength {
                    field,
                    expected_length: n_rows,
                    actual_length: len,
                });
            }
        }
        let mut affected = vec![false; n_rows];
        let mut w = vec![0.0; n_rows];
        let mut controls_without_age = 0;
        for (row, (&s, &age)) in values.values.iter().zip(age).enumerate() {
            if !age.is_nan() && !(age.is_finite() && age >= 0.0) {
                return Err(Error::InvalidAge {
                    field: age_field,
                    position: row,
                    value: age,
                });
            }
            if s.is_nan() {
                continue;
            }
            if s == 1.0 {
                affected[row] = true;
                w[row] = 1.0;
            } else if s == 0.0 {
                if age.is_nan() {
                    controls_without_age += 1;
                } else {
                    w[row] = cip.w(age);
                }
            } else {
                return Err(Error::InvalidTraitCode {
                    field: status_field,
                    kind: TraitKind::Binary,
                    position: row,
                    value: s,
                });
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
        Cip::new(vec![10.0, 20.0, 40.0], vec![0.0, 0.05, 0.1]).unwrap()
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
                Cip::new(ages, values).unwrap_err(),
                Error::InvalidCip { reason, position }
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
}
