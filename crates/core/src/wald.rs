//! Wald intervals: the confidence level, its normal quantile, and the
//! interval from an estimate and its SE on the Fisher-z or the log scale.

/// The confidence level of every interval.
pub(crate) const CI_LEVEL: f64 = 0.95;

/// The two-sided normal quantile of [`CI_LEVEL`] for Wald intervals: SciPy's
/// cephes `ndtri((1 + 0.95) / 2)`, as pedsum computes it, one ulp above
/// pedsum's kernel AS 241.  A literal for parity; `ci_level_and_wald_z_agree` ties
/// it to `CI_LEVEL`.
pub(crate) const WALD_Z: f64 = 1.959_963_984_540_054;

/// The scale of an estimator's sandwich SE and Wald interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CiScale {
    /// Fisher z, `atanh` of a correlation.
    FisherZ,
    /// `log` of the odds ratio.
    Log,
}

/// How a CI was built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CiMethod {
    /// Percentile interval of the Mate Network bootstrap.
    Bootstrap,
    /// Wald interval from the sandwich SE, on the estimator's [`CiScale`].
    Sandwich,
}

impl CiMethod {
    /// The name hosts use.
    pub fn name(self) -> &'static str {
        match self {
            CiMethod::Bootstrap => "bootstrap",
            CiMethod::Sandwich => "sandwich",
        }
    }
}

/// An interval at the result's `ci_level` and how it was built.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ci {
    pub bounds: [f64; 2],
    pub method: CiMethod,
}

/// The [`CI_LEVEL`] Wald interval on the Fisher-z or the log scale.
pub(crate) fn wald_ci(value: f64, se: f64, scale: CiScale) -> [f64; 2] {
    let z = WALD_Z;
    match scale {
        CiScale::Log => [(value.ln() - z * se).exp(), (value.ln() + z * se).exp()],
        CiScale::FisherZ => {
            let centre = value.atanh();
            let half = z * se / ((1.0 - value) * (1.0 + value));
            [(centre - half).tanh(), (centre + half).tanh()]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normal::{as241, As241Rounding};

    #[test]
    fn ci_level_and_wald_z_agree() {
        let z = as241(0.5 + CI_LEVEL / 2.0, As241Rounding::Pedsum);
        assert!(
            (z - WALD_Z).abs() <= 2.0 * f64::EPSILON * WALD_Z,
            "{z} vs {WALD_Z}"
        );
    }
}
