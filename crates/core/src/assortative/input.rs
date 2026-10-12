//! The input boundary: traits, strata and settings checked once, then trusted.

use crate::error::Error;
use crate::input::{admit, Admits, Admitted, Trait, TraitKind};

/// Field names of the traits, as errors report them.
const FIELDS: [&str; 2] = ["traits[0]", "traits[1]"];

/// One stratum label per pedigree row: Depth, a birth-year bin, or any
/// grouping whose strata a cell is standardised within.
#[derive(Clone, Copy, Debug)]
pub struct Strata<'a> {
    pub labels: &'a [i64],
    /// `false` where the label is unknown; the label there is ignored.
    pub known: &'a [bool],
}

impl Strata<'_> {
    /// Each row's code, the rank of its label among the distinct known
    /// labels (`None` where unknown), and those labels ascending.
    pub(crate) fn codes(self) -> (Vec<Option<usize>>, Vec<i64>) {
        let mut labels: Vec<i64> = self
            .labels
            .iter()
            .zip(self.known)
            .filter(|(_, &k)| k)
            .map(|(&l, _)| l)
            .collect();
        labels.sort_unstable();
        labels.dedup();
        let codes = self
            .labels
            .iter()
            .zip(self.known)
            .map(|(l, &k)| k.then(|| labels.binary_search(l).unwrap_or(0)))
            .collect();
        (codes, labels)
    }
}

/// Father permutations per primary estimate when inference is on and none
/// are given (pedsum's CLI default).
pub const DEFAULT_PERMUTATIONS: i64 = 999;

/// Draw counts, the seed, the thin-stratum rule, and the optional estimators.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Father permutations per primary estimate; 0 turns them off.  `None`
    /// is [`DEFAULT_PERMUTATIONS`] with inference and 0 without.
    pub permutations: Option<i64>,
    /// Mate Network bootstrap draws; 0 gives sandwich (Wald) CIs.
    pub bootstrap: i64,
    /// Keys every permutation and bootstrap draw.
    pub seed: i64,
    /// With strata, a sex x stratum is kept only when a cell's pairs in it
    /// span at least this many Mate Networks.
    pub min_stratum_networks: i64,
    /// Add Spearman to a continuous x continuous cell's crude estimators.
    pub spearman: bool,
    /// Compute inference: Mate Networks, SEs, CIs, the permutation records
    /// and the Within-Person Cross-Trait Correlation.  Off, a call returns
    /// the point estimates and counts alone, and takes no draws.
    pub inference: bool,
}

impl Default for Settings {
    /// pedsum's CLI defaults, without Spearman, which pedsum always computes.
    fn default() -> Settings {
        Settings {
            permutations: None,
            bootstrap: 0,
            seed: 0,
            min_stratum_networks: 10,
            spearman: false,
            inference: true,
        }
    }
}

/// [`Settings`] checked.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Checked {
    pub permutations: u64,
    pub bootstrap: u64,
    pub seed: i64,
    pub min_stratum_networks: u64,
    pub spearman: bool,
    pub inference: bool,
}

impl Settings {
    pub(crate) fn check(self) -> Result<Checked, Error> {
        let count = |name, value: i64, minimum: i64| {
            if value >= minimum {
                Ok(value as u64)
            } else {
                Err(Error::ParameterOutOfRange {
                    name,
                    value: value as f64,
                    domain: if minimum == 0 { "[0, inf)" } else { "[1, inf)" },
                })
            }
        };
        let permutations = self.permutations.unwrap_or(if self.inference {
            DEFAULT_PERMUTATIONS
        } else {
            0
        });
        let checked = Checked {
            permutations: count("permutations", permutations, 0)?,
            bootstrap: count("bootstrap", self.bootstrap, 0)?,
            seed: self.seed,
            min_stratum_networks: count("min_stratum_networks", self.min_stratum_networks, 1)?,
            spearman: self.spearman,
            inference: self.inference,
        };
        if !self.inference {
            for (name, value) in [
                ("permutations", permutations),
                ("bootstrap", self.bootstrap),
            ] {
                if value != 0 {
                    return Err(Error::ParameterOutOfRange {
                        name,
                        value: value as f64,
                        domain: "{0} without inference",
                    });
                }
            }
        }
        Ok(checked)
    }
}

/// Continuous, binary or ordinal, strict: a cell needs each trait to vary
/// and, for a discrete one, every level taken.
const ADMITS: Admits = Admits::new(
    &[TraitKind::Continuous, TraitKind::Binary, TraitKind::Ordinal],
    true,
);

/// Check one or two traits against `n_rows` (D4, D8, D9): their count, then
/// each trait's Admission in turn.
pub(crate) fn check_traits<'a>(
    traits: &[Trait<'a>],
    n_rows: usize,
) -> Result<Vec<Admitted<'a>>, Error> {
    if traits.is_empty() || traits.len() > 2 {
        return Err(Error::TraitCount {
            actual: traits.len(),
            minimum: 1,
            maximum: 2,
        });
    }
    traits
        .iter()
        .zip(FIELDS)
        .map(|(t, field)| admit(*t, field, n_rows, ADMITS))
        .collect()
}

/// Check that `strata` has one label per row.
pub(crate) fn check_strata(strata: Strata<'_>, n_rows: usize) -> Result<(), Error> {
    for len in [strata.labels.len(), strata.known.len()] {
        if len != n_rows {
            return Err(Error::StratumLength {
                expected_length: n_rows,
                actual_length: len,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::Kind;

    const NA: f64 = f64::NAN;

    #[test]
    fn without_inference_every_draw_count_is_zero() {
        let on = Settings::default().check().unwrap();
        assert_eq!(on.permutations, DEFAULT_PERMUTATIONS as u64);
        let off = Settings {
            inference: false,
            ..Settings::default()
        };
        let checked = off.check().unwrap();
        assert_eq!((checked.permutations, checked.inference), (0, false));
        for settings in [
            Settings {
                permutations: Some(1),
                ..off
            },
            Settings {
                bootstrap: 1,
                ..off
            },
        ] {
            let err = settings.check().unwrap_err();
            assert_eq!(err.code(), "parameter_out_of_range");
        }
    }

    fn t(values: &[f64], kind: TraitKind, n_levels: Option<usize>) -> Trait<'_> {
        Trait {
            values,
            kind,
            n_levels,
        }
    }

    #[test]
    fn one_or_two_traits_each_admitted_strictly() {
        let v = [0.0, 1.0];
        let b = t(&v, TraitKind::Binary, None);
        let err = |traits: &[Trait<'_>], n| check_traits(traits, n).unwrap_err();
        assert_eq!(err(&[], 2).code(), "trait_count");
        assert_eq!(err(&[b, b, b], 2).code(), "trait_count");
        // The count comes before any trait.
        assert_eq!(err(&[b, b, b], 3).code(), "trait_count");
        assert_eq!(
            err(&[b, t(&[1.0, 1.0], TraitKind::Binary, None)], 2),
            Error::ConstantTrait {
                field: "traits[1]",
                value: 1.0
            }
        );
        assert_eq!(
            err(&[t(&[NA, NA], TraitKind::Continuous, None), b], 2),
            Error::AllMissingTrait { field: "traits[0]" }
        );
        assert_eq!(
            err(&[t(&[0.0, 2.0], TraitKind::Ordinal, None)], 2).code(),
            "sparse_ordinal_codes"
        );
        let ok = check_traits(&[b, t(&v, TraitKind::Ordinal, None)], 2).unwrap();
        assert_eq!(ok[0].kind, Kind::Binary);
        assert_eq!(ok[1].kind, Kind::Ordinal { k: 2 });
    }

    #[test]
    fn settings_and_strata() {
        let mut s = Settings::default();
        assert!(s.check().is_ok());
        s.min_stratum_networks = 0;
        assert_eq!(s.check().unwrap_err().code(), "parameter_out_of_range");
        s = Settings {
            bootstrap: -1,
            ..Settings::default()
        };
        assert_eq!(s.check().unwrap_err().code(), "parameter_out_of_range");
        let strata = Strata {
            labels: &[1, 2],
            known: &[true],
        };
        assert_eq!(
            check_strata(strata, 2).unwrap_err().code(),
            "stratum_length_mismatch"
        );
    }
}
