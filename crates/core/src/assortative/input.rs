//! The input boundary: traits, strata and settings checked once, then trusted.

use crate::error::Error;
use crate::input::{Trait, TraitKind};

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

/// Draw counts, the seed, and the thin-stratum rule.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Father permutations per primary estimate; 0 turns them off.
    pub permutations: i64,
    /// Mate Network bootstrap draws; 0 gives sandwich (Wald) CIs.
    pub bootstrap: i64,
    /// Keys every permutation and bootstrap draw.
    pub seed: i64,
    /// With strata, a sex x stratum is kept only when a cell's pairs in it
    /// span at least this many Mate Networks.
    pub min_stratum_networks: i64,
}

impl Default for Settings {
    /// pedsum's CLI defaults.
    fn default() -> Settings {
        Settings {
            permutations: 999,
            bootstrap: 0,
            seed: 0,
            min_stratum_networks: 10,
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
        Ok(Checked {
            permutations: count("permutations", self.permutations, 0)?,
            bootstrap: count("bootstrap", self.bootstrap, 0)?,
            seed: self.seed,
            min_stratum_networks: count("min_stratum_networks", self.min_stratum_networks, 1)?,
        })
    }
}

/// How a checked trait's values are read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Continuous,
    /// Codes 0 and 1.
    Binary,
    /// Codes `0..k`, every one taken by some row.
    Ordinal {
        k: usize,
    },
}

impl Kind {
    /// The number of level codes of a discrete trait.
    pub fn levels(self) -> Option<usize> {
        match self {
            Kind::Continuous => None,
            Kind::Binary => Some(2),
            Kind::Ordinal { k } => Some(k),
        }
    }
}

/// A trait that passed [`check_traits`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Column<'a> {
    /// One value per row, NaN where missing.
    pub values: &'a [f64],
    pub kind: Kind,
}

/// Check one or two traits against `n_rows` (D4, D8, D9).
///
/// Per trait, in order: length, kind, values, all missing, constant, then
/// every level used.
pub(crate) fn check_traits<'a>(
    traits: &[Trait<'a>],
    n_rows: usize,
) -> Result<Vec<Column<'a>>, Error> {
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
        .map(|(t, field)| check_trait(*t, field, n_rows))
        .collect()
}

fn check_trait<'a>(t: Trait<'a>, field: &'static str, n_rows: usize) -> Result<Column<'a>, Error> {
    if t.values.len() != n_rows {
        return Err(Error::TraitLength {
            field,
            expected_length: n_rows,
            actual_length: t.values.len(),
        });
    }
    let declared = match t.kind {
        TraitKind::Categorical => {
            return Err(Error::UnsupportedTraitKind {
                field,
                kind: t.kind.name(),
            })
        }
        TraitKind::Continuous => None,
        TraitKind::Binary | TraitKind::Ordinal => t.n_levels,
    };
    let ceiling = match t.kind {
        TraitKind::Binary => 2.0_f64.min(declared.map_or(2.0, |k| k as f64)),
        _ => declared.map_or(f64::INFINITY, |k| k as f64),
    };
    let mut first: Option<f64> = None;
    let mut constant = true;
    let mut max_code = 0.0_f64;
    let mut n_present = 0usize;
    for (position, &value) in t.values.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        n_present += 1;
        let valid = match t.kind {
            TraitKind::Continuous => value.is_finite(),
            _ => value >= 0.0 && value < ceiling && value.fract() == 0.0,
        };
        if !valid {
            return Err(Error::InvalidTraitCode {
                field,
                kind: t.kind.name(),
                position,
                value,
            });
        }
        match first {
            None => first = Some(value),
            Some(f) => constant &= f == value,
        }
        max_code = max_code.max(value);
    }
    let Some(value) = first else {
        return Err(Error::AllMissingTrait { field });
    };
    if constant {
        return Err(Error::ConstantTrait { field, value });
    }
    let kind = match t.kind {
        TraitKind::Continuous => {
            return Ok(Column {
                values: t.values,
                kind: Kind::Continuous,
            })
        }
        TraitKind::Binary => Kind::Binary,
        _ => Kind::Ordinal {
            // Saturating: a huge code fails the level check below before `k`
            // is used.
            k: declared.unwrap_or((max_code as usize).saturating_add(1)),
        },
    };
    // A binary trait declared with more than two levels leaves one unused.
    let k = declared.or(kind.levels()).unwrap_or(0);
    // `n_present` values take at most `n_present` levels, so the first unused
    // level is at most `n_present`: the bitmap never needs more slots than
    // that, whatever the codes or the declared count.
    let mut used = vec![false; k.min(n_present + 1)];
    for &value in t.values.iter().filter(|v| !v.is_nan()) {
        if let Some(slot) = used.get_mut(value as usize) {
            *slot = true;
        }
    }
    if let Some(level) = used.iter().position(|&u| !u) {
        return Err(match declared {
            Some(n_levels) => Error::UnusedLevel {
                field,
                level,
                n_levels,
            },
            None => Error::SparseOrdinalCodes { field, level },
        });
    }
    Ok(Column {
        values: t.values,
        kind,
    })
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

    const NA: f64 = f64::NAN;

    fn t(values: &[f64], kind: TraitKind, n_levels: Option<usize>) -> Trait<'_> {
        Trait {
            values,
            kind,
            n_levels,
        }
    }

    fn code(traits: &[Trait<'_>], n: usize) -> &'static str {
        check_traits(traits, n).unwrap_err().code()
    }

    #[test]
    fn counts_lengths_and_kinds() {
        let v = [0.0, 1.0];
        let b = t(&v, TraitKind::Binary, None);
        assert_eq!(code(&[], 2), "trait_count");
        assert_eq!(code(&[b, b, b], 2), "trait_count");
        assert_eq!(code(&[b], 3), "trait_length_mismatch");
        assert_eq!(
            code(&[t(&v, TraitKind::Categorical, None)], 2),
            "unsupported_trait_kind"
        );
        let ok = check_traits(&[b, t(&v, TraitKind::Ordinal, None)], 2).unwrap();
        assert_eq!(ok[0].kind, Kind::Binary);
        assert_eq!(ok[1].kind, Kind::Ordinal { k: 2 });
    }

    #[test]
    fn values_must_be_codes_or_finite() {
        assert_eq!(
            code(&[t(&[0.0, 2.0], TraitKind::Binary, None)], 2),
            "invalid_trait_value"
        );
        assert_eq!(
            code(&[t(&[0.0, 0.5], TraitKind::Ordinal, None)], 2),
            "invalid_trait_value"
        );
        assert_eq!(
            code(&[t(&[0.0, -1.0], TraitKind::Ordinal, None)], 2),
            "invalid_trait_value"
        );
        assert_eq!(
            code(&[t(&[0.0, 3.0], TraitKind::Ordinal, Some(3))], 2),
            "invalid_trait_value"
        );
        assert_eq!(
            code(&[t(&[0.0, f64::INFINITY], TraitKind::Continuous, None)], 2),
            "invalid_trait_value"
        );
    }

    #[test]
    fn missing_constant_then_levels() {
        assert_eq!(
            code(&[t(&[NA, NA], TraitKind::Continuous, None)], 2),
            "all_missing_trait"
        );
        assert_eq!(
            code(&[t(&[1.0, NA, 1.0], TraitKind::Binary, None)], 3),
            "constant_trait"
        );
        assert_eq!(
            code(&[t(&[2.0, 2.0], TraitKind::Ordinal, Some(5))], 2),
            "constant_trait"
        );
        assert_eq!(
            code(&[t(&[10.0, 20.0, 40.0], TraitKind::Ordinal, None)], 3),
            "sparse_ordinal_codes"
        );
        assert_eq!(
            code(&[t(&[0.0, 2.0], TraitKind::Ordinal, None)], 2),
            "sparse_ordinal_codes"
        );
        let err = check_traits(&[t(&[0.0, 1.0], TraitKind::Ordinal, Some(3))], 2).unwrap_err();
        assert_eq!(
            err,
            Error::UnusedLevel {
                field: "traits[0]",
                level: 2,
                n_levels: 3
            }
        );
        assert_eq!(
            code(&[t(&[0.0, 1.0], TraitKind::Binary, Some(3))], 2),
            "unused_level"
        );
        let ok = check_traits(&[t(&[0.0, 2.0, NA, 1.0], TraitKind::Ordinal, Some(3))], 4).unwrap();
        assert_eq!(ok[0].kind, Kind::Ordinal { k: 3 });
    }

    #[test]
    fn huge_codes_and_level_counts_are_rejected_without_allocating() {
        // Undeclared levels: the bitmap would have been sized by the code.
        let err = check_traits(&[t(&[0.0, 1e12], TraitKind::Ordinal, None)], 2).unwrap_err();
        assert_eq!(
            err,
            Error::SparseOrdinalCodes {
                field: "traits[0]",
                level: 1
            }
        );
        assert_eq!(
            code(&[t(&[0.0, 1e300], TraitKind::Ordinal, None)], 2),
            "sparse_ordinal_codes"
        );
        // A huge declared count reports the first unused level.
        let err =
            check_traits(&[t(&[0.0, 1.0], TraitKind::Ordinal, Some(usize::MAX))], 2).unwrap_err();
        assert_eq!(
            err,
            Error::UnusedLevel {
                field: "traits[0]",
                level: 2,
                n_levels: usize::MAX
            }
        );
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
