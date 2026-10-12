//! What every method takes: the pedigree and the traits on its rows.

use crate::error::Error;

/// The pedigree columns a caller passes, one entry per row; a method or
/// [`crate::Pedigree::new`] validates them.
#[derive(Clone, Copy, Debug)]
pub struct PedigreeInput<'a> {
    pub ids: &'a [i64],
    /// Mother id, `-1` when missing.
    pub mother: &'a [i64],
    /// Father id, `-1` when missing.
    pub father: &'a [i64],
    /// MZ co-twin id, `-1` when none.
    pub twin: Option<&'a [i64]>,
    /// `0` female, `1` male, `-1` unknown.
    pub sex: Option<&'a [i64]>,
}

/// How a trait's values are read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraitKind {
    Continuous,
    /// `0` or `1`.
    Binary,
    /// Ordered integer codes.
    Ordinal,
    /// Unordered integer codes.
    Categorical,
}

impl TraitKind {
    /// Every kind, in the order hosts list them.
    pub const ALL: [TraitKind; 4] = [
        TraitKind::Continuous,
        TraitKind::Binary,
        TraitKind::Ordinal,
        TraitKind::Categorical,
    ];

    /// The kind a host names, `None` for an unknown name.
    pub fn from_name(name: &str) -> Option<TraitKind> {
        TraitKind::ALL.into_iter().find(|k| k.name() == name)
    }

    /// The name hosts use.
    pub fn name(self) -> &'static str {
        match self {
            TraitKind::Continuous => "continuous",
            TraitKind::Binary => "binary",
            TraitKind::Ordinal => "ordinal",
            TraitKind::Categorical => "categorical",
        }
    }
}

/// One phenotype column aligned to pedigree rows, NaN where missing.
#[derive(Clone, Copy, Debug)]
pub struct Trait<'a> {
    pub values: &'a [f64],
    pub kind: TraitKind,
    /// The declared number of levels of a binary, ordinal or categorical
    /// trait (codes `0..n_levels`), when the host knows it: Python
    /// `Trait.levels`, R factor levels.  `None` means max code + 1.
    pub n_levels: Option<usize>,
}

/// What a method admits of a Trait: the kinds it reads, and whether the
/// trait must vary.  Each method declares its own as a const.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Admits {
    kinds: &'static [TraitKind],
    strict: bool,
}

impl Admits {
    /// A method reading `kinds`, never categorical: no method reads one, so
    /// an admitted trait has no categorical [`Kind`].  `strict` requires a
    /// present value, two distinct values, and every level taken.
    pub(crate) const fn new(kinds: &'static [TraitKind], strict: bool) -> Admits {
        let mut i = 0;
        while i < kinds.len() {
            assert!(
                !matches!(kinds[i], TraitKind::Categorical),
                "no method admits a categorical trait"
            );
            i += 1;
        }
        Admits { kinds, strict }
    }
}

/// How an admitted trait's values are read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Continuous,
    /// Codes 0 and 1.
    Binary,
    /// Codes `0..k`: the declared levels, else up to the largest code.
    /// Under a strict [`Admits`] some row takes every one.
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

/// A trait that passed [`admit`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Admitted<'a> {
    /// One value per row, NaN where missing.
    pub values: &'a [f64],
    /// Only a strict [`Admits`] checks an ordinal `k` against the rows;
    /// otherwise it is the declared count or the largest code + 1,
    /// saturating, which a huge code makes `usize::MAX`.
    pub kind: Kind,
}

/// The Admission of `t` to a method declaring `admits`, its errors naming
/// `field`.  In order: `trait_length_mismatch`; for a kind the method does
/// not read, `trait_kind_mismatch` when it reads one kind and
/// `unsupported_trait_kind` otherwise; `invalid_trait_value` for a present
/// value that is not finite, or not a level code `0..k` (binary: at most
/// 2); then, when strict, `all_missing_trait`, `constant_trait`, and the
/// first level no row takes: `unused_level` when levels were declared,
/// `sparse_ordinal_codes` otherwise.
pub(crate) fn admit<'a>(
    t: Trait<'a>,
    field: &'static str,
    n_rows: usize,
    admits: Admits,
) -> Result<Admitted<'a>, Error> {
    if t.values.len() != n_rows {
        return Err(Error::TraitLength {
            field,
            expected_length: n_rows,
            actual_length: t.values.len(),
        });
    }
    if !admits.kinds.contains(&t.kind) {
        return Err(match *admits.kinds {
            [expected] => Error::TraitKindMismatch {
                field,
                expected,
                actual: t.kind,
            },
            _ => Error::UnsupportedTraitKind {
                field,
                kind: t.kind,
            },
        });
    }
    let declared = match t.kind {
        TraitKind::Continuous => None,
        _ => t.n_levels,
    };
    // The bound a code stays below; `None` for a continuous trait.
    let ceiling = match t.kind {
        TraitKind::Continuous => None,
        TraitKind::Binary => Some(declared.map_or(2.0, |k| k as f64).min(2.0)),
        TraitKind::Ordinal | TraitKind::Categorical => {
            Some(declared.map_or(f64::INFINITY, |k| k as f64))
        }
    };
    let mut first: Option<f64> = None;
    let mut constant = true;
    let mut max_code = 0.0_f64;
    let mut n_present = 0usize;
    for (position, &value) in t.values.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        let valid = match ceiling {
            None => value.is_finite(),
            Some(ceiling) => value >= 0.0 && value < ceiling && value.fract() == 0.0,
        };
        if !valid {
            return Err(Error::InvalidTraitCode {
                field,
                kind: t.kind,
                position: position.into(),
                value,
            });
        }
        n_present += 1;
        match first {
            None => first = Some(value),
            Some(f) => constant &= f == value,
        }
        max_code = max_code.max(value);
    }
    let kind = match t.kind {
        TraitKind::Continuous => Kind::Continuous,
        TraitKind::Binary => Kind::Binary,
        // Ordinal: `Admits::new` refuses to declare a categorical trait.
        TraitKind::Ordinal | TraitKind::Categorical => Kind::Ordinal {
            // Saturating: under a strict Admits a huge code fails the level
            // check below before `k` is used.
            k: declared.unwrap_or((max_code as usize).saturating_add(1)),
        },
    };
    let admitted = Admitted {
        values: t.values,
        kind,
    };
    if !admits.strict {
        return Ok(admitted);
    }
    let Some(value) = first else {
        return Err(Error::AllMissingTrait { field });
    };
    if constant {
        return Err(Error::ConstantTrait { field, value });
    }
    if kind == Kind::Continuous {
        return Ok(admitted);
    }
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
    Ok(admitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use TraitKind::{Binary, Categorical, Continuous, Ordinal};

    const NA: f64 = f64::NAN;
    const ALL: &[TraitKind] = &[Continuous, Binary, Ordinal];
    const LOOSE: Admits = Admits::new(ALL, false);
    const STRICT: Admits = Admits::new(ALL, true);
    const NUMERIC: Admits = Admits::new(&[Continuous, Binary], false);
    const BINARY: Admits = Admits::new(&[Binary], false);

    fn run(
        values: &[f64],
        kind: TraitKind,
        n_levels: Option<usize>,
        n_rows: usize,
        admits: Admits,
    ) -> Result<Kind, Error> {
        let t = Trait {
            values,
            kind,
            n_levels,
        };
        admit(t, "t", n_rows, admits).map(|a| a.kind)
    }

    /// Each rule, and the order between them: a row with two faults
    /// reports the earlier check's.
    #[test]
    fn admission_rules_in_order() {
        type Row<'a> = (
            &'a [f64],
            TraitKind,
            Option<usize>,
            Admits,
            Result<Kind, &'static str>,
        );
        #[rustfmt::skip]
        let rows: &[Row<'_>] = &[
            // The kind, before the values.
            (&[0.0, 0.5], Categorical, None, LOOSE, Err("unsupported_trait_kind")),
            (&[0.0, 5.0], Ordinal, None, NUMERIC, Err("unsupported_trait_kind")),
            (&[0.0, 0.5], Continuous, None, BINARY, Err("trait_kind_mismatch")),
            (&[0.0, 1.0], Ordinal, None, BINARY, Err("trait_kind_mismatch")),
            // Every present value: finite, or a code below the ceiling.
            (&[0.0, f64::INFINITY], Continuous, None, LOOSE, Err("invalid_trait_value")),
            (&[0.0, f64::NEG_INFINITY], Continuous, None, LOOSE, Err("invalid_trait_value")),
            (&[0.0, 2.0], Binary, None, LOOSE, Err("invalid_trait_value")),
            (&[0.0, 2.0], Binary, Some(3), LOOSE, Err("invalid_trait_value")),
            (&[1.0, 0.0], Binary, Some(1), BINARY, Err("invalid_trait_value")),
            (&[0.0, 0.5], Ordinal, None, LOOSE, Err("invalid_trait_value")),
            (&[0.0, -1.0], Ordinal, None, LOOSE, Err("invalid_trait_value")),
            (&[0.0, 3.0], Ordinal, Some(3), LOOSE, Err("invalid_trait_value")),
            // A bad value, before a strict rule.
            (&[1.0, 1.0, 2.0], Binary, None, STRICT, Err("invalid_trait_value")),
            // Strict: present, then not constant, then every level taken.
            (&[NA, NA], Continuous, None, STRICT, Err("all_missing_trait")),
            (&[1.0, NA, 1.0], Binary, None, STRICT, Err("constant_trait")),
            (&[2.0, 2.0], Ordinal, Some(5), STRICT, Err("constant_trait")),
            (&[10.0, 20.0, 40.0], Ordinal, None, STRICT, Err("sparse_ordinal_codes")),
            (&[0.0, 2.0], Ordinal, None, STRICT, Err("sparse_ordinal_codes")),
            (&[0.0, 1e300], Ordinal, None, STRICT, Err("sparse_ordinal_codes")),
            (&[0.0, 1.0], Ordinal, Some(3), STRICT, Err("unused_level")),
            (&[0.0, 1.0], Binary, Some(3), STRICT, Err("unused_level")),
            // Admitted.
            (&[0.5, NA, -3.0], Continuous, None, STRICT, Ok(Kind::Continuous)),
            (&[0.0, 1.0], Binary, None, BINARY, Ok(Kind::Binary)),
            (&[0.0, 1.0], Ordinal, None, STRICT, Ok(Kind::Ordinal { k: 2 })),
            (&[0.0, 2.0, NA, 1.0], Ordinal, Some(3), STRICT, Ok(Kind::Ordinal { k: 3 })),
            // Without strict, missing, constant and sparse traits pass.
            (&[NA, NA], Binary, None, LOOSE, Ok(Kind::Binary)),
            (&[1.0, 1.0], Continuous, None, LOOSE, Ok(Kind::Continuous)),
            (&[0.0, 1.0], Binary, Some(3), LOOSE, Ok(Kind::Binary)),
            (&[0.0, 2.0], Ordinal, None, LOOSE, Ok(Kind::Ordinal { k: 3 })),
        ];
        for (i, &(values, kind, n_levels, admits, expected)) in rows.iter().enumerate() {
            let got = run(values, kind, n_levels, values.len(), admits);
            assert_eq!(got.map_err(|e| e.code()), expected, "row {i}");
        }
    }

    #[test]
    fn the_length_comes_before_the_kind() {
        for (kind, admits) in [(Categorical, LOOSE), (Continuous, BINARY)] {
            let err = run(&[0.5], kind, None, 2, admits).unwrap_err();
            assert_eq!(
                err,
                Error::TraitLength {
                    field: "t",
                    expected_length: 2,
                    actual_length: 1
                }
            );
        }
    }

    #[test]
    fn errors_name_the_field_and_what_failed() {
        assert_eq!(
            run(&[0.0, 0.5], Continuous, None, 2, BINARY).unwrap_err(),
            Error::TraitKindMismatch {
                field: "t",
                expected: Binary,
                actual: Continuous
            }
        );
        assert_eq!(
            run(&[0.0, 1.0, 7.0], Binary, None, 3, LOOSE).unwrap_err(),
            Error::InvalidTraitCode {
                field: "t",
                kind: Binary,
                position: 2.into(),
                value: 7.0
            }
        );
        assert_eq!(
            run(&[0.0, 1.0], Ordinal, Some(3), 2, STRICT).unwrap_err(),
            Error::UnusedLevel {
                field: "t",
                level: 2,
                n_levels: 3
            }
        );
    }

    #[test]
    fn huge_codes_and_level_counts_are_rejected_without_allocating() {
        // Undeclared levels: the bitmap would have been sized by the code.
        assert_eq!(
            run(&[0.0, 1e12], Ordinal, None, 2, STRICT).unwrap_err(),
            Error::SparseOrdinalCodes {
                field: "t",
                level: 1
            }
        );
        // A huge declared count reports the first unused level.
        assert_eq!(
            run(&[0.0, 1.0], Ordinal, Some(usize::MAX), 2, STRICT).unwrap_err(),
            Error::UnusedLevel {
                field: "t",
                level: 2,
                n_levels: usize::MAX
            }
        );
    }

    #[test]
    #[should_panic(expected = "no method admits a categorical trait")]
    fn no_method_declares_a_categorical_trait() {
        Admits::new(&[Continuous, Categorical], false);
    }
}
