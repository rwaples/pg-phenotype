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

/// What a present value of a continuous, binary or ordinal trait must be:
/// finite, or a level code `0..k` (binary: at most 2).  Methods check each
/// value in their own pass over the trait.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValueRule {
    field: &'static str,
    kind: TraitKind,
    /// The bound a code stays below; `None` for a continuous trait.
    ceiling: Option<f64>,
}

impl ValueRule {
    /// The rule of `t`, after its length and kind: `trait_length_mismatch`,
    /// then `unsupported_trait_kind` for a categorical trait.
    pub(crate) fn of(
        t: &Trait<'_>,
        field: &'static str,
        n_rows: usize,
    ) -> Result<ValueRule, Error> {
        if t.values.len() != n_rows {
            return Err(Error::TraitLength {
                field,
                expected_length: n_rows,
                actual_length: t.values.len(),
            });
        }
        let ceiling = match t.kind {
            TraitKind::Categorical => {
                return Err(Error::UnsupportedTraitKind {
                    field,
                    kind: t.kind,
                })
            }
            TraitKind::Continuous => None,
            TraitKind::Binary => Some(t.n_levels.map_or(2.0, |k| k as f64).min(2.0)),
            TraitKind::Ordinal => Some(t.n_levels.map_or(f64::INFINITY, |k| k as f64)),
        };
        Ok(ValueRule {
            field,
            kind: t.kind,
            ceiling,
        })
    }

    /// `invalid_trait_value` unless the present `value` at `position` obeys the rule.
    pub(crate) fn check(self, position: usize, value: f64) -> Result<(), Error> {
        let valid = match self.ceiling {
            None => value.is_finite(),
            Some(ceiling) => value >= 0.0 && value < ceiling && value.fract() == 0.0,
        };
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidTraitCode {
                field: self.field,
                kind: self.kind,
                position: position.into(),
                value,
            })
        }
    }
}
