//! Structured errors: a stable code and its operands, never parsed prose.
//!
//! A pedigree failure is pedigree-graph-core's own error, passed through with
//! its code and fields so a host can raise the same class pedigree-graph
//! raises.  Everything pg-phenotype checks itself is a variant here.

use pedigree_graph_core::error::{Error as PgError, ErrorClass, FieldValue as PgField};
use std::fmt;

/// The host exception family a variant maps onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// The pedigree, a trait, or the proband list violates the input contract.
    Validation,
    /// A scoring parameter or CIP table is outside its domain.
    Parameter,
    /// A capacity or allocation limit was hit.
    Resource,
    /// API misuse a host maps onto its plain argument error.
    Usage,
}

/// One keyword field of an error.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    Int(i64),
    Float(f64),
    Str(&'static str),
    Ints(Vec<i64>),
    Strs(Vec<&'static str>),
}

/// A structured pg-phenotype failure.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// pedigree-graph-core rejected the pedigree or ran out of a resource.
    Pedigree(PgError),
    /// `ndegree` is outside `1..=5`.
    DegreeOutOfRange {
        value: i64,
        minimum: i64,
        maximum: i64,
    },
    /// A proband id names no pedigree row.
    UnknownProband { id: i64, position: usize },
    /// A proband id is listed twice.
    DuplicateProband { id: i64, positions: [usize; 2] },
    /// A trait column is not one entry per pedigree row.
    TraitLength {
        field: &'static str,
        expected_length: usize,
        actual_length: usize,
    },
    /// A method needs a trait of another kind.
    TraitKindMismatch {
        field: &'static str,
        expected: &'static str,
        actual: &'static str,
    },
    /// A binary trait value is not 0, 1 or missing.
    InvalidTraitValue {
        field: &'static str,
        position: usize,
        value: f64,
    },
    /// A present age is negative or not finite.
    InvalidAge {
        field: &'static str,
        position: usize,
        value: f64,
    },
    /// A scalar parameter lies outside its domain, written as an interval.
    ParameterOutOfRange {
        name: &'static str,
        value: f64,
        domain: &'static str,
    },
    /// The CIP table breaks its contract.
    InvalidCip {
        reason: &'static str,
        position: usize,
    },
    /// Parameters each in their domain that together give a covariance
    /// that is not positive semidefinite.
    InconsistentParameters { reason: &'static str },
    /// A method takes another number of traits.
    TraitCount {
        actual: usize,
        minimum: usize,
        maximum: usize,
    },
    /// A method does not take traits of this kind.
    UnsupportedTraitKind {
        field: &'static str,
        kind: &'static str,
    },
    /// A trait value that is not a level code (binary, ordinal) or not
    /// finite (continuous).
    InvalidTraitCode {
        field: &'static str,
        kind: &'static str,
        position: usize,
        value: f64,
    },
    /// An ordinal trait with no declared levels skips a code below its largest.
    SparseOrdinalCodes { field: &'static str, level: usize },
    /// A declared level no row takes.
    UnusedLevel {
        field: &'static str,
        level: usize,
        n_levels: usize,
    },
    /// A trait missing in every row.
    AllMissingTrait { field: &'static str },
    /// A trait with one value in every non-missing row.
    ConstantTrait { field: &'static str, value: f64 },
    /// The stratum labels are not one per pedigree row.
    StratumLength {
        expected_length: usize,
        actual_length: usize,
    },
}

impl From<PgError> for Error {
    fn from(err: PgError) -> Error {
        Error::Pedigree(err)
    }
}

impl Error {
    /// The host exception family.
    pub fn class(&self) -> Class {
        match self {
            Error::Pedigree(err) => match err.class() {
                ErrorClass::Validation | ErrorClass::Metadata => Class::Validation,
                ErrorClass::Resource => Class::Resource,
                ErrorClass::Usage => Class::Usage,
            },
            Error::DegreeOutOfRange { .. }
            | Error::UnknownProband { .. }
            | Error::DuplicateProband { .. }
            | Error::TraitLength { .. }
            | Error::TraitKindMismatch { .. }
            | Error::InvalidTraitValue { .. }
            | Error::InvalidAge { .. }
            | Error::TraitCount { .. }
            | Error::UnsupportedTraitKind { .. }
            | Error::InvalidTraitCode { .. }
            | Error::SparseOrdinalCodes { .. }
            | Error::UnusedLevel { .. }
            | Error::AllMissingTrait { .. }
            | Error::ConstantTrait { .. }
            | Error::StratumLength { .. } => Class::Validation,
            Error::ParameterOutOfRange { .. }
            | Error::InvalidCip { .. }
            | Error::InconsistentParameters { .. } => Class::Parameter,
        }
    }

    /// The stable code a host branches on.
    pub fn code(&self) -> &'static str {
        match self {
            // pedigree-graph-core leaves its pool errors uncoded (they map to
            // a plain RuntimeError there); name them so hosts can branch.
            Error::Pedigree(PgError::ThreadPoolConflict { .. }) => "thread_pool_conflict",
            Error::Pedigree(PgError::ThreadPoolUnavailable { .. }) => "thread_pool_unavailable",
            Error::Pedigree(err) => err.code(),
            Error::DegreeOutOfRange { .. } => "degree_out_of_range",
            Error::UnknownProband { .. } => "unknown_proband",
            Error::DuplicateProband { .. } => "duplicate_proband",
            Error::TraitLength { .. } => "trait_length_mismatch",
            Error::TraitKindMismatch { .. } => "trait_kind_mismatch",
            Error::InvalidTraitValue { .. } => "invalid_trait_value",
            Error::InvalidAge { .. } => "invalid_age",
            Error::ParameterOutOfRange { .. } => "parameter_out_of_range",
            Error::InvalidCip { .. } => "invalid_cip",
            Error::InconsistentParameters { .. } => "inconsistent_parameters",
            Error::TraitCount { .. } => "trait_count",
            Error::UnsupportedTraitKind { .. } => "unsupported_trait_kind",
            Error::InvalidTraitCode { .. } => "invalid_trait_value",
            Error::SparseOrdinalCodes { .. } => "sparse_ordinal_codes",
            Error::UnusedLevel { .. } => "unused_level",
            Error::AllMissingTrait { .. } => "all_missing_trait",
            Error::ConstantTrait { .. } => "constant_trait",
            Error::StratumLength { .. } => "stratum_length_mismatch",
        }
    }

    /// The operands, as keyword fields.
    pub fn fields(&self) -> Vec<(&'static str, FieldValue)> {
        use FieldValue::{Float, Int, Ints, Str};
        let int = |n: usize| Int(n as i64);
        match self {
            Error::Pedigree(err) => err
                .fields()
                .into_iter()
                .map(|(name, value)| {
                    let value = match value {
                        PgField::Int(v) => Int(v),
                        PgField::Str(v) => Str(v),
                        PgField::Ints(v) => Ints(v),
                        PgField::Strs(v) => FieldValue::Strs(v),
                    };
                    (name, value)
                })
                .collect(),
            Error::DegreeOutOfRange {
                value,
                minimum,
                maximum,
            } => vec![
                ("value", Int(*value)),
                ("minimum", Int(*minimum)),
                ("maximum", Int(*maximum)),
            ],
            Error::UnknownProband { id, position } => {
                vec![("id", Int(*id)), ("position", int(*position))]
            }
            Error::DuplicateProband { id, positions } => vec![
                ("id", Int(*id)),
                (
                    "positions",
                    Ints(positions.iter().map(|&p| p as i64).collect()),
                ),
            ],
            Error::TraitLength {
                field,
                expected_length,
                actual_length,
            } => vec![
                ("field", Str(field)),
                ("expected_length", int(*expected_length)),
                ("actual_length", int(*actual_length)),
            ],
            Error::TraitKindMismatch {
                field,
                expected,
                actual,
            } => vec![
                ("field", Str(field)),
                ("expected", Str(expected)),
                ("actual", Str(actual)),
            ],
            Error::InvalidTraitValue {
                field,
                position,
                value,
            }
            | Error::InvalidAge {
                field,
                position,
                value,
            } => vec![
                ("field", Str(field)),
                ("position", int(*position)),
                ("value", Float(*value)),
            ],
            Error::ParameterOutOfRange {
                name,
                value,
                domain,
            } => vec![
                ("name", Str(name)),
                ("value", Float(*value)),
                ("domain", Str(domain)),
            ],
            Error::InvalidCip { reason, position } => {
                vec![("reason", Str(reason)), ("position", int(*position))]
            }
            Error::InconsistentParameters { reason } => vec![("reason", Str(reason))],
            Error::TraitCount {
                actual,
                minimum,
                maximum,
            } => vec![
                ("actual", int(*actual)),
                ("minimum", int(*minimum)),
                ("maximum", int(*maximum)),
            ],
            Error::UnsupportedTraitKind { field, kind } => {
                vec![("field", Str(field)), ("kind", Str(kind))]
            }
            Error::InvalidTraitCode {
                field,
                kind,
                position,
                value,
            } => vec![
                ("field", Str(field)),
                ("kind", Str(kind)),
                ("position", int(*position)),
                ("value", Float(*value)),
            ],
            Error::SparseOrdinalCodes { field, level } => {
                vec![("field", Str(field)), ("level", int(*level))]
            }
            Error::UnusedLevel {
                field,
                level,
                n_levels,
            } => vec![
                ("field", Str(field)),
                ("level", int(*level)),
                ("n_levels", int(*n_levels)),
            ],
            Error::AllMissingTrait { field } => vec![("field", Str(field))],
            Error::ConstantTrait { field, value } => {
                vec![("field", Str(field)), ("value", Float(*value))]
            }
            Error::StratumLength {
                expected_length,
                actual_length,
            } => vec![
                ("expected_length", int(*expected_length)),
                ("actual_length", int(*actual_length)),
            ],
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Pedigree(err) => write!(f, "{err}"),
            Error::DegreeOutOfRange { value, minimum, maximum } => {
                write!(f, "ndegree must be in {minimum}..={maximum}, got {value}")
            }
            Error::UnknownProband { id, position } => {
                write!(f, "probands[{position}] = {id} is not a pedigree id")
            }
            Error::DuplicateProband { id, positions } => write!(
                f,
                "proband id {id} is listed at positions {} and {}",
                positions[0], positions[1]
            ),
            Error::TraitLength { field, expected_length, actual_length } => write!(
                f,
                "{field} must have one entry per pedigree row ({expected_length}), got {actual_length}"
            ),
            Error::TraitKindMismatch { field, expected, actual } => {
                write!(f, "{field} must be a {expected} trait, got {actual}")
            }
            Error::InvalidTraitValue { field, position, value } => {
                write!(f, "{field}[{position}] = {value} is not 0, 1 or missing")
            }
            Error::InvalidAge { field, position, value } => {
                write!(f, "{field}[{position}] = {value} is not a finite age >= 0")
            }
            Error::ParameterOutOfRange { name, value, domain } => {
                write!(f, "{name} = {value} is outside {domain}")
            }
            Error::InvalidCip { reason, position } => {
                write!(f, "invalid CIP table at position {position}: {reason}")
            }
            Error::InconsistentParameters { reason } => write!(f, "{reason}"),
            Error::TraitCount { actual, minimum, maximum } => {
                write!(f, "pass {minimum} to {maximum} traits, got {actual}")
            }
            Error::UnsupportedTraitKind { field, kind } => {
                write!(f, "{field} is a {kind} trait, which this method does not take")
            }
            Error::InvalidTraitCode { field, kind, position, value } => match *kind {
                "continuous" => write!(f, "{field}[{position}] = {value} is not finite"),
                "binary" => write!(f, "{field}[{position}] = {value} is not 0, 1 or missing"),
                _ => write!(f, "{field}[{position}] = {value} is not a {kind} level code"),
            },
            Error::SparseOrdinalCodes { field, level } => write!(
                f,
                "{field} has no row at code {level} below its largest code: ordinal codes must be 0..k-1, \
                 or declare the levels"
            ),
            Error::UnusedLevel { field, level, n_levels } => write!(
                f,
                "{field} declares {n_levels} levels but no row has level {level}"
            ),
            Error::AllMissingTrait { field } => write!(f, "{field} is missing in every row"),
            Error::ConstantTrait { field, value } => {
                write!(f, "{field} is constant ({value} in every non-missing row)")
            }
            Error::StratumLength { expected_length, actual_length } => write!(
                f,
                "stratum must have one label per pedigree row ({expected_length}), got {actual_length}"
            ),
        }
    }
}

impl std::error::Error for Error {}
