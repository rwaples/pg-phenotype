//! Structured errors: a stable code and its operands, never parsed prose.
//!
//! A pedigree failure is pedigree-graph-core's own error, passed through with
//! its code and fields so a host can raise the same class pedigree-graph
//! raises.  Everything pg-phenotype checks itself is a variant here.
//!
//! A host renders an error under its [`IndexBase`]: Python counts positions
//! from 0, R from 1, and the message and the fields count alike.  A pedigree
//! error is the exception: pedigree-graph-core writes its message, from 0 in
//! every host, while its fields count from the host's base.

use crate::input::TraitKind;
use pedigree_graph_core::error::{Error as PgError, ErrorClass, FieldValue as PgField};
use std::fmt;

/// Which part of a CIP table's contract it breaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipProblem {
    Empty,
    LengthMismatch,
    AgeNotFinite,
    AgesNotIncreasing,
    CipOutOfRange,
    CipDecreases,
    PrevalenceNotPositive,
}

impl CipProblem {
    /// The stable `reason` a host branches on.
    pub fn name(self) -> &'static str {
        match self {
            CipProblem::Empty => "empty",
            CipProblem::LengthMismatch => "length_mismatch",
            CipProblem::AgeNotFinite => "age_not_finite",
            CipProblem::AgesNotIncreasing => "ages_not_increasing",
            CipProblem::CipOutOfRange => "cip_out_of_range",
            CipProblem::CipDecreases => "cip_decreases",
            CipProblem::PrevalenceNotPositive => "prevalence_not_positive",
        }
    }

    fn describe(self) -> &'static str {
        match self {
            CipProblem::Empty => "the table is empty",
            CipProblem::LengthMismatch => "ages and cip differ in length",
            CipProblem::AgeNotFinite => "an age is not finite",
            CipProblem::AgesNotIncreasing => "ages are not strictly increasing",
            CipProblem::CipOutOfRange => "a cip value is outside [0, 1)",
            CipProblem::CipDecreases => "cip decreases",
            CipProblem::PrevalenceNotPositive => "the last cip value (K) is not positive",
        }
    }
}

/// Why data give no CIP table: the estimated `K` is outside `(0, 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipEstimate {
    /// No case while anyone was at risk, so `K = 0`.
    ZeroPrevalence,
    /// Everyone at risk became a case, so `K = 1`.
    PrevalenceOne,
}

impl CipEstimate {
    /// The stable `reason` a host branches on.
    pub fn name(self) -> &'static str {
        match self {
            CipEstimate::ZeroPrevalence => "zero_prevalence",
            CipEstimate::PrevalenceOne => "prevalence_one",
        }
    }

    fn describe(self) -> &'static str {
        match self {
            CipEstimate::ZeroPrevalence => "no case while anyone was at risk: K is 0",
            CipEstimate::PrevalenceOne => "everyone at risk became a case: K is 1",
        }
    }
}

/// Which joint constraint parameters each in their domain break together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inconsistency {
    /// `rho_within - rg sqrt(h2_1 h2_2)` exceeds `sqrt((1 - h2_1)(1 - h2_2))`.
    NonGeneticCovarianceNotPsd,
}

impl Inconsistency {
    /// The stable `reason` a host branches on.
    pub fn name(self) -> &'static str {
        match self {
            Inconsistency::NonGeneticCovarianceNotPsd => "non_genetic_covariance_not_psd",
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Inconsistency::NonGeneticCovarianceNotPsd => {
                "rho_within - rg*sqrt(h2_1*h2_2) exceeds sqrt((1 - h2_1)(1 - h2_2)): \
                 the non-genetic cross-trait covariance is not positive semidefinite"
            }
        }
    }
}

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

/// Where an entry sits in a list the caller passed or among a Pedigree's
/// rows, counted from 0.
///
/// It has no `Display` and no integer conversion: a message or field reads
/// it only through [`Position::at`], so it always appears in the host's base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position(usize);

impl From<usize> for Position {
    fn from(position: usize) -> Position {
        Position(position)
    }
}

impl Position {
    /// The position as a host counting from `base` writes it.
    pub fn at(self, base: IndexBase) -> i64 {
        self.0 as i64 + base.offset()
    }
}

/// Where a host starts counting positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexBase {
    /// Python and Rust.
    Zero,
    /// R.
    One,
}

impl IndexBase {
    fn offset(self) -> i64 {
        match self {
            IndexBase::Zero => 0,
            IndexBase::One => 1,
        }
    }
}

/// Whether a pedigree-graph-core field holds 0-based rows or positions.
///
/// pedigree-graph-core does not mark its positions, so they are known by name,
/// by the rule of pedigree-graph's R binding (`is_index_field`), so a pedigree
/// error's fields read the same from either package.
fn is_index_name(name: &str) -> bool {
    matches!(name, "row" | "rows" | "column" | "columns" | "position")
        || ["_row", "_rows", "_column", "_columns"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

/// One keyword field of an error.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    Int(i64),
    Float(f64),
    Str(&'static str),
    Ints(Vec<i64>),
    Strs(Vec<&'static str>),
    /// A string the caller passed, such as a category code it misspelt.
    Text(String),
    /// Strings the caller passed, such as two entries' names.
    Texts(Vec<String>),
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
    UnknownProband { id: i64, position: Position },
    /// A proband id is listed twice.
    DuplicateProband { id: i64, positions: [Position; 2] },
    /// An id looked up on a Pedigree names no row.
    UnknownId { id: i64, position: Position },
    /// An id looked up on a Pedigree is listed twice.
    DuplicateLookupId { id: i64, positions: [Position; 2] },
    /// A trait column is not one entry per pedigree row.
    TraitLength {
        field: &'static str,
        expected_length: usize,
        actual_length: usize,
    },
    /// A method needs a trait of another kind.
    TraitKindMismatch {
        field: &'static str,
        expected: TraitKind,
        actual: TraitKind,
    },
    /// A present age is negative or not finite.
    InvalidAge {
        field: &'static str,
        position: Position,
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
        reason: CipProblem,
        position: Position,
    },
    /// Parameters each in their domain that together give a covariance
    /// that is not positive semidefinite.
    InconsistentParameters { reason: Inconsistency },
    /// A method takes another number of traits.
    TraitCount {
        actual: usize,
        minimum: usize,
        maximum: usize,
    },
    /// A method does not take traits of this kind.
    UnsupportedTraitKind {
        field: &'static str,
        kind: TraitKind,
    },
    /// A trait value that is not a level code (binary: 0, 1; ordinal) or
    /// not finite (continuous).
    InvalidTraitCode {
        field: &'static str,
        kind: TraitKind,
        position: Position,
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
    /// A requested category is not a Relationship Category code.
    UnknownCategory {
        category: String,
        position: Position,
    },
    /// A Relationship Category requested twice, in two entries or in one
    /// Category Group; `positions` and `entries` are the entries' positions
    /// in the request and their names (a code, or a group's name).
    DuplicateCategory {
        category: &'static str,
        positions: [Position; 2],
        entries: [String; 2],
    },
    /// A Category Group named like a Relationship Category code, which would
    /// label another category's pairs with that code.
    ReservedGroupName { group: String, position: Position },
    /// A Category Group mixes symmetric and directed categories, or directed
    /// categories with different roles.
    MixedCategoryGroup {
        group: String,
        categories: [&'static str; 2],
    },
    /// Two entries of a category request share a name.
    DuplicateEntryName {
        name: String,
        positions: [Position; 2],
    },
    /// A Category Group with no categories.
    EmptyCategoryGroup { group: String, position: Position },
    /// A continuous trait's quantization step is too coarse for its spread.
    InsufficientPrecision {
        field: &'static str,
        step: f64,
        spread: f64,
    },
    /// The second of two paired columns is not the first's length.
    PairLength {
        /// The two columns' names, first then second.
        columns: [&'static str; 2],
        expected_length: usize,
        actual_length: usize,
    },
    /// A survival status that is not 0 (censored), 1 (case) or 2 (death).
    InvalidStatus { position: Position, value: f64 },
    /// A row that enters the risk set after the age it leaves it, or a case
    /// or death at its entry age.
    EntryNotBeforeAge {
        position: Position,
        entry: f64,
        age: f64,
    },
    /// Data whose estimated CIP is not a valid table.
    CipNotEstimable { reason: CipEstimate },
    /// A parent-offspring call that requests no regression line.
    NoLines,
}

impl From<PgError> for Error {
    fn from(err: PgError) -> Error {
        Error::Pedigree(err)
    }
}

impl Error {
    /// The host exception family.
    pub fn class(&self) -> Class {
        self.spec().0
    }

    /// The stable code a host branches on.
    pub fn code(&self) -> &'static str {
        self.spec().1
    }

    /// Each variant's class and code, in one table so the two cannot drift.
    fn spec(&self) -> (Class, &'static str) {
        use Class::{Parameter, Validation};
        match self {
            Error::Pedigree(err) => {
                let class = match err.class() {
                    ErrorClass::Validation | ErrorClass::Metadata => Class::Validation,
                    ErrorClass::Resource => Class::Resource,
                    ErrorClass::Usage => Class::Usage,
                };
                // pedigree-graph-core leaves its pool errors uncoded (they map
                // to a plain RuntimeError there); name them so hosts can branch.
                let code = match err {
                    PgError::ThreadPoolConflict { .. } => "thread_pool_conflict",
                    PgError::ThreadPoolUnavailable { .. } => "thread_pool_unavailable",
                    _ => err.code(),
                };
                (class, code)
            }
            Error::DegreeOutOfRange { .. } => (Validation, "degree_out_of_range"),
            Error::UnknownProband { .. } => (Validation, "unknown_proband"),
            Error::DuplicateProband { .. } => (Validation, "duplicate_proband"),
            Error::UnknownId { .. } => (Validation, "unknown_id"),
            Error::DuplicateLookupId { .. } => (Validation, "duplicate_lookup_id"),
            Error::TraitLength { .. } => (Validation, "trait_length_mismatch"),
            Error::TraitKindMismatch { .. } => (Validation, "trait_kind_mismatch"),
            Error::InvalidAge { .. } => (Validation, "invalid_age"),
            Error::ParameterOutOfRange { .. } => (Parameter, "parameter_out_of_range"),
            Error::InvalidCip { .. } => (Parameter, "invalid_cip"),
            Error::InconsistentParameters { .. } => (Parameter, "inconsistent_parameters"),
            Error::TraitCount { .. } => (Validation, "trait_count"),
            Error::UnsupportedTraitKind { .. } => (Validation, "unsupported_trait_kind"),
            Error::InvalidTraitCode { .. } => (Validation, "invalid_trait_value"),
            Error::SparseOrdinalCodes { .. } => (Validation, "sparse_ordinal_codes"),
            Error::UnusedLevel { .. } => (Validation, "unused_level"),
            Error::AllMissingTrait { .. } => (Validation, "all_missing_trait"),
            Error::ConstantTrait { .. } => (Validation, "constant_trait"),
            Error::StratumLength { .. } => (Validation, "stratum_length_mismatch"),
            Error::PairLength { .. } => (Validation, "pair_length_mismatch"),
            Error::UnknownCategory { .. } => (Validation, "unknown_category"),
            Error::DuplicateCategory { .. } => (Validation, "duplicate_category"),
            Error::ReservedGroupName { .. } => (Validation, "reserved_group_name"),
            Error::MixedCategoryGroup { .. } => (Validation, "mixed_category_group"),
            Error::DuplicateEntryName { .. } => (Validation, "duplicate_entry_name"),
            Error::EmptyCategoryGroup { .. } => (Validation, "empty_category_group"),
            Error::InsufficientPrecision { .. } => (Validation, "insufficient_precision"),
            Error::InvalidStatus { .. } => (Validation, "invalid_status"),
            Error::EntryNotBeforeAge { .. } => (Validation, "entry_not_before_age"),
            Error::CipNotEstimable { .. } => (Validation, "cip_not_estimable"),
            Error::NoLines => (Class::Usage, "no_lines"),
        }
    }

    /// The operands, as keyword fields, positions counted from `base`.
    pub fn fields(&self, base: IndexBase) -> Vec<(&'static str, FieldValue)> {
        use FieldValue::{Float, Int, Ints, Str, Text};
        let at = |p: &Position| Int(p.at(base));
        let ats = |p: &[Position; 2]| Ints(p.iter().map(|p| p.at(base)).collect());
        let int = |n: usize| Int(n as i64);
        match self {
            Error::Pedigree(err) => err
                .fields()
                .into_iter()
                .map(|(name, value)| {
                    let shift = if is_index_name(name) {
                        base.offset()
                    } else {
                        0
                    };
                    let value = match value {
                        PgField::Int(v) => Int(v + shift),
                        PgField::Str(v) => Str(v),
                        PgField::Ints(v) => Ints(v.into_iter().map(|v| v + shift).collect()),
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
            Error::UnknownProband { id, position } | Error::UnknownId { id, position } => {
                vec![("id", Int(*id)), ("position", at(position))]
            }
            Error::DuplicateProband { id, positions }
            | Error::DuplicateLookupId { id, positions } => {
                vec![("id", Int(*id)), ("positions", ats(positions))]
            }
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
                ("expected", Str(expected.name())),
                ("actual", Str(actual.name())),
            ],
            Error::InvalidAge {
                field,
                position,
                value,
            } => vec![
                ("field", Str(field)),
                ("position", at(position)),
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
                vec![("reason", Str(reason.name())), ("position", at(position))]
            }
            Error::InconsistentParameters { reason } => vec![("reason", Str(reason.name()))],
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
                vec![("field", Str(field)), ("kind", Str(kind.name()))]
            }
            Error::InvalidTraitCode {
                field,
                kind,
                position,
                value,
            } => vec![
                ("field", Str(field)),
                ("kind", Str(kind.name())),
                ("position", at(position)),
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
            Error::UnknownCategory { category, position } => vec![
                ("category", Text(category.clone())),
                ("position", at(position)),
            ],
            Error::DuplicateCategory {
                category,
                positions,
                entries,
            } => vec![
                ("category", Str(category)),
                ("positions", ats(positions)),
                ("entries", FieldValue::Texts(entries.to_vec())),
            ],
            Error::ReservedGroupName { group, position } => {
                vec![("group", Text(group.clone())), ("position", at(position))]
            }
            Error::MixedCategoryGroup { group, categories } => vec![
                ("group", Text(group.clone())),
                ("categories", FieldValue::Strs(categories.to_vec())),
            ],
            Error::DuplicateEntryName { name, positions } => {
                vec![("name", Text(name.clone())), ("positions", ats(positions))]
            }
            Error::EmptyCategoryGroup { group, position } => {
                vec![("group", Text(group.clone())), ("position", at(position))]
            }
            Error::InsufficientPrecision {
                field,
                step,
                spread,
            } => vec![
                ("field", Str(field)),
                ("step", Float(*step)),
                ("spread", Float(*spread)),
            ],
            Error::PairLength {
                columns,
                expected_length,
                actual_length,
            } => vec![
                ("columns", FieldValue::Strs(columns.to_vec())),
                ("expected_length", int(*expected_length)),
                ("actual_length", int(*actual_length)),
            ],
            Error::InvalidStatus { position, value } => {
                vec![("position", at(position)), ("value", Float(*value))]
            }
            Error::EntryNotBeforeAge {
                position,
                entry,
                age,
            } => vec![
                ("position", at(position)),
                ("entry", Float(*entry)),
                ("age", Float(*age)),
            ],
            Error::CipNotEstimable { reason } => vec![("reason", Str(reason.name()))],
            Error::NoLines => vec![],
        }
    }

    /// The message, positions counted from `base`.
    pub fn message(&self, base: IndexBase) -> String {
        Message { err: self, base }.to_string()
    }
}

/// An error's message with its positions counted from `base`.
struct Message<'a> {
    err: &'a Error,
    base: IndexBase,
}

impl fmt::Display for Message<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let base = self.base;
        match self.err {
            // pedigree-graph-core writes this message, counting from 0 in
            // every host, as pedigree-graph's own R binding shows it.
            Error::Pedigree(err) => write!(f, "{err}"),
            Error::DegreeOutOfRange { value, minimum, maximum } => {
                write!(f, "ndegree must be in {minimum}..={maximum}, got {value}")
            }
            Error::UnknownProband { id, position } => {
                write!(f, "probands[{}] = {id} is not a pedigree id", position.at(base))
            }
            Error::DuplicateProband { id, positions } => write!(
                f,
                "proband id {id} is listed at positions {} and {}",
                positions[0].at(base),
                positions[1].at(base)
            ),
            Error::UnknownId { id, position } => {
                write!(f, "ids[{}] = {id} is not a pedigree id", position.at(base))
            }
            Error::DuplicateLookupId { id, positions } => write!(
                f,
                "id {id} is listed at positions {} and {}",
                positions[0].at(base),
                positions[1].at(base)
            ),
            Error::TraitLength { field, expected_length, actual_length } => write!(
                f,
                "{field} must have one entry per pedigree row ({expected_length}), got {actual_length}"
            ),
            Error::TraitKindMismatch { field, expected, actual } => {
                write!(f, "{field} must be a {} trait, got {}", expected.name(), actual.name())
            }
            Error::InvalidAge { field, position, value } => {
                write!(f, "{field}[{}] = {value} is not a finite age >= 0", position.at(base))
            }
            Error::ParameterOutOfRange { name, value, domain } => {
                write!(f, "{name} = {value} is outside {domain}")
            }
            Error::InvalidCip { reason, position } => {
                write!(f, "invalid CIP table at position {}: {}", position.at(base), reason.describe())
            }
            Error::InconsistentParameters { reason } => write!(f, "{}", reason.describe()),
            Error::TraitCount { actual, minimum, maximum } => {
                write!(f, "pass {minimum} to {maximum} traits, got {actual}")
            }
            Error::UnsupportedTraitKind { field, kind } => {
                write!(f, "{field} is a {} trait, which this method does not take", kind.name())
            }
            Error::InvalidTraitCode { field, kind, position, value } => {
                let position = position.at(base);
                match kind {
                    TraitKind::Continuous => write!(f, "{field}[{position}] = {value} is not finite"),
                    TraitKind::Binary => {
                        write!(f, "{field}[{position}] = {value} is not 0, 1 or missing")
                    }
                    _ => write!(
                        f,
                        "{field}[{position}] = {value} is not a {} level code",
                        kind.name()
                    ),
                }
            }
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
            Error::UnknownCategory { category, position } => write!(
                f,
                "categories[{}] names {category:?}, which is not a Relationship Category code",
                position.at(base)
            ),
            Error::DuplicateCategory { category, positions: [p, q], entries: [a, b] } => {
                let (p, q) = (p.at(base), q.at(base));
                if p == q && a == b {
                    write!(f, "category {category} is listed twice in group {a:?} (categories[{p}])")
                } else {
                    write!(
                        f,
                        "category {category} is in both {a:?} (categories[{p}]) and {b:?} (categories[{q}])"
                    )
                }
            }
            Error::ReservedGroupName { group, position } => write!(
                f,
                "category group {group:?} (categories[{}]) is named like a Relationship \
                 Category code; name it apart from every code",
                position.at(base)
            ),
            Error::MixedCategoryGroup { group, categories: [a, b] } => write!(
                f,
                "category group {group:?} mixes {a} and {b}: a group's categories are all symmetric, \
                 or all directed with the same roles"
            ),
            Error::DuplicateEntryName { name, positions } => write!(
                f,
                "categories[{}] and categories[{}] are both named {name:?}",
                positions[0].at(base),
                positions[1].at(base)
            ),
            Error::EmptyCategoryGroup { group, position } => {
                write!(f, "category group {group:?} (categories[{}]) is empty", position.at(base))
            }
            Error::InsufficientPrecision { field, step, spread } => write!(
                f,
                "{field} is quantized in steps of {step:e}, more than 2^-20 of its spread {spread:e}: \
                 its range is too wide for its spread"
            ),
            Error::PairLength { columns: [first, second], expected_length, actual_length } => write!(
                f,
                "{second} must have one entry per entry of {first} ({expected_length}), got {actual_length}"
            ),
            Error::InvalidStatus { position, value } => {
                write!(
                    f,
                    "status[{}] = {value} is not 0 (censored), 1 (case) or 2 (death)",
                    position.at(base)
                )
            }
            Error::EntryNotBeforeAge { position, entry, age } => write!(
                f,
                "row {} enters at {entry}, not before its age {age}: \
                 only a censored row may leave at its entry age",
                position.at(base)
            ),
            Error::CipNotEstimable { reason } => {
                write!(f, "no CIP table: {}", reason.describe())
            }
            Error::NoLines => write!(f, "request at least one of midparent, mother and father"),
        }
    }
}

impl fmt::Display for Error {
    /// The message as Python and Rust count, from 0.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = Message {
            err: self,
            base: IndexBase::Zero,
        };
        message.fmt(f)
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    /// How many variants `Error` has.
    const VARIANTS: usize = 32;

    /// Each variant's number, so `samples` can be checked to hold every
    /// variant: a new variant does not compile until it has an arm here.
    fn variant(err: &Error) -> usize {
        match err {
            Error::Pedigree(_) => 0,
            Error::DegreeOutOfRange { .. } => 1,
            Error::UnknownProband { .. } => 2,
            Error::DuplicateProband { .. } => 3,
            Error::UnknownId { .. } => 4,
            Error::DuplicateLookupId { .. } => 5,
            Error::TraitLength { .. } => 6,
            Error::TraitKindMismatch { .. } => 7,
            Error::InvalidAge { .. } => 8,
            Error::ParameterOutOfRange { .. } => 9,
            Error::InvalidCip { .. } => 10,
            Error::InconsistentParameters { .. } => 11,
            Error::TraitCount { .. } => 12,
            Error::UnsupportedTraitKind { .. } => 13,
            Error::InvalidTraitCode { .. } => 14,
            Error::SparseOrdinalCodes { .. } => 15,
            Error::UnusedLevel { .. } => 16,
            Error::AllMissingTrait { .. } => 17,
            Error::ConstantTrait { .. } => 18,
            Error::StratumLength { .. } => 19,
            Error::UnknownCategory { .. } => 20,
            Error::DuplicateCategory { .. } => 21,
            Error::ReservedGroupName { .. } => 22,
            Error::MixedCategoryGroup { .. } => 23,
            Error::DuplicateEntryName { .. } => 24,
            Error::EmptyCategoryGroup { .. } => 25,
            Error::InsufficientPrecision { .. } => 26,
            Error::PairLength { .. } => 27,
            Error::InvalidStatus { .. } => 28,
            Error::EntryNotBeforeAge { .. } => 29,
            Error::CipNotEstimable { .. } => 30,
            Error::NoLines => 31,
        }
    }

    /// One error of every variant.  Positions are 61 and 93, numbers no
    /// other operand contains, so a message shows which base wrote them.
    fn samples() -> Vec<Error> {
        let (p, q) = (Position(61), Position(93));
        vec![
            Error::Pedigree(PgError::MzSelfReference { row: 61, id: 5 }),
            Error::Pedigree(PgError::DuplicateId {
                id: 5,
                rows: vec![61, 93],
                duplicate_count: 1,
            }),
            Error::DegreeOutOfRange {
                value: 7,
                minimum: 1,
                maximum: 5,
            },
            Error::UnknownProband { id: 5, position: p },
            Error::DuplicateProband {
                id: 5,
                positions: [p, q],
            },
            Error::UnknownId { id: 5, position: p },
            Error::DuplicateLookupId {
                id: 5,
                positions: [p, q],
            },
            Error::TraitLength {
                field: "trait",
                expected_length: 4,
                actual_length: 3,
            },
            Error::TraitKindMismatch {
                field: "trait",
                expected: TraitKind::Binary,
                actual: TraitKind::Continuous,
            },
            Error::InvalidAge {
                field: "age",
                position: p,
                value: -2.5,
            },
            Error::ParameterOutOfRange {
                name: "h2",
                value: 2.5,
                domain: "[0, 1]",
            },
            Error::InvalidCip {
                reason: CipProblem::CipDecreases,
                position: p,
            },
            Error::InconsistentParameters {
                reason: Inconsistency::NonGeneticCovarianceNotPsd,
            },
            Error::TraitCount {
                actual: 3,
                minimum: 1,
                maximum: 2,
            },
            Error::UnsupportedTraitKind {
                field: "trait",
                kind: TraitKind::Categorical,
            },
            Error::InvalidTraitCode {
                field: "trait",
                kind: TraitKind::Continuous,
                position: p,
                value: f64::INFINITY,
            },
            Error::InvalidTraitCode {
                field: "trait",
                kind: TraitKind::Binary,
                position: p,
                value: 2.0,
            },
            Error::InvalidTraitCode {
                field: "trait",
                kind: TraitKind::Ordinal,
                position: p,
                value: 2.5,
            },
            Error::SparseOrdinalCodes {
                field: "trait",
                level: 2,
            },
            Error::UnusedLevel {
                field: "trait",
                level: 2,
                n_levels: 4,
            },
            Error::AllMissingTrait { field: "trait" },
            Error::ConstantTrait {
                field: "trait",
                value: 0.5,
            },
            Error::StratumLength {
                expected_length: 4,
                actual_length: 3,
            },
            Error::UnknownCategory {
                category: "XX".to_string(),
                position: p,
            },
            Error::DuplicateCategory {
                category: "MO",
                positions: [p, q],
                entries: ["A".to_string(), "B".to_string()],
            },
            Error::DuplicateCategory {
                category: "MO",
                positions: [p, p],
                entries: ["A".to_string(), "A".to_string()],
            },
            Error::ReservedGroupName {
                group: "FS".to_string(),
                position: p,
            },
            Error::MixedCategoryGroup {
                group: "A".to_string(),
                categories: ["FS", "MO"],
            },
            Error::DuplicateEntryName {
                name: "A".to_string(),
                positions: [p, q],
            },
            Error::EmptyCategoryGroup {
                group: "A".to_string(),
                position: p,
            },
            Error::InsufficientPrecision {
                field: "trait",
                step: 0.25,
                spread: 2.0,
            },
            Error::PairLength {
                columns: ["ids", "values"],
                expected_length: 4,
                actual_length: 3,
            },
            Error::InvalidStatus {
                position: p,
                value: 3.0,
            },
            Error::EntryNotBeforeAge {
                position: p,
                entry: 2.5,
                age: 2.0,
            },
            Error::CipNotEstimable {
                reason: CipEstimate::ZeroPrevalence,
            },
            Error::NoLines,
        ]
    }

    #[test]
    fn samples_hold_every_variant() {
        let mut seen = [false; VARIANTS];
        for err in samples() {
            let i = variant(&err);
            assert!(i < VARIANTS, "count variant {i} in VARIANTS");
            seen[i] = true;
        }
        let missing: Vec<usize> = (0..VARIANTS).filter(|&i| !seen[i]).collect();
        assert!(missing.is_empty(), "add a sample of variant(s) {missing:?}");
    }

    /// An integer field's values, if it has any.
    fn ints(value: &FieldValue) -> Option<Vec<i64>> {
        match value {
            FieldValue::Int(n) => Some(vec![*n]),
            FieldValue::Ints(v) => Some(v.clone()),
            _ => None,
        }
    }

    #[test]
    fn positions_count_from_the_hosts_base() {
        for err in samples() {
            let zero = err.fields(IndexBase::Zero);
            let one = err.fields(IndexBase::One);
            let (message_zero, message_one) =
                (err.message(IndexBase::Zero), err.message(IndexBase::One));
            assert_eq!(message_zero, err.to_string(), "{err:?}");
            let mut shifted = false;
            for ((name, v0), (_, v1)) in zero.iter().zip(&one) {
                // pg-phenotype names its pairs `positions`, which
                // pedigree-graph-core never does.
                let positions = (is_index_name(name) || *name == "positions")
                    .then(|| ints(v0).zip(ints(v1)))
                    .flatten();
                let Some((p0, p1)) = positions else {
                    assert_eq!(v0, v1, "{name} of {err:?} is not a position");
                    continue;
                };
                shifted = true;
                for (p0, p1) in p0.iter().zip(&p1) {
                    assert_eq!(*p1, p0 + 1, "{name} of {err:?}");
                    // pedigree-graph-core writes its own message, from 0.
                    if !matches!(err, Error::Pedigree(_)) {
                        assert!(message_zero.contains(&p0.to_string()), "{message_zero}");
                        assert!(message_one.contains(&p1.to_string()), "{message_one}");
                        assert!(!message_one.contains(&p0.to_string()), "{message_one}");
                    }
                }
            }
            if !shifted || matches!(err, Error::Pedigree(_)) {
                assert_eq!(message_zero, message_one, "{err:?}");
            }
        }
    }
}
