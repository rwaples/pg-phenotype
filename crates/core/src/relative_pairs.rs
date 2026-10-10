//! The Relative-Pair Correlation: one trait's correlation between the
//! members of each Relationship Category's (or Category Group's) Relative
//! Pairs (ADR 0009).
//!
//! The pairs come from pedigree-graph-core's `relationship_moments`, which
//! reduces them to exact per-cell sums and never stores them.  A symmetric
//! category's pairs enter in both orders, so its Pearson is intraclass and
//! its tetrachoric fits one threshold; a directed category's pairs enter
//! once, in its roles' order.  There is no SE yet (ADR 0009).

use crate::assortative::Reason as FitReason;
use crate::correlation;
use crate::error::Error;
use crate::input::{Trait, TraitKind, ValueRule};
use crate::pedigree::PedigreeArg;
use crate::value::Value;
use pedigree_graph_core::error::Error as PgError;
use pedigree_graph_core::relationships::{
    quantize_column, relationship_moments, Category, CategorySet, MaxDegree, MomentsInput,
    MomentsTable, Operand, Product, Progress, Receiver, Side, Statistic, Symmetric, N_CATEGORIES,
};
use std::borrow::Cow;
use std::num::NonZeroUsize;

/// The trait's field name, as errors report it.
const FIELD: &str = "trait";

/// The deepest degree the default request covers.
pub const DEFAULT_DEGREE: u8 = 3;

/// The default bound on the engine's accumulators: 1 GiB, as in
/// pedigree-graph's Python.
pub const DEFAULT_MEMORY_BUDGET_BYTES: i64 = 1 << 30;

/// The coarsest quantization step a continuous trait takes, as a fraction of
/// its spread.
const PRECISION: f64 = 1.0 / (1u64 << 20) as f64;

/// One entry of a category request: a Relationship Category code, or a
/// named Category Group.  Each entry gives one cell.  `position` is its
/// index in the caller's request, as errors report it; groups a host gives
/// together (one Python mapping) share one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entry {
    Category {
        code: String,
        position: usize,
    },
    Group {
        name: String,
        members: Vec<String>,
        position: usize,
    },
}

/// A correlation estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimator {
    Pearson,
    Tetrachoric,
    Phi,
}

impl Estimator {
    pub fn name(self) -> &'static str {
        match self {
            Estimator::Pearson => "pearson",
            Estimator::Tetrachoric => "tetrachoric",
            Estimator::Phi => "phi",
        }
    }
}

/// Why an estimate is undefined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// No pair has both values observed.
    NoCompletePairs,
    /// A binary side shows one level.
    ConstantMargin,
    /// A continuous side is constant at the quantization step.
    ConstantValue,
    /// Any other reason the tetrachoric fit gives.
    Fit(FitReason),
}

impl Reason {
    pub fn name(self) -> &'static str {
        match self {
            Reason::NoCompletePairs => "no_complete_pairs",
            Reason::ConstantMargin => "constant_margin",
            Reason::ConstantValue => "constant_value",
            Reason::Fit(reason) => reason.name(),
        }
    }
}

impl From<FitReason> for Reason {
    fn from(reason: FitReason) -> Reason {
        match reason {
            FitReason::NoCompletePairs => Reason::NoCompletePairs,
            FitReason::ConstantMargin => Reason::ConstantMargin,
            other => Reason::Fit(other),
        }
    }
}

/// One estimator's value in a cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub estimator: Estimator,
    pub value: Result<f64, Reason>,
    /// Whether the tetrachoric fit stopped at a bound; `None` otherwise.
    pub boundary: Option<bool>,
}

/// A cell's binary pair counts, as the caller sees them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counts {
    /// A continuous trait: no counts.
    None,
    /// A symmetric category's Concordance: both 0, discordant, both 1.
    Concordance([u64; 3]),
    /// A directed category's 2 x 2 table, rows by the first role's value.
    Table([[u64; 2]; 2]),
}

/// One entry's correlation.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    /// The category's code, or the group's name.
    pub category: String,
    pub members: Vec<Category>,
    pub symmetric: bool,
    /// A directed category's roles, first then second; a group's with
    /// mother and father read as parent.
    pub roles: Option<[&'static str; 2]>,
    /// Unordered pairs with both values observed.
    pub n: u64,
    /// Unordered pairs with either value missing.
    pub n_dropped: u64,
    pub counts: Counts,
    /// The primary estimate first.
    pub estimates: Vec<Estimate>,
}

/// The result of [`correlation`].
#[derive(Clone, Debug, PartialEq)]
pub struct RelativePairCorrelation {
    /// One per entry, in request order.
    pub cells: Vec<Cell>,
    pub kind: TraitKind,
    /// A continuous trait's quantization exponent `e`: its centred values
    /// are integers in steps of `2^-e`.
    pub quantization_exponent: Option<i64>,
    pub memory_budget_bytes: u64,
    pub threads: usize,
}

/// One entry, resolved.
#[derive(Clone, Debug, PartialEq)]
struct Request {
    name: String,
    members: Vec<Category>,
    symmetric: bool,
    roles: Option<[&'static str; 2]>,
}

/// The Relative-Pair Correlation of `trait_` over `pedigree`, one cell per
/// entry of `categories` (default: each category up to degree 3).
///
/// # Errors
///
/// Any pedigree-graph-core validation error of columns, then
/// `trait_length_mismatch`, `unsupported_trait_kind` (ordinal,
/// categorical), `invalid_trait_value`; then `unknown_category`,
/// `duplicate_category`, `duplicate_entry_name`, `empty_category_group`,
/// `mixed_category_group`; then `parameter_out_of_range` for a budget below
/// one byte; `insufficient_precision`; and the engine's errors
/// (`memory_budget_exceeded`, ...).
pub fn correlation(
    pedigree: PedigreeArg<'_>,
    trait_: Trait<'_>,
    categories: Option<&[Entry]>,
    memory_budget_bytes: i64,
) -> Result<RelativePairCorrelation, Error> {
    let pedigree = pedigree.resolve()?;
    let n_rows = pedigree.len();
    check_trait(trait_, n_rows)?;
    let requests = match categories {
        Some(entries) => resolve(entries)?,
        None => default_requests(),
    };
    let budget = u64::try_from(memory_budget_bytes)
        .ok()
        .filter(|&b| b >= 1)
        .ok_or(Error::ParameterOutOfRange {
            name: "memory_budget_bytes",
            value: memory_budget_bytes as f64,
            domain: "[1, 2^63)",
        })?;
    let threads = rayon::current_num_threads();
    let side = Member::of(trait_)?;
    let mut result = RelativePairCorrelation {
        cells: Vec::with_capacity(requests.len()),
        kind: trait_.kind,
        quantization_exponent: side.exponent,
        memory_budget_bytes: budget,
        threads,
    };
    let union: CategorySet = requests.iter().flat_map(|r| r.members.clone()).collect();
    let Some(top) = union.top_degree() else {
        return Ok(result);
    };
    let ped = pedigree.relationships()?;
    let input = side.input();
    let moments = relationship_moments(
        &ped,
        MaxDegree::try_new(top)?,
        union,
        Receiver::Graph,
        &input,
        Symmetric::Both,
        NonZeroUsize::new(threads).unwrap_or(NonZeroUsize::MIN),
        budget,
        &Progress::default(),
    )?;
    let order: Vec<Category> = union.iter().collect();
    let table = MomentsTable::from_bytes(
        vec![order.len(), side.n_labels, side.n_labels],
        input.n_columns,
        input.products.to_vec(),
        side.exponent.into_iter().collect(),
        moments.width,
        Cow::Owned(moments.table),
    )?;
    for request in requests {
        let positions: Vec<usize> = request
            .members
            .iter()
            .filter_map(|m| order.iter().position(|c| c == m))
            .collect();
        let summed = table.select(0, &positions)?.sum(0)?;
        result.cells.push(cell(request, &summed, trait_.kind)?);
    }
    Ok(result)
}

/// The trait's checks: its length and kind (categorical raises in
/// [`ValueRule::of`], ordinal here), then every present value.  An
/// all-missing or constant trait is not an error: its cells carry a reason.
fn check_trait(t: Trait<'_>, n_rows: usize) -> Result<(), Error> {
    let rule = ValueRule::of(&t, FIELD, n_rows)?;
    if t.kind == TraitKind::Ordinal {
        return Err(Error::UnsupportedTraitKind {
            field: FIELD,
            kind: t.kind,
        });
    }
    for (position, &value) in t.values.iter().enumerate() {
        if !value.is_nan() {
            rule.check(position, value)?;
        }
    }
    Ok(())
}

/// Each category up to [`DEFAULT_DEGREE`], one entry each, in registry order.
fn default_requests() -> Vec<Request> {
    Category::ALL
        .into_iter()
        .filter(|c| c.degree() <= DEFAULT_DEGREE)
        .map(single)
        .collect()
}

fn single(category: Category) -> Request {
    Request {
        name: category.code().to_string(),
        members: vec![category],
        symmetric: category.symmetric(),
        roles: category.roles().map(|(a, b)| [a, b]),
    }
}

/// A directed category's roles with mother and father read as parent.
fn group_roles(category: Category) -> Option<[&'static str; 2]> {
    let parent = |role: &'static str| match role {
        "mother" | "father" => "parent",
        other => other,
    };
    category.roles().map(|(a, b)| [parent(a), parent(b)])
}

/// The entries, checked in order: each is a known code, no category is in
/// two entries (or twice in one group), names are distinct, and a group is
/// named apart from every category code, non-empty and of one kind.
fn resolve(entries: &[Entry]) -> Result<Vec<Request>, Error> {
    fn claim<'a>(
        owner: &mut [Option<(usize, &'a str)>; N_CATEGORIES],
        category: Category,
        position: usize,
        name: &'a str,
    ) -> Result<(), Error> {
        if let Some((first, first_name)) = owner[category.index()] {
            return Err(Error::DuplicateCategory {
                category: category.code(),
                positions: [first, position],
                entries: [first_name.to_string(), name.to_string()],
            });
        }
        owner[category.index()] = Some((position, name));
        Ok(())
    }
    // The entry, by position and name, that first claimed each category.
    let mut owner: [Option<(usize, &str)>; N_CATEGORIES] = [None; N_CATEGORIES];
    let mut requests: Vec<(usize, Request)> = Vec::with_capacity(entries.len());
    let parse = |code: &str, position: usize| {
        Category::parse(code).ok_or_else(|| Error::UnknownCategory {
            category: code.to_string(),
            position,
        })
    };
    for entry in entries {
        let request = match *entry {
            Entry::Category { ref code, position } => {
                let category = parse(code, position)?;
                claim(&mut owner, category, position, code)?;
                (position, single(category))
            }
            Entry::Group {
                ref name,
                ref members,
                position,
            } => {
                if Category::parse(name).is_some() {
                    return Err(Error::ReservedGroupName {
                        group: name.clone(),
                        position,
                    });
                }
                let Some(first) = members.first() else {
                    return Err(Error::EmptyCategoryGroup {
                        group: name.clone(),
                        position,
                    });
                };
                let first = parse(first, position)?;
                claim(&mut owner, first, position, name)?;
                let mut cats = vec![first];
                for code in &members[1..] {
                    let category = parse(code, position)?;
                    claim(&mut owner, category, position, name)?;
                    if category.symmetric() != first.symmetric()
                        || group_roles(category) != group_roles(first)
                    {
                        return Err(Error::MixedCategoryGroup {
                            group: name.clone(),
                            categories: [first.code(), category.code()],
                        });
                    }
                    cats.push(category);
                }
                (
                    position,
                    Request {
                        name: name.clone(),
                        members: cats,
                        symmetric: first.symmetric(),
                        roles: group_roles(first),
                    },
                )
            }
        };
        let (position, request) = request;
        if let Some(&(first, _)) = requests.iter().find(|(_, r)| r.name == request.name) {
            return Err(Error::DuplicateEntryName {
                name: request.name,
                positions: [first, position],
            });
        }
        requests.push((position, request));
    }
    Ok(requests.into_iter().map(|(_, r)| r).collect())
}

/// One member's inputs to the engine: the same labels and values on both
/// sides of a pair.
struct Member {
    labels: Vec<i32>,
    n_labels: usize,
    /// One quantized column for a continuous trait, none for a binary one.
    values: Vec<i64>,
    exponent: Option<i64>,
}

/// The one product a continuous trait takes: first x second.
const PRODUCT: [Product; 1] = [Product {
    a: Operand {
        side: Side::First,
        column: 0,
    },
    b: Operand {
        side: Side::Second,
        column: 0,
    },
}];

impl Member {
    /// Binary: labels 0 and 1, 2 for missing.  Continuous: label 0 observed,
    /// 1 missing, and the values centred on their midrange and quantized.
    fn of(t: Trait<'_>) -> Result<Member, Error> {
        if t.kind == TraitKind::Binary {
            return Ok(Member {
                labels: t
                    .values
                    .iter()
                    .map(|&v| if v.is_nan() { 2 } else { v as i32 })
                    .collect(),
                n_labels: 3,
                values: Vec::new(),
                exponent: None,
            });
        }
        let (min, max) = t
            .values
            .iter()
            .filter(|v| !v.is_nan())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        // No overflow near f64::MAX; Pearson is shift-invariant.
        let centre = if min <= max {
            min / 2.0 + max / 2.0
        } else {
            0.0
        };
        let mut scratch: Vec<f64> = t
            .values
            .iter()
            .map(|&v| if v.is_nan() { 0.0 } else { v - centre })
            .collect();
        let mut values = vec![0i64; scratch.len()];
        let exponent = quantize_column(&scratch, &mut values, 1, 0, FIELD)?;
        // The present centred values, compacted in place.
        let mut kept = 0;
        for (i, &v) in t.values.iter().enumerate() {
            if !v.is_nan() {
                scratch[kept] = scratch[i];
                kept += 1;
            }
        }
        scratch.truncate(kept);
        check_precision(&mut scratch, exponent)?;
        Ok(Member {
            labels: t.values.iter().map(|&v| i32::from(v.is_nan())).collect(),
            n_labels: 2,
            values,
            exponent: Some(exponent),
        })
    }

    fn input(&self) -> MomentsInput<'_> {
        let continuous = self.exponent.is_some();
        MomentsInput {
            labels_first: &self.labels,
            n_labels_first: self.n_labels,
            labels_second: &self.labels,
            n_labels_second: self.n_labels,
            values: &self.values,
            n_columns: usize::from(continuous),
            products: if continuous { &PRODUCT } else { &[] },
            same: &[],
            n_same: 0,
        }
    }
}

/// `2^-e`, which underflows to 0 past `e = 1074`.
fn step_of(exponent: i64) -> f64 {
    libm::ldexp(1.0, -(exponent.clamp(-2000, 2000) as i32))
}

/// `insufficient_precision` when the step exceeds `2^-20` of the present
/// values' interquartile range, or of their SD when the IQR is 0.  A
/// constant (or empty) trait has no spread and passes.
fn check_precision(present: &mut [f64], exponent: i64) -> Result<(), Error> {
    let spread = match iqr(present) {
        0.0 => sd(present),
        iqr => iqr,
    };
    let step = step_of(exponent);
    if spread > 0.0 && step > PRECISION * spread {
        return Err(Error::InsufficientPrecision {
            field: FIELD,
            step,
            spread,
        });
    }
    Ok(())
}

/// The interquartile range with linear interpolation (NumPy's default); 0
/// for fewer than two values.
/// Reorders `values` (a selection, not a sort).
fn iqr(values: &mut [f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let last = values.len() - 1;
    let mut quantile = |p: f64| {
        let h = last as f64 * p;
        let (lo, frac) = (h.floor() as usize, h - h.floor());
        let (_, &mut below, above) = values.select_nth_unstable_by(lo, f64::total_cmp);
        // The next order statistic is the smallest value above position lo.
        let next = above.iter().copied().fold(f64::INFINITY, f64::min);
        if lo == last {
            below
        } else {
            below + frac * (next - below)
        }
    };
    let q3 = quantile(0.75);
    q3 - quantile(0.25)
}

/// The sample SD, scaled by the largest magnitude so squares cannot
/// overflow; 0 for fewer than two values.
fn sd(values: &[f64]) -> f64 {
    let scale = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if values.len() < 2 || scale == 0.0 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|v| v / scale).sum::<f64>() / n;
    let ss: f64 = values.iter().map(|v| (v / scale - mean).powi(2)).sum();
    scale * (ss / (n - 1.0)).sqrt()
}

/// One entry's cell from its summed `[labels, labels]` table.
fn cell(request: Request, table: &MomentsTable<'_>, kind: TraitKind) -> Result<Cell, Error> {
    let counts: Vec<u64> = table.counts().iter().map(|&c| c as u64).collect();
    let n_labels = table.shape()[0];
    let at = |i: usize, j: usize| counts[i * n_labels + j];
    let total: u64 = counts.iter().sum();
    let observed: u64 = if kind == TraitKind::Binary {
        at(0, 0) + at(0, 1) + at(1, 0) + at(1, 1)
    } else {
        at(0, 0)
    };
    // Each symmetric pair entered in both orders, so its table is symmetric
    // with an even diagonal, and halving is exact.  Anything else is an
    // engine contract break, raised rather than rounded.
    if request.symmetric {
        let mirrored =
            (0..n_labels).all(|i| at(i, i) % 2 == 0 && (0..i).all(|j| at(i, j) == at(j, i)));
        if !mirrored {
            return Err(PgError::InvalidMomentsTable {
                reason: format!(
                    "the double-entered counts of symmetric {} are not mirrored: {counts:?}",
                    request.name
                ),
            }
            .into());
        }
    }
    let halve = |count: u64| if request.symmetric { count / 2 } else { count };
    let (n, n_dropped) = (halve(observed), halve(total - observed));
    let (counts, estimates) = if kind == TraitKind::Binary {
        let fitted = [[at(0, 0), at(0, 1)], [at(1, 0), at(1, 1)]];
        let shown = if request.symmetric {
            Counts::Concordance([at(0, 0) / 2, at(0, 1), at(1, 1) / 2])
        } else {
            Counts::Table(fitted)
        };
        (shown, vec![tetrachoric(fitted), phi(fitted)])
    } else {
        let r = table.derive(Statistic::Pearson, 0, FIELD)?[0];
        let value = match (n, r.is_nan()) {
            (0, _) => Err(Reason::NoCompletePairs),
            (_, true) => Err(Reason::ConstantValue),
            _ => Ok(r),
        };
        let pearson = Estimate {
            estimator: Estimator::Pearson,
            value,
            boundary: None,
        };
        (Counts::None, vec![pearson])
    };
    Ok(Cell {
        category: request.name,
        members: request.members,
        symmetric: request.symmetric,
        roles: request.roles,
        n,
        n_dropped,
        counts,
        estimates,
    })
}

/// ADR 0006's two-step point estimate of the table, without its SE.
fn tetrachoric(table: [[u64; 2]; 2]) -> Estimate {
    let point = correlation::point(table);
    Estimate {
        estimator: Estimator::Tetrachoric,
        value: point.map(|p| p.value).map_err(Reason::from),
        boundary: point.ok().and_then(|p| p.boundary),
    }
}

/// `(N00 N11 - N01 N10) / sqrt(r0 r1 c0 c1)`, the numerator exact.
fn phi(table: [[u64; 2]; 2]) -> Estimate {
    let [[a, b], [c, d]] = table.map(|row| row.map(i128::from));
    let margins = [a + b, c + d, a + c, b + d];
    let value = if a + b + c + d == 0 {
        Err(Reason::NoCompletePairs)
    } else if margins.contains(&0) {
        Err(Reason::ConstantMargin)
    } else {
        let [r0, r1, c0, c1] = margins.map(|m| m as f64);
        Ok(((a * d - b * c) as f64 / ((r0 * r1).sqrt() * (c0 * c1).sqrt())).clamp(-1.0, 1.0))
    };
    Estimate {
        estimator: Estimator::Phi,
        value,
        boundary: None,
    }
}

const METHOD: [(&str, &str); 4] = [
    (
        "entry",
        "a symmetric category's pairs enter in both orders (intraclass); a directed \
         category's once, in its roles' order",
    ),
    (
        "estimators",
        "continuous: pearson; binary: tetrachoric (primary, ADR 0006's two-step fit) and phi",
    ),
    (
        "quantization",
        "continuous values centred on their midrange and quantized to 43 bits of their \
         largest magnitude; constant_value means constant at that step",
    ),
    ("se", "none (ADR 0009)"),
];

impl Estimate {
    fn to_value(self, primary: bool) -> Value {
        let (value, reason) = match self.value {
            Ok(v) => (Value::Float(v), Value::Null),
            Err(r) => (Value::Null, r.name().into()),
        };
        Value::Map(vec![
            ("estimator", self.estimator.name().into()),
            ("primary", primary.into()),
            ("value", value),
            ("reason", reason),
            ("boundary", self.boundary.into()),
        ])
    }
}

impl Cell {
    pub fn to_value(&self) -> Value {
        let codes = |cats: &[Category]| Value::List(cats.iter().map(|c| c.code().into()).collect());
        let (concordance, table) = match self.counts {
            Counts::None => (Value::Null, Value::Null),
            Counts::Concordance(c) => (Value::Counts(c.to_vec()), Value::Null),
            Counts::Table(t) => (
                Value::Null,
                Value::List(t.iter().map(|row| Value::Counts(row.to_vec())).collect()),
            ),
        };
        Value::Map(vec![
            ("category", Value::Text(self.category.clone())),
            ("members", codes(&self.members)),
            ("symmetric", self.symmetric.into()),
            (
                "roles",
                self.roles
                    .map_or(Value::Null, |[a, b]| Value::List(vec![a.into(), b.into()])),
            ),
            ("n", self.n.into()),
            ("n_dropped", self.n_dropped.into()),
            ("concordance", concordance),
            ("table", table),
            (
                "crude",
                Value::List(
                    self.estimates
                        .iter()
                        .enumerate()
                        .map(|(i, e)| e.to_value(i == 0))
                        .collect(),
                ),
            ),
        ])
    }
}

impl RelativePairCorrelation {
    /// The result as a host-neutral tree; its keys are the hosts' contract.
    pub fn to_value(&self) -> Value {
        let names = self
            .cells
            .iter()
            .map(|c| Value::Text(c.category.clone()))
            .collect();
        let exponent = self.quantization_exponent;
        Value::Map(vec![
            (
                "cells",
                Value::List(self.cells.iter().map(Cell::to_value).collect()),
            ),
            (
                "settings",
                Value::Map(vec![
                    ("categories", Value::List(names)),
                    ("trait_kind", self.kind.name().into()),
                    ("memory_budget_bytes", self.memory_budget_bytes.into()),
                    ("threads", (self.threads as u64).into()),
                    (
                        "quantization_exponent",
                        exponent.map_or(Value::Null, Value::Int),
                    ),
                    ("quantization_step", exponent.map(step_of).into()),
                ]),
            ),
            (
                "method",
                Value::Map(METHOD.iter().map(|&(k, v)| (k, v.into())).collect()),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::PedigreeInput;

    const NA: f64 = f64::NAN;

    /// Grandparents 0 (F), 1 (M); their daughter 2 with founders 3 and 8;
    /// full sibs 4-6 of (2, 3); 7, maternal half-sib of 4-6, of (2, 8).
    fn family() -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let ids: Vec<i64> = (0..9).collect();
        let mother = vec![-1, -1, 0, -1, 2, 2, 2, 2, -1];
        let father = vec![-1, -1, 1, -1, 3, 3, 3, 8, -1];
        (ids, mother, father)
    }

    fn run(
        values: &[f64],
        kind: TraitKind,
        categories: Option<&[Entry]>,
    ) -> Result<RelativePairCorrelation, Error> {
        let (ids, mother, father) = family();
        run_on((&ids, &mother, &father), values, kind, categories)
    }

    fn run_on(
        ped: (&[i64], &[i64], &[i64]),
        values: &[f64],
        kind: TraitKind,
        categories: Option<&[Entry]>,
    ) -> Result<RelativePairCorrelation, Error> {
        let input = PedigreeInput {
            ids: ped.0,
            mother: ped.1,
            father: ped.2,
            twin: None,
            sex: None,
        };
        let trait_ = Trait {
            values,
            kind,
            n_levels: None,
        };
        let categories = categories.map(numbered);
        correlation(
            PedigreeArg::Columns(input),
            trait_,
            categories.as_deref(),
            DEFAULT_MEMORY_BUDGET_BYTES,
        )
    }

    fn cell<'a>(res: &'a RelativePairCorrelation, name: &str) -> &'a Cell {
        res.cells.iter().find(|c| c.category == name).unwrap()
    }

    /// A code entry; [`numbered`] gives it its position.
    fn code(c: &str) -> Entry {
        Entry::Category {
            code: c.to_string(),
            position: 0,
        }
    }

    /// A group entry; [`numbered`] gives it its position.
    fn group(name: &str, members: &[&str]) -> Entry {
        Entry::Group {
            name: name.to_string(),
            members: members.iter().map(|m| m.to_string()).collect(),
            position: 0,
        }
    }

    /// `entries` at positions 0, 1, ..., as a host passes a list.
    fn numbered(entries: &[Entry]) -> Vec<Entry> {
        entries
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, e)| match e {
                Entry::Category { code, .. } => Entry::Category { code, position: i },
                Entry::Group { name, members, .. } => Entry::Group {
                    name,
                    members,
                    position: i,
                },
            })
            .collect()
    }

    /// Pearson's r of two columns, two-pass.
    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
        for (a, b) in x.iter().zip(y) {
            sxx += (a - mx) * (a - mx);
            syy += (b - my) * (b - my);
            sxy += (a - mx) * (b - my);
        }
        sxy / (sxx * syy).sqrt()
    }

    #[test]
    fn default_cells_count_each_category_in_registry_order() {
        let v = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let res = run(&v, TraitKind::Binary, None).unwrap();
        let names: Vec<&str> = res.cells.iter().map(|c| c.category.as_str()).collect();
        assert_eq!(
            names,
            ["MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "Av", "GGP", "HAv", "GAv", "1C"]
        );
        let n = |name| cell(&res, name).n;
        assert_eq!(
            [
                n("MZ"),
                n("MO"),
                n("FO"),
                n("FS"),
                n("MHS"),
                n("PHS"),
                n("GP"),
                n("Av")
            ],
            [0, 5, 5, 3, 3, 0, 8, 0]
        );
        // Sibs 4, 5, 6 are 1, 1, 0: one concordant affected pair, two discordant.
        let fs = cell(&res, "FS");
        assert_eq!(fs.counts, Counts::Concordance([0, 2, 1]));
        assert!(fs.symmetric && fs.roles.is_none());
        // MO rows are offspring: (2, 0) is 1, 0; (6, 2) is 0, 1; (4, 5, 7; 2)
        // are 1, 1.
        let mo = cell(&res, "MO");
        assert_eq!(mo.roles, Some(["offspring", "mother"]));
        assert_eq!(mo.counts, Counts::Table([[0, 1], [1, 3]]));
        // An empty category still has its cell.
        let mz = cell(&res, "MZ");
        assert_eq!((mz.n, mz.n_dropped), (0, 0));
        assert_eq!(mz.estimates[0].value, Err(Reason::NoCompletePairs));
        assert_eq!(mz.estimates[1].value, Err(Reason::NoCompletePairs));
        assert_eq!(res.quantization_exponent, None);
    }

    #[test]
    fn missing_values_drop_pairs_and_count_them() {
        let v = [0.0, 1.0, 1.0, 0.0, NA, 1.0, 0.0, 1.0, 0.0];
        let res = run(&v, TraitKind::Binary, None).unwrap();
        let fs = cell(&res, "FS");
        assert_eq!((fs.n, fs.n_dropped), (1, 2));
        assert_eq!(fs.counts, Counts::Concordance([0, 1, 0]));
        let mo = cell(&res, "MO");
        assert_eq!((mo.n, mo.n_dropped), (4, 1));
        let gp = cell(&res, "GP");
        assert_eq!((gp.n, gp.n_dropped), (6, 2));
    }

    #[test]
    fn symmetric_pearson_is_the_double_entry_pearson() {
        let v = [0.3, -1.2, 0.8, 2.0, 1.5, -0.4, 0.9, 0.1, 1.1];
        let res = run(&v, TraitKind::Continuous, None).unwrap();
        let (mut x, mut y) = (vec![], vec![]);
        for (a, b) in [(4, 5), (4, 6), (5, 6)] {
            x.extend([v[a], v[b]]);
            y.extend([v[b], v[a]]);
        }
        let fs = cell(&res, "FS");
        assert_eq!((fs.n, fs.n_dropped, fs.counts), (3, 0, Counts::None));
        let r = fs.estimates[0].value.unwrap();
        assert!((r - pearson(&x, &y)).abs() < 1e-12, "{r}");
        // Directed: once, offspring first.
        let x: Vec<f64> = [2, 4, 5, 6, 7].iter().map(|&i| v[i]).collect();
        let y = [v[0], v[2], v[2], v[2], v[2]];
        let r = cell(&res, "MO").estimates[0].value.unwrap();
        assert!((r - pearson(&x, &y)).abs() < 1e-12, "{r}");
        assert!(res.quantization_exponent.is_some());
    }

    #[test]
    fn a_parent_offspring_group_pools_mother_and_father_tables() {
        let v = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let entries = [group("PO", &["MO", "FO"]), code("MO"), code("FO")];
        let err = run(&v, TraitKind::Binary, Some(&entries)).unwrap_err();
        assert_eq!(err.code(), "duplicate_category");
        let all = run(&v, TraitKind::Binary, None).unwrap();
        let po = run(&v, TraitKind::Binary, Some(&entries[..1])).unwrap();
        let po = &po.cells[0];
        assert_eq!(po.roles, Some(["offspring", "parent"]));
        assert_eq!(po.members, [Category::MO, Category::FO]);
        let table = |c: &Cell| match c.counts {
            Counts::Table(t) => t,
            _ => panic!("a directed cell has a table"),
        };
        let (mo, fo) = (table(cell(&all, "MO")), table(cell(&all, "FO")));
        let pooled: Vec<u64> = mo
            .iter()
            .flatten()
            .zip(fo.iter().flatten())
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(
            table(po).iter().flatten().copied().collect::<Vec<_>>(),
            pooled
        );
        assert_eq!(po.n, 10);
        let point = correlation::point(table(po)).unwrap();
        assert_eq!(po.estimates[0].value, Ok(point.value));
    }

    #[test]
    fn groups_keep_roles_and_reject_mixtures() {
        let v = [0.3, -1.2, 0.8, 2.0, 1.5, -0.4, 0.9, 0.1, 1.1];
        let entries = [
            group("ancestors", &["GP", "GGP"]),
            group("sibs", &["FS", "MHS"]),
        ];
        let res = run(&v, TraitKind::Continuous, Some(&entries)).unwrap();
        assert_eq!(res.cells[0].roles, Some(["descendant", "ancestor"]));
        assert_eq!(res.cells[0].n, 8);
        assert!(res.cells[1].symmetric);
        assert_eq!(res.cells[1].n, 6);
        let mixed = [group("bad", &["MO", "GP"])];
        let err = run(&v, TraitKind::Continuous, Some(&mixed)).unwrap_err();
        assert_eq!(err.code(), "mixed_category_group");
        let mixed = [group("bad", &["FS", "MO"])];
        let err = run(&v, TraitKind::Continuous, Some(&mixed)).unwrap_err();
        assert_eq!(err.code(), "mixed_category_group");
    }

    #[test]
    fn category_requests_are_checked() {
        let v = [0.0; 9];
        let cases: [(&[Entry], &str); 9] = [
            (&[code("XX")], "unknown_category"),
            (&[group("G", &["FS", "nope"])], "unknown_category"),
            (&[code("FS"), code("FS")], "duplicate_category"),
            (&[group("G", &["FS", "FS"])], "duplicate_category"),
            (
                &[group("G", &["FS"]), group("G", &["MHS"])],
                "duplicate_entry_name",
            ),
            (&[code("FS"), group("FS", &["MHS"])], "reserved_group_name"),
            (&[group("MO", &["FO"])], "reserved_group_name"),
            (&[group("G", &[])], "empty_category_group"),
            (&[group("G", &["MO", "GP"])], "mixed_category_group"),
        ];
        for (entries, expected) in cases {
            let err = run(&v, TraitKind::Binary, Some(entries)).unwrap_err();
            assert_eq!(err.code(), expected, "{entries:?}");
        }
        let res = run(&v, TraitKind::Binary, Some(&[])).unwrap();
        assert!(res.cells.is_empty());
        let twice = run(&v, TraitKind::Binary, Some(&[group("G", &["FS", "FS"])])).unwrap_err();
        assert_eq!(
            twice.to_string(),
            "category FS is listed twice in group \"G\" (categories[0])"
        );
    }

    #[test]
    fn errors_report_the_callers_positions() {
        // Two groups a host passed as one mapping share position 1.
        let entries = [
            code("FS"),
            Entry::Group {
                name: "A".to_string(),
                members: vec!["MO".to_string()],
                position: 1,
            },
            Entry::Group {
                name: "B".to_string(),
                members: vec!["XX".to_string()],
                position: 1,
            },
        ];
        let err = run_raw(&entries).unwrap_err();
        assert_eq!(
            err,
            Error::UnknownCategory {
                category: "XX".to_string(),
                position: 1
            }
        );
        let mut entries = entries;
        entries[2] = Entry::Group {
            name: "B".to_string(),
            members: vec!["MO".to_string()],
            position: 1,
        };
        assert_eq!(
            run_raw(&entries).unwrap_err().to_string(),
            "category MO is in both \"A\" (categories[1]) and \"B\" (categories[1])"
        );
    }

    /// [`correlation`] on the family with `entries` as given, positions kept.
    fn run_raw(entries: &[Entry]) -> Result<RelativePairCorrelation, Error> {
        let (ids, mother, father) = family();
        let input = PedigreeInput {
            ids: &ids,
            mother: &mother,
            father: &father,
            twin: None,
            sex: None,
        };
        let values = [0.0; 9];
        let trait_ = Trait {
            values: &values,
            kind: TraitKind::Binary,
            n_levels: None,
        };
        correlation(
            PedigreeArg::Columns(input),
            trait_,
            Some(entries),
            DEFAULT_MEMORY_BUDGET_BYTES,
        )
    }

    #[test]
    fn the_trait_and_budget_are_checked_in_order() {
        let (ids, mother, father) = family();
        let input = PedigreeInput {
            ids: &ids,
            mother: &mother,
            father: &father,
            twin: None,
            sex: None,
        };
        let call = |values: &[f64], kind, categories: Option<&[Entry]>, budget| {
            correlation(
                PedigreeArg::Columns(input),
                Trait {
                    values,
                    kind,
                    n_levels: None,
                },
                categories,
                budget,
            )
            .unwrap_err()
            .code()
        };
        let bad = [code("XX")];
        assert_eq!(
            call(&[0.0; 8], TraitKind::Ordinal, Some(&bad), 0),
            "trait_length_mismatch"
        );
        assert_eq!(
            call(&[0.0; 9], TraitKind::Categorical, Some(&bad), 0),
            "unsupported_trait_kind"
        );
        assert_eq!(
            call(&[0.0; 9], TraitKind::Ordinal, Some(&bad), 0),
            "unsupported_trait_kind"
        );
        let mut v = [0.0; 9];
        v[3] = 2.0;
        assert_eq!(
            call(&v, TraitKind::Binary, Some(&bad), 0),
            "invalid_trait_value"
        );
        v[3] = f64::INFINITY;
        assert_eq!(
            call(&v, TraitKind::Continuous, Some(&bad), 0),
            "invalid_trait_value"
        );
        assert_eq!(
            call(&[0.0; 9], TraitKind::Binary, Some(&bad), 0),
            "unknown_category"
        );
        assert_eq!(
            call(&[0.0; 9], TraitKind::Binary, None, 0),
            "parameter_out_of_range"
        );
        assert_eq!(
            call(&[0.0; 9], TraitKind::Binary, None, 16),
            "memory_budget_exceeded"
        );
    }

    #[test]
    fn centring_keeps_close_values_near_a_large_offset_distinct() {
        // Four full sibs (rows 2-5) of founders 0, 1.
        let ids: Vec<i64> = (0..6).collect();
        let mother = [-1, -1, 0, 0, 0, 0];
        let father = [-1, -1, 1, 1, 1, 1];
        let v = [NA, NA, 1e12, 1e12 + 0.01, 1e12 + 0.02, 1e12 + 0.03];
        let res = run_on((&ids, &mother, &father), &v, TraitKind::Continuous, None).unwrap();
        let fs = cell(&res, "FS");
        assert_eq!((fs.n, fs.n_dropped), (6, 0));
        let r = fs.estimates[0].value.unwrap();
        assert!((r + 1.0 / 3.0).abs() < 1e-9, "{r}");
        let mo = cell(&res, "MO");
        assert_eq!((mo.n, mo.n_dropped), (0, 4));
        assert_eq!(mo.estimates[0].value, Err(Reason::NoCompletePairs));
    }

    #[test]
    fn a_constant_or_missing_trait_gives_reasons() {
        let res = run(&[2.5; 9], TraitKind::Continuous, None).unwrap();
        assert_eq!(
            cell(&res, "FS").estimates[0].value,
            Err(Reason::ConstantValue)
        );
        let res = run(&[NA; 9], TraitKind::Continuous, None).unwrap();
        let fs = cell(&res, "FS");
        assert_eq!((fs.n, fs.n_dropped), (0, 3));
        assert_eq!(fs.estimates[0].value, Err(Reason::NoCompletePairs));
        let res = run(&[1.0; 9], TraitKind::Binary, None).unwrap();
        let fs = cell(&res, "FS");
        assert_eq!(fs.estimates[0].value, Err(Reason::ConstantMargin));
        assert_eq!(fs.estimates[1].value, Err(Reason::ConstantMargin));
    }

    #[test]
    fn an_outlier_far_past_the_spread_is_insufficient_precision() {
        let mut v: Vec<f64> = (0..9).map(|i| f64::from(i) * 0.1).collect();
        v[0] = 1e12;
        let err = run(&v, TraitKind::Continuous, None).unwrap_err();
        assert_eq!(err.code(), "insufficient_precision");
    }

    #[test]
    fn a_symmetric_table_that_is_not_mirrored_raises() {
        // Binary labels 0, 1, missing: N[0][1] = 3 but N[1][0] = 2.
        let counts: [i128; 9] = [4, 3, 0, 2, 6, 0, 0, 0, 0];
        let table = MomentsTable::from_i128(vec![3, 3], 0, vec![], vec![], &counts).unwrap();
        let err = super::cell(single(Category::FS), &table, TraitKind::Binary).unwrap_err();
        assert!(
            matches!(err, Error::Pedigree(PgError::InvalidMomentsTable { .. })),
            "{err:?}"
        );
        // An odd diagonal is not a double entry either.
        let counts: [i128; 9] = [5, 2, 0, 2, 6, 0, 0, 0, 0];
        let table = MomentsTable::from_i128(vec![3, 3], 0, vec![], vec![], &counts).unwrap();
        assert!(super::cell(single(Category::FS), &table, TraitKind::Binary).is_err());
        // A directed category takes any table.
        let table =
            MomentsTable::from_i128(vec![3, 3], 0, vec![], vec![], &[5, 3, 0, 2, 6, 0, 0, 0, 0])
                .unwrap();
        let mo = super::cell(single(Category::MO), &table, TraitKind::Binary).unwrap();
        assert_eq!(mo.counts, Counts::Table([[5, 3], [2, 6]]));
    }

    #[test]
    fn phi_matches_its_formula() {
        let e = phi([[30, 10], [5, 15]]);
        let expected = (30.0 * 15.0 - 10.0 * 5.0) / (40.0f64 * 20.0 * 35.0 * 25.0).sqrt();
        assert!((e.value.unwrap() - expected).abs() < 1e-15);
        assert_eq!(phi([[0, 0], [0, 0]]).value, Err(Reason::NoCompletePairs));
        assert_eq!(phi([[3, 0], [4, 0]]).value, Err(Reason::ConstantMargin));
    }

    #[test]
    fn quantile_and_sd_helpers() {
        assert_eq!(iqr(&mut [5.0, 1.0, 4.0, 2.0, 3.0]), 2.0);
        // Interpolated: q1 = 1.75, q3 = 3.25.
        assert_eq!(iqr(&mut [4.0, 3.0, 2.0, 1.0]), 1.5);
        assert_eq!(iqr(&mut [7.0]), 0.0);
        assert!((sd(&[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-15);
        assert!(sd(&[-1e300, 1e300]).is_finite());
        assert_eq!(step_of(0), 1.0);
        assert_eq!(step_of(2000), 0.0);
    }
}
