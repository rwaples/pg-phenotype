//! The Relative-Pair Correlation: one or two traits' correlation between
//! the members of each Relationship Category's (or Category Group's)
//! Relative Pairs (ADR 0009), and with two traits the Within-Person
//! Cross-Trait Correlation.
//!
//! The pairs come from one pass of pedigree-graph-core's engine, which
//! reduces them to exact per-cell sums and never stores them: its
//! `relationship_moments` table and, when the SE is asked for, beside it
//! the per-cluster sums the cluster sandwich reads (ADR 0013, `clusters`).
//! A symmetric category's pairs enter in both orders, so its same-trait
//! Pearson is intraclass and its tetrachoric fits one threshold; a directed
//! category's pairs enter once, in its roles' order.
//!
//! Each member's engine label is its stratum code, then each trait's label
//! (trait 0 most significant), so the one traversal gives a table by (first
//! stratum, first labels, second stratum, second labels), and every trait
//! combination's pooled cell, stratum cells and stratified fit read it.

use crate::assortative::{check_strata, Strata};
use crate::correlation;
use crate::error::Error;
use crate::input::{admit, Admits, Trait, TraitKind};
use crate::pedigree::PedigreeArg;
use crate::polychoric::table::Table;
use crate::polychoric::{Point, Reason as FitReason};
use crate::value::Value;
use clusters::{Clusters, Meat, Pass};
use influence::{Influence, StratifiedBlock, Weights};
use mrca::Ancestry;
use pedigree_graph_core::error::Error as PgError;
use pedigree_graph_core::relationships::{
    host_bytes, quantize_column, relationship_moments, Category, CategorySet, MaxDegree,
    MomentsInput, MomentsTable, Operand, Product, Progress, Receiver, Side, Statistic, Symmetric,
    N_CATEGORIES,
};
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Reverse;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

mod clusters;
mod influence;
mod mrca;

/// The trait's field name with one trait, as errors report it.
const FIELD: &str = "trait";

/// Each trait's field name with two traits.
const FIELDS: [&str; 2] = ["traits[0]", "traits[1]"];

/// Continuous or binary; an all-missing or constant trait is not an error
/// here: its cells carry a reason.
const ADMITS: Admits = Admits::new(&[TraitKind::Continuous, TraitKind::Binary], false);

/// The deepest degree the default request covers.
pub const DEFAULT_DEGREE: u8 = 3;

/// The default bound on the engine's accumulators: 1 GiB, as in
/// pedigree-graph's Python.
pub const DEFAULT_MEMORY_BUDGET_BYTES: i64 = 1 << 30;

/// The coarsest quantization step a continuous trait takes, as a fraction of
/// its spread.
const PRECISION: f64 = 1.0 / (1u64 << 20) as f64;

/// The fewest `f64::EPSILON`s of its scale a meat must pass not to be
/// zero ([`degenerate`]).
const DEGENERATE_FLOOR: f64 = 1024.0;

/// A meat of an estimate over `entries` entries within this fraction of
/// its scale is zero (`degenerate_meat`): `max(2^10, 4 √entries) ε`.  The
/// scale is the meat's three sums of squares, added, had each entry the
/// influence `1/n`.  A meat that is zero in exact arithmetic (a cell's
/// pairs in one MRCA cluster, or `r = ±1`) comes out as its floats'
/// rounding.  Over `n` entries the meat is at most `16n` additions: each
/// of at most `2n` rows into its MRCA, member and (member, MRCA) sums,
/// then at most `5n` of those sums' squares.  Each rounds by about `ε` of
/// the scale and their errors add as a random walk, to `√(16n) ε = 4√n ε`.
/// Below `2^16` entries the floor `2^10 ε` holds, 2e4 times the rounding
/// seen on the test pedigrees (`0.05 ε`), whose smallest true meat is
/// `1e-5` of its scale: a meat that small has no digit left to report.
fn degenerate(entries: u64) -> f64 {
    (4.0 * (entries as f64).sqrt()).max(DEGENERATE_FLOOR) * f64::EPSILON
}

/// The category name of the Within-Person Cross-Trait Correlation's cell.
const WITHIN_PERSON: &str = "within_person";

/// One entry of a category request: a Relationship Category code, or a
/// named Category Group.  Each entry gives one cell per trait combination.
/// `position` is its index in the caller's request, as errors report it;
/// groups a host gives together (one Python mapping) share one.
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
    /// The cluster sandwich SE (ADR 0013), or why there is none; `None`
    /// unless the call asked for it.
    pub se: Option<Result<f64, SeReason>>,
}

/// Why an estimate has no SE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeReason {
    /// The estimate is undefined; its `value` says why.
    NoEstimate,
    /// The tetrachoric fit stopped at a bound.
    Boundary,
    /// The two-step Hessian is not negative definite.
    SandwichUndefined,
    /// The meat is zero to within rounding ([`degenerate`]), as when a
    /// cell's pairs share one MRCA or its estimate is ±1.
    DegenerateMeat,
    /// The inclusion-exclusion meat came out negative.
    NegativeMeat,
    /// The Within-Person Cross-Trait Correlation has no cluster design.
    WithinPerson,
}

impl SeReason {
    pub fn name(self) -> &'static str {
        match self {
            SeReason::NoEstimate => "no_estimate",
            SeReason::Boundary => "boundary",
            SeReason::SandwichUndefined => "sandwich_undefined",
            SeReason::DegenerateMeat => "degenerate_meat",
            SeReason::NegativeMeat => "negative_meat",
            SeReason::WithinPerson => "within_person",
        }
    }

    /// `√meat`, or why there is none: a meat within rounding of zero is
    /// degenerate, a clearly negative one negative.
    fn of_meat(meat: Meat) -> Result<f64, SeReason> {
        let Meat {
            value,
            scale,
            entries,
        } = meat;
        if value.is_nan() || scale.is_nan() {
            Err(SeReason::SandwichUndefined)
        } else if value.abs() <= degenerate(entries) * scale {
            Err(SeReason::DegenerateMeat)
        } else if value < 0.0 {
            Err(SeReason::NegativeMeat)
        } else {
            Ok(value.sqrt())
        }
    }
}

/// A cell's binary pair counts, as the caller sees them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counts {
    /// A continuous trait: no counts.
    None,
    /// A symmetric category's same-trait Concordance: both 0, discordant,
    /// both 1.
    Concordance([u64; 3]),
    /// Every other cell's 2 x 2 table: rows by trait `first_trait` of one
    /// member (a directed category's first role), columns by trait
    /// `second_trait` of the other; a symmetric cell's both orientations
    /// summed.
    Table([[u64; 2]; 2]),
}

/// One entry's correlation of one trait combination, or the Within-Person
/// Cross-Trait Correlation.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell<E = Estimate> {
    /// The category's code, the group's name, or `within_person`.
    pub category: String,
    pub members: Vec<Category>,
    pub symmetric: bool,
    /// A directed category's roles, first then second; a group's with
    /// mother and father read as parent.
    pub roles: Option<[&'static str; 2]>,
    /// The trait read in each pair's first member (role), 0-based.
    pub first_trait: usize,
    /// The trait read in each pair's second member (role).
    pub second_trait: usize,
    /// Unordered pairs with a complete orientation: trait `first_trait`
    /// of one member and `second_trait` of the other both observed.
    pub n: u64,
    /// The entry's other unordered pairs.
    pub n_dropped: u64,
    /// A symmetric cross-trait cell's complete orientations, between `n`
    /// and `2n`; `None` elsewhere.
    pub n_entries: Option<u64>,
    pub counts: Counts,
    /// The primary estimate first.
    pub estimates: Vec<E>,
    /// Each known stratum label of any row, ascending; `None` without strata.
    pub strata: Option<Vec<StratumCell<E>>>,
    /// Complete symmetric pairs whose members are in two known strata: in
    /// no stratum's cell.
    pub n_cross_stratum: u64,
    /// Complete pairs in no stratum's cell for an unknown stratum: a
    /// directed pair's first member's, either member's of a symmetric pair.
    pub n_unknown_stratum: u64,
    /// `None` without strata.
    pub stratified: Option<Stratified<E>>,
}

/// One known stratum's pairs of a cell: a directed category's whose first
/// member is in it (the second's stratum may be any, or unknown), a
/// symmetric category's whose members are both in it.  The fields are a
/// [`Cell`]'s pooled ones.
#[derive(Clone, Debug, PartialEq)]
pub struct StratumCell<E = Estimate> {
    pub label: i64,
    pub n: u64,
    pub n_dropped: u64,
    pub n_entries: Option<u64>,
    pub counts: Counts,
    pub estimates: Vec<E>,
}

/// The primary estimator fitted over every block of known strata at once,
/// rows by the first member's stratum and columns by the second's (both
/// orders for a symmetric category): a binary side gets a threshold per
/// stratum, a continuous side is standardised within each stratum, and one
/// correlation is fitted.  A stratum whose side is constant over the
/// blocks kept carries no information on it, so its blocks are left out,
/// until no stratum's side is constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stratified<E = Estimate> {
    /// Unordered pairs with a complete orientation fitted.
    pub n: u64,
    /// A symmetric cross-trait cell's complete orientations fitted, those
    /// in kept blocks; `None` elsewhere.
    pub n_entries: Option<u64>,
    /// Unordered pairs with a complete orientation and both strata known
    /// that a constant side's stratum left out.
    pub n_degenerate_stratum: u64,
    pub estimate: E,
}

/// An estimate as its fit gave it and, with the SE, the index of its
/// influence in its request's [`Influences`].  [`Influences::made`] makes
/// both in one call, so an estimate cannot meet another's SE.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fit {
    estimate: Estimate,
    influence: Option<usize>,
}

impl Fit {
    /// The estimate with its SE, `ses` being its request's in
    /// [`Influences`] order; none without the SE.
    fn finish(self, ses: &[Result<f64, SeReason>]) -> Estimate {
        Estimate {
            se: self.influence.map(|e| ses[e]),
            ..self.estimate
        }
    }
}

/// A request's influences in the order its fits made them; `None` without
/// the SE.
struct Influences(Option<Vec<Result<Influence, SeReason>>>);

impl Influences {
    fn on(&self) -> bool {
        self.0.is_some()
    }

    /// `estimate`'s fit; with the SE its influence too, which `influence`
    /// gives.
    fn made(
        &mut self,
        estimate: Estimate,
        influence: impl FnOnce() -> Result<Influence, SeReason>,
    ) -> Fit {
        let influence = self.0.as_mut().map(|list| {
            list.push(influence());
            list.len() - 1
        });
        Fit {
            estimate,
            influence,
        }
    }

    /// `estimate`'s fit, which has no influence.
    fn none(&mut self, estimate: Estimate) -> Fit {
        self.made(estimate, || Err(SeReason::NoEstimate))
    }
}

/// The result of [`correlation`].
#[derive(Clone, Debug, PartialEq)]
pub struct RelativePairCorrelation {
    /// Each entry's cells in request order: with two traits, a symmetric
    /// entry's (0, 0), (1, 1), (0, 1) and a directed entry's (0, 0),
    /// (0, 1), (1, 0), (1, 1).
    pub cells: Vec<Cell>,
    /// With two traits, the Within-Person Cross-Trait Correlation: each
    /// row's trait 0 against its trait 1, a row's stratum its own.
    pub within_person: Option<Cell>,
    pub kind: TraitKind,
    /// Each continuous trait's quantization exponent `e`: its centred
    /// values are integers in steps of `2^-e`.
    pub quantization_exponents: Option<Vec<i64>>,
    pub memory_budget_bytes: u64,
    pub threads: usize,
    /// Whether the call asked for the SE.
    pub se: bool,
}

/// One entry, resolved.
#[derive(Clone, Debug, PartialEq)]
struct Request {
    name: String,
    members: Vec<Category>,
    symmetric: bool,
    roles: Option<[&'static str; 2]>,
}

/// The Relative-Pair Correlation of `traits` (one, or two of one kind)
/// over `pedigree`, the cells of each entry of `categories` (default: each
/// category up to degree 3), with two traits the Within-Person Cross-Trait
/// Correlation, and with `strata` each cell's stratum cells and stratified
/// fit.  With `se`, every estimate's cluster sandwich SE (ADR 0013); without
/// it the call is `relationship_moments` and the fits alone.
///
/// # Errors
///
/// Any pedigree-graph-core validation error of columns, then
/// `trait_count`, then each trait's `trait_length_mismatch`,
/// `unsupported_trait_kind` (ordinal, categorical), `invalid_trait_value`,
/// then `unsupported_trait_kind` for two traits of different kinds; then
/// `stratum_length_mismatch`; then `unknown_category`,
/// `duplicate_category`, `duplicate_entry_name`, `empty_category_group`,
/// `mixed_category_group`; then `parameter_out_of_range` for a budget below
/// one byte; `insufficient_precision`; with two traits,
/// `memory_budget_exceeded` for the Within-Person accumulator; the
/// engine's errors (`memory_budget_exceeded`, ...); and with `se`,
/// `memory_budget_exceeded` (operation `relative_pairs_se`) or
/// `allocation_failed` for the per-cluster sums.
pub fn correlation(
    pedigree: PedigreeArg<'_>,
    traits: &[Trait<'_>],
    strata: Option<Strata<'_>>,
    categories: Option<&[Entry]>,
    memory_budget_bytes: i64,
    se: bool,
) -> Result<RelativePairCorrelation, Error> {
    let pedigree = pedigree.resolve()?;
    let n_rows = pedigree.len();
    check_traits(traits, n_rows)?;
    if let Some(s) = strata {
        check_strata(s, n_rows)?;
    }
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
    let (codes, labels) = strata.map(Strata::codes).unzip();
    let mut members = Members::of(traits)?;
    if let (Some(codes), Some(labels)) = (&codes, &labels) {
        members = members.stratified(codes, labels.len() + 1);
    }
    let layout = members.layout;
    let within_person = if layout.n_traits == 2 {
        let request = Request {
            name: WITHIN_PERSON.to_string(),
            members: Vec::new(),
            symmetric: false,
            roles: None,
        };
        let cell = Blocks::within_person(&members, budget)?.cell(
            request,
            labels.as_deref(),
            &mut Influences(None),
        )?;
        Some(cell.map(|fit| Estimate {
            se: se.then_some(Err(SeReason::WithinPerson)),
            ..fit.estimate
        }))
    } else {
        None
    };
    let mut result = RelativePairCorrelation {
        cells: Vec::new(),
        within_person,
        kind: layout.kind,
        quantization_exponents: (layout.kind == TraitKind::Continuous)
            .then(|| members.exponents.clone()),
        memory_budget_bytes: budget,
        threads,
        se,
    };
    let union: CategorySet = requests.iter().flat_map(|r| r.members.clone()).collect();
    let Some(top) = union.top_degree() else {
        return Ok(result);
    };
    let ped = pedigree.relationships()?;
    let max_degree = MaxDegree::try_new(top)?;
    let lanes = NonZeroUsize::new(threads).unwrap_or(NonZeroUsize::MIN);
    let ancestry = se.then(|| Ancestry::of(&pedigree, budget)).transpose()?;
    let order: Vec<Category> = union.iter().collect();
    let (table, clusters) = match &ancestry {
        Some(ancestry) => {
            let (reserve, batch) = influence::reserve(
                layout,
                &requests,
                labels.as_ref().map(Vec::len),
                n_rows,
                budget,
                ancestry.bytes(),
            );
            let pass = Pass::run(
                &ped, max_degree, union, &requests, &members, ancestry, lanes, budget, reserve,
            )?;
            (pass.table, Some((pass.clusters, batch)))
        }
        None => {
            let input = members.input();
            let moments = relationship_moments(
                &ped,
                max_degree,
                union,
                Receiver::Graph,
                &input,
                Symmetric::Both,
                lanes,
                budget,
                &Progress::default(),
            )?;
            let table = MomentsTable::from_bytes(
                members.shape(order.len()),
                input.n_columns,
                input.products.to_vec(),
                members.exponents.clone(),
                moments.width,
                Cow::Owned(moments.table),
            )?;
            (table, None)
        }
    };
    let (mut cells, mut influences) = (Vec::new(), Vec::new());
    for (r, request) in requests.iter().enumerate() {
        let positions: Vec<usize> = request
            .members
            .iter()
            .filter_map(|m| order.iter().position(|c| c == m))
            .collect();
        let joint = table.select(0, &positions)?.sum(0)?;
        let counts = joint.counts();
        if request.symmetric {
            check_mirrored(&request.name, &counts, layout.n_codes * layout.n_joint())?;
        }
        let mut fitted = Influences(se.then(Vec::new));
        for view in View::of(layout.n_traits, request.symmetric) {
            let blocks = Blocks::pairs(&joint, &counts, layout, view)?;
            cells.push((
                r,
                blocks.cell(request.clone(), labels.as_deref(), &mut fitted)?,
            ));
        }
        influences.push(fitted.0.unwrap_or_default());
    }
    let ses = match &clusters {
        Some((clusters, batch)) => {
            let held = (table.bytes().len() as u64)
                .saturating_add(ancestry.as_ref().map_or(0, Ancestry::bytes));
            let sweeps = Sweeps {
                clusters,
                members: &members,
                influences: &influences,
                batch: *batch,
            };
            sweeps.standard_errors(budget.saturating_sub(held), threads)
        }
        None => Vec::new(),
    };
    drop(influences);
    result.cells = cells
        .into_iter()
        .map(|(r, cell)| cell.map(|fit| fit.finish(ses.get(r).map_or(&[], Vec::as_slice))))
        .collect();
    Ok(result)
}

/// The SE's last step: each request's [`Weights`] and sweeps over the
/// store ([`Clusters::meats`]).
struct Sweeps<'a> {
    clusters: &'a Clusters<'a>,
    members: &'a Members,
    /// Per request, its fits' influences in the order they made them.
    influences: &'a [Vec<Result<Influence, SeReason>>],
    /// The estimates a sweep takes as the pass planned it.
    batch: usize,
}

impl Sweeps<'_> {
    /// Each request's SEs, in its influences' order, sweeping as many
    /// requests at once as [`influence::sweeps`] fits in `free` (the
    /// budget less the table and the ancestry), at most `threads`.  One
    /// thread sweeps a request whole, its floats in key order, so no SE
    /// depends on the threads; requests with the most rows go first.
    fn standard_errors(&self, free: u64, threads: usize) -> Vec<Vec<Result<f64, SeReason>>> {
        let layout = self.members.layout;
        let held: u64 = self
            .influences
            .iter()
            .flatten()
            .map(Influence::bytes)
            .sum::<u64>()
            .saturating_add(self.clusters.bytes());
        let shapes: Vec<(u64, usize)> = self
            .influences
            .iter()
            .map(|list| (Weights::bytes(layout, list), list.len()))
            .collect();
        let (width, batch) = influence::sweeps(
            free.saturating_sub(held),
            threads,
            &shapes,
            self.members.labels.len(),
            self.batch,
        );
        let mut order: Vec<usize> = (0..self.influences.len()).collect();
        order.sort_by_key(|&r| Reverse(self.clusters.n_rows(r as u32)));
        let next = AtomicUsize::new(0);
        let mut meats: Vec<(usize, Vec<Meat>)> = (0..width)
            .into_par_iter()
            .flat_map_iter(|_| {
                let mut done = Vec::new();
                while let Some(&r) = order.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let weights =
                        Weights::build(layout, &self.members.exponents, &self.influences[r]);
                    done.push((r, self.clusters.meats(r as u32, &weights, batch)));
                }
                done
            })
            .collect();
        meats.sort_unstable_by_key(|&(r, _)| r);
        meats
            .into_iter()
            .zip(self.influences)
            .map(|((_, meats), list)| {
                list.iter()
                    .zip(meats)
                    .map(|(influence, meat)| match influence {
                        Ok(_) => SeReason::of_meat(meat),
                        Err(reason) => Err(*reason),
                    })
                    .collect()
            })
            .collect()
    }
}

/// The traits' errors in order: their count, each one's Admission, then a
/// second trait of another kind, which no estimator here pairs yet.
fn check_traits(traits: &[Trait<'_>], n_rows: usize) -> Result<(), Error> {
    if traits.is_empty() || traits.len() > 2 {
        return Err(Error::TraitCount {
            actual: traits.len(),
            minimum: 1,
            maximum: 2,
        });
    }
    for (t, &trait_) in traits.iter().enumerate() {
        admit(trait_, field(traits.len(), t), n_rows, ADMITS)?;
    }
    match traits {
        [first, second] if second.kind != first.kind => Err(Error::MixedTraitKinds {
            kinds: [first.kind, second.kind],
        }),
        _ => Ok(()),
    }
}

/// Trait `t`'s field name: `trait` when it is the only one, so one-trait
/// messages read as they always have.
fn field(n_traits: usize, t: usize) -> &'static str {
    if n_traits == 1 {
        FIELD
    } else {
        FIELDS[t]
    }
}

/// Each Relationship Category's code and its meioses `up` from the first
/// member to the MRCA and `down` to the second, as the SE's MRCA walk
/// reads them, in registry order.
pub fn meioses() -> impl Iterator<Item = (&'static str, u8, u8)> {
    Category::ALL.into_iter().map(|c| {
        let (up, down) = mrca::MEIOSES[c.index()];
        (c.code(), up, down)
    })
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
/// named apart from every category code and the Within-Person cell,
/// non-empty and of one kind.
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
                positions: [first.into(), position.into()],
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
            position: position.into(),
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
                if Category::parse(name).is_some() || name == WITHIN_PERSON {
                    return Err(Error::ReservedGroupName {
                        group: name.clone(),
                        position: position.into(),
                    });
                }
                let Some(first) = members.first() else {
                    return Err(Error::EmptyCategoryGroup {
                        group: name.clone(),
                        position: position.into(),
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
                positions: [first.into(), position.into()],
            });
        }
        requests.push((position, request));
    }
    Ok(requests.into_iter().map(|(_, r)| r).collect())
}

/// How the engine labels a member: its stratum code, then each trait's
/// label, trait 0 most significant.  A binary trait's labels are 0, 1 and
/// missing (2); a continuous trait's observed (0) and missing (1).
#[derive(Clone, Copy, Debug)]
struct Layout {
    kind: TraitKind,
    n_traits: usize,
    /// Labels per trait.
    n_labels: usize,
    /// One code without strata; with them, codes `0..k` are the known
    /// labels ascending and `k` is unknown.
    n_codes: usize,
}

impl Layout {
    /// The joint labels of a member's traits.
    fn n_joint(self) -> usize {
        self.n_labels.pow(self.n_traits as u32)
    }

    /// Trait `t`'s label in joint label `joint`.
    fn digit(self, joint: usize, t: usize) -> usize {
        joint / self.n_labels.pow((self.n_traits - 1 - t) as u32) % self.n_labels
    }

    /// Whether joint label `joint` has trait `t` observed.
    fn shows(self, joint: usize, t: usize) -> bool {
        self.digit(joint, t) < observed(self.kind)
    }
}

/// A trait's labels for an observed value: binary 0 and 1, continuous 0.
fn observed(kind: TraitKind) -> usize {
    if kind == TraitKind::Binary {
        2
    } else {
        1
    }
}

/// The traits a cell reads: `first` in its pairs' first member, `second`
/// in the second.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct View {
    first: usize,
    second: usize,
    symmetric: bool,
}

impl View {
    /// An entry's cells in order: with one trait (0, 0); with two, a
    /// symmetric entry's (0, 0), (1, 1), (0, 1) and a directed entry's
    /// (0, 0), (0, 1), (1, 0), (1, 1).
    fn of(n_traits: usize, symmetric: bool) -> Vec<View> {
        let traits: &[(usize, usize)] = match (n_traits, symmetric) {
            (1, _) => &[(0, 0)],
            (_, true) => &[(0, 0), (1, 1), (0, 1)],
            (_, false) => &[(0, 0), (0, 1), (1, 0), (1, 1)],
        };
        traits
            .iter()
            .map(|&(first, second)| View {
                first,
                second,
                symmetric,
            })
            .collect()
    }

    /// A symmetric pair's two orientations read different traits.
    fn cross(self) -> bool {
        self.symmetric && self.first != self.second
    }

    /// The trait `side` reads, which is its column.
    fn column(self, side: Side) -> usize {
        match side {
            Side::First => self.first,
            Side::Second => self.second,
        }
    }

    /// The index in [`PRODUCTS`] of first `first` times second `second`.
    fn product(self) -> usize {
        if self.first == self.second {
            self.first
        } else {
            2 + self.first
        }
    }
}

const fn product(first: usize, second: usize) -> Product {
    Product {
        a: Operand {
            side: Side::First,
            column: first,
        },
        b: Operand {
            side: Side::Second,
            column: second,
        },
    }
}

/// Continuous traits' products, first x second: each trait with itself,
/// then trait 0 with trait 1 and trait 1 with trait 0.  One trait takes
/// the first.
const PRODUCTS: [Product; 4] = [product(0, 0), product(1, 1), product(0, 1), product(1, 0)];

/// Every member's inputs to the engine: the same labels and values on both
/// sides of a pair.
struct Members {
    layout: Layout,
    /// `code * n_joint + joint label`.
    labels: Vec<i32>,
    /// Each continuous trait's values centred on their midrange and
    /// quantized, row-major `[rows, traits]`; empty for binary traits.
    values: Vec<i64>,
    /// Each continuous trait's quantization exponent; empty for binary.
    exponents: Vec<i64>,
}

impl Members {
    fn of(traits: &[Trait<'_>]) -> Result<Members, Error> {
        let (n_traits, kind) = (traits.len(), traits[0].kind);
        let binary = kind == TraitKind::Binary;
        let layout = Layout {
            kind,
            n_traits,
            n_labels: if binary { 3 } else { 2 },
            n_codes: 1,
        };
        let own = |v: f64| match kind {
            TraitKind::Binary if v.is_nan() => 2,
            TraitKind::Binary => v as i32,
            _ => i32::from(v.is_nan()),
        };
        let mut labels = vec![0i32; traits[0].values.len()];
        for t in traits {
            for (label, &v) in labels.iter_mut().zip(t.values) {
                *label = *label * layout.n_labels as i32 + own(v);
            }
        }
        let (mut values, mut exponents) = (Vec::new(), Vec::new());
        if !binary {
            values = vec![0i64; labels.len() * n_traits];
            for (j, t) in traits.iter().enumerate() {
                exponents.push(quantize(
                    t.values,
                    &mut values,
                    n_traits,
                    j,
                    field(n_traits, j),
                )?);
            }
        }
        Ok(Members {
            layout,
            labels,
            values,
            exponents,
        })
    }

    /// Each label moved to its row's stratum code, `n_codes - 1` where the
    /// stratum is unknown.
    fn stratified(mut self, codes: &[Option<usize>], n_codes: usize) -> Members {
        let n_joint = self.layout.n_joint();
        for (label, code) in self.labels.iter_mut().zip(codes) {
            *label += (code.unwrap_or(n_codes - 1) * n_joint) as i32;
        }
        self.layout.n_codes = n_codes;
        self
    }

    /// The continuous traits' products; none for binary traits.
    fn products(&self) -> &'static [Product] {
        match self.exponents.len() {
            0 => &[],
            1 => &PRODUCTS[..1],
            _ => &PRODUCTS,
        }
    }

    fn input(&self) -> MomentsInput<'_> {
        let n_labels = self.layout.n_codes * self.layout.n_joint();
        MomentsInput {
            labels_first: &self.labels,
            n_labels_first: n_labels,
            labels_second: &self.labels,
            n_labels_second: n_labels,
            values: &self.values,
            n_columns: self.exponents.len(),
            products: self.products(),
            same: &[],
            n_same: 0,
        }
    }

    /// The engine table's axes: the category, then each member's code and
    /// its traits' labels.
    fn shape(&self, n_categories: usize) -> Vec<usize> {
        let Layout {
            n_traits,
            n_labels,
            n_codes,
            ..
        } = self.layout;
        let member: Vec<usize> = std::iter::once(n_codes)
            .chain(std::iter::repeat_n(n_labels, n_traits))
            .collect();
        [vec![n_categories], member.clone(), member].concat()
    }

    /// Each row of two traits paired with itself, trait 0 read in the first
    /// member and trait 1 in the second, as the engine would sum it: a
    /// table by (code, trait 0's label, code, trait 1's label) with the
    /// pairs' columns and products.
    ///
    /// The accumulator is dense over every (code, code) block though only
    /// the diagonal ones fill, so it is sized as one engine lane is:
    /// `memory_budget_exceeded` before allocating when it does not fit.
    fn within_person(&self, budget_bytes: u64) -> Result<MomentsTable<'static>, Error> {
        let Layout {
            n_labels, n_codes, ..
        } = self.layout;
        let n_joint = self.layout.n_joint();
        let k = self.exponents.len();
        let products = self.products();
        // pedigree-graph's cell layout: the count, the first member's sums,
        // the second's, their sums of squares, then the cross sums.
        let stride = 1 + 4 * k + products.len();
        let side = (n_codes * n_labels) as u64;
        let estimated_bytes = side
            .checked_mul(side)
            .and_then(|cells| cells.checked_mul(stride as u64))
            .and_then(|a| a.checked_mul(16)?.checked_add(host_bytes(a)?))
            .unwrap_or(u64::MAX);
        if estimated_bytes > budget_bytes {
            return Err(PgError::MemoryBudgetExceeded {
                operation: WITHIN_PERSON,
                estimated_bytes,
                budget_bytes,
            }
            .into());
        }
        // A row adds at most 2^86 to a sum, so i128 holds 2^41 rows.
        let mut acc = vec![0i128; (n_codes * n_labels).pow(2) * stride];
        for (row, &label) in self.labels.iter().enumerate() {
            let (code, joint) = (label as usize / n_joint, label as usize % n_joint);
            let (first, second) = (joint / n_labels, joint % n_labels);
            let cell = ((code * n_labels + first) * n_codes + code) * n_labels + second;
            let slots = &mut acc[cell * stride..(cell + 1) * stride];
            slots[0] += 1;
            let x = &self.values[row * k..(row + 1) * k];
            for (c, &v) in x.iter().enumerate() {
                let v = i128::from(v);
                for side in 0..2 {
                    slots[1 + side * k + c] += v;
                    slots[1 + (2 + side) * k + c] += v * v;
                }
            }
            for (p, product) in products.iter().enumerate() {
                slots[1 + 4 * k + p] +=
                    i128::from(x[product.a.column]) * i128::from(x[product.b.column]);
            }
        }
        Ok(MomentsTable::from_i128(
            vec![n_codes, n_labels, n_codes, n_labels],
            k,
            products.to_vec(),
            self.exponents.clone(),
            &acc,
        )?)
    }
}

/// `values` centred on their midrange and quantized into column `j` of the
/// row-major `[rows, k]` `out`, a missing value as 0; returns the exponent.
fn quantize(
    values: &[f64],
    out: &mut [i64],
    k: usize,
    j: usize,
    field: &'static str,
) -> Result<i64, Error> {
    let (min, max) = values
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
    let mut scratch: Vec<f64> = values
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v - centre })
        .collect();
    let exponent = quantize_column(&scratch, out, k, j, field)?;
    // The present centred values, compacted in place.
    let mut kept = 0;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_nan() {
            scratch[kept] = scratch[i];
            kept += 1;
        }
    }
    scratch.truncate(kept);
    check_precision(&mut scratch, exponent, field)?;
    Ok(exponent)
}

/// `2^-e`, which underflows to 0 past `e = 1074`.
fn step_of(exponent: i64) -> f64 {
    libm::ldexp(1.0, -(exponent.clamp(-2000, 2000) as i32))
}

/// `insufficient_precision` when the step exceeds `2^-20` of the present
/// values' interquartile range, or of their SD when the IQR is 0.  A
/// constant (or empty) trait has no spread and passes.
fn check_precision(present: &mut [f64], exponent: i64, field: &'static str) -> Result<(), Error> {
    let spread = match iqr(present) {
        0.0 => sd(present),
        iqr => iqr,
    };
    let step = step_of(exponent);
    if spread > 0.0 && step > PRECISION * spread {
        return Err(Error::InsufficientPrecision {
            field,
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

/// `invalid_moments_table` unless a symmetric entry's `counts`, square of
/// side `width`, are mirrored with an even diagonal, as entering each pair
/// in both orders makes them; this makes halving entries to pairs exact.
/// Anything else is an engine contract break, raised rather than rounded.
fn check_mirrored(name: &str, counts: &[i64], width: usize) -> Result<(), Error> {
    for i in 0..width {
        for j in 0..=i {
            let (a, b) = (counts[i * width + j], counts[j * width + i]);
            if a != b || (i == j && a % 2 != 0) {
                return Err(PgError::InvalidMomentsTable {
                    reason: format!(
                        "the double-entered counts of symmetric {name} are not mirrored: \
                         {a} at ({i}, {j}), {b} at ({j}, {i})"
                    ),
                }
                .into());
            }
        }
    }
    Ok(())
}

/// One block's entries by the orientations in which they are complete.  A
/// directed pair is read one way; a symmetric pair, entered in both
/// orders, both ways, which agree in a same-trait cell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Entries {
    total: u64,
    /// Complete read forward only: the first member's trait `first` and
    /// the second's `second`.
    forward: u64,
    /// Complete read reversed only: the first member's `second` and the
    /// second's `first`.
    reverse: u64,
    /// Complete both ways.
    both: u64,
}

impl Entries {
    fn complete(self) -> u64 {
        self.forward + self.reverse + self.both
    }

    /// Complete in a reading kept: forward where `forward`, reversed where
    /// `reverse`.
    fn kept(self, forward: bool, reverse: bool) -> u64 {
        u64::from(forward) * self.forward
            + u64::from(reverse) * self.reverse
            + u64::from(forward || reverse) * self.both
    }

    fn add(self, other: Entries) -> Entries {
        Entries {
            total: self.total + other.total,
            forward: self.forward + other.forward,
            reverse: self.reverse + other.reverse,
            both: self.both + other.both,
        }
    }
}

/// Each block's [`Entries`], row-major over (first code, second code), of
/// `counts` by (code, label, code, label) of `n_codes` codes and `width`
/// labels; `complete(x, y)` says whether an entry labelled `x`, `y` is
/// complete read forward and reversed.
fn tally(
    counts: &[i64],
    n_codes: usize,
    width: usize,
    complete: impl Fn(usize, usize) -> (bool, bool),
) -> Vec<Entries> {
    let mut blocks = vec![Entries::default(); n_codes * n_codes];
    for (cell, &count) in counts.iter().enumerate() {
        let count = count as u64;
        let (s, x) = (
            cell / (width * n_codes * width),
            cell / (n_codes * width) % width,
        );
        let (u, y) = (cell / width % n_codes, cell % width);
        let block = &mut blocks[s * n_codes + u];
        block.total += count;
        match complete(x, y) {
            (true, true) => block.both += count,
            (true, false) => block.forward += count,
            (false, true) => block.reverse += count,
            (false, false) => {}
        }
    }
    blocks
}

/// A cell's (or a stratum's) counts and fits.
struct Fields {
    n: u64,
    n_dropped: u64,
    n_entries: Option<u64>,
    counts: Counts,
    estimates: Vec<Fit>,
}

/// One cell's table by (first member's stratum code, its trait `first`'s
/// label, second member's stratum code, its trait `second`'s label), and
/// each block's entries by completeness.
struct Blocks {
    table: MomentsTable<'static>,
    counts: Vec<u64>,
    entries: Vec<Entries>,
    n_codes: usize,
    n_labels: usize,
    kind: TraitKind,
    view: View,
    /// The first and second traits' field names, as errors report them.
    names: [&'static str; 2],
}

impl Blocks {
    fn new(
        table: MomentsTable<'static>,
        entries: Vec<Entries>,
        kind: TraitKind,
        view: View,
        names: [&'static str; 2],
    ) -> Blocks {
        let shape = table.shape();
        let (n_codes, n_labels) = (shape[0], shape[1]);
        Blocks {
            counts: table.counts().iter().map(|&c| c as u64).collect(),
            table,
            entries,
            n_codes,
            n_labels,
            kind,
            view,
            names,
        }
    }

    /// The cell `view` of an entry from its table by (code, labels, code,
    /// labels) and that table's `counts`: the other trait summed out of
    /// each member, and each entry's completeness read from both traits'
    /// labels.
    fn pairs(
        joint: &MomentsTable<'static>,
        counts: &[i64],
        layout: Layout,
        view: View,
    ) -> Result<Blocks, Error> {
        let read = |x: usize, y: usize, first: usize, second: usize| {
            layout.shows(x, first) && layout.shows(y, second)
        };
        let entries = tally(counts, layout.n_codes, layout.n_joint(), |x, y| {
            (
                read(x, y, view.first, view.second),
                view.symmetric && read(x, y, view.second, view.first),
            )
        });
        // A member's axes are its code, trait 0's label, trait 1's label.
        let table = if layout.n_traits == 1 {
            joint.clone()
        } else {
            joint.sum(5 - view.second)?.sum(2 - view.first)?
        };
        let names = [view.first, view.second].map(|t| field(layout.n_traits, t));
        Ok(Blocks::new(table, entries, layout.kind, view, names))
    }

    /// The Within-Person Cross-Trait Correlation's blocks: each row once,
    /// in its own stratum's diagonal block.
    fn within_person(members: &Members, budget_bytes: u64) -> Result<Blocks, Error> {
        let table = members.within_person(budget_bytes)?;
        let Layout {
            kind,
            n_labels,
            n_codes,
            ..
        } = members.layout;
        let n_observed = observed(kind);
        let entries = tally(&table.counts(), n_codes, n_labels, |x, y| {
            (x < n_observed && y < n_observed, false)
        });
        let view = View {
            first: 0,
            second: 1,
            symmetric: false,
        };
        Ok(Blocks::new(table, entries, kind, view, FIELDS))
    }

    /// The flat index of (first code, first label, second code, second label).
    fn at(&self, s: usize, i: usize, u: usize, j: usize) -> usize {
        ((s * self.n_labels + i) * self.n_codes + u) * self.n_labels + j
    }

    fn entries(&self, s: usize, u: usize) -> Entries {
        self.entries[s * self.n_codes + u]
    }

    /// Entries of block `(s, u)` with a complete orientation.
    fn complete(&self, s: usize, u: usize) -> u64 {
        self.entries(s, u).complete()
    }

    /// Unordered pairs of `entries`: a symmetric pair entered twice.
    fn pairs_of(&self, entries: u64) -> u64 {
        if self.view.symmetric {
            entries / 2
        } else {
            entries
        }
    }

    fn name(&self, side: Side) -> &'static str {
        match side {
            Side::First => self.names[0],
            Side::Second => self.names[1],
        }
    }

    /// The `[labels, labels]` table of the blocks with first code in `rows`
    /// and second code in `cols`.
    fn block(&self, rows: &[usize], cols: &[usize]) -> Result<MomentsTable<'static>, Error> {
        Ok(self
            .table
            .select(0, rows)?
            .sum(0)?
            .select(1, cols)?
            .sum(1)?)
    }

    /// The fields of the blocks with first code in `rows` and second code
    /// in `cols`; a symmetric cell's set is closed under swapping them.
    /// Each fit's influence goes to `influences`.
    fn fields(
        &self,
        rows: &[usize],
        cols: &[usize],
        influences: &mut Influences,
    ) -> Result<Fields, Error> {
        let entries = rows
            .iter()
            .flat_map(|&s| cols.iter().map(move |&u| (s, u)))
            .map(|(s, u)| self.entries(s, u))
            .fold(Entries::default(), Entries::add);
        let n = self.pairs_of(entries.complete());
        let n_dropped = self.pairs_of(entries.total) - n;
        let n_entries = self.view.cross().then_some(entries.forward + entries.both);
        let table = self.block(rows, cols)?;
        let view = self.view;
        let (counts, estimates) = if self.kind == TraitKind::Binary {
            let counts = table.counts();
            let at = |i: usize, j: usize| counts[i * self.n_labels + j] as u64;
            let fitted = [[at(0, 0), at(0, 1)], [at(1, 0), at(1, 1)]];
            let shown = if self.view.symmetric && !self.view.cross() {
                Counts::Concordance([at(0, 0) / 2, at(0, 1), at(1, 1) / 2])
            } else {
                Counts::Table(fitted)
            };
            let fit = correlation::fitted(fitted);
            let point = fit.as_ref().map(|&(_, point)| point).map_err(|&r| r);
            let two_step = influences.made(tetrachoric(point), || {
                influence::two_step(&fit, |at| Influence::tetrachoric(view, (rows, cols), at))
            });
            let phi = phi(fitted);
            let phi = influences.made(phi, || {
                phi.value
                    .map(|value| Influence::phi(view, (rows, cols), fitted, value))
                    .map_err(|_| SeReason::NoEstimate)
            });
            (shown, vec![two_step, phi])
        } else {
            let r = table.derive(Statistic::Pearson, self.view.product(), self.names[0])?[0];
            let value = match (n, r.is_nan()) {
                (0, _) => Err(Reason::NoCompletePairs),
                (_, true) => Err(Reason::ConstantValue),
                _ => Ok(r),
            };
            let influence = match (influences.on(), value) {
                (true, Ok(r)) => Some(Influence::pearson(
                    view,
                    (rows, cols),
                    &table,
                    r,
                    self.names[0],
                )?),
                _ => None,
            };
            let pearson = influences.made(pearson(value), || influence.ok_or(SeReason::NoEstimate));
            (Counts::None, vec![pearson])
        };
        Ok(Fields {
            n,
            n_dropped,
            n_entries,
            counts,
            estimates,
        })
    }

    /// The cell; with `labels` (the known strata, ascending) its stratum
    /// cells, counts and stratified fit too.  Each fit's influence goes to
    /// `influences`.
    fn cell(
        &self,
        request: Request,
        labels: Option<&[i64]>,
        influences: &mut Influences,
    ) -> Result<Cell<Fit>, Error> {
        let codes: Vec<usize> = (0..self.n_codes).collect();
        let Fields {
            n,
            n_dropped,
            n_entries,
            counts,
            estimates,
        } = self.fields(&codes, &codes, influences)?;
        let pooled = Cell {
            category: request.name,
            members: request.members,
            symmetric: request.symmetric,
            roles: request.roles,
            first_trait: self.view.first,
            second_trait: self.view.second,
            n,
            n_dropped,
            n_entries,
            counts,
            estimates,
            strata: None,
            n_cross_stratum: 0,
            n_unknown_stratum: 0,
            stratified: None,
        };
        let Some(labels) = labels else {
            return Ok(pooled);
        };
        let (k, symmetric) = (labels.len(), self.view.symmetric);
        let strata = labels
            .iter()
            .enumerate()
            .map(|(s, &label)| {
                let seconds = if symmetric { &codes[s..=s] } else { &codes };
                let Fields {
                    n,
                    n_dropped,
                    n_entries,
                    counts,
                    estimates,
                } = self.fields(&[s], seconds, influences)?;
                Ok(StratumCell {
                    label,
                    n,
                    n_dropped,
                    n_entries,
                    counts,
                    estimates,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        // Code k is the unknown stratum.  A symmetric pair across strata s
        // and u is in block (s, u) once and (u, s) once, complete in both
        // or neither; one within a stratum is in its diagonal block twice.
        let (n_cross_stratum, n_unknown_stratum) = if symmetric {
            let cross = (0..k)
                .flat_map(|s| (s + 1..k).map(move |u| (s, u)))
                .map(|(s, u)| self.complete(s, u))
                .sum();
            let known: u64 = (0..k).map(|u| self.complete(k, u)).sum();
            (cross, known + self.complete(k, k) / 2)
        } else {
            (0, (0..=k).map(|u| self.complete(k, u)).sum())
        };
        Ok(Cell {
            strata: Some(strata),
            n_cross_stratum,
            n_unknown_stratum,
            stratified: Some(self.stratified(k, influences)?),
            ..pooled
        })
    }

    /// [`Stratified`] over the blocks of the known codes `0..k`.  A
    /// symmetric pair across strata s and u is fitted when an orientation
    /// complete in a kept block is: its forward reading in (s, u), its
    /// reverse in (u, s).
    fn stratified(&self, k: usize, influences: &mut Influences) -> Result<Stratified<Fit>, Error> {
        let fitted = |kept: &[bool]| -> u64 {
            let entries: u64 = (0..k * k)
                .map(|b| {
                    let (s, u) = (b / k, b % k);
                    self.entries(s, u).kept(kept[b], kept[u * k + s])
                })
                .sum();
            self.pairs_of(entries)
        };
        let n_known = fitted(&vec![true; k * k]);
        let kept = self.drop_degenerate(k)?;
        let n = fitted(&kept);
        let n_entries = self.view.cross().then(|| {
            (0..k * k)
                .filter(|&b| kept[b])
                .map(|b| self.entries(b / k, b % k).kept(true, false))
                .sum()
        });
        let estimate = match (self.kind == TraitKind::Binary, n_known, n) {
            (true, 0, _) => influences.none(tetrachoric(Err(FitReason::NoCompletePairs))),
            (true, _, 0) => influences.none(tetrachoric(Err(FitReason::ConstantMargin))),
            (true, _, _) => self.stratified_tetrachoric(k, &kept, influences),
            (false, 0, _) => influences.none(pearson(Err(Reason::NoCompletePairs))),
            (false, _, 0) => influences.none(pearson(Err(Reason::ConstantValue))),
            (false, _, _) => self.stratified_pearson(k, &kept, influences)?,
        };
        Ok(Stratified {
            n,
            n_entries,
            n_degenerate_stratum: n_known - n,
            estimate,
        })
    }

    /// The `k x k` known blocks a stratified fit keeps, row-major: every
    /// block of a stratum whose side is constant over the blocks kept is
    /// left out, until none is.  Leaving blocks out never makes a constant
    /// side vary, so the blocks kept do not depend on the order strata are
    /// found in.
    fn drop_degenerate(&self, k: usize) -> Result<Vec<bool>, Error> {
        let mut kept = vec![true; k * k];
        loop {
            let mut constant = Vec::new();
            for side in [Side::First, Side::Second] {
                for s in 0..k {
                    if self.constant(side, s, k, &kept)? {
                        constant.push((side, s));
                    }
                }
            }
            if constant.is_empty() {
                return Ok(kept);
            }
            for (side, s) in constant {
                for u in 0..k {
                    kept[block_index(side, s, u, k)] = false;
                }
            }
        }
    }

    /// The `[labels, labels]` table of stratum `s` of `side` over the kept
    /// blocks.
    fn side_block(
        &self,
        side: Side,
        s: usize,
        k: usize,
        kept: &[bool],
    ) -> Result<MomentsTable<'static>, Error> {
        let others: Vec<usize> = (0..k)
            .filter(|&u| kept[block_index(side, s, u, k)])
            .collect();
        match side {
            Side::First => self.block(&[s], &others),
            Side::Second => self.block(&others, &[s]),
        }
    }

    /// Whether stratum `s` of `side` has complete pairs in the kept blocks
    /// and its values there are constant: one binary level, or zero spread
    /// at the quantization step.
    fn constant(&self, side: Side, s: usize, k: usize, kept: &[bool]) -> Result<bool, Error> {
        let table = self.side_block(side, s, k, kept)?;
        let counts = table.counts();
        if self.kind == TraitKind::Binary {
            let at = |i: usize, j: usize| counts[i * self.n_labels + j];
            let shows = [0, 1].map(|v| match side {
                Side::First => at(v, 0) + at(v, 1) > 0,
                Side::Second => at(0, v) + at(1, v) > 0,
            });
            return Ok(shows[0] != shows[1]);
        }
        let m2 = table.derive(Statistic::M2(side), self.view.column(side), self.name(side))?;
        Ok(counts[0] > 0 && m2[0] == 0.0)
    }

    /// The Polychoric Fit of the kept blocks' 2 x 2 tables, a threshold
    /// per stratum and side.
    fn stratified_tetrachoric(&self, k: usize, kept: &[bool], influences: &mut Influences) -> Fit {
        let mut table = Table::zeros(k, k, 2, 2);
        for (b, _) in kept.iter().enumerate().filter(|(_, &kept)| kept) {
            let (s, u) = (b / k, b % k);
            for i in 0..2 {
                for j in 0..2 {
                    let at = table.index(s, u, i, j);
                    table.data[at] = self.counts[self.at(s, i, u, j)] as f64;
                }
            }
        }
        let fit = correlation::fit_counts(table);
        let point = fit.as_ref().map(|&(_, point)| point).map_err(|&r| r);
        influences.made(tetrachoric(point), || {
            influence::two_step(&fit, |at| {
                Influence::stratified_tetrachoric(self.view, k, kept, at)
            })
        })
    }

    /// `Σ z₁ z₂ / N` over the kept blocks' pairs, each side standardised
    /// within its stratum (mean and `1/N` variance of that side's values
    /// over the kept blocks), so `Σ z₁² = Σ z₂² = N`.  A block's
    /// cross-product about its strata's means is its own comoment plus its
    /// count times the product of its means' offsets from them.
    fn stratified_pearson(
        &self,
        k: usize,
        kept: &[bool],
        influences: &mut Influences,
    ) -> Result<Fit, Error> {
        let scale = |side: Side| {
            let (column, name) = (self.view.column(side), self.name(side));
            (0..k)
                .map(|s| {
                    let table = self.side_block(side, s, k, kept)?;
                    let n = table.counts()[0] as f64;
                    let mean = table.derive(Statistic::Mean(side), column, name)?[0];
                    let m2 = table.derive(Statistic::M2(side), column, name)?[0];
                    Ok((mean, (m2 / n).sqrt()))
                })
                .collect::<Result<Vec<_>, Error>>()
        };
        let (rows, cols) = (scale(Side::First)?, scale(Side::Second)?);
        let comoment =
            self.table
                .derive(Statistic::Comoment, self.view.product(), self.names[0])?;
        let first =
            self.table
                .derive(Statistic::Mean(Side::First), self.view.first, self.names[0])?;
        let second = self.table.derive(
            Statistic::Mean(Side::Second),
            self.view.second,
            self.names[1],
        )?;
        let (mut cross, mut n) = (0.0, 0u64);
        let mut blocks = Vec::new();
        for (b, _) in kept.iter().enumerate().filter(|(_, &kept)| kept) {
            let (s, u) = (b / k, b % k);
            let at = self.at(s, 0, u, 0);
            let count = self.counts[at];
            if count == 0 {
                continue;
            }
            let ((m1, sd1), (m2, sd2)) = (rows[s], cols[u]);
            let about = comoment[at] + count as f64 * (first[at] - m1) * (second[at] - m2);
            cross += about / (sd1 * sd2);
            n += count;
            if influences.on() {
                blocks.push(StratifiedBlock {
                    s,
                    u,
                    n: count as f64,
                    za: count as f64 * (first[at] - m1) / sd1,
                    zb: count as f64 * (second[at] - m2) / sd2,
                    zab: about / (sd1 * sd2),
                });
            }
        }
        let rho = (cross / n as f64).clamp(-1.0, 1.0);
        Ok(influences.made(pearson(Ok(rho)), || {
            Ok(Influence::stratified_pearson(
                self.view,
                [&rows, &cols],
                &blocks,
                rho,
            ))
        }))
    }
}

/// The row-major index of the block of stratum `s` of `side` and stratum
/// `u` of the other side, of `k x k`.
fn block_index(side: Side, s: usize, u: usize, k: usize) -> usize {
    match side {
        Side::First => s * k + u,
        Side::Second => u * k + s,
    }
}

/// ADR 0011's two-step point estimate, without its SE.
fn tetrachoric(point: Result<Point, FitReason>) -> Estimate {
    Estimate {
        estimator: Estimator::Tetrachoric,
        value: point.map(|p| p.value).map_err(Reason::from),
        boundary: point.ok().and_then(|p| p.boundary),
        se: None,
    }
}

fn pearson(value: Result<f64, Reason>) -> Estimate {
    Estimate {
        estimator: Estimator::Pearson,
        value,
        boundary: None,
        se: None,
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
        se: None,
    }
}

const METHOD: [(&str, &str); 7] = [
    (
        "entry",
        "a symmetric category's pairs enter in both orders (a same-trait cell is intraclass); \
         a directed category's once, in its roles' order",
    ),
    (
        "traits",
        "with two traits, each entry's cells read trait first_trait of the first member \
         against second_trait of the second: a symmetric category's (0, 0), (1, 1) and the \
         double-entered (0, 1), a threshold or a mean and SD per trait; a directed \
         category's (0, 0), (0, 1), (1, 0), (1, 1); n counts the pairs with a complete \
         orientation, n_entries a symmetric cross-trait cell's complete orientations",
    ),
    (
        "within_person",
        "with two traits, each row's trait 0 against its trait 1, a row's stratum its own",
    ),
    (
        "estimators",
        "continuous: pearson; binary: tetrachoric (primary, ADR 0011's two-step fit) and phi",
    ),
    (
        "quantization",
        "continuous values centred on their midrange and quantized to 43 bits of their \
         largest magnitude, each trait on its own; constant_value means constant at that step",
    ),
    (
        "strata",
        "a directed pair is in its first role's stratum, a symmetric pair in a stratum \
         only when both members are; stratified fits the primary estimator once over every \
         block of known strata (a threshold, or a mean and SD, per stratum and side), \
         leaving out the blocks of a stratum whose side is constant",
    ),
    (
        "se",
        "with se, a cluster sandwich (ADR 0013): each entry's influence on the estimate summed \
         over its pair's MRCA and over each member, meat = sum over MRCAs + sum over members - \
         sum over (member, MRCA), SE = sqrt(meat), no small-sample factor; none within_person, \
         nor when |meat| <= max(1024, 4 sqrt(n)) eps x its scale, the three sums of squares at \
         influence 1/n over the n entries (degenerate_meat)",
    ),
];

impl Estimate {
    fn to_value(self, primary: bool) -> Value {
        let (value, reason) = match self.value {
            Ok(v) => (Value::Float(v), Value::Null),
            Err(r) => (Value::Null, r.name().into()),
        };
        let mut entries = vec![
            ("estimator", self.estimator.name().into()),
            ("primary", primary.into()),
            ("value", value),
            ("reason", reason),
            ("boundary", self.boundary.into()),
        ];
        if let Some(se) = self.se {
            entries.extend(match se {
                Ok(se) => [("se", Value::Float(se)), ("se_reason", Value::Null)],
                Err(r) => [("se", Value::Null), ("se_reason", r.name().into())],
            });
        }
        Value::Map(entries)
    }
}

/// A cell's or a stratum's counts and estimates, in the key order hosts
/// present.
fn field_entries(
    n: u64,
    n_dropped: u64,
    n_entries: Option<u64>,
    counts: Counts,
    estimates: &[Estimate],
) -> Vec<(&'static str, Value)> {
    let (concordance, table) = match counts {
        Counts::None => (Value::Null, Value::Null),
        Counts::Concordance(c) => (Value::Counts(c.to_vec()), Value::Null),
        Counts::Table(t) => (
            Value::Null,
            Value::List(t.iter().map(|row| Value::Counts(row.to_vec())).collect()),
        ),
    };
    vec![
        ("n", n.into()),
        ("n_dropped", n_dropped.into()),
        ("n_entries", n_entries.into()),
        ("concordance", concordance),
        ("table", table),
        (
            "crude",
            Value::List(
                estimates
                    .iter()
                    .enumerate()
                    .map(|(i, e)| e.to_value(i == 0))
                    .collect(),
            ),
        ),
    ]
}

impl StratumCell {
    fn to_value(&self) -> Value {
        let mut entries = vec![("stratum", Value::Int(self.label))];
        entries.extend(field_entries(
            self.n,
            self.n_dropped,
            self.n_entries,
            self.counts,
            &self.estimates,
        ));
        Value::Map(entries)
    }
}

impl Stratified {
    fn to_value(self) -> Value {
        Value::Map(vec![
            ("n", self.n.into()),
            ("n_entries", self.n_entries.into()),
            ("n_degenerate_stratum", self.n_degenerate_stratum.into()),
            ("estimate", self.estimate.to_value(true)),
        ])
    }
}

impl<E> Cell<E> {
    /// The same cell, each estimate through `f`.
    fn map<F>(self, mut f: impl FnMut(E) -> F) -> Cell<F> {
        let estimates = self.estimates.into_iter().map(&mut f).collect();
        let strata = self.strata.map(|strata| {
            strata
                .into_iter()
                .map(|s| StratumCell {
                    label: s.label,
                    n: s.n,
                    n_dropped: s.n_dropped,
                    n_entries: s.n_entries,
                    counts: s.counts,
                    estimates: s.estimates.into_iter().map(&mut f).collect(),
                })
                .collect()
        });
        let stratified = self.stratified.map(|s| Stratified {
            n: s.n,
            n_entries: s.n_entries,
            n_degenerate_stratum: s.n_degenerate_stratum,
            estimate: f(s.estimate),
        });
        Cell {
            category: self.category,
            members: self.members,
            symmetric: self.symmetric,
            roles: self.roles,
            first_trait: self.first_trait,
            second_trait: self.second_trait,
            n: self.n,
            n_dropped: self.n_dropped,
            n_entries: self.n_entries,
            counts: self.counts,
            estimates,
            strata,
            n_cross_stratum: self.n_cross_stratum,
            n_unknown_stratum: self.n_unknown_stratum,
            stratified,
        }
    }
}

impl Cell {
    pub fn to_value(&self) -> Value {
        let codes = |cats: &[Category]| Value::List(cats.iter().map(|c| c.code().into()).collect());
        let mut entries = vec![
            ("category", Value::Text(self.category.clone())),
            ("members", codes(&self.members)),
            ("symmetric", self.symmetric.into()),
            (
                "roles",
                self.roles
                    .map_or(Value::Null, |[a, b]| Value::List(vec![a.into(), b.into()])),
            ),
            ("first_trait", (self.first_trait as u64).into()),
            ("second_trait", (self.second_trait as u64).into()),
        ];
        entries.extend(field_entries(
            self.n,
            self.n_dropped,
            self.n_entries,
            self.counts,
            &self.estimates,
        ));
        let strata = self
            .strata
            .as_ref()
            .map(|strata| Value::List(strata.iter().map(StratumCell::to_value).collect()));
        entries.extend([
            ("strata", strata.into()),
            ("n_cross_stratum", self.n_cross_stratum.into()),
            ("n_unknown_stratum", self.n_unknown_stratum.into()),
            (
                "stratified",
                self.stratified.map(Stratified::to_value).into(),
            ),
        ]);
        Value::Map(entries)
    }
}

impl RelativePairCorrelation {
    /// The result as a host-neutral tree; its keys are the hosts' contract.
    pub fn to_value(&self) -> Value {
        let names = self
            .cells
            .iter()
            .filter(|c| (c.first_trait, c.second_trait) == (0, 0))
            .map(|c| Value::Text(c.category.clone()))
            .collect();
        let exponents = self.quantization_exponents.as_deref();
        Value::Map(vec![
            (
                "cells",
                Value::List(self.cells.iter().map(Cell::to_value).collect()),
            ),
            (
                "within_person",
                self.within_person.as_ref().map(Cell::to_value).into(),
            ),
            (
                "settings",
                Value::Map(vec![
                    ("categories", Value::List(names)),
                    ("trait_kind", self.kind.name().into()),
                    ("memory_budget_bytes", self.memory_budget_bytes.into()),
                    ("threads", (self.threads as u64).into()),
                    ("se", self.se.into()),
                    (
                        "quantization_exponents",
                        exponents
                            .map(|e| Value::List(e.iter().copied().map(Value::Int).collect()))
                            .into(),
                    ),
                    (
                        "quantization_steps",
                        exponents
                            .map(|e| Value::Floats(e.iter().copied().map(step_of).collect()))
                            .into(),
                    ),
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

    fn point(table: [[u64; 2]; 2]) -> Result<Point, FitReason> {
        correlation::fitted(table).map(|(_, point)| point)
    }

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
            &[trait_],
            None,
            categories.as_deref(),
            DEFAULT_MEMORY_BUDGET_BYTES,
            false,
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
        assert_eq!(res.quantization_exponents, None);
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
        assert!(res.quantization_exponents.is_some());
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
        let point = point(table(po)).unwrap();
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
        let cases: [(&[Entry], &str); 10] = [
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
            (&[group("within_person", &["FS"])], "reserved_group_name"),
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
                position: 1.into()
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
            &[trait_],
            None,
            Some(entries),
            DEFAULT_MEMORY_BUDGET_BYTES,
            false,
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
                &[Trait {
                    values,
                    kind,
                    n_levels: None,
                }],
                None,
                categories,
                budget,
                false,
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
        let err = check_mirrored("FS", &[4, 3, 0, 2, 6, 0, 0, 0, 0], 3).unwrap_err();
        assert!(
            matches!(err, Error::Pedigree(PgError::InvalidMomentsTable { .. })),
            "{err:?}"
        );
        // An odd diagonal is not a double entry either.
        assert!(check_mirrored("FS", &[5, 2, 0, 2, 6, 0, 0, 0, 0], 3).is_err());
        assert!(check_mirrored("FS", &[4, 2, 0, 2, 6, 0, 0, 0, 0], 3).is_ok());
        // A directed category takes any table.
        let counts: [i64; 9] = [5, 3, 0, 2, 6, 0, 0, 0, 0];
        let table =
            MomentsTable::from_i128(vec![1, 3, 1, 3], 0, vec![], vec![], &counts.map(i128::from))
                .unwrap();
        let layout = Layout {
            kind: TraitKind::Binary,
            n_traits: 1,
            n_labels: 3,
            n_codes: 1,
        };
        let blocks = Blocks::pairs(&table, &counts, layout, View::of(1, false)[0]).unwrap();
        let mo = blocks
            .cell(single(Category::MO), None, &mut Influences(None))
            .unwrap();
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

    /// [`correlation`] on `ped` with one stratum per row, `NaN` unknown.
    fn run_strata(
        ped: (&[i64], &[i64], &[i64]),
        values: &[f64],
        kind: TraitKind,
        stratum: &[f64],
        categories: Option<&[Entry]>,
    ) -> Result<RelativePairCorrelation, Error> {
        run_strata_se(ped, values, kind, stratum, categories, false)
    }

    fn run_strata_se(
        ped: (&[i64], &[i64], &[i64]),
        values: &[f64],
        kind: TraitKind,
        stratum: &[f64],
        categories: Option<&[Entry]>,
        se: bool,
    ) -> Result<RelativePairCorrelation, Error> {
        let labels: Vec<i64> = stratum
            .iter()
            .map(|&s| if s.is_nan() { 0 } else { s as i64 })
            .collect();
        let known: Vec<bool> = stratum.iter().map(|s| !s.is_nan()).collect();
        let input = PedigreeInput {
            ids: ped.0,
            mother: ped.1,
            father: ped.2,
            twin: None,
            sex: None,
        };
        let categories = categories.map(numbered);
        correlation(
            PedigreeArg::Columns(input),
            &[Trait {
                values,
                kind,
                n_levels: None,
            }],
            Some(Strata {
                labels: &labels,
                known: &known,
            }),
            categories.as_deref(),
            DEFAULT_MEMORY_BUDGET_BYTES,
            se,
        )
    }

    /// Every complete pair of a stratified cell is in one stratum's cell,
    /// across two known strata, or of unknown stratum.
    fn assert_partition(cell: &Cell) {
        let in_strata: u64 = cell.strata.as_ref().unwrap().iter().map(|s| s.n).sum();
        assert_eq!(
            in_strata + cell.n_cross_stratum + cell.n_unknown_stratum,
            cell.n,
            "{}",
            cell.category
        );
    }

    fn stratum_n(cell: &Cell) -> Vec<(i64, u64, u64)> {
        let strata = cell.strata.as_ref().unwrap();
        strata.iter().map(|s| (s.label, s.n, s.n_dropped)).collect()
    }

    /// `families` couples (mother, father, then `kids` children each), in
    /// row blocks of `2 + kids`.
    fn sibships(families: usize, kids: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let width = 2 + kids;
        let n = families * width;
        let (mut mother, mut father) = (vec![-1; n], vec![-1; n]);
        for f in 0..families {
            let base = f * width;
            for kid in base + 2..base + width {
                mother[kid] = base as i64;
                father[kid] = base as i64 + 1;
            }
        }
        ((0..n as i64).collect(), mother, father)
    }

    /// A deterministic value in `[0, 1)` per row.
    fn noise(i: usize) -> f64 {
        ((i * 7919 + 13) % 101) as f64 / 101.0
    }

    #[test]
    fn without_strata_the_new_fields_are_empty() {
        let v = [0.3, -1.2, 0.8, 2.0, 1.5, -0.4, 0.9, 0.1, 1.1];
        let res = run(&v, TraitKind::Continuous, None).unwrap();
        for cell in &res.cells {
            assert_eq!(cell.strata, None);
            assert_eq!((cell.n_cross_stratum, cell.n_unknown_stratum), (0, 0));
            assert_eq!(cell.stratified, None);
        }
    }

    #[test]
    fn strata_assign_directed_pairs_at_the_first_role_and_symmetric_within_one() {
        // Rows 3 (a father) and 5 (a full sib) are of unknown stratum; row
        // 8's value is missing.
        let stratum = [10.0, 10.0, 20.0, NA, 30.0, NA, 31.0, 30.0, 20.0];
        let v = [0.3, -1.2, 0.8, 2.0, 1.5, -0.4, 0.9, 0.1, NA];
        let (ids, mother, father) = family();
        let ped = (&ids[..], &mother[..], &father[..]);
        let res = run_strata(ped, &v, TraitKind::Continuous, &stratum, None).unwrap();
        let plain = run_on(ped, &v, TraitKind::Continuous, None).unwrap();
        for (cell, plain) in res.cells.iter().zip(&plain.cells) {
            assert_partition(cell);
            let pooled = Cell {
                strata: None,
                n_cross_stratum: 0,
                n_unknown_stratum: 0,
                stratified: None,
                ..cell.clone()
            };
            assert_eq!(&pooled, plain, "strata leave the pooled cell alone");
        }
        // MO, offspring first: (2, 0) at 20; (4, 2), (7, 2) at 30; (6, 2) at
        // 31; (5, 2) unknown.
        let mo = cell(&res, "MO");
        assert_eq!(
            stratum_n(mo),
            [(10, 0, 0), (20, 1, 0), (30, 2, 0), (31, 1, 0)]
        );
        assert_eq!((mo.n_cross_stratum, mo.n_unknown_stratum), (0, 1));
        // FO: (4, 3) and (6, 3) stay at the offspring's stratum though the
        // father's is unknown; (7, 8) is dropped at 30.
        let fo = cell(&res, "FO");
        assert_eq!(
            stratum_n(fo),
            [(10, 0, 0), (20, 1, 0), (30, 1, 1), (31, 1, 0)]
        );
        assert_eq!((fo.n_cross_stratum, fo.n_unknown_stratum), (0, 1));
        // FS {4, 6} spans 30 and 31; {4, 5} and {5, 6} have an unknown member.
        let fs = cell(&res, "FS");
        assert!(stratum_n(fs).iter().all(|&(_, n, _)| n == 0));
        assert_eq!((fs.n_cross_stratum, fs.n_unknown_stratum), (1, 2));
        // MHS {4, 7} is within 30; {6, 7} spans 31 and 30; {5, 7} unknown.
        let mhs = cell(&res, "MHS");
        assert_eq!(
            stratum_n(mhs),
            [(10, 0, 0), (20, 0, 0), (30, 1, 0), (31, 0, 0)]
        );
        assert_eq!((mhs.n_cross_stratum, mhs.n_unknown_stratum), (1, 1));
        let mhs_30 = &mhs.strata.as_ref().unwrap()[2];
        let r = mhs_30.estimates[0].value.unwrap();
        assert!((r - pearson(&[v[4], v[7]], &[v[7], v[4]])).abs() < 1e-12);
        // GP: descendants 4, 7 at 30, 6 at 31, 5 unknown, two ancestors each.
        let gp = cell(&res, "GP");
        assert_eq!(
            stratum_n(gp),
            [(10, 0, 0), (20, 0, 0), (30, 4, 0), (31, 2, 0)]
        );
        assert_eq!(gp.n_unknown_stratum, 2);
        // With row 6 unknown too, {5, 6} has both members unknown: one pair.
        let mut both = stratum;
        both[6] = NA;
        let res = run_strata(ped, &v, TraitKind::Continuous, &both, Some(&[code("FS")])).unwrap();
        let fs = &res.cells[0];
        assert_eq!((fs.n_cross_stratum, fs.n_unknown_stratum), (0, 3));
        assert_partition(fs);
    }

    #[test]
    fn a_directed_pair_with_its_second_stratum_unknown_is_out_of_the_stratified_fit() {
        // Three families of two children; the third father's stratum is
        // unknown.
        let (ids, mother, father) = sibships(3, 2);
        let stratum = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, NA, 1.0, 1.0];
        let v = [0.0, 0.4, 1.1, -0.3, 0.0, 1.7, 0.2, 0.9, 0.0, 2.5, 1.4, -0.8];
        let ped = (&ids[..], &mother[..], &father[..]);
        let res = run_strata(ped, &v, TraitKind::Continuous, &stratum, None).unwrap();
        let fo = cell(&res, "FO");
        assert_eq!(stratum_n(fo), [(0, 0, 0), (1, 6, 0)]);
        assert_eq!(fo.n_unknown_stratum, 0);
        let stratified = fo.stratified.unwrap();
        assert_eq!((stratified.n, stratified.n_degenerate_stratum), (4, 0));
        // One block (1, 0): the stratified r is that block's Pearson.
        let r = stratified.estimate.value.unwrap();
        let expected = pearson(&[v[2], v[3], v[6], v[7]], &[v[1], v[1], v[5], v[5]]);
        assert!((r - expected).abs() < 1e-12, "{r} {expected}");
        assert_eq!(stratified.estimate.estimator, Estimator::Pearson);
    }

    #[test]
    fn one_stratum_over_every_row_gives_the_pooled_primary() {
        let (ids, mother, father) = sibships(30, 4);
        let ped = (&ids[..], &mother[..], &father[..]);
        let stratum = vec![5.0; ids.len()];
        let binary: Vec<f64> = (0..ids.len())
            .map(|i| f64::from(u8::from(noise(i) < 0.4)))
            .collect();
        let continuous: Vec<f64> = (0..ids.len()).map(noise).collect();
        let entries = [code("MO"), code("FS")];
        for (values, kind) in [
            (&binary, TraitKind::Binary),
            (&continuous, TraitKind::Continuous),
        ] {
            let res = run_strata(ped, values, kind, &stratum, Some(&entries)).unwrap();
            for cell in &res.cells {
                let stratified = cell.stratified.unwrap();
                assert_eq!((stratified.n, stratified.n_degenerate_stratum), (cell.n, 0));
                let (got, pooled) = (stratified.estimate, cell.estimates[0]);
                if kind == TraitKind::Binary {
                    assert_eq!(got, pooled, "{}", cell.category);
                } else {
                    let gap = got.value.unwrap() - pooled.value.unwrap();
                    assert!(gap.abs() < 1e-12, "{}: {gap}", cell.category);
                }
            }
        }
    }

    #[test]
    fn the_within_person_accumulator_is_held_to_the_budget() {
        let Fixture {
            ids,
            mother,
            father,
            values: x,
            ..
        } = stratified_fixture(TraitKind::Binary);
        let y = second_trait(TraitKind::Binary, x.len());
        let labels: Vec<i64> = (0..x.len() as i64).collect();
        let known = vec![true; x.len()];
        let call = |budget: u64| {
            let input = PedigreeInput {
                ids: &ids,
                mother: &mother,
                father: &father,
                twin: None,
                sex: None,
            };
            let traits = [&x, &y].map(|values| Trait {
                values,
                kind: TraitKind::Binary,
                n_levels: None,
            });
            let strata = Strata {
                labels: &labels,
                known: &known,
            };
            correlation(
                PedigreeArg::Columns(input),
                &traits,
                Some(strata),
                Some(&[]),
                budget as i64,
                false,
            )
        };
        // A stratum per row and the unknown code, three labels each, one
        // count per cell: 16 bytes a lane and 32 a host.
        let side = 3 * (x.len() as u64 + 1);
        let estimated_bytes = side * side * 48;
        assert_eq!(
            call(estimated_bytes - 1).unwrap_err(),
            Error::Pedigree(PgError::MemoryBudgetExceeded {
                operation: "within_person",
                estimated_bytes,
                budget_bytes: estimated_bytes - 1,
            })
        );
        assert!(call(estimated_bytes).unwrap().within_person.is_some());
    }

    struct Fixture {
        ids: Vec<i64>,
        mother: Vec<i64>,
        father: Vec<i64>,
        stratum: Vec<f64>,
        values: Vec<f64>,
    }

    /// The stratified fixture: 40 families of four; a child's stratum
    /// cycles through 10, 11, 12 within its family, a mother's is 0 or 1,
    /// fathers' and a few others' are unknown, and some values are missing.
    fn stratified_fixture(kind: TraitKind) -> Fixture {
        let (ids, mother, father) = sibships(40, 4);
        let stratum: Vec<f64> = (0..ids.len())
            .map(|i| {
                let (f, slot) = (i / 6, i % 6);
                match slot {
                    0 if f % 5 == 4 => NA,
                    0 => (f % 2) as f64,
                    1 => NA,
                    _ if i % 11 == 0 => NA,
                    _ => 10.0 + ((f + slot) % 3) as f64,
                }
            })
            .collect();
        let values = (0..ids.len())
            .map(|i| match (i % 17, kind) {
                (5, _) => NA,
                (_, TraitKind::Binary) => f64::from(u8::from(noise(i) < 0.45)),
                _ => noise(i) + 0.3 * noise(i / 6),
            })
            .collect();
        Fixture {
            ids,
            mother,
            father,
            stratum,
            values,
        }
    }

    /// MO's (child, mother) rows, or FS's sibs in both orders.
    fn ordered_pairs(mother: &[i64], symmetric: bool) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for (child, &m) in mother.iter().enumerate() {
            if m < 0 {
                continue;
            }
            if !symmetric {
                pairs.push((child, m as usize));
                continue;
            }
            for (sib, &m2) in mother.iter().enumerate() {
                if m2 == m && sib != child {
                    pairs.push((child, sib));
                }
            }
        }
        pairs
    }

    /// The complete pairs with both strata known, as (first, second) rows:
    /// MO's (child, mother), or FS's sibs in both orders.
    fn known_pairs(
        mother: &[i64],
        stratum: &[f64],
        values: &[f64],
        symmetric: bool,
    ) -> Vec<(usize, usize)> {
        let kept = |r: usize| !stratum[r].is_nan() && !values[r].is_nan();
        let mut pairs = ordered_pairs(mother, symmetric);
        pairs.retain(|&(a, b)| kept(a) && kept(b));
        pairs
    }

    /// The dense code of each known stratum label, ascending.
    fn code_of(stratum: &[f64]) -> impl Fn(usize) -> usize + '_ {
        let mut labels: Vec<i64> = stratum
            .iter()
            .filter(|s| !s.is_nan())
            .map(|&s| s as i64)
            .collect();
        labels.sort_unstable();
        labels.dedup();
        move |r| labels.binary_search(&(stratum[r] as i64)).unwrap()
    }

    #[test]
    fn binary_stratified_is_the_polychoric_fit_of_the_pairs_strata_table() {
        use crate::polychoric::table::Grid;
        use crate::polychoric::Tables;
        let Fixture {
            ids,
            mother,
            father,
            stratum,
            values: v,
        } = stratified_fixture(TraitKind::Binary);
        let ped = (&ids[..], &mother[..], &father[..]);
        let res = run_strata(ped, &v, TraitKind::Binary, &stratum, None).unwrap();
        let code = code_of(&stratum);
        let k = 5;
        for (name, symmetric) in [("MO", false), ("FS", true)] {
            let cell = cell(&res, name);
            assert_partition(cell);
            let mut table = Table::zeros(k, k, 2, 2);
            let pairs = known_pairs(&mother, &stratum, &v, symmetric);
            for &(a, b) in &pairs {
                let at = table.index(code(a), code(b), v[a] as usize, v[b] as usize);
                table.data[at] += 1.0;
            }
            let shown = |margin: Grid<f64>| Grid {
                rows: margin.rows,
                cols: margin.cols,
                data: margin.data.iter().map(|&m| m > 0.0).collect(),
            };
            let (rows, cols) = (shown(table.row_margin()), shown(table.col_margin()));
            let point = Tables::from_counts(table, &rows, &cols).unwrap().fit(0.0);
            let stratified = cell.stratified.unwrap();
            let halve = if symmetric { 2 } else { 1 };
            assert_eq!(stratified.n, pairs.len() as u64 / halve, "{name}");
            assert_eq!(stratified.n_degenerate_stratum, 0, "{name}");
            assert_eq!(stratified.estimate.value, Ok(point.value), "{name}");
            assert_eq!(stratified.estimate.boundary, point.boundary, "{name}");
        }
    }

    /// Pearson of the pairs' values, `x` of the first and `y` of the
    /// second, each side standardised (mean and `1/N` variance) within its
    /// stratum: two passes in `f64`.
    fn stratified_oracle(
        pairs: &[(usize, usize)],
        x: &[f64],
        y: &[f64],
        code: impl Fn(usize) -> usize,
        k: usize,
    ) -> f64 {
        let side = |pick: &dyn Fn(&(usize, usize)) -> usize, v: &[f64]| {
            let (mut n, mut sum) = (vec![0.0; k], vec![0.0; k]);
            for p in pairs {
                let r = pick(p);
                n[code(r)] += 1.0;
                sum[code(r)] += v[r];
            }
            let mean: Vec<f64> = sum.iter().zip(&n).map(|(s, n)| s / n).collect();
            let mut ss = vec![0.0; k];
            for p in pairs {
                let r = pick(p);
                ss[code(r)] += (v[r] - mean[code(r)]).powi(2);
            }
            let sd: Vec<f64> = ss.iter().zip(&n).map(|(s, n)| (s / n).sqrt()).collect();
            pairs
                .iter()
                .map(|p| {
                    let r = pick(p);
                    (v[r] - mean[code(r)]) / sd[code(r)]
                })
                .collect::<Vec<f64>>()
        };
        let (z1, z2) = (side(&|p| p.0, x), side(&|p| p.1, y));
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        dot(&z1, &z2) / (dot(&z1, &z1) * dot(&z2, &z2)).sqrt()
    }

    #[test]
    fn continuous_stratified_is_the_pearson_of_values_standardised_per_stratum() {
        let Fixture {
            ids,
            mother,
            father,
            stratum,
            values: v,
        } = stratified_fixture(TraitKind::Continuous);
        let ped = (&ids[..], &mother[..], &father[..]);
        let res = run_strata(ped, &v, TraitKind::Continuous, &stratum, None).unwrap();
        for (name, symmetric) in [("MO", false), ("FS", true)] {
            let cell = cell(&res, name);
            assert_partition(cell);
            let pairs = known_pairs(&mother, &stratum, &v, symmetric);
            let stratified = cell.stratified.unwrap();
            let halve = if symmetric { 2 } else { 1 };
            assert_eq!(stratified.n, pairs.len() as u64 / halve, "{name}");
            let expected = stratified_oracle(&pairs, &v, &v, code_of(&stratum), 5);
            let r = stratified.estimate.value.unwrap();
            assert!((r - expected).abs() < 1e-12, "{name}: {r} {expected}");
            // Standardising per stratum moves r off the pooled Pearson.
            assert!(
                (r - cell.estimates[0].value.unwrap()).abs() > 1e-6,
                "{name}"
            );
        }
        for cell in &res.cells {
            assert_partition(cell);
        }
    }

    /// The stratified fits' SE against brute force on the entries with
    /// both strata known: each entry's influence from an independent fit
    /// (the Polychoric Fit's cell influence; assortative's per-pair Pearson
    /// influence), and the meat summed over every two entries that share a
    /// member or the MRCA, here the mother.
    #[test]
    fn the_stratified_fits_se_is_the_brute_force_cluster_sum() {
        use crate::assortative::{pearson_influence, InfluenceSide};
        use crate::polychoric::table::Grid;
        use crate::polychoric::{cell_influence, Tables};
        let k = 5;
        for kind in [TraitKind::Binary, TraitKind::Continuous] {
            let Fixture {
                ids,
                mother,
                father,
                stratum,
                values: v,
            } = stratified_fixture(kind);
            let ped = (&ids[..], &mother[..], &father[..]);
            let res = run_strata_se(ped, &v, kind, &stratum, None, true).unwrap();
            let code = code_of(&stratum);
            for (name, symmetric) in [("MO", false), ("FS", true)] {
                let entries = known_pairs(&mother, &stratum, &v, symmetric);
                let phi: Vec<f64> = if kind == TraitKind::Binary {
                    let mut table = Table::zeros(k, k, 2, 2);
                    for &(a, b) in &entries {
                        let at = table.index(code(a), code(b), v[a] as usize, v[b] as usize);
                        table.data[at] += 1.0;
                    }
                    let shown = |margin: Grid<f64>| Grid {
                        rows: margin.rows,
                        cols: margin.cols,
                        data: margin.data.iter().map(|&m| m > 0.0).collect(),
                    };
                    let (rows, cols) = (shown(table.row_margin()), shown(table.col_margin()));
                    let tables = Tables::from_counts(table, &rows, &cols).unwrap();
                    let point = tables.fit(0.0);
                    let influence = cell_influence(&tables, point.value).unwrap();
                    entries
                        .iter()
                        .map(|&(a, b)| influence.at(code(a), code(b), v[a] as usize, v[b] as usize))
                        .collect()
                } else {
                    let side = |pick: fn(&(usize, usize)) -> usize| {
                        let rows: Vec<usize> = entries.iter().map(pick).collect();
                        let x: Vec<f64> = rows.iter().map(|&r| v[r]).collect();
                        let s: Vec<usize> = rows.iter().map(|&r| code(r)).collect();
                        let (mut n, mut sum, mut ss) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
                        for (&x, &s) in x.iter().zip(&s) {
                            n[s] += 1.0;
                            sum[s] += x;
                        }
                        let mean: Vec<f64> = sum.iter().zip(&n).map(|(s, n)| s / n).collect();
                        for (&x, &s) in x.iter().zip(&s) {
                            ss[s] += (x - mean[s]).powi(2);
                        }
                        let var: Vec<f64> = ss.iter().zip(&n).map(|(s, n)| s / n).collect();
                        (x, s, mean, var)
                    };
                    let ((xa, sa, ma, va), (xb, sb, mb, vb)) = (side(|p| p.0), side(|p| p.1));
                    pearson_influence(
                        InfluenceSide {
                            x: &xa,
                            stratum: &sa,
                            mean: &ma,
                            var: &va,
                        },
                        InfluenceSide {
                            x: &xb,
                            stratum: &sb,
                            mean: &mb,
                            var: &vb,
                        },
                        &vec![1.0; entries.len()],
                    )
                };
                let mut meat = 0.0;
                for (e, &(a, b)) in entries.iter().enumerate() {
                    for (f, &(c, d)) in entries.iter().enumerate() {
                        if a == c || a == d || b == c || b == d || mother[a] == mother[c] {
                            meat += phi[e] * phi[f];
                        }
                    }
                }
                let stratified = cell(&res, name).stratified.unwrap();
                assert_eq!(stratified.n_degenerate_stratum, 0, "{kind:?} {name}");
                let se = stratified.estimate.se.unwrap().unwrap();
                assert!(
                    (se * se - meat).abs() <= 1e-10 * meat,
                    "{kind:?} {name}: {} {}",
                    se * se,
                    meat
                );
            }
        }
    }

    #[test]
    fn a_meat_within_rounding_of_zero_is_degenerate_and_one_below_it_negative() {
        let at = |value: f64, scale, entries| {
            SeReason::of_meat(Meat {
                value,
                scale,
                entries,
            })
        };
        let floor = DEGENERATE_FLOOR * f64::EPSILON;
        for entries in [0, 1, 1 << 16] {
            assert_eq!(degenerate(entries), floor);
            assert_eq!(at(floor, 1.0, entries), Err(SeReason::DegenerateMeat));
            assert_eq!(at(-floor, 1.0, entries), Err(SeReason::DegenerateMeat));
            assert_eq!(at(2.0 * floor, 1.0, entries), Ok((2.0 * floor).sqrt()));
            assert_eq!(at(-2.0 * floor, 1.0, entries), Err(SeReason::NegativeMeat));
        }
        // 4 √(2^20) = 2^12: a million entries' rounding passes the floor.
        assert_eq!(degenerate(1 << 20), 4.0 * floor);
        assert_eq!(at(2.0 * floor, 1.0, 1 << 20), Err(SeReason::DegenerateMeat));
        assert_eq!(
            at(-4.0 * floor, 1.0, 1 << 20),
            Err(SeReason::DegenerateMeat)
        );
        assert_eq!(at(5.0 * floor, 1.0, 1 << 20), Ok((5.0 * floor).sqrt()));
        assert_eq!(at(0.0, 0.0, 0), Err(SeReason::DegenerateMeat));
        assert_eq!(at(0.25, 0.5, 9), Ok(0.5));
        assert_eq!(at(f64::NAN, 1.0, 9), Err(SeReason::SandwichUndefined));
    }

    #[test]
    fn a_meat_of_rounding_noise_or_below_zero_has_no_se() {
        // FS: sibs 4-6 share one MRCA, so their influences cancel inside
        // it.  MO: mother 2's entry as daughter of 0 cancels her four as
        // mother.  GP: the grandchildren's sums net to less than their
        // per-grandparent parts.
        let (ids, mother, father) = family();
        let ped = (&ids[..], &mother[..], &father[..]);
        let one_stratum = [0.0; 9];
        for (kind, values) in [
            (
                TraitKind::Continuous,
                [0.3, 1.2, -0.5, 0.9, 0.1, 2.0, -1.1, 0.4, 0.7],
            ),
            (
                TraitKind::Binary,
                [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            ),
        ] {
            let res = run_strata_se(ped, &values, kind, &one_stratum, None, true).unwrap();
            let reasons: Vec<(&str, Option<&str>)> = res
                .cells
                .iter()
                .filter(|c| c.n > 0)
                .map(|c| {
                    let estimate = c.estimates.last().unwrap();
                    let se = estimate.se.unwrap();
                    (c.category.as_str(), se.err().map(SeReason::name))
                })
                .collect();
            assert_eq!(
                reasons,
                [
                    ("MO", Some("degenerate_meat")),
                    ("FO", None),
                    ("FS", Some("degenerate_meat")),
                    ("MHS", Some("degenerate_meat")),
                    ("GP", Some("negative_meat")),
                ],
                "{kind:?}"
            );
        }
    }

    /// The SE's reserve ([`influence::reserve`]: the ancestry, the
    /// influences and one sweep) is charged in full before the pass: one
    /// byte short of the engine's one lane and the reserve is refused with
    /// exactly their sum, and the same plus a MiB for the store's rows gives
    /// the SEs of an ample budget.
    #[test]
    fn the_se_reserve_is_held_back_from_the_budget() {
        let Fixture {
            ids,
            mother,
            father,
            stratum,
            values,
        } = stratified_fixture(TraitKind::Continuous);
        let labels: Vec<i64> = stratum
            .iter()
            .map(|&s| if s.is_nan() { 0 } else { s as i64 })
            .collect();
        let known: Vec<bool> = stratum.iter().map(|s| !s.is_nan()).collect();
        let strata = Strata {
            labels: &labels,
            known: &known,
        };
        let trait_ = Trait {
            values: &values,
            kind: TraitKind::Continuous,
            n_levels: None,
        };
        let input = PedigreeInput {
            ids: &ids,
            mother: &mother,
            father: &father,
            twin: None,
            sex: None,
        };
        let pedigree = PedigreeArg::Columns(input).resolve().unwrap();
        let ancestry = Ancestry::of(&pedigree, 1 << 30).unwrap().bytes();
        assert_eq!(ancestry, 8 * ids.len() as u64);
        let run = |budget: u64| {
            correlation(
                PedigreeArg::Columns(input),
                &[trait_],
                Some(strata),
                None,
                budget as i64,
                true,
            )
        };
        let (codes, strata_labels) = strata.codes();
        let members = Members::of(&[trait_])
            .unwrap()
            .stratified(&codes, strata_labels.len() + 1);
        let requests = default_requests();
        let union: CategorySet = requests.iter().flat_map(|r| r.members.clone()).collect();
        let reserve = |budget| {
            influence::reserve(
                members.layout,
                &requests,
                Some(strata_labels.len()),
                ids.len(),
                budget,
                ancestry,
            )
            .0
        };
        let one_lane = pedigree_graph_core::relationships::MomentsPlan::new(
            members.input().shape(union),
            NonZeroUsize::MIN,
            1 << 30,
        )
        .unwrap()
        .estimated_peak_bytes;
        let fixed = one_lane + reserve(1 << 30);
        assert_eq!(reserve(fixed - 1), reserve(1 << 30));
        assert!(reserve(fixed) > 16 * ids.len() as u64);
        match run(fixed - 1) {
            Err(Error::Pedigree(PgError::MemoryBudgetExceeded {
                operation,
                estimated_bytes,
                budget_bytes,
            })) => {
                assert_eq!(operation, "relative_pairs_se");
                assert_eq!((estimated_bytes, budget_bytes), (fixed, fixed - 1));
            }
            other => panic!("{other:?}"),
        }
        let ample = run(1 << 30).unwrap();
        assert_eq!(run(fixed + (1 << 20)).unwrap().cells, ample.cells);
    }

    #[test]
    fn dropping_a_constant_stratum_can_make_another_constant() {
        // Five families of two (mother, unknown father, two children).
        // Mothers of stratum 0 are all 1, so column 0 goes first; child
        // stratum 2 then shows only 1 and goes next; block (3, 1) is left.
        let (ids, mother, father) = sibships(5, 2);
        let nan = NA;
        #[rustfmt::skip]
        let stratum = [
            0.0, nan, 2.0, 2.0,
            1.0, nan, 2.0, 2.0,
            1.0, nan, 3.0, 3.0,
            1.0, nan, 3.0, 3.0,
            1.0, nan, 3.0, 3.0,
        ];
        #[rustfmt::skip]
        let v = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 1.0,
        ];
        let ped = (&ids[..], &mother[..], &father[..]);
        let mo = [code("MO")];
        // Block (3, 1), child by mother: (0, 1), (1, 1), (1, 0), (0, 0), (1, 1), (1, 1).
        let left = [[1, 1], [1, 3]];
        let (x, y) = (
            [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        );
        for kind in [TraitKind::Binary, TraitKind::Continuous] {
            let res = run_strata(ped, &v, kind, &stratum, Some(&mo)).unwrap();
            let stratified = res.cells[0].stratified.unwrap();
            assert_eq!((stratified.n, stratified.n_degenerate_stratum), (6, 4));
            let r = stratified.estimate.value.unwrap();
            let expected = if kind == TraitKind::Binary {
                point(left).unwrap().value
            } else {
                pearson(&x, &y)
            };
            assert!((r - expected).abs() < 1e-12, "{kind:?}: {r} {expected}");
        }
        // Every stratum constant: nothing is left.
        let ones = [1.0; 20];
        for (kind, reason) in [
            (TraitKind::Binary, Reason::ConstantMargin),
            (TraitKind::Continuous, Reason::ConstantValue),
        ] {
            let res = run_strata(ped, &ones, kind, &stratum, Some(&mo)).unwrap();
            let stratified = res.cells[0].stratified.unwrap();
            assert_eq!((stratified.n, stratified.n_degenerate_stratum), (0, 10));
            assert_eq!(stratified.estimate.value, Err(reason));
        }
        // No pair with both strata known.
        let res = run_strata(ped, &v, TraitKind::Binary, &stratum, Some(&[code("FO")])).unwrap();
        let stratified = res.cells[0].stratified.unwrap();
        assert_eq!((stratified.n, stratified.n_degenerate_stratum), (0, 0));
        assert_eq!(stratified.estimate.value, Err(Reason::NoCompletePairs));
        assert_eq!(stratified.estimate.estimator, Estimator::Tetrachoric);
    }

    #[test]
    fn strata_are_checked_after_the_trait_and_before_the_request() {
        let (ids, mother, father) = family();
        let ped = (&ids[..], &mother[..], &father[..]);
        let bad = [code("XX")];
        let short = [0.0; 8];
        let err = run_strata(ped, &[0.0; 8], TraitKind::Binary, &short, Some(&bad)).unwrap_err();
        assert_eq!(err.code(), "trait_length_mismatch");
        let err = run_strata(ped, &[0.0; 9], TraitKind::Binary, &short, Some(&bad)).unwrap_err();
        assert_eq!(err.code(), "stratum_length_mismatch");
        let err = run_strata(ped, &[0.0; 9], TraitKind::Binary, &[0.0; 9], Some(&bad)).unwrap_err();
        assert_eq!(err.code(), "unknown_category");
    }

    /// [`correlation`] of `traits`, each of `kind`, on `ped`, with one
    /// stratum per row (`NaN` unknown) when given.
    fn run_traits(
        ped: (&[i64], &[i64], &[i64]),
        traits: &[&[f64]],
        kind: TraitKind,
        stratum: Option<&[f64]>,
        categories: Option<&[Entry]>,
    ) -> Result<RelativePairCorrelation, Error> {
        let input = PedigreeInput {
            ids: ped.0,
            mother: ped.1,
            father: ped.2,
            twin: None,
            sex: None,
        };
        let traits: Vec<Trait> = traits
            .iter()
            .map(|&values| Trait {
                values,
                kind,
                n_levels: None,
            })
            .collect();
        let labels: Option<Vec<i64>> = stratum.map(|s| {
            s.iter()
                .map(|&l| if l.is_nan() { 0 } else { l as i64 })
                .collect()
        });
        let known: Option<Vec<bool>> = stratum.map(|s| s.iter().map(|l| !l.is_nan()).collect());
        let strata = labels
            .as_deref()
            .zip(known.as_deref())
            .map(|(labels, known)| Strata { labels, known });
        let categories = categories.map(numbered);
        correlation(
            PedigreeArg::Columns(input),
            &traits,
            strata,
            categories.as_deref(),
            DEFAULT_MEMORY_BUDGET_BYTES,
            false,
        )
    }

    fn cell_of<'a>(
        res: &'a RelativePairCorrelation,
        name: &str,
        first: usize,
        second: usize,
    ) -> &'a Cell {
        res.cells
            .iter()
            .find(|c| c.category == name && (c.first_trait, c.second_trait) == (first, second))
            .unwrap()
    }

    /// A second trait on the stratified fixture: a family effect shared
    /// with the first, and its own missing rows.
    fn second_trait(kind: TraitKind, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let liability = noise(i + 7) + 0.4 * noise(i / 6);
                match (i % 13, kind) {
                    (3, _) => NA,
                    (_, TraitKind::Binary) => f64::from(u8::from(liability < 0.65)),
                    _ => liability,
                }
            })
            .collect()
    }

    /// The double-entered cross list of unordered `pairs`: each
    /// orientation (a, b), trait `x` of a against `y` of b, where both are
    /// observed.
    fn cross_entries(pairs: &[(usize, usize)], x: &[f64], y: &[f64]) -> Vec<(usize, usize)> {
        pairs
            .iter()
            .flat_map(|&(a, b)| [(a, b), (b, a)])
            .filter(|&(a, b)| !x[a].is_nan() && !y[b].is_nan())
            .collect()
    }

    fn table_of(entries: &[(usize, usize)], x: &[f64], y: &[f64]) -> [[u64; 2]; 2] {
        let mut table = [[0u64; 2]; 2];
        for &(a, b) in entries {
            table[x[a] as usize][y[b] as usize] += 1;
        }
        table
    }

    #[test]
    fn two_traits_same_trait_cells_are_the_single_trait_cells() {
        for kind in [TraitKind::Binary, TraitKind::Continuous] {
            let Fixture {
                ids,
                mother,
                father,
                stratum,
                values: x,
            } = stratified_fixture(kind);
            let y = second_trait(kind, x.len());
            let ped = (&ids[..], &mother[..], &father[..]);
            for stratum in [None, Some(&stratum[..])] {
                let both = run_traits(ped, &[&x, &y], kind, stratum, None).unwrap();
                // Seven directed default categories and five symmetric.
                assert_eq!(both.cells.len(), 7 * 4 + 5 * 3);
                for (t, values) in [&x, &y].into_iter().enumerate() {
                    let one = run_traits(ped, &[values], kind, stratum, None).unwrap();
                    assert_eq!(one.within_person, None);
                    let same: Vec<Cell> = both
                        .cells
                        .iter()
                        .filter(|c| (c.first_trait, c.second_trait) == (t, t))
                        .map(|c| Cell {
                            first_trait: 0,
                            second_trait: 0,
                            ..c.clone()
                        })
                        .collect();
                    assert_eq!(same, one.cells, "{kind:?} trait {t}");
                    let exponent = |r: &RelativePairCorrelation, t: usize| {
                        r.quantization_exponents.as_ref().map(|e| e[t])
                    };
                    assert_eq!(exponent(&both, t), exponent(&one, 0));
                }
            }
        }
    }

    #[test]
    fn a_symmetric_cross_cell_is_the_double_entered_cross_list() {
        // Row 4's trait 1 is missing: sib pairs {4, 5} and {4, 6} have one
        // complete orientation, {5, 6} two.
        let x = [0.3, -1.2, 0.8, 2.0, 1.5, -0.4, 0.9, 0.1, 1.1];
        let y = [1.0, 0.2, -0.5, 0.7, NA, 1.3, -0.8, 0.6, 0.4];
        let (ids, mother, father) = family();
        let ped = (&ids[..], &mother[..], &father[..]);
        let fs = [code("FS")];
        let res = run_traits(ped, &[&x, &y], TraitKind::Continuous, None, Some(&fs)).unwrap();
        let views: Vec<_> = res
            .cells
            .iter()
            .map(|c| (c.first_trait, c.second_trait, c.n_entries))
            .collect();
        assert_eq!(views, [(0, 0, None), (1, 1, None), (0, 1, Some(4))]);
        let cross = &res.cells[2];
        let entries = cross_entries(&[(4, 5), (4, 6), (5, 6)], &x, &y);
        assert_eq!((cross.n, cross.n_dropped, entries.len()), (3, 0, 4));
        let (a, b): (Vec<f64>, Vec<f64>) = entries.iter().map(|&(i, j)| (x[i], y[j])).unzip();
        let r = cross.estimates[0].value.unwrap();
        assert!((r - pearson(&a, &b)).abs() < 1e-12, "{r}");
        assert_eq!(cross.counts, Counts::None);

        let x = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let y = [1.0, 0.0, 1.0, 1.0, NA, 1.0, 0.0, 0.0, 1.0];
        let entries = [code("FS"), code("MHS")];
        let res = run_traits(ped, &[&x, &y], TraitKind::Binary, None, Some(&entries)).unwrap();
        for (name, pairs) in [
            ("FS", [(4, 5), (4, 6), (5, 6)]),
            ("MHS", [(4, 7), (5, 7), (6, 7)]),
        ] {
            let cell = cell_of(&res, name, 0, 1);
            let entries = cross_entries(&pairs, &x, &y);
            let table = table_of(&entries, &x, &y);
            assert_eq!(cell.counts, Counts::Table(table), "{name}");
            assert_eq!((cell.n, cell.n_entries), (3, Some(entries.len() as u64)));
            assert_eq!(cell.estimates[0], tetrachoric(point(table)));
            assert_eq!(cell.estimates[1], phi(table));
            // The same-trait cells keep their Concordance.
            assert!(matches!(
                cell_of(&res, name, 1, 1).counts,
                Counts::Concordance(_)
            ));
        }
    }

    #[test]
    fn a_pair_counts_once_when_either_orientation_is_complete() {
        // Row 4 lacks trait 1, row 6 both: of the sibs, {4, 5} is complete
        // one way, {4, 6} and {5, 6} neither; of 7's half-sib pairs, {4, 7}
        // one way, {5, 7} both, {6, 7} neither.
        let x = [0.3, -1.2, 0.8, 2.0, 1.5, -0.4, NA, 0.1, 1.1];
        let y = [1.0, 0.2, -0.5, 0.7, NA, 1.3, NA, 0.6, 0.4];
        let (ids, mother, father) = family();
        let ped = (&ids[..], &mother[..], &father[..]);
        let entries = [code("FS"), code("MHS")];
        let res = run_traits(ped, &[&x, &y], TraitKind::Continuous, None, Some(&entries)).unwrap();
        let counts = |c: &Cell| (c.n, c.n_dropped, c.n_entries);
        assert_eq!(counts(cell_of(&res, "FS", 0, 1)), (1, 2, Some(1)));
        assert_eq!(counts(cell_of(&res, "MHS", 0, 1)), (2, 1, Some(3)));
        assert_eq!(counts(cell_of(&res, "FS", 1, 1)), (0, 3, None));
        assert_eq!(counts(cell_of(&res, "MHS", 0, 0)), (2, 1, None));
        // Stratum 30 holds 4, 5 and 7, 31 holds 6: each count is in one
        // place.
        let stratum = [10.0, 10.0, 20.0, NA, 30.0, 30.0, 31.0, 30.0, 20.0];
        let res = run_traits(
            ped,
            &[&x, &y],
            TraitKind::Continuous,
            Some(&stratum),
            Some(&entries),
        )
        .unwrap();
        for cell in &res.cells {
            assert_partition(cell);
        }
        let mhs = cell_of(&res, "MHS", 0, 1);
        let s30 = &mhs.strata.as_ref().unwrap()[2];
        assert_eq!(
            (s30.label, s30.n, s30.n_dropped, s30.n_entries),
            (30, 2, 0, Some(3))
        );
        assert_eq!((mhs.n_cross_stratum, mhs.n_unknown_stratum), (0, 0));
    }

    #[test]
    fn directed_cross_cells_read_each_role_at_its_own_trait() {
        let x = [0.3, -1.2, 0.8, 2.0, 1.5, -0.4, 0.9, 0.1, 1.1];
        let y = [1.0, 0.2, -0.5, 0.7, NA, 1.3, -0.8, 0.6, 0.4];
        let (ids, mother, father) = family();
        let ped = (&ids[..], &mother[..], &father[..]);
        let res = run_traits(
            ped,
            &[&x, &y],
            TraitKind::Continuous,
            None,
            Some(&[code("MO")]),
        )
        .unwrap();
        let views: Vec<_> = res
            .cells
            .iter()
            .map(|c| (c.first_trait, c.second_trait))
            .collect();
        assert_eq!(views, [(0, 0), (0, 1), (1, 0), (1, 1)]);
        // Offspring first: (2, 0), (4, 2), (5, 2), (6, 2), (7, 2).
        let pairs = [(2, 0), (4, 2), (5, 2), (6, 2), (7, 2)];
        let traits = [&x, &y];
        for (first, second, n) in [(0, 1, 5), (1, 0, 4)] {
            let cell = cell_of(&res, "MO", first, second);
            let (u, w) = (traits[first], traits[second]);
            let (a, b): (Vec<f64>, Vec<f64>) = pairs
                .iter()
                .filter(|&&(o, m)| !u[o].is_nan() && !w[m].is_nan())
                .map(|&(o, m)| (u[o], w[m]))
                .unzip();
            assert_eq!((cell.n, cell.n_dropped, cell.n_entries), (n, 5 - n, None));
            let r = cell.estimates[0].value.unwrap();
            assert!(
                (r - pearson(&a, &b)).abs() < 1e-12,
                "({first}, {second}): {r}"
            );
            assert_eq!(cell.roles, Some(["offspring", "mother"]));
        }
    }

    #[test]
    fn cross_trait_strata_partition_and_fit_each_orientation() {
        use crate::polychoric::table::Grid;
        use crate::polychoric::Tables;
        for kind in [TraitKind::Binary, TraitKind::Continuous] {
            let Fixture {
                ids,
                mother,
                father,
                stratum,
                values: x,
            } = stratified_fixture(kind);
            let y = second_trait(kind, x.len());
            let ped = (&ids[..], &mother[..], &father[..]);
            let res = run_traits(ped, &[&x, &y], kind, Some(&stratum), None).unwrap();
            for cell in &res.cells {
                assert_partition(cell);
            }
            let code = code_of(&stratum);
            let known = |r: usize| !stratum[r].is_nan();
            for (name, symmetric) in [("MO", false), ("FS", true)] {
                let cell = cell_of(&res, name, 0, 1);
                let entries: Vec<(usize, usize)> = ordered_pairs(&mother, symmetric)
                    .into_iter()
                    .filter(|&(a, b)| known(a) && known(b) && !x[a].is_nan() && !y[b].is_nan())
                    .collect();
                let mut unordered: Vec<(usize, usize)> =
                    entries.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
                unordered.sort_unstable();
                unordered.dedup();
                let stratified = cell.stratified.unwrap();
                assert_eq!(stratified.n, unordered.len() as u64, "{kind:?} {name}");
                assert_eq!(stratified.n_degenerate_stratum, 0, "{kind:?} {name}");
                let value = stratified.estimate.value.unwrap();
                let expected = if kind == TraitKind::Binary {
                    let mut table = Table::zeros(5, 5, 2, 2);
                    for &(a, b) in &entries {
                        let at = table.index(code(a), code(b), x[a] as usize, y[b] as usize);
                        table.data[at] += 1.0;
                    }
                    let shown = |margin: Grid<f64>| Grid {
                        rows: margin.rows,
                        cols: margin.cols,
                        data: margin.data.iter().map(|&m| m > 0.0).collect(),
                    };
                    let (rows, cols) = (shown(table.row_margin()), shown(table.col_margin()));
                    Tables::from_counts(table, &rows, &cols)
                        .unwrap()
                        .fit(0.0)
                        .value
                } else {
                    stratified_oracle(&entries, &x, &y, &code, 5)
                };
                assert!(
                    (value - expected).abs() < 1e-12,
                    "{kind:?} {name}: {value} {expected}"
                );
            }
            let fs = cell_of(&res, "FS", 0, 1);
            let entries = fs.n_entries.unwrap();
            assert!(fs.n < entries && entries < 2 * fs.n, "{kind:?}");
        }
    }

    #[test]
    fn stratified_n_entries_counts_the_complete_orientations_fitted() {
        for kind in [TraitKind::Binary, TraitKind::Continuous] {
            let Fixture {
                ids,
                mother,
                father,
                stratum,
                values: mut x,
            } = stratified_fixture(kind);
            let y = second_trait(kind, x.len());
            let ped = (&ids[..], &mother[..], &father[..]);
            let at = code_of(&stratum);
            let known = |r: usize| !stratum[r].is_nan();
            let complete = |x: &[f64], a: usize, b: usize| !x[a].is_nan() && !y[b].is_nan();
            let one_way = ordered_pairs(&mother, true).into_iter().any(|(a, b)| {
                a < b
                    && known(a)
                    && known(b)
                    && at(a) != at(b)
                    && complete(&x, a, b) != complete(&x, b, a)
            });
            assert!(one_way, "{kind:?}: no cross-stratum pair complete one way");
            // Stratum 11's trait 0 made constant: its blocks as first
            // member are left out.
            let degenerate = at(stratum.iter().position(|&s| s == 11.0).unwrap());
            for (r, v) in x.iter_mut().enumerate() {
                if known(r) && at(r) == degenerate && !v.is_nan() {
                    *v = 1.0;
                }
            }
            let res =
                run_traits(ped, &[&x, &y], kind, Some(&stratum), Some(&[code("FS")])).unwrap();
            let kept: Vec<(usize, usize)> = ordered_pairs(&mother, true)
                .into_iter()
                .filter(|&(a, b)| known(a) && known(b) && at(a) != degenerate && complete(&x, a, b))
                .collect();
            let mut pairs: Vec<(usize, usize)> =
                kept.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
            pairs.sort_unstable();
            pairs.dedup();
            let cross = cell_of(&res, "FS", 0, 1).stratified.unwrap();
            assert!(cross.n_degenerate_stratum > 0, "{kind:?}");
            assert!(cross.estimate.value.is_ok(), "{kind:?}");
            assert_eq!(
                (cross.n, cross.n_entries),
                (pairs.len() as u64, Some(kept.len() as u64)),
                "{kind:?}"
            );
            for (first, second) in [(0, 0), (1, 1)] {
                let same = cell_of(&res, "FS", first, second).stratified.unwrap();
                assert_eq!(same.n_entries, None);
            }
            let res =
                run_traits(ped, &[&x, &y], kind, Some(&stratum), Some(&[code("MO")])).unwrap();
            for (first, second) in [(0, 1), (1, 0)] {
                let directed = cell_of(&res, "MO", first, second).stratified.unwrap();
                assert_eq!(directed.n_entries, None);
            }
            let within = res.within_person.unwrap().stratified.unwrap();
            assert_eq!(within.n_entries, None);
        }
    }

    #[test]
    fn within_person_is_each_rows_trait_0_against_its_trait_1() {
        use crate::polychoric::table::Grid;
        use crate::polychoric::Tables;
        for kind in [TraitKind::Binary, TraitKind::Continuous] {
            let Fixture {
                ids,
                mother,
                father,
                stratum,
                values: x,
            } = stratified_fixture(kind);
            let y = second_trait(kind, x.len());
            let ped = (&ids[..], &mother[..], &father[..]);
            let complete = |r: &usize| !x[*r].is_nan() && !y[*r].is_nan();
            let rows: Vec<usize> = (0..x.len()).filter(complete).collect();
            let expected = |rows: &[usize]| -> (Counts, Vec<Estimate>) {
                let entries: Vec<(usize, usize)> = rows.iter().map(|&r| (r, r)).collect();
                if kind == TraitKind::Binary {
                    let table = table_of(&entries, &x, &y);
                    let estimates = vec![tetrachoric(point(table)), phi(table)];
                    (Counts::Table(table), estimates)
                } else {
                    let (a, b): (Vec<f64>, Vec<f64>) = rows.iter().map(|&r| (x[r], y[r])).unzip();
                    (Counts::None, vec![super::pearson(Ok(pearson(&a, &b)))])
                }
            };
            let close = |got: &[Estimate], want: &[Estimate]| {
                for (g, w) in got.iter().zip(want) {
                    assert_eq!(g.estimator, w.estimator);
                    let gap = g.value.unwrap() - w.value.unwrap();
                    assert!(gap.abs() < 1e-12, "{kind:?}: {gap}");
                }
            };

            let res = run_traits(ped, &[&x, &y], kind, None, Some(&[])).unwrap();
            assert!(res.cells.is_empty());
            let w = res.within_person.unwrap();
            assert_eq!(
                (w.category.as_str(), w.members.len(), w.symmetric, w.roles),
                ("within_person", 0, false, None)
            );
            assert_eq!((w.first_trait, w.second_trait, w.n_entries), (0, 1, None));
            assert_eq!(
                (w.n, w.n_dropped),
                (rows.len() as u64, (x.len() - rows.len()) as u64)
            );
            let (counts, estimates) = expected(&rows);
            assert_eq!(w.counts, counts);
            close(&w.estimates, &estimates);
            assert_eq!(w.strata, None);

            let res = run_traits(ped, &[&x, &y], kind, Some(&stratum), Some(&[])).unwrap();
            let w = res.within_person.unwrap();
            assert_partition(&w);
            let code = code_of(&stratum);
            let known: Vec<usize> = rows
                .iter()
                .copied()
                .filter(|&r| !stratum[r].is_nan())
                .collect();
            assert_eq!(w.n_cross_stratum, 0);
            assert_eq!(w.n_unknown_stratum, (rows.len() - known.len()) as u64);
            for (s, cell) in w.strata.as_ref().unwrap().iter().enumerate() {
                let in_s: Vec<usize> = known.iter().copied().filter(|&r| code(r) == s).collect();
                assert_eq!(cell.n, in_s.len() as u64, "{kind:?} stratum {}", cell.label);
                if in_s.len() > 2 {
                    let (counts, estimates) = expected(&in_s);
                    assert_eq!(cell.counts, counts);
                    if estimates.iter().all(|e| e.value.is_ok()) {
                        close(&cell.estimates, &estimates);
                    }
                }
            }
            let stratified = w.stratified.unwrap();
            assert_eq!(
                stratified.n + stratified.n_degenerate_stratum,
                known.len() as u64
            );
            let fitted: Vec<(usize, usize)> = known.iter().map(|&r| (r, r)).collect();
            let value = stratified.estimate.value.unwrap();
            let expected = if kind == TraitKind::Binary {
                let mut table = Table::zeros(5, 5, 2, 2);
                for &(r, _) in &fitted {
                    let at = table.index(code(r), code(r), x[r] as usize, y[r] as usize);
                    table.data[at] += 1.0;
                }
                let shown = |margin: Grid<f64>| Grid {
                    rows: margin.rows,
                    cols: margin.cols,
                    data: margin.data.iter().map(|&m| m > 0.0).collect(),
                };
                let (rows, cols) = (shown(table.row_margin()), shown(table.col_margin()));
                Tables::from_counts(table, &rows, &cols)
                    .unwrap()
                    .fit(0.0)
                    .value
            } else {
                stratified_oracle(&fitted, &x, &y, &code, 5)
            };
            assert_eq!(stratified.n_degenerate_stratum, 0, "{kind:?}");
            assert!(
                (value - expected).abs() < 1e-12,
                "{kind:?}: {value} {expected}"
            );
        }
    }

    #[test]
    fn traits_are_counted_then_admitted_then_matched_in_kind() {
        let (ids, mother, father) = family();
        let z = [0.0; 9];
        let short = [0.0; 8];
        let call = |traits: &[(&[f64], TraitKind)], stratum: Option<&[f64]>, m: &[i64]| {
            let input = PedigreeInput {
                ids: &ids,
                mother: m,
                father: &father,
                twin: None,
                sex: None,
            };
            let traits: Vec<Trait> = traits
                .iter()
                .map(|&(values, kind)| Trait {
                    values,
                    kind,
                    n_levels: None,
                })
                .collect();
            let known = stratum.map(|s| vec![true; s.len()]);
            let labels = stratum.map(|s| vec![0i64; s.len()]);
            let strata = labels
                .as_deref()
                .zip(known.as_deref())
                .map(|(labels, known)| Strata { labels, known });
            correlation(
                PedigreeArg::Columns(input),
                &traits,
                strata,
                Some(&[code("XX")]),
                DEFAULT_MEMORY_BUDGET_BYTES,
                false,
            )
            .unwrap_err()
        };
        let continuous = TraitKind::Continuous;
        assert!(matches!(call(&[], None, &mother[..8]), Error::Pedigree(_)));
        assert_eq!(
            call(&[], None, &mother),
            Error::TraitCount {
                actual: 0,
                minimum: 1,
                maximum: 2
            }
        );
        let three = [(&z[..], continuous); 3];
        assert_eq!(call(&three, None, &mother).code(), "trait_count");
        assert_eq!(
            call(&[(&z, continuous), (&short, continuous)], None, &mother),
            Error::TraitLength {
                field: "traits[1]",
                expected_length: 9,
                actual_length: 8
            }
        );
        assert_eq!(
            call(&[(&short, TraitKind::Binary)], None, &mother),
            Error::TraitLength {
                field: "trait",
                expected_length: 9,
                actual_length: 8
            }
        );
        let mut two = z;
        two[3] = 2.0;
        assert_eq!(
            call(
                &[(&z, continuous), (&two, TraitKind::Binary)],
                None,
                &mother
            )
            .code(),
            "invalid_trait_value"
        );
        let mixed = [(&z[..], continuous), (&z[..], TraitKind::Binary)];
        let expected = Error::MixedTraitKinds {
            kinds: [TraitKind::Continuous, TraitKind::Binary],
        };
        assert_eq!(call(&mixed, None, &mother), expected);
        assert_eq!(call(&mixed, Some(&short), &mother), expected);
        let same = [(&z[..], continuous), (&z[..], continuous)];
        assert_eq!(
            call(&same, Some(&short), &mother).code(),
            "stratum_length_mismatch"
        );
        assert_eq!(call(&same, None, &mother).code(), "unknown_category");
    }
}
