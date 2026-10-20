//! The ACE Fit: additive genetic (A), common environment (C) and unique
//! environment (E) shares solved from a Relative-Pair Correlation's cells.
//!
//! On the correlation scale (each trait's variance 1) a cell k of trait t
//! expects `r_k = 2 phi_k A_t + s_k C_t`, with `phi_k = 2^-(degree+1)` the
//! nominal kinship of its categories and `s_k = 1` for a Shared-Environment
//! Category, else 0.  Each trait's unknowns are the weighted least-squares
//! solution over its same-trait cells; `E = 1 - A - C`.  Two traits add the
//! cross solve over the cross-trait cells, `r12_k = 2 phi_k sA + s_k sC`,
//! and the Within-Person Cross-Trait Correlation, `r_wp = sA + sC + sE`;
//! then `rA = sA / sqrt(A_1 A_2)` and so on.  Pairs weights weigh a
//! same-trait cell by its `n` and a cross-trait cell by its complete
//! orientations, so each counts once: a symmetric cell's `n_entries`, a
//! directed cell's `n` (one orientation per cell).  Values are as solved,
//! with no constraint.
//!
//! Given the result's covariance `Σ` of the cells (ADR 0014), each solve is
//! linear in the cells it reads, `θ = G r` with `G = X̃⁺ T` (`T` the
//! whitening, `TᵀT` the weights, `X̃ = T X`), so a value's SE is
//! `sqrt(g Σ gᵀ)`, `g` its row of `G` over the cells; E and the cross-trait
//! correlations take the gradient through those rows (the delta method).
//! The rows of every solve index one `Σ`, so the trait and cross solves'
//! covariance enters `rA`, `rC` and `rE` with no special case.
//!
//! The function is pure over cell rows as a host holds them, so a host
//! passes the cells of the result it already has.

mod linear;

use crate::error::{Error, Position};
use crate::relative_pairs::{self, Covariance, Form, Pick};
use crate::value::Value;
use pedigree_graph_core::relationships::{Category, CategorySet};

/// The variance components solved for; E is always the remainder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Components {
    A,
    C,
    AC,
}

impl Components {
    fn a(self) -> bool {
        !matches!(self, Components::C)
    }

    fn c(self) -> bool {
        !matches!(self, Components::A)
    }

    fn names(self) -> &'static [&'static str] {
        match self {
            Components::A => &["A"],
            Components::C => &["C"],
            Components::AC => &["A", "C"],
        }
    }

    /// The components a host names, in either order; `None` for an empty,
    /// repeated or unknown name.
    pub fn parse(names: &[&str]) -> Option<Components> {
        match names {
            ["A"] => Some(Components::A),
            ["C"] => Some(Components::C),
            ["A", "C"] | ["C", "A"] => Some(Components::AC),
            _ => None,
        }
    }
}

/// Which crude estimate a pooled cell gives the fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimator {
    /// The cell's primary estimate (its first).
    Primary,
    /// The observed-scale phi, which binary cells carry.
    Phi,
}

impl Estimator {
    pub fn name(self) -> &'static str {
        match self {
            Estimator::Primary => "primary",
            Estimator::Phi => "phi",
        }
    }

    pub fn parse(name: &str) -> Option<Estimator> {
        [Estimator::Primary, Estimator::Phi]
            .into_iter()
            .find(|e| e.name() == name)
    }
}

/// Whether the fit reads each cell's pooled estimate or its stratified form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimate {
    Pooled,
    Stratified,
}

impl Estimate {
    pub fn name(self) -> &'static str {
        match self {
            Estimate::Pooled => "pooled",
            Estimate::Stratified => "stratified",
        }
    }

    pub fn parse(name: &str) -> Option<Estimate> {
        [Estimate::Pooled, Estimate::Stratified]
            .into_iter()
            .find(|e| e.name() == name)
    }
}

/// How the rows of a solve are weighted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weights {
    /// What entered each row's estimate, counted once.
    Pairs,
    /// Each row's `1 / Var(r_k)`, from the covariance's diagonal.
    InverseVariance,
    /// The inverse of the covariance of the rows that entered.
    Gls,
}

impl Weights {
    pub fn name(self) -> &'static str {
        match self {
            Weights::Pairs => "pairs",
            Weights::InverseVariance => "inverse_variance",
            Weights::Gls => "gls",
        }
    }

    pub fn parse(name: &str) -> Option<Weights> {
        [Weights::Pairs, Weights::InverseVariance, Weights::Gls]
            .into_iter()
            .find(|w| w.name() == name)
    }
}

/// Whose values a cell correlates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pairs {
    /// Relative Pairs of these categories (one, or a Category Group's).
    Relatives(Vec<Category>),
    /// One person's two traits.
    WithinPerson,
}

impl Pairs {
    /// A cell's pairs from its member codes, as a host holds them.
    ///
    /// # Errors
    ///
    /// `unknown_category` for a code that is not a Relationship Category's,
    /// at its position in `codes`.
    pub fn relatives(codes: &[String]) -> Result<Pairs, Error> {
        codes
            .iter()
            .enumerate()
            .map(|(i, code)| {
                Category::parse(code).ok_or_else(|| Error::UnknownCategory {
                    category: code.clone(),
                    position: i.into(),
                })
            })
            .collect::<Result<_, _>>()
            .map(Pairs::Relatives)
    }
}

/// One crude estimate of a cell, as a host holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct CrudeEstimate {
    pub estimator: relative_pairs::Estimator,
    /// The value, or the reason it is undefined.
    pub value: Result<f64, String>,
}

/// A cell's stratified form.
#[derive(Clone, Debug, PartialEq)]
pub struct StratifiedEstimate {
    pub value: Result<f64, String>,
    /// The unordered pairs that entered it.
    pub n: u64,
    /// A symmetric cross-trait cell's complete orientations that entered
    /// it; `None` elsewhere.
    pub n_entries: Option<u64>,
}

/// One cell of a Relative-Pair Correlation result.
#[derive(Clone, Debug, PartialEq)]
pub struct AceCell {
    /// The category's code, the group's name, or the within-person entry's.
    pub name: String,
    pub pairs: Pairs,
    /// The first and second member's trait; `(0, 0)` with one trait.
    pub first_trait: usize,
    pub second_trait: usize,
    /// The primary estimate first.
    pub crude: Vec<CrudeEstimate>,
    /// The unordered pairs that entered the pooled estimate.
    pub n: u64,
    /// A symmetric cross-trait cell's complete orientations that entered
    /// the pooled estimate; `None` elsewhere.
    pub n_entries: Option<u64>,
    pub stratified: Option<StratifiedEstimate>,
}

/// What to fit, as the caller asks.
#[derive(Clone, Debug, PartialEq)]
pub struct AceRequest {
    /// The entries to fit by name; `None` is every cell.  The within-person
    /// cell enters the cross solve either way.
    pub categories: Option<Vec<String>>,
    pub components: Components,
    /// The Shared-Environment Categories' codes; required with C.
    pub shared_environment: Option<Vec<String>>,
    pub estimator: Estimator,
    /// `None` is the default: [`Estimate::Stratified`] when any relatives
    /// cell carries a stratified form, else [`Estimate::Pooled`].
    pub estimate: Option<Estimate>,
    /// The weighting of every solve; `None` is the default (ADR 0014):
    /// [`Weights::Pairs`] for phi or without the covariance of the form
    /// read, else per fit [`Weights::Gls`] when the fit's entered rows'
    /// block of the covariance has a Cholesky factor and pairs when it has
    /// none.  Each fit records the weighting it used.
    pub weights: Option<Weights>,
}

/// Why a fitted value or its SE is undefined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The component is not among those requested.
    NotRequested,
    /// The cells that entered do not determine the solve's unknowns.
    NotIdentified,
    /// A cross-trait correlation's component is <= 0 in a trait.
    NonpositiveComponent,
    /// GLS: the covariance of the rows that entered has no Cholesky factor,
    /// so the solve's values and SEs are undefined.
    CovarianceNotPositiveDefinite,
    /// An SE without the result's covariance of the form read.
    NoCovariance,
    /// An SE that reads a row with no SE (pairs weights).
    RowWithoutSe,
    /// An SE whose `g Σ gᵀ` is negative: the covariance is not positive
    /// semidefinite in that direction.
    NegativeVariance,
}

impl Reason {
    pub fn name(self) -> &'static str {
        match self {
            Reason::NotRequested => "not_requested",
            Reason::NotIdentified => "not_identified",
            Reason::NonpositiveComponent => "nonpositive_component",
            Reason::CovarianceNotPositiveDefinite => "covariance_not_positive_definite",
            Reason::NoCovariance => "no_covariance",
            Reason::RowWithoutSe => "row_without_se",
            Reason::NegativeVariance => "negative_variance",
        }
    }
}

/// What a row's model coefficients come from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RowKind {
    Relatives {
        /// The nominal kinship `2^-(degree+1)`.
        kinship: f64,
        /// Whether it is a Shared-Environment Category; `None` when the
        /// call declared none.
        shared: Option<bool>,
    },
    WithinPerson,
}

/// One candidate cell of the fit.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub entry: String,
    pub first_trait: usize,
    pub second_trait: usize,
    pub kind: RowKind,
    /// The value the fit reads; an undefined one names why, and the row is
    /// left out.  A solve weighted by the covariance leaves out a row with
    /// no `variance` as `no_se`.
    pub value: Result<f64, String>,
    /// The estimate's `n_entries` where it has one (a symmetric
    /// cross-trait cell's complete orientations), else its `n`.
    pub weight: u64,
    /// The estimate's variance, the covariance's diagonal entry; `None`
    /// without the covariance or where the estimate has no SE.
    pub variance: Option<f64>,
}

/// A solved value and its SE, or why the SE is undefined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solved {
    pub value: f64,
    pub se: Result<f64, Reason>,
}

/// One trait's shares.
#[derive(Clone, Debug, PartialEq)]
pub struct TraitFit {
    pub a: Result<Solved, Reason>,
    pub c: Result<Solved, Reason>,
    pub e: Result<Solved, Reason>,
    /// Indices into [`AceFit::rows`] of the rows that entered.
    pub rows: Vec<usize>,
    /// The weighting its solve used.
    pub weights: Weights,
}

/// The cross-trait covariances and correlations of two traits.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossFit {
    pub s_a: Result<Solved, Reason>,
    pub s_c: Result<Solved, Reason>,
    pub s_e: Result<Solved, Reason>,
    pub r_a: Result<Solved, Reason>,
    pub r_c: Result<Solved, Reason>,
    pub r_e: Result<Solved, Reason>,
    /// Indices into [`AceFit::rows`] of the rows that entered.
    pub rows: Vec<usize>,
    /// The weighting the cross solve used.
    pub weights: Weights,
}

/// The result of [`fit`].
#[derive(Clone, Debug, PartialEq)]
pub struct AceFit {
    /// One per trait.
    pub traits: Vec<TraitFit>,
    /// With two traits.
    pub cross: Option<CrossFit>,
    /// Every candidate cell, in the order given.
    pub rows: Vec<Row>,
    pub request: AceRequest,
    /// The estimate read: the request's, or its default.
    pub estimate: Estimate,
}

/// How the fit was computed, as the tree's `method` reports it.
fn method(request: &AceRequest) -> [(&'static str, &'static str); 6] {
    let weights = match request.weights {
        None => {
            "default: pairs for phi or without the result's covariance of the estimates read; \
             else gls for each fit (each trait's and the cross solve) whose entered rows' \
             covariance has a Cholesky factor, and pairs for a fit whose has none; each fit's \
             weights records the weighting it used"
        }
        Some(Weights::Pairs) => {
            "pairs: what entered the estimate read, the pooled cell or the stratified form: a \
             same-trait row's n (unordered pairs) and the within-person row's n (rows); a \
             cross-trait row's complete orientations, each once: a symmetric cell's n_entries, a \
             directed cell's n"
        }
        Some(Weights::InverseVariance) => {
            "inverse_variance: each row's 1 / Var(r_k), the diagonal of the result's covariance \
             of the estimates read; a row without an SE is left out (no_se)"
        }
        Some(Weights::Gls) => {
            "gls: the inverse of the result's covariance of the estimates of the rows that \
             entered a solve, through its Cholesky factor L (rows whitened by L^-1); a row \
             without an SE is left out (no_se); a covariance with no Cholesky factor leaves the \
             solve's values and SEs covariance_not_positive_definite"
        }
    };
    [
        (
            "model",
            "correlation scale, each trait's variance 1: r_k = 2 phi_k A + s_k C over a trait's \
             same-trait cells; with two traits r12_k = 2 phi_k sA + s_k sC over the cross-trait \
             cells and r_wp = sA + sC + sE within person. phi_k = 2^-(degree+1), the nominal \
             kinship; s_k = 1 for a Shared-Environment Category, else 0. Components not \
             requested are fixed at 0",
        ),
        ("weights", weights),
        (
            "solve",
            "least squares by Householder QR with column pivoting of the whitened design (each \
             row scaled by sqrt(weight), or under gls premultiplied by L^-1), rows sorted by \
             decreasing largest entry; a requested component zero in every row that entered is \
             not_identified and left out of the solve; in the cross solve, sC zero in every \
             relative row leaves sC and sE to the within-person row alone, so both are \
             not_identified and that row is left out; any other rank deficiency is \
             not_identified for the solve's unknowns; values unconstrained",
        ),
        ("E", "the remainder: E = 1 - A - C"),
        (
            "cross",
            "rA = sA / sqrt(A_1 A_2), rC = sC / sqrt(C_1 C_2), rE = sE / sqrt(E_1 E_2); \
             nonpositive_component when a trait's component is <= 0; |r| > 1 is flagged \
             out_of_range. sE needs the within-person cell",
        ),
        (
            "se",
            "G Sigma G', G the solve's linear map of the cells read and Sigma their covariance \
             (ADR 0014): a value's SE is sqrt(g Sigma g') over its row g; E and the cross-trait \
             correlations by the delta method through the stacked solves; weights treated as \
             fixed; none without the result's covariance (no_covariance); row_without_se when g \
             reads a row with no SE (pairs weights); negative_variance when g Sigma g' < 0",
        ),
    ]
}

/// The ACE Fit of `cells`, one cell row each, as `request` asks.
///
/// A cell whose selected value is undefined (`n` = 0 included) does not
/// enter; its row names the reason.  `covariance`, the result's
/// ([`crate::relative_pairs::RelativePairCorrelation::covariance`]), indexes
/// `cells` in order; without it every SE is `no_covariance`.
///
/// # Errors
///
/// `trait_count` (a trait index past 1); `no_stratified_form` (stratified
/// with phi); `missing_shared_environment` (C without
/// `shared_environment`); `unknown_category` (a `shared_environment` code);
/// `unknown_entry` (a `categories` name no cell has); then over the
/// entries that enter: `empty_category_group`, `mixed_kinship`,
/// `mixed_shared_environment`, `missing_estimator`; and
/// `no_stratified_form` when stratified and none has a stratified form;
/// then `covariance_shape_mismatch` (a covariance not over `cells`) and
/// `missing_covariance` (inverse-variance or GLS weights asked for without
/// the covariance of the form read).
pub fn fit(
    cells: &[AceCell],
    request: &AceRequest,
    covariance: Option<&Covariance>,
) -> Result<AceFit, Error> {
    let top = cells
        .iter()
        .map(|c| c.first_trait.max(c.second_trait))
        .max()
        .unwrap_or(0);
    if top > 1 {
        return Err(Error::TraitCount {
            actual: top + 1,
            minimum: 1,
            maximum: 2,
        });
    }
    let estimate = request.estimate.unwrap_or_else(|| {
        let stratified = cells
            .iter()
            .any(|c| c.pairs != Pairs::WithinPerson && c.stratified.is_some());
        if stratified {
            Estimate::Stratified
        } else {
            Estimate::Pooled
        }
    });
    let form = form(estimate, request.estimator).ok_or(Error::NoStratifiedForm {
        estimator: Estimator::Phi,
    })?;
    let shared = shared_set(request)?;
    if let Some(names) = &request.categories {
        for (i, name) in names.iter().enumerate() {
            if !cells.iter().any(|c| c.name == *name) {
                return Err(Error::UnknownEntry {
                    name: name.clone(),
                    position: i.into(),
                });
            }
        }
    }
    let candidates: Vec<(usize, &AceCell)> = cells
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            c.pairs == Pairs::WithinPerson
                || request
                    .categories
                    .as_ref()
                    .is_none_or(|n| n.contains(&c.name))
        })
        .collect();
    let mut rows = candidates
        .iter()
        .map(|&(i, c)| row(c, i, request.estimator, form, shared.as_ref()))
        .collect::<Result<Vec<Row>, Error>>()?;
    if estimate == Estimate::Stratified && candidates.iter().all(|(_, c)| c.stratified.is_none()) {
        return Err(Error::NoStratifiedForm {
            estimator: Estimator::Primary,
        });
    }
    let sigma = sigma(cells.len(), form, covariance)?;
    let rule = rule(request, sigma)?;
    for (row, &(i, _)) in rows.iter_mut().zip(&candidates) {
        row.variance = sigma.and_then(|s| s.variance(i));
    }
    let problem = Problem {
        rows: &rows,
        slots: candidates.iter().map(|&(i, _)| i).collect(),
        width: if sigma.is_some() { cells.len() } else { 0 },
    };
    let components = request.components;
    let parts: Vec<(TraitParts, Weighting)> = (0..=top)
        .map(|t| rule.run(|w| trait_fit(&problem, w, t, components), |p| &p.shares))
        .collect();
    let cross = (top == 1).then(|| {
        let traits: Vec<&TraitParts> = parts.iter().map(|(p, _)| p).collect();
        rule.run(
            |w| cross_fit(&problem, w, &traits, components),
            |p| &p.covariances,
        )
    });
    let solved_by = |row: &Row| match &cross {
        Some((_, w)) if in_cross(row) => *w,
        _ => parts[row.first_trait].1,
    };
    for row in &mut rows {
        if row.value.is_ok() && !takes(solved_by(row), row) {
            row.value = Err("no_se".to_string());
        }
    }
    let se = |q: Quantity| q.map(|l| l.solved(sigma));
    let traits = parts
        .into_iter()
        .map(|(p, w)| {
            let [a, c, e] = p.shares;
            TraitFit {
                a: se(a),
                c: se(c),
                e: se(e),
                rows: p.rows,
                weights: w.weights(),
            }
        })
        .collect();
    let cross = cross.map(|(p, w)| {
        let [s_a, s_c, s_e] = p.covariances;
        let [r_a, r_c, r_e] = p.correlations;
        CrossFit {
            s_a: se(s_a),
            s_c: se(s_c),
            s_e: se(s_e),
            r_a: se(r_a),
            r_c: se(r_c),
            r_e: se(r_e),
            rows: p.rows,
            weights: w.weights(),
        }
    });
    Ok(AceFit {
        traits,
        cross,
        rows,
        request: request.clone(),
        estimate,
    })
}

/// The covariance of the form a fit reads: `n x n`, row-major, by cell.
#[derive(Clone, Copy)]
struct Sigma<'a> {
    n: usize,
    matrix: &'a [f64],
}

impl Sigma<'_> {
    fn at(self, i: usize, j: usize) -> f64 {
        self.matrix[i * self.n + j]
    }

    /// Cell `i`'s variance; `None` where its estimate has no SE (NaN), or
    /// none a weight can take.
    fn variance(self, i: usize) -> Option<f64> {
        Some(self.at(i, i)).filter(|v| *v > 0.0)
    }
}

/// The weighting of a solve, with the covariance it reads.
#[derive(Clone, Copy)]
enum Weighting<'a> {
    Pairs,
    InverseVariance(Sigma<'a>),
    Gls(Sigma<'a>),
}

impl Weighting<'_> {
    fn weights(self) -> Weights {
        match self {
            Weighting::Pairs => Weights::Pairs,
            Weighting::InverseVariance(_) => Weights::InverseVariance,
            Weighting::Gls(_) => Weights::Gls,
        }
    }
}

/// How each fit's solve is weighted.
#[derive(Clone, Copy)]
enum Rule<'a> {
    /// The one weighting of every solve.
    Every(Weighting<'a>),
    /// The default with the covariance read: GLS, and pairs for a fit
    /// whose entered rows' covariance has no Cholesky factor.
    GlsElsePairs(Sigma<'a>),
}

impl<'a> Rule<'a> {
    /// A fit under this rule, with the weighting it used.  `unknowns` reads
    /// the fit's solved unknowns, which GLS refuses together.
    fn run<T>(
        self,
        fit: impl Fn(Weighting<'a>) -> T,
        unknowns: impl Fn(&T) -> &[Quantity; 3],
    ) -> (T, Weighting<'a>) {
        let weighting = match self {
            Rule::Every(weighting) => weighting,
            Rule::GlsElsePairs(sigma) => Weighting::Gls(sigma),
        };
        let out = fit(weighting);
        let refused = unknowns(&out)
            .iter()
            .any(|q| matches!(q, Err(Reason::CovarianceNotPositiveDefinite)));
        match self {
            Rule::GlsElsePairs(_) if refused => (fit(Weighting::Pairs), Weighting::Pairs),
            _ => (out, weighting),
        }
    }
}

/// The form a fit reads for `estimate` and `estimator`; phi has no
/// stratified form.
fn form(estimate: Estimate, estimator: Estimator) -> Option<Form> {
    match (estimate, estimator) {
        (Estimate::Pooled, Estimator::Primary) => Some(Form::Pooled),
        (Estimate::Pooled, Estimator::Phi) => Some(Form::Phi),
        (Estimate::Stratified, Estimator::Primary) => Some(Form::Stratified),
        (Estimate::Stratified, Estimator::Phi) => None,
    }
}

/// The covariance of `form`, when `covariance` has it.
///
/// # Errors
///
/// `covariance_shape_mismatch` for a covariance not over `cells` cells.
fn sigma<'a>(
    cells: usize,
    form: Form,
    covariance: Option<&'a Covariance>,
) -> Result<Option<Sigma<'a>>, Error> {
    if let Some(covariance) = covariance.filter(|c| c.n() != cells) {
        return Err(Error::CovarianceShapeMismatch {
            cells,
            expected_length: cells.saturating_mul(cells),
            actual_length: covariance.n().saturating_mul(covariance.n()),
        });
    }
    Ok(covariance
        .and_then(|c| c.of(form))
        .map(|matrix| Sigma { n: cells, matrix }))
}

/// The request's rule over the covariance of the form read.
///
/// # Errors
///
/// `missing_covariance` for weights other than pairs without that
/// covariance.
fn rule<'a>(request: &AceRequest, sigma: Option<Sigma<'a>>) -> Result<Rule<'a>, Error> {
    Ok(match (request.weights, sigma) {
        (None, Some(sigma)) if request.estimator == Estimator::Primary => Rule::GlsElsePairs(sigma),
        (None | Some(Weights::Pairs), _) => Rule::Every(Weighting::Pairs),
        (Some(Weights::InverseVariance), Some(sigma)) => {
            Rule::Every(Weighting::InverseVariance(sigma))
        }
        (Some(Weights::Gls), Some(sigma)) => Rule::Every(Weighting::Gls(sigma)),
        (Some(weights), None) => return Err(Error::MissingCovariance { weights }),
    })
}

/// The declared Shared-Environment Categories, required with C.
fn shared_set(request: &AceRequest) -> Result<Option<CategorySet>, Error> {
    let Some(codes) = &request.shared_environment else {
        if request.components.c() {
            return Err(Error::MissingSharedEnvironment);
        }
        return Ok(None);
    };
    let mut set = CategorySet::EMPTY;
    for (i, code) in codes.iter().enumerate() {
        let category = Category::parse(code).ok_or_else(|| Error::UnknownSharedEnvironment {
            category: code.clone(),
            position: Position::from(i),
        })?;
        set.insert(category);
    }
    Ok(Some(set))
}

/// A cell's candidate row: its coefficients and the value of `form` it
/// gives.
fn row(
    cell: &AceCell,
    position: usize,
    estimator: Estimator,
    form: Form,
    declared: Option<&CategorySet>,
) -> Result<Row, Error> {
    let kind = match &cell.pairs {
        Pairs::WithinPerson => RowKind::WithinPerson,
        Pairs::Relatives(members) => {
            let &[first, ..] = members.as_slice() else {
                return Err(Error::EmptyCategoryGroup {
                    group: cell.name.clone(),
                    position: position.into(),
                });
            };
            if let Some(&other) = members.iter().find(|m| m.degree() != first.degree()) {
                return Err(Error::MixedKinship {
                    group: cell.name.clone(),
                    categories: [first.code(), other.code()],
                });
            }
            if let Some(set) = declared {
                let mixed = members
                    .iter()
                    .find(|&&m| set.contains(m) != set.contains(first));
                if let Some(&other) = mixed {
                    return Err(Error::MixedSharedEnvironment {
                        group: cell.name.clone(),
                        categories: [first.code(), other.code()],
                    });
                }
            }
            RowKind::Relatives {
                kinship: 0.5f64.powi(i32::from(first.degree()) + 1),
                shared: declared.map(|set| set.contains(first)),
            }
        }
    };
    let (value, weight) = match form.pick(&cell.crude, |e| e.estimator) {
        Pick::Stratified => match &cell.stratified {
            Some(s) => (s.value.clone(), s.n_entries.unwrap_or(s.n)),
            None => (Err("no_stratified_form".to_string()), 0),
        },
        Pick::Pooled(estimate) => {
            let estimate = estimate.ok_or_else(|| Error::MissingEstimator {
                estimator,
                entry: cell.name.clone(),
            })?;
            (estimate.value.clone(), cell.n_entries.unwrap_or(cell.n))
        }
    };
    let value = match value {
        Ok(_) if weight == 0 => Err("no_complete_pairs".to_string()),
        other => other,
    };
    Ok(Row {
        entry: cell.name.clone(),
        first_trait: cell.first_trait,
        second_trait: cell.second_trait,
        kind,
        value,
        weight,
        variance: None,
    })
}

/// A value and its coefficients over the cells: its row of a solve's
/// linear map, or its gradient through such rows.
#[derive(Clone, Debug)]
struct Linear {
    value: f64,
    g: Vec<f64>,
    /// A bound on `g`'s rounding: an entry no larger is indistinguishable
    /// from an exact zero.  A solve's row has `m eps max|g|`, the bound
    /// [`linear::least_squares`] puts on its rank; E and the ratios carry
    /// their operands' bounds.  Within-person against sA is such a zero.
    noise: f64,
}

impl Linear {
    fn zero(cells: usize) -> Linear {
        Linear {
            value: 0.0,
            g: vec![0.0; cells],
            noise: 0.0,
        }
    }

    /// The value with its SE, `sqrt(g Σ gᵀ)` over `g`'s support: a cell
    /// with no SE is NaN in `Σ`, and `NaN * 0` would reach every value.
    fn solved(self, sigma: Option<Sigma<'_>>) -> Solved {
        let se = sigma.ok_or(Reason::NoCovariance).and_then(|sigma| {
            let support: Vec<usize> = (0..self.g.len())
                .filter(|&i| self.g[i].abs() > self.noise)
                .collect();
            let variance: f64 = support
                .iter()
                .map(|&i| {
                    let row: f64 = support.iter().map(|&j| sigma.at(i, j) * self.g[j]).sum();
                    self.g[i] * row
                })
                .sum();
            if variance.is_nan() {
                Err(Reason::RowWithoutSe)
            } else if variance < 0.0 {
                Err(Reason::NegativeVariance)
            } else {
                Ok(variance.sqrt())
            }
        });
        Solved {
            value: self.value,
            se,
        }
    }
}

/// A solved unknown or a function of solved unknowns, or why it is
/// undefined.
type Quantity = Result<Linear, Reason>;

/// What every solve of one fit reads.
struct Problem<'a> {
    rows: &'a [Row],
    /// Each row's cell: its index in the fit's `cells` and the covariance.
    slots: Vec<usize>,
    /// The length of each value's `g`: the cells with the covariance, 0
    /// without, when no SE reads it.
    width: usize,
}

/// A row-scaling `T` of a solve's rows, `TᵀT` its weights.
enum Whitening {
    Diagonal(Vec<f64>),
    /// Lower-triangular, row-major.
    Lower(Vec<f64>),
}

impl Whitening {
    /// `T v`.
    fn apply(&self, v: &[f64]) -> Vec<f64> {
        match self {
            Whitening::Diagonal(d) => d.iter().zip(v).map(|(d, v)| d * v).collect(),
            Whitening::Lower(t) => {
                let m = v.len();
                (0..m)
                    .map(|i| (0..=i).map(|j| t[i * m + j] * v[j]).sum())
                    .collect()
            }
        }
    }

    /// `T`'s columns.
    fn columns(&self, m: usize) -> impl Iterator<Item = Vec<f64>> + '_ {
        (0..m).map(move |j| match self {
            Whitening::Diagonal(d) => (0..m).map(|i| if i == j { d[j] } else { 0.0 }).collect(),
            Whitening::Lower(t) => (0..m).map(|i| t[i * m + j]).collect(),
        })
    }
}

impl Problem<'_> {
    /// The whitening of the rows `entered` under `weighting`.
    fn whitening(&self, weighting: Weighting, entered: &[usize]) -> Result<Whitening, Reason> {
        let slot = |i: usize| self.slots[entered[i]];
        Ok(match weighting {
            Weighting::Pairs => Whitening::Diagonal(
                entered
                    .iter()
                    .map(|&i| (self.rows[i].weight as f64).sqrt())
                    .collect(),
            ),
            Weighting::InverseVariance(sigma) => Whitening::Diagonal(
                (0..entered.len())
                    .map(|i| 1.0 / sigma.at(slot(i), slot(i)).sqrt())
                    .collect(),
            ),
            Weighting::Gls(sigma) => {
                let m = entered.len();
                let block: Vec<f64> = (0..m * m)
                    .map(|x| sigma.at(slot(x / m), slot(x % m)))
                    .collect();
                let l = linear::cholesky(&block, m).ok_or(Reason::CovarianceNotPositiveDefinite)?;
                Whitening::Lower(linear::lower_inverse(&l, m))
            }
        })
    }

    /// The least-squares solution under `weighting` over `entered` of the
    /// present `columns`, each present column's value and its row of
    /// `G = X̃⁺ T` scattered to the cells (with the covariance), in column
    /// order; or why there is none.  Pairs weights take the operations of
    /// the fit before the SE, so their values are the same bits.
    fn solve(
        &self,
        weighting: Weighting,
        entered: &[usize],
        columns: &Columns,
    ) -> Result<Vec<Linear>, Reason> {
        let p = columns.iter().filter(|c| c.is_ok()).count();
        let whitening = self.whitening(weighting, entered)?;
        let rows: Vec<&Row> = entered.iter().map(|&i| &self.rows[i]).collect();
        let x: Vec<Vec<f64>> = rows.iter().map(|r| coefficients(r.kind, columns)).collect();
        let design = (0..p)
            .map(|k| whitening.apply(&x.iter().map(|x| x[k]).collect::<Vec<f64>>()))
            .collect();
        let y: Vec<f64> = rows
            .iter()
            .map(|r| r.value.as_ref().map_or(f64::NAN, |v| *v))
            .collect();
        let mut rhs = vec![whitening.apply(&y)];
        if self.width > 0 {
            rhs.extend(whitening.columns(entered.len()));
        }
        let solutions = linear::least_squares(design, rhs).ok_or(Reason::NotIdentified)?;
        let Some((values, map)) = solutions.split_first() else {
            return Err(Reason::NotIdentified);
        };
        Ok((0..p)
            .map(|k| {
                let mut g = vec![0.0; self.width];
                for (column, &i) in map.iter().zip(entered) {
                    g[self.slots[i]] = column[k];
                }
                let largest = g.iter().fold(0.0, |a: f64, b| a.max(b.abs()));
                Linear {
                    value: values[k],
                    g,
                    noise: entered.len() as f64 * f64::EPSILON * largest,
                }
            })
            .collect())
    }
}

/// Each design column's presence: `Ok` to solve for it, else the reason
/// its unknown carries.  A, C, then E (the cross solve's sE).
type Columns = [Result<(), Reason>; 3];

fn requested(on: bool) -> Result<(), Reason> {
    if on {
        Ok(())
    } else {
        Err(Reason::NotRequested)
    }
}

/// A row's coefficients on A, C and E.
fn design(kind: RowKind) -> [f64; 3] {
    match kind {
        RowKind::Relatives { kinship, shared } => {
            let s = if shared == Some(true) { 1.0 } else { 0.0 };
            [2.0 * kinship, s, 0.0]
        }
        RowKind::WithinPerson => [1.0, 1.0, 1.0],
    }
}

/// A row's coefficients on the columns present.
fn coefficients(kind: RowKind, columns: &Columns) -> Vec<f64> {
    design(kind)
        .into_iter()
        .zip(columns)
        .filter(|(_, column)| column.is_ok())
        .map(|(x, _)| x)
        .collect()
}

/// Whether a solve under `weighting` takes a row whose value is defined:
/// weights by the covariance leave out a row with no variance (`no_se`).
fn takes(weighting: Weighting, row: &Row) -> bool {
    matches!(weighting, Weighting::Pairs) || row.variance.is_some()
}

/// Whether the cross solve reads `row`; a trait's solve reads the others
/// of that trait.
fn in_cross(row: &Row) -> bool {
    row.first_trait != row.second_trait || row.kind == RowKind::WithinPerson
}

/// The indices of the defined rows a solve under `weighting` takes among
/// those it `reads`.
fn entered(rows: &[Row], weighting: Weighting, reads: impl Fn(&Row) -> bool) -> Vec<usize> {
    (0..rows.len())
        .filter(|&i| rows[i].value.is_ok() && takes(weighting, &rows[i]) && reads(&rows[i]))
        .collect()
}

/// `columns` with each present one that is zero in every row of
/// `entered` marked not_identified: no row informs it, and leaving it out
/// solves the others exactly as if it were absent.
fn supported(rows: &[Row], entered: &[usize], mut columns: Columns) -> Columns {
    for (j, column) in columns.iter_mut().enumerate() {
        if column.is_ok() && zero_in(rows, entered, j) {
            *column = Err(Reason::NotIdentified);
        }
    }
    columns
}

/// Whether column `j` is zero in every row of `entered`.
fn zero_in(rows: &[Row], entered: &[usize], j: usize) -> bool {
    entered.iter().all(|&i| design(rows[i].kind)[j] == 0.0)
}

/// Each column's unknown, in A, C, E order: a present one's from the
/// solve, an absent one with its reason.
fn unknowns(solution: Result<Vec<Linear>, Reason>, columns: Columns) -> [Quantity; 3] {
    let mut values = solution.map(Vec::into_iter);
    columns.map(|column| {
        column.and_then(|()| match &mut values {
            Ok(values) => values.next().ok_or(Reason::NotIdentified),
            Err(reason) => Err(*reason),
        })
    })
}

/// One trait's A, C and E before their SEs, and the rows that entered.
struct TraitParts {
    shares: [Quantity; 3],
    rows: Vec<usize>,
}

fn trait_fit(
    problem: &Problem,
    weighting: Weighting,
    t: usize,
    components: Components,
) -> TraitParts {
    let rows = problem.rows;
    let rows_in = entered(rows, weighting, |r| !in_cross(r) && r.first_trait == t);
    let columns = supported(
        rows,
        &rows_in,
        [
            requested(components.a()),
            requested(components.c()),
            Err(Reason::NotRequested),
        ],
    );
    let [a, c, _] = unknowns(problem.solve(weighting, &rows_in, &columns), columns);
    let share = |x: &Quantity| match x {
        Err(Reason::NotRequested) => Ok(Linear::zero(problem.width)),
        other => other.clone(),
    };
    let e = share(&a).and_then(|a| {
        share(&c).map(|c| Linear {
            value: 1.0 - a.value - c.value,
            g: a.g.iter().zip(&c.g).map(|(a, c)| -a - c).collect(),
            noise: a.noise + c.noise,
        })
    });
    TraitParts {
        shares: [a, c, e],
        rows: rows_in,
    }
}

/// The cross solve's sA, sC and sE, and rA, rC and rE, before their SEs.
struct CrossParts {
    covariances: [Quantity; 3],
    correlations: [Quantity; 3],
    rows: Vec<usize>,
}

/// The cross solve.  sE is in the within-person row alone, so with sC
/// zero in every relative row the two share that row and neither is
/// identified; the row is then left out and sA solved from the relatives.
fn cross_fit(
    problem: &Problem,
    weighting: Weighting,
    traits: &[&TraitParts],
    components: Components,
) -> CrossParts {
    let rows = problem.rows;
    let mut rows_in = entered(rows, weighting, in_cross);
    let within_person = rows_in
        .iter()
        .any(|&i| rows[i].kind == RowKind::WithinPerson);
    let mut columns = [
        requested(components.a()),
        requested(components.c()),
        if within_person {
            Ok(())
        } else {
            Err(Reason::NotIdentified)
        },
    ];
    let relatives: Vec<usize> = rows_in
        .iter()
        .copied()
        .filter(|&i| rows[i].kind != RowKind::WithinPerson)
        .collect();
    if columns[1].is_ok() && zero_in(rows, &relatives, 1) {
        columns[1] = Err(Reason::NotIdentified);
        columns[2] = Err(Reason::NotIdentified);
        rows_in = relatives;
    }
    let columns = supported(rows, &rows_in, columns);
    let covariances = unknowns(problem.solve(weighting, &rows_in, &columns), columns);
    let (one, two) = (&traits[0].shares, &traits[1].shares);
    let correlations = [0, 1, 2].map(|k| ratio(&covariances[k], &one[k], &two[k]));
    CrossParts {
        covariances,
        correlations,
        rows: rows_in,
    }
}

/// `s / sqrt(x1 x2)` and its gradient, undefined with the first undefined
/// operand's reason or when a share is <= 0.
fn ratio(s: &Quantity, x1: &Quantity, x2: &Quantity) -> Quantity {
    let (s, x1, x2) = (
        s.as_ref().map_err(|r| *r)?,
        x1.as_ref().map_err(|r| *r)?,
        x2.as_ref().map_err(|r| *r)?,
    );
    if x1.value <= 0.0 || x2.value <= 0.0 {
        return Err(Reason::NonpositiveComponent);
    }
    let root = (x1.value * x2.value).sqrt();
    let r = s.value / root;
    let (d1, d2) = (r / (2.0 * x1.value), r / (2.0 * x2.value));
    let g = (0..s.g.len())
        .map(|i| s.g[i] / root - d1 * x1.g[i] - d2 * x2.g[i])
        .collect();
    Ok(Linear {
        value: r,
        g,
        noise: s.noise / root + d1.abs() * x1.noise + d2.abs() * x2.noise,
    })
}

/// The entries of `x` under `keys`: value, reason, SE, SE reason.  An
/// undefined value's SE carries its reason.
fn either(
    out: &mut Vec<(&'static str, Value)>,
    keys: [&'static str; 4],
    x: Result<Solved, Reason>,
) {
    let [value, reason, se, se_reason] = keys;
    let (v, r, s) = match x {
        Ok(Solved { value, se }) => (Value::Float(value), Value::Null, se),
        Err(r) => (Value::Null, r.name().into(), Err(r)),
    };
    out.extend([(value, v), (reason, r)]);
    match s {
        Ok(s) => out.extend([(se, s.into()), (se_reason, Value::Null)]),
        Err(r) => out.extend([(se, Value::Null), (se_reason, r.name().into())]),
    }
}

fn texts(names: &[String]) -> Value {
    Value::List(names.iter().map(|n| Value::Text(n.clone())).collect())
}

impl Row {
    fn to_value(&self) -> Value {
        let (kinship, shared) = match self.kind {
            RowKind::Relatives { kinship, shared } => (Value::Float(kinship), shared.into()),
            RowKind::WithinPerson => (Value::Null, Value::Null),
        };
        let (value, reason) = match &self.value {
            Ok(v) => (Value::Float(*v), Value::Null),
            Err(r) => (Value::Null, Value::Text(r.clone())),
        };
        Value::Map(vec![
            ("entry", Value::Text(self.entry.clone())),
            ("first_trait", (self.first_trait as u64).into()),
            ("second_trait", (self.second_trait as u64).into()),
            ("within_person", (self.kind == RowKind::WithinPerson).into()),
            ("kinship", kinship),
            ("shared", shared),
            ("value", value),
            ("weight", self.weight.into()),
            ("variance", self.variance.into()),
            ("used", self.value.is_ok().into()),
            ("reason", reason),
        ])
    }
}

impl AceFit {
    /// The entries of the rows at `indices`, in row order.
    fn entries(&self, indices: &[usize]) -> Value {
        Value::List(
            indices
                .iter()
                .map(|&i| Value::Text(self.rows[i].entry.clone()))
                .collect(),
        )
    }

    /// The result as a host-neutral tree; its keys are the hosts' contract.
    pub fn to_value(&self) -> Value {
        let traits = self
            .traits
            .iter()
            .enumerate()
            .map(|(t, fit)| {
                let mut out = vec![("trait", (t as u64).into())];
                either(&mut out, ["A", "A_reason", "A_se", "A_se_reason"], fit.a);
                either(&mut out, ["C", "C_reason", "C_se", "C_se_reason"], fit.c);
                either(&mut out, ["E", "E_reason", "E_se", "E_se_reason"], fit.e);
                out.push(("rows", self.entries(&fit.rows)));
                out.push(("weights", fit.weights.name().into()));
                Value::Map(out)
            })
            .collect();
        let cross = self.cross.as_ref().map_or(Value::Null, |fit| {
            let mut out = vec![];
            for (keys, s) in [
                (["sA", "sA_reason", "sA_se", "sA_se_reason"], fit.s_a),
                (["sC", "sC_reason", "sC_se", "sC_se_reason"], fit.s_c),
                (["sE", "sE_reason", "sE_se", "sE_se_reason"], fit.s_e),
            ] {
                either(&mut out, keys, s);
            }
            for (keys, flag, r) in [
                (
                    ["rA", "rA_reason", "rA_se", "rA_se_reason"],
                    "rA_out_of_range",
                    fit.r_a,
                ),
                (
                    ["rC", "rC_reason", "rC_se", "rC_se_reason"],
                    "rC_out_of_range",
                    fit.r_c,
                ),
                (
                    ["rE", "rE_reason", "rE_se", "rE_se_reason"],
                    "rE_out_of_range",
                    fit.r_e,
                ),
            ] {
                either(&mut out, keys, r);
                out.push((flag, r.ok().map(|r| r.value.abs() > 1.0).into()));
            }
            out.push(("rows", self.entries(&fit.rows)));
            out.push(("weights", fit.weights.name().into()));
            Value::Map(out)
        });
        let request = &self.request;
        Value::Map(vec![
            ("traits", Value::List(traits)),
            ("cross", cross),
            (
                "rows",
                Value::List(self.rows.iter().map(Row::to_value).collect()),
            ),
            (
                "settings",
                Value::Map(vec![
                    (
                        "categories",
                        request.categories.as_deref().map_or(Value::Null, texts),
                    ),
                    (
                        "components",
                        Value::List(
                            request
                                .components
                                .names()
                                .iter()
                                .map(|&c| c.into())
                                .collect(),
                        ),
                    ),
                    (
                        "shared_environment",
                        request
                            .shared_environment
                            .as_deref()
                            .map_or(Value::Null, texts),
                    ),
                    ("estimator", request.estimator.name().into()),
                    ("estimate", self.estimate.name().into()),
                    (
                        "weights",
                        request.weights.map_or("default", Weights::name).into(),
                    ),
                ]),
            ),
            (
                "method",
                Value::Map(
                    method(request)
                        .iter()
                        .map(|&(k, v)| (k, v.into()))
                        .collect(),
                ),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Category::{Av, C1, FS, MHS, MO, MZ, PHS};

    /// The fit without a covariance.
    fn fit(cells: &[AceCell], request: &AceRequest) -> Result<AceFit, Error> {
        super::fit(cells, request, None)
    }

    fn crude(estimator: &str, value: Result<f64, &str>) -> CrudeEstimate {
        CrudeEstimate {
            estimator: relative_pairs::Estimator::parse(estimator).unwrap(),
            value: value.map_err(str::to_string),
        }
    }

    fn cell(members: &[Category], traits: (usize, usize), r: f64, n: u64) -> AceCell {
        AceCell {
            name: members[0].code().to_string(),
            pairs: Pairs::Relatives(members.to_vec()),
            first_trait: traits.0,
            second_trait: traits.1,
            crude: vec![crude("tetrachoric", Ok(r)), crude("phi", Ok(r / 2.0))],
            n,
            n_entries: None,
            stratified: None,
        }
    }

    fn group(name: &str, members: &[Category], r: f64) -> AceCell {
        AceCell {
            name: name.to_string(),
            pairs: Pairs::Relatives(members.to_vec()),
            ..cell(&[MZ], (0, 0), r, 500)
        }
    }

    fn within_person(r: f64, n: u64) -> AceCell {
        AceCell {
            name: "within_person".to_string(),
            pairs: Pairs::WithinPerson,
            ..cell(&[MZ], (0, 1), r, n)
        }
    }

    fn request(components: Components, shared: Option<&[&str]>) -> AceRequest {
        AceRequest {
            categories: None,
            components,
            shared_environment: shared.map(|s| s.iter().map(|c| c.to_string()).collect()),
            estimator: Estimator::Primary,
            estimate: Some(Estimate::Pooled),
            weights: Some(Weights::Pairs),
        }
    }

    fn close(actual: Result<Solved, Reason>, expected: f64, relative: f64) {
        let actual = actual.unwrap().value;
        assert!(
            (actual - expected).abs() <= relative * expected.abs(),
            "{actual} vs {expected}: relative error {:e}",
            (actual - expected).abs() / expected.abs()
        );
    }

    fn operand_close(actual: Result<Solved, Reason>, expected: f64, scale: f64) {
        let error = (actual.unwrap().value - expected).abs() / scale;
        assert!(error <= 1e-15, "error {error:e} of the operands' scale");
    }

    fn one_trait(cells: &[AceCell], request: &AceRequest) -> TraitFit {
        let fit = fit(cells, request).unwrap();
        assert!(fit.cross.is_none());
        fit.traits.into_iter().next().unwrap()
    }

    #[test]
    fn recipes_match_their_closed_forms() {
        let r_fs = 0.312_7;
        // A closed form that subtracts cannot be held to 1e-15 of its result
        // when that result is small beside its operands: 1e-15 is of the
        // operands' scale, the size of a rounding of either.
        let sibs = one_trait(
            &[cell(&[FS], (0, 0), r_fs, 4813)],
            &request(Components::A, None),
        );
        close(sibs.a, 2.0 * r_fs, 1e-15);
        close(sibs.e, 1.0 - 2.0 * r_fs, 1e-15);
        assert_eq!(sibs.c, Err(Reason::NotRequested));

        let (r_mz, r_fs) = (0.621_3, 0.351_8);
        for (n_mz, n_fs) in [(911, 5222), (3, 1_000_000_000), (1_000_000_000, 3)] {
            let falconer = one_trait(
                &[
                    cell(&[MZ], (0, 0), r_mz, n_mz),
                    cell(&[FS], (0, 0), r_fs, n_fs),
                ],
                &request(Components::AC, Some(&["MZ", "FS"])),
            );
            operand_close(falconer.a, 2.0 * (r_mz - r_fs), 2.0 * (r_mz + r_fs));
            operand_close(falconer.c, 2.0 * r_fs - r_mz, 2.0 * r_fs + r_mz);
        }

        let r_hs = 0.109_3;
        let half_sibs = one_trait(
            &[cell(&[PHS], (0, 0), r_hs, 731)],
            &request(Components::A, None),
        );
        close(half_sibs.a, 4.0 * r_hs, 1e-15);

        let r_1c = 0.057_1;
        let cousins = one_trait(
            &[cell(&[C1], (0, 0), r_1c, 20_417)],
            &request(Components::A, None),
        );
        close(cousins.a, 8.0 * r_1c, 1e-15);
    }

    /// Weighted normal equations of `r = a x_a + c x_c`, solved by Cramer's rule.
    fn normal_equations(rows: &[(f64, f64, f64, f64)]) -> (f64, f64) {
        let (mut aa, mut ac, mut cc, mut ay, mut cy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &(x_a, x_c, y, w) in rows {
            aa += w * x_a * x_a;
            ac += w * x_a * x_c;
            cc += w * x_c * x_c;
            ay += w * x_a * y;
            cy += w * x_c * y;
        }
        let det = aa * cc - ac * ac;
        ((ay * cc - ac * cy) / det, (aa * cy - ac * ay) / det)
    }

    #[test]
    fn overidentified_fit_is_the_weighted_least_squares_solution() {
        let data = [
            (MZ, 1.0, 1.0, 0.71, 400),
            (FS, 0.5, 1.0, 0.42, 3000),
            (MHS, 0.25, 1.0, 0.27, 800),
            (PHS, 0.25, 0.0, 0.11, 650),
        ];
        let req = request(Components::AC, Some(&["MZ", "FS", "MHS"]));
        let fit_with = |fs_n: u64| {
            let cells: Vec<AceCell> = data
                .iter()
                .map(|&(cat, _, _, r, n)| cell(&[cat], (0, 0), r, if cat == FS { fs_n } else { n }))
                .collect();
            let rows: Vec<_> = data
                .iter()
                .map(|&(cat, x_a, x_c, r, n)| {
                    (x_a, x_c, r, if cat == FS { fs_n } else { n } as f64)
                })
                .collect();
            (one_trait(&cells, &req), normal_equations(&rows))
        };
        let (fit, (a, c)) = fit_with(3000);
        close(fit.a, a, 1e-12);
        close(fit.c, c, 1e-12);
        assert_eq!(fit.rows, [0, 1, 2, 3]);
        let (refit, (a2, c2)) = fit_with(30);
        close(refit.a, a2, 1e-12);
        close(refit.c, c2, 1e-12);
        assert!(
            (refit.c.unwrap().value - fit.c.unwrap().value).abs() > 1e-3,
            "weights do not matter"
        );
    }

    /// Each trait's A and C, and the cross covariances sA, sC, sE.
    struct Truth {
        a: [f64; 2],
        c: [f64; 2],
        s: [f64; 3],
    }

    /// A truth with A = (0.5, 0.4), C = (0.2, 0.15) and rA, rC, rE.
    fn truth(r: [f64; 3]) -> Truth {
        let (a, c): ([f64; 2], [f64; 2]) = ([0.5, 0.4], [0.2, 0.15]);
        let e = [1.0 - a[0] - c[0], 1.0 - a[1] - c[1]];
        Truth {
            a,
            c,
            s: [
                r[0] * (a[0] * a[1]).sqrt(),
                r[1] * (c[0] * c[1]).sqrt(),
                r[2] * (e[0] * e[1]).sqrt(),
            ],
        }
    }

    impl Truth {
        fn e(&self, t: usize) -> f64 {
            1.0 - self.a[t] - self.c[t]
        }

        /// Each category's exact model correlation for every trait pair,
        /// both orientations of a directed category's cross cell, and the
        /// within-person cell if asked.
        fn cells(&self, within: bool) -> Vec<AceCell> {
            let [s_a, s_c, s_e] = self.s;
            let mut cells = vec![];
            let mut n = 100;
            for cat in [MZ, FS, MHS, PHS, MO, C1] {
                let k = 0.5f64.powi(i32::from(cat.degree()));
                let s = if [MZ, FS, MHS].contains(&cat) {
                    1.0
                } else {
                    0.0
                };
                let mut traits = vec![(0, 0), (1, 1), (0, 1)];
                if !cat.symmetric() {
                    traits.push((1, 0));
                }
                for (i, j) in traits {
                    let r = if i == j {
                        k * self.a[i] + s * self.c[i]
                    } else {
                        k * s_a + s * s_c
                    };
                    n += 137;
                    cells.push(cell(&[cat], (i, j), r, n));
                }
            }
            if within {
                cells.push(within_person(s_a + s_c + s_e, 5000));
            }
            cells
        }
    }

    const SHARED: &[&str] = &["MZ", "FS", "MHS"];

    #[test]
    fn exact_model_correlations_recover_every_parameter() {
        let r = [0.6, 0.5, 0.3];
        let truth = truth(r);
        let fit = fit(&truth.cells(true), &request(Components::AC, Some(SHARED))).unwrap();
        for t in 0..2 {
            close(fit.traits[t].a, truth.a[t], 1e-12);
            close(fit.traits[t].c, truth.c[t], 1e-12);
            close(fit.traits[t].e, truth.e(t), 1e-12);
            assert_eq!(fit.traits[t].rows.len(), 6);
        }
        let cross = fit.cross.unwrap();
        let [s_a, s_c, s_e] = truth.s;
        close(cross.s_a, s_a, 1e-12);
        close(cross.s_c, s_c, 1e-12);
        close(cross.s_e, s_e, 1e-12);
        close(cross.r_a, r[0], 1e-12);
        close(cross.r_c, r[1], 1e-12);
        close(cross.r_e, r[2], 1e-12);
        assert_eq!(cross.rows.len(), 8);
    }

    #[test]
    fn without_the_within_person_cell_se_is_not_identified() {
        let truth = truth([0.6, 0.5, 0.3]);
        let cross = fit(&truth.cells(false), &request(Components::AC, Some(SHARED)))
            .unwrap()
            .cross
            .unwrap();
        let [s_a, s_c, _] = truth.s;
        close(cross.s_a, s_a, 1e-12);
        close(cross.s_c, s_c, 1e-12);
        close(cross.r_a, 0.6, 1e-12);
        assert_eq!(cross.s_e, Err(Reason::NotIdentified));
        assert_eq!(cross.r_e, Err(Reason::NotIdentified));
    }

    #[test]
    fn a_nonpositive_share_leaves_its_correlation_undefined() {
        let truth = Truth {
            a: [0.5, 0.6],
            c: [0.2, -0.05],
            s: [0.3, 0.02, 0.1],
        };
        let fit = fit(&truth.cells(true), &request(Components::AC, Some(SHARED))).unwrap();
        close(fit.traits[1].c, -0.05, 1e-12);
        let cross = fit.cross.unwrap();
        close(cross.s_c, 0.02, 1e-12);
        assert_eq!(cross.r_c, Err(Reason::NonpositiveComponent));
        close(cross.r_a, 0.3 / (0.5f64 * 0.6).sqrt(), 1e-12);
        close(cross.r_e, 0.1 / (0.3f64 * 0.45).sqrt(), 1e-12);
    }

    fn lookup<'a>(value: &'a Value, key: &str) -> &'a Value {
        let Value::Map(entries) = value else {
            panic!("{value:?} is not a map")
        };
        &entries.iter().find(|(k, _)| *k == key).unwrap().1
    }

    #[test]
    fn a_correlation_past_one_is_reported_and_flagged() {
        let truth = truth([2.0, 0.5, 0.3]);
        let fit = fit(&truth.cells(true), &request(Components::AC, Some(SHARED))).unwrap();
        close(fit.cross.as_ref().unwrap().r_a, 2.0, 1e-12);
        let tree = fit.to_value();
        let cross = lookup(&tree, "cross");
        assert_eq!(lookup(cross, "rA_out_of_range"), &Value::Bool(true));
        assert_eq!(lookup(cross, "rC_out_of_range"), &Value::Bool(false));
        assert_eq!(lookup(cross, "rA_reason"), &Value::Null);
    }

    #[test]
    fn rank_deficient_designs_are_not_identified() {
        let ac = |shared: &[&str], cells: &[AceCell]| {
            one_trait(cells, &request(Components::AC, Some(shared)))
        };
        let not_identified = |fit: TraitFit| {
            assert_eq!(fit.a, Err(Reason::NotIdentified));
            assert_eq!(fit.c, Err(Reason::NotIdentified));
            assert_eq!(fit.e, Err(Reason::NotIdentified));
        };
        not_identified(ac(&["FS"], &[cell(&[FS], (0, 0), 0.4, 900)]));
        // MHS and Av share a degree and are both declared shared: two rows,
        // one direction.
        not_identified(ac(
            &["MHS", "Av"],
            &[
                cell(&[MHS], (0, 0), 0.3, 900),
                cell(&[Av], (0, 0), 0.2, 300),
            ],
        ));
        not_identified(ac(&["FS"], &[]));
    }

    #[test]
    fn a_trait_without_a_shared_row_solves_a_alone() {
        let mut cells = vec![
            cell(&[MZ], (0, 0), 0.7, 400),
            cell(&[FS], (0, 0), 0.4, 900),
            cell(&[PHS], (0, 0), 0.1, 300),
        ];
        cells[0].crude[0].value = Err("constant_margin".to_string());
        let fit = one_trait(&cells, &request(Components::AC, Some(&["MZ"])));
        let a = (900.0 * 0.5 * 0.4 + 300.0 * 0.25 * 0.1) / (900.0 * 0.25 + 300.0 * 0.0625);
        close(fit.a, a, 1e-14);
        assert_eq!(fit.c, Err(Reason::NotIdentified));
        assert_eq!(fit.e, Err(Reason::NotIdentified));
        assert_eq!(fit.rows, [1, 2]);
    }

    #[test]
    fn a_cross_solve_without_a_shared_row_solves_sa_alone() {
        let (a, c) = ([0.5, 0.4], [0.2, 0.15]);
        let (s_a, s_c, s_e) = (0.3, 0.1, 0.05);
        let mut cells = vec![];
        for (cat, k, s) in [(FS, 0.5, 1.0), (PHS, 0.25, 0.0), (C1, 0.125, 0.0)] {
            for t in 0..2 {
                cells.push(cell(&[cat], (t, t), k * a[t] + s * c[t], 700));
            }
            cells.push(cell(&[cat], (0, 1), k * s_a + s * s_c, 500));
        }
        cells[2].crude[0].value = Err("constant_margin".to_string());
        cells.push(within_person(s_a + s_c + s_e, 5000));
        let fit = fit(&cells, &request(Components::AC, Some(&["FS"]))).unwrap();
        for t in 0..2 {
            close(fit.traits[t].a, a[t], 1e-12);
            close(fit.traits[t].c, c[t], 1e-12);
        }
        let cross = fit.cross.unwrap();
        close(cross.s_a, s_a, 1e-12);
        close(cross.r_a, s_a / (a[0] * a[1]).sqrt(), 1e-12);
        for x in [cross.s_c, cross.s_e, cross.r_c, cross.r_e] {
            assert_eq!(x, Err(Reason::NotIdentified));
        }
        assert_eq!(cross.rows, [5, 8]);
    }

    #[test]
    fn undefined_cells_are_rows_with_their_reason() {
        let mut cells = vec![
            cell(&[MZ], (0, 0), 0.7, 400),
            cell(&[FS], (0, 0), 0.4, 900),
            cell(&[PHS], (0, 0), 0.1, 300),
            cell(&[MHS], (0, 0), 0.3, 0),
        ];
        cells[2].crude[0].value = Err("constant_margin".to_string());
        let fit = fit(&cells, &request(Components::AC, Some(SHARED))).unwrap();
        assert_eq!(fit.rows.len(), 4);
        assert_eq!(fit.rows[2].value, Err("constant_margin".to_string()));
        assert_eq!(fit.rows[3].value, Err("no_complete_pairs".to_string()));
        assert_eq!(fit.traits[0].rows, [0, 1]);
        close(fit.traits[0].a, 2.0 * (0.7 - 0.4), 1e-14);
        let tree = fit.to_value();
        let Value::List(rows) = lookup(&tree, "rows") else {
            panic!("rows is not a list")
        };
        assert_eq!(lookup(&rows[2], "used"), &Value::Bool(false));
        assert_eq!(
            lookup(&rows[2], "reason"),
            &Value::Text("constant_margin".to_string())
        );
        assert_eq!(lookup(&rows[0], "kinship"), &Value::Float(0.5));
        assert_eq!(lookup(&rows[0], "shared"), &Value::Bool(true));
    }

    #[test]
    fn cross_rows_weigh_each_complete_orientation_once() {
        let cells = |n: [u64; 7], entries: [u64; 3]| {
            let symmetric = |c: AceCell, e: u64| AceCell {
                n_entries: Some(e),
                ..c
            };
            vec![
                cell(&[FS], (0, 0), 0.4, n[0]),
                cell(&[MO], (0, 1), 0.21, n[1]),
                cell(&[MO], (1, 0), 0.17, n[2]),
                symmetric(cell(&[FS], (0, 1), 0.33, n[3]), entries[0]),
                symmetric(cell(&[MHS], (0, 1), 0.12, n[4]), entries[1]),
                symmetric(cell(&[PHS], (0, 1), 0.08, n[5]), entries[2]),
                within_person(0.3, n[6]),
            ]
        };
        // MO's two directed cross cells hold 300 + 250 orientations, FS's
        // symmetric cross cell the same 550 over 400 pairs; stratified,
        // 280 + 230 and 510 over 380.
        let pooled = cells([700, 300, 250, 400, 600, 900, 5000], [550, 800, 1100]);
        let stratified: Vec<AceCell> = pooled
            .iter()
            .zip(cells(
                [650, 280, 230, 380, 560, 850, 4900],
                [510, 740, 1020],
            ))
            .map(|(c, s)| AceCell {
                stratified: Some(StratifiedEstimate {
                    value: c.crude[0].value.clone(),
                    n: s.n,
                    n_entries: s.n_entries,
                }),
                ..c.clone()
            })
            .collect();
        let design = [(0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (0.25, 1.0), (0.25, 0.0)];
        let wls = |cells: &[AceCell], weights: [u64; 5]| {
            let rows: Vec<_> = design
                .iter()
                .zip(&cells[1..6])
                .zip(weights)
                .map(|((&(x_a, x_c), c), w)| {
                    (x_a, x_c, c.crude[0].value.clone().unwrap(), w as f64)
                })
                .collect();
            normal_equations(&rows)
        };
        let req = request(Components::AC, Some(SHARED));
        for (cells, estimate, weights, by_pairs) in [
            (
                pooled,
                Estimate::Pooled,
                [700, 300, 250, 550, 800, 1100, 5000],
                [300, 250, 400, 600, 900],
            ),
            (
                stratified,
                Estimate::Stratified,
                [650, 280, 230, 510, 740, 1020, 4900],
                [280, 230, 380, 560, 850],
            ),
        ] {
            let fit = fit(
                &cells,
                &AceRequest {
                    estimate: Some(estimate),
                    ..req.clone()
                },
            )
            .unwrap();
            let got: Vec<u64> = fit.rows.iter().map(|r| r.weight).collect();
            assert_eq!(got, weights, "{estimate:?}");
            assert_eq!(got[1] + got[2], got[3], "{estimate:?}");
            let (s_a, s_c) = wls(&cells, [1, 2, 3, 4, 5].map(|i| weights[i]));
            let cross = fit.cross.unwrap();
            assert_eq!(cross.rows, [1, 2, 3, 4, 5, 6]);
            close(cross.s_a, s_a, 1e-12);
            close(cross.s_c, s_c, 1e-12);
            let (a_by_pairs, c_by_pairs) = wls(&cells, by_pairs);
            assert!(
                (s_a - a_by_pairs).abs() > 1e-4 && (s_c - c_by_pairs).abs() > 1e-4,
                "{estimate:?}: weights do not matter"
            );
        }
    }

    #[test]
    fn the_estimate_and_estimator_pick_each_cells_value() {
        let mut cells = vec![
            cell(&[FS], (0, 0), 0.4, 900),
            cell(&[PHS], (0, 0), 0.1, 300),
        ];
        cells[0].stratified = Some(StratifiedEstimate {
            value: Ok(0.35),
            n: 850,
            n_entries: None,
        });
        let mut req = request(Components::A, None);
        req.categories = Some(vec!["FS".to_string()]);
        close(one_trait(&cells, &req).a, 0.8, 1e-15);
        req.estimator = Estimator::Phi;
        close(one_trait(&cells, &req).a, 0.4, 1e-15);
        req.estimator = Estimator::Primary;
        req.estimate = Some(Estimate::Stratified);
        let fit = fit(&cells, &req).unwrap();
        assert_eq!(fit.rows.len(), 1);
        assert_eq!(fit.rows[0].weight, 850);
        close(fit.traits[0].a, 0.7, 1e-15);
        req.categories = None;
        let fit = self::fit(&cells, &req).unwrap();
        assert_eq!(fit.rows[1].value, Err("no_stratified_form".to_string()));
        assert_eq!(fit.traits[0].rows, [0]);
    }

    #[test]
    fn the_default_estimate_is_stratified_when_a_relatives_cell_has_one() {
        let stratified = StratifiedEstimate {
            value: Ok(0.3),
            n: 400,
            n_entries: None,
        };
        let mut req = request(Components::A, None);
        req.estimate = None;
        let mut wp = within_person(0.2, 900);
        wp.stratified = Some(stratified.clone());
        let mut cells = vec![cell(&[FS], (0, 0), 0.4, 500), wp];
        assert_eq!(fit(&cells, &req).unwrap().estimate, Estimate::Pooled);
        cells[0].stratified = Some(stratified);
        let fitted = fit(&cells, &req).unwrap();
        assert_eq!(fitted.estimate, Estimate::Stratified);
        assert_eq!(fitted.rows[0].value, Ok(0.3));
    }

    fn code(cells: &[AceCell], request: &AceRequest) -> &'static str {
        fit(cells, request).unwrap_err().code()
    }

    #[test]
    fn every_ace_error() {
        let fs = || cell(&[FS], (0, 0), 0.4, 900);
        assert_eq!(
            code(&[fs()], &request(Components::AC, None)),
            "missing_shared_environment"
        );
        let err = fit(&[fs()], &request(Components::AC, Some(&["MZ", "XX"]))).unwrap_err();
        assert_eq!(
            err,
            Error::UnknownSharedEnvironment {
                category: "XX".to_string(),
                position: 1.into()
            }
        );
        assert_eq!(err.code(), "unknown_category");
        let mut req = request(Components::A, None);
        req.categories = Some(vec!["FS".to_string(), "sibs".to_string()]);
        assert_eq!(code(&[fs()], &req), "unknown_entry");
        assert_eq!(
            code(
                &[group("hs", &[MHS, PHS], 0.2)],
                &request(Components::AC, Some(&["MHS"]))
            ),
            "mixed_shared_environment"
        );
        assert_eq!(
            code(
                &[group("sibs", &[FS, MHS], 0.3)],
                &request(Components::A, None)
            ),
            "mixed_kinship"
        );
        assert_eq!(
            code(&[group("none", &[], 0.3)], &request(Components::A, None)),
            "empty_category_group"
        );
        let mut req = request(Components::A, None);
        req.estimate = Some(Estimate::Stratified);
        assert_eq!(
            fit(&[fs()], &req).unwrap_err(),
            Error::NoStratifiedForm {
                estimator: Estimator::Primary
            }
        );
        req.estimator = Estimator::Phi;
        assert_eq!(
            fit(&[fs()], &req).unwrap_err(),
            Error::NoStratifiedForm {
                estimator: Estimator::Phi
            }
        );
        let mut continuous = fs();
        continuous.crude = vec![crude("pearson", Ok(0.4))];
        let mut req = request(Components::A, None);
        req.estimator = Estimator::Phi;
        assert_eq!(code(&[continuous], &req), "missing_estimator");
        let mut third = fs();
        third.second_trait = 2;
        assert_eq!(code(&[third], &request(Components::A, None)), "trait_count");
        assert_eq!(
            Pairs::relatives(&["FS".to_string(), "XX".to_string()]),
            Err(Error::UnknownCategory {
                category: "XX".to_string(),
                position: 1.into()
            })
        );
    }

    #[test]
    fn hosts_name_the_request_as_the_tree_does() {
        assert_eq!(
            Pairs::relatives(&["MZ".to_string()]),
            Ok(Pairs::Relatives(vec![MZ]))
        );
        for (names, components) in [
            (&["A"][..], Some(Components::A)),
            (&["C"], Some(Components::C)),
            (&["C", "A"], Some(Components::AC)),
            (&["A", "C"], Some(Components::AC)),
            (&["A", "A"], None),
            (&["E"], None),
            (&[], None),
        ] {
            assert_eq!(Components::parse(names), components, "{names:?}");
        }
        for estimator in [Estimator::Primary, Estimator::Phi] {
            assert_eq!(Estimator::parse(estimator.name()), Some(estimator));
        }
        for estimate in [Estimate::Pooled, Estimate::Stratified] {
            assert_eq!(Estimate::parse(estimate.name()), Some(estimate));
        }
        assert_eq!(Estimator::parse("tetrachoric"), None);
        assert_eq!(Estimate::parse("both"), None);
        for weights in [Weights::Pairs, Weights::InverseVariance, Weights::Gls] {
            assert_eq!(Weights::parse(weights.name()), Some(weights));
        }
        assert_eq!(Weights::parse("iv"), None);
    }

    #[test]
    fn the_tree_echoes_the_request() {
        let tree = fit(
            &truth([0.6, 0.5, 0.3]).cells(true),
            &request(Components::A, Some(SHARED)),
        )
        .unwrap()
        .to_value();
        let settings = lookup(&tree, "settings");
        assert_eq!(
            lookup(settings, "components"),
            &Value::List(vec![Value::Str("A")])
        );
        assert_eq!(lookup(settings, "estimate"), &Value::Str("pooled"));
        let Value::List(traits) = lookup(&tree, "traits") else {
            panic!("traits is not a list")
        };
        assert_eq!(lookup(&traits[0], "C_reason"), &Value::Str("not_requested"));
        assert_eq!(lookup(settings, "weights"), &Value::Str("pairs"));
        assert!(matches!(lookup(&tree, "method"), Value::Map(m) if m.len() == 6));
    }

    /// Two traits' cells of every kind, values off the model, and the
    /// within-person cell last.
    fn noisy_cells() -> Vec<AceCell> {
        let mut cells = vec![];
        let mut k = 0u64;
        for cat in [MZ, FS, MHS, PHS, MO, C1] {
            let mut traits = vec![(0, 0), (1, 1), (0, 1)];
            if !cat.symmetric() {
                traits.push((1, 0));
            }
            for t in traits {
                k += 1;
                let r = 0.9f64.powi(i32::from(cat.degree()) + 1)
                    * (0.5 + 0.37 * ((k * 7919 % 13) as f64 / 13.0));
                cells.push(cell(&[cat], t, r, 100 + 997 * (k * 31 % 17)));
            }
        }
        cells.push(within_person(0.41, 5000));
        cells
    }

    #[test]
    fn pairs_weights_give_the_bits_of_the_fit_before_the_se() {
        let fit = fit(&noisy_cells(), &request(Components::AC, Some(SHARED))).unwrap();
        let mut values = vec![];
        for t in &fit.traits {
            values.extend([t.a, t.c, t.e]);
        }
        let c = fit.cross.unwrap();
        values.extend([c.s_a, c.s_c, c.s_e, c.r_a, c.r_c, c.r_e]);
        let bits: Vec<u64> = values.iter().map(|v| v.unwrap().value.to_bits()).collect();
        assert_eq!(
            bits,
            [
                0x3fdf0738c589ad45,
                0x3fc5a367023ae3ac,
                0x3fd62713b958e0e6,
                0x3fe47cfd305cd9b9,
                0x3fba2a0ff213ea68,
                0x3fd07b81a2c151f4,
                0x3fef38ac2acf9dbc,
                0x3fabcf1f0ea17eb7,
                0xbfe3d6e5c9ce308a,
                0x3ffc04ecf8302dbd,
                0x3fda721a6e9d5f2e,
                0xc0009cb5db8ef2d3,
            ]
        );
    }

    /// A positive definite `n x n` covariance, row-major: variances
    /// `(0.01 (1 + i mod 5))²`, correlations `0.3^|i-j|`.
    fn positive_definite(n: usize) -> Vec<f64> {
        let sd = |i: usize| 0.01 * (1 + i % 5) as f64;
        (0..n * n)
            .map(|x| {
                let (i, j) = (x / n, x % n);
                sd(i) * sd(j) * 0.3f64.powi(i.abs_diff(j) as i32)
            })
            .collect()
    }

    /// A positive definite covariance of `cells` like a result's:
    /// variances `1 / n`, correlations `0.3^|i-j|`.
    fn like_a_result(cells: &[AceCell]) -> Vec<f64> {
        let n = cells.len();
        let sd = |i: usize| 1.0 / (cells[i].n as f64).sqrt();
        (0..n * n)
            .map(|x| {
                let (i, j) = (x / n, x % n);
                sd(i) * sd(j) * 0.3f64.powi(i.abs_diff(j) as i32)
            })
            .collect()
    }

    fn with(n: usize, matrix: Vec<f64>) -> Covariance {
        Covariance::new(n, matrix, None, None).unwrap()
    }

    fn weighted(weights: Weights, req: &AceRequest) -> AceRequest {
        AceRequest {
            weights: Some(weights),
            ..req.clone()
        }
    }

    const WEIGHTS: [Weights; 3] = [Weights::Pairs, Weights::InverseVariance, Weights::Gls];

    fn se(x: Result<Solved, Reason>) -> f64 {
        x.unwrap().se.unwrap()
    }

    fn near(actual: f64, expected: f64, relative: f64) {
        assert!(
            (actual - expected).abs() <= relative * expected.abs(),
            "{actual} vs {expected}"
        );
    }

    #[test]
    fn falconer_ses_are_the_closed_forms_under_every_weighting() {
        let cells = [
            cell(&[MZ], (0, 0), 0.62, 911),
            cell(&[FS], (0, 0), 0.35, 5222),
        ];
        let (mm, mf, ff) = (0.0021, 0.0004, 0.0009);
        let covariance = with(2, vec![mm, mf, mf, ff]);
        let req = request(Components::AC, Some(&["MZ", "FS"]));
        for weights in WEIGHTS {
            let fit = super::fit(&cells, &weighted(weights, &req), Some(&covariance)).unwrap();
            let t = &fit.traits[0];
            close(t.a, 2.0 * (0.62 - 0.35), 1e-14);
            near(se(t.a), (4.0 * mm - 8.0 * mf + 4.0 * ff).sqrt(), 1e-13);
            near(se(t.c), (mm - 4.0 * mf + 4.0 * ff).sqrt(), 1e-13);
            near(se(t.e), mm.sqrt(), 1e-13);
            assert_eq!(fit.rows[1].variance, Some(ff));
        }
    }

    /// `a⁻¹` of the `m x m` `a` by Gauss-Jordan elimination.
    fn inverse(a: &[f64], m: usize) -> Vec<f64> {
        let mut a = a.to_vec();
        let mut out: Vec<f64> = (0..m * m)
            .map(|x| if x / m == x % m { 1.0 } else { 0.0 })
            .collect();
        for k in 0..m {
            let p = (k..m)
                .max_by(|&i, &j| a[i * m + k].abs().total_cmp(&a[j * m + k].abs()))
                .unwrap();
            for j in 0..m {
                a.swap(k * m + j, p * m + j);
                out.swap(k * m + j, p * m + j);
            }
            let d = a[k * m + k];
            for j in 0..m {
                a[k * m + j] /= d;
                out[k * m + j] /= d;
            }
            for i in (0..m).filter(|&i| i != k) {
                let f = a[i * m + k];
                for j in 0..m {
                    a[i * m + j] -= f * a[k * m + j];
                    out[i * m + j] -= f * out[k * m + j];
                }
            }
        }
        out
    }

    /// `X_rows`, `y` and `Σ` of the four one-trait rows, and the cells.
    fn overidentified() -> ([[f64; 2]; 4], [f64; 4], Vec<f64>, Vec<AceCell>) {
        let x = [[1.0, 1.0], [0.5, 1.0], [0.25, 1.0], [0.25, 0.0]];
        let y = [0.71, 0.42, 0.27, 0.11];
        let mut sigma = positive_definite(4);
        sigma[1] += 0.000_02;
        sigma[4] += 0.000_02;
        let cells = [MZ, FS, MHS, PHS]
            .iter()
            .zip(y)
            .map(|(&cat, r)| cell(&[cat], (0, 0), r, 500))
            .collect();
        (x, y, sigma, cells)
    }

    /// `(Xᵀ W X)⁻¹ Xᵀ W` of the four rows, `W` given whole.
    fn wls_map(x: &[[f64; 2]; 4], w: &[f64]) -> [[f64; 4]; 2] {
        let mut xtwx = [0.0; 4];
        let mut xtw = [[0.0; 4]; 2];
        for a in 0..2 {
            for j in 0..4 {
                xtw[a][j] = (0..4).map(|i| x[i][a] * w[i * 4 + j]).sum();
            }
            for b in 0..2 {
                xtwx[a * 2 + b] = (0..4).map(|j| xtw[a][j] * x[j][b]).sum();
            }
        }
        let inv = inverse(&xtwx, 2);
        let mut map = [[0.0; 4]; 2];
        for a in 0..2 {
            for j in 0..4 {
                map[a][j] = (0..2).map(|b| inv[a * 2 + b] * xtw[b][j]).sum();
            }
        }
        map
    }

    /// `g Σ hᵀ` over four cells.
    fn quadratic(g: &[f64], sigma: &[f64], h: &[f64]) -> f64 {
        (0..4)
            .map(|i| (0..4).map(|j| g[i] * sigma[i * 4 + j] * h[j]).sum::<f64>())
            .sum()
    }

    #[test]
    fn inverse_variance_is_weighted_least_squares_with_a_sandwich() {
        let (x, y, sigma, cells) = overidentified();
        let w: Vec<f64> = (0..16)
            .map(|k| if k / 4 == k % 4 { 1.0 / sigma[k] } else { 0.0 })
            .collect();
        let map = wls_map(&x, &w);
        let req = weighted(
            Weights::InverseVariance,
            &request(Components::AC, Some(SHARED)),
        );
        let fit = super::fit(&cells, &req, Some(&with(4, sigma.clone()))).unwrap();
        let t = &fit.traits[0];
        let e: Vec<f64> = (0..4).map(|j| -map[0][j] - map[1][j]).collect();
        for (got, g, constant) in [
            (t.a, &map[0][..], 0.0),
            (t.c, &map[1][..], 0.0),
            (t.e, &e[..], 1.0),
        ] {
            let value: f64 = (0..4).map(|j| g[j] * y[j]).sum();
            near(got.unwrap().value, constant + value, 1e-12);
            near(se(got), quadratic(g, &sigma, g).sqrt(), 1e-12);
        }
        let pairs = self::fit(&cells, &request(Components::AC, Some(SHARED))).unwrap();
        let by_pairs = pairs.traits[0].a.unwrap().value;
        assert!(
            (t.a.unwrap().value - by_pairs).abs() > 1e-4,
            "weights do not matter"
        );
    }

    #[test]
    fn gls_is_the_generalized_least_squares_fit_and_its_variance() {
        let (x, y, sigma, cells) = overidentified();
        let precision = inverse(&sigma, 4);
        let map = wls_map(&x, &precision);
        let mut information = [0.0; 4];
        for a in 0..2 {
            for b in 0..2 {
                information[a * 2 + b] = (0..4)
                    .map(|i| {
                        (0..4)
                            .map(|j| x[i][a] * precision[i * 4 + j] * x[j][b])
                            .sum::<f64>()
                    })
                    .sum();
            }
        }
        let variance = inverse(&information, 2);
        let req = weighted(Weights::Gls, &request(Components::AC, Some(SHARED)));
        let fit = super::fit(&cells, &req, Some(&with(4, sigma.clone()))).unwrap();
        let t = &fit.traits[0];
        for (k, got) in [t.a, t.c].into_iter().enumerate() {
            near(
                got.unwrap().value,
                (0..4).map(|j| map[k][j] * y[j]).sum(),
                1e-12,
            );
            near(se(got), variance[k * 2 + k].sqrt(), 1e-12);
            near(se(got), quadratic(&map[k], &sigma, &map[k]).sqrt(), 1e-12);
        }
        near(se(t.e), variance.iter().sum::<f64>().sqrt(), 1e-12);
    }

    /// Every value of a fit, by name, with its SE.
    fn quantities(fit: &AceFit) -> Vec<(String, Result<Solved, Reason>)> {
        let mut out = vec![];
        for (t, shares) in fit.traits.iter().enumerate() {
            for (name, x) in [("A", shares.a), ("C", shares.c), ("E", shares.e)] {
                out.push((format!("{name}{t}"), x));
            }
        }
        if let Some(c) = &fit.cross {
            for (name, x) in [
                ("sA", c.s_a),
                ("sC", c.s_c),
                ("sE", c.s_e),
                ("rA", c.r_a),
                ("rC", c.r_c),
                ("rE", c.r_e),
            ] {
                out.push((name.to_string(), x));
            }
        }
        out
    }

    #[test]
    fn every_se_is_the_finite_difference_gradients_variance() {
        let cells = noisy_cells();
        let n = cells.len();
        let sigma = like_a_result(&cells);
        let covariance = with(n, sigma.clone());
        let h = 1e-6;
        for weights in WEIGHTS {
            let req = weighted(weights, &request(Components::AC, Some(SHARED)));
            let at = |cells: &[AceCell]| {
                quantities(&super::fit(cells, &req, Some(&covariance)).unwrap())
            };
            let base = at(&cells);
            let shifted = |i: usize, by: f64| {
                let mut moved = cells.clone();
                let value = moved[i].crude[0].value.as_mut().unwrap();
                *value += by;
                at(&moved)
            };
            let steps: Vec<_> = (0..n).map(|i| (shifted(i, h), shifted(i, -h))).collect();
            for (q, (name, x)) in base.iter().enumerate() {
                let g: Vec<f64> = steps
                    .iter()
                    .map(|(up, down)| {
                        (up[q].1.unwrap().value - down[q].1.unwrap().value) / (2.0 * h)
                    })
                    .collect();
                let variance: f64 = (0..n)
                    .map(|i| (0..n).map(|j| g[i] * sigma[i * n + j] * g[j]).sum::<f64>())
                    .sum();
                let got = se(*x);
                assert!(
                    (got - variance.sqrt()).abs() <= 1e-6 * got,
                    "{weights:?} {name}: {got} vs {}",
                    variance.sqrt()
                );
            }
        }
    }

    #[test]
    fn the_within_person_row_carries_se_and_re() {
        let cells = noisy_cells();
        let n = cells.len();
        let mut sigma = like_a_result(&cells);
        let base = super::fit(
            &cells,
            &request(Components::AC, Some(SHARED)),
            Some(&with(n, sigma.clone())),
        )
        .unwrap();
        let wp = n - 1;
        sigma[wp * n + wp] *= 4.0;
        let wider = super::fit(
            &cells,
            &request(Components::AC, Some(SHARED)),
            Some(&with(n, sigma.clone())),
        )
        .unwrap();
        let (b, w) = (base.cross.unwrap(), wider.cross.unwrap());
        assert!(se(w.s_e) > se(b.s_e) && se(w.r_e) > se(b.r_e));
        near(se(w.s_a), se(b.s_a), 1e-9);
        assert_eq!(base.traits, wider.traits);
        assert_eq!(wider.rows[wp].variance, Some(sigma[wp * n + wp]));
        for k in 0..n {
            sigma[wp * n + k] = f64::NAN;
            sigma[k * n + wp] = f64::NAN;
        }
        let without = super::fit(
            &cells,
            &request(Components::AC, Some(SHARED)),
            Some(&with(n, sigma)),
        )
        .unwrap();
        let cross = without.cross.unwrap();
        assert_eq!(cross.s_e.unwrap().se, Err(Reason::RowWithoutSe));
        assert_eq!(cross.r_e.unwrap().se, Err(Reason::RowWithoutSe));
        assert_eq!(cross.s_a.unwrap().se, b.s_a.unwrap().se);
    }

    #[test]
    fn a_categories_subset_reads_its_cells_slots() {
        let cells = [
            cell(&[PHS], (0, 0), 0.1, 300),
            cell(&[MZ], (0, 0), 0.62, 911),
            cell(&[C1], (0, 0), 0.05, 2000),
            cell(&[FS], (0, 0), 0.35, 5222),
        ];
        let (mm, mf, ff) = (0.0021, 0.0004, 0.0009);
        let mut sigma = positive_definite(4);
        for (i, j, v) in [(1, 1, mm), (1, 3, mf), (3, 1, mf), (3, 3, ff)] {
            sigma[i * 4 + j] = v;
        }
        let mut req = request(Components::AC, Some(&["MZ", "FS"]));
        req.categories = Some(vec!["FS".to_string(), "MZ".to_string()]);
        for weights in WEIGHTS {
            let fit = super::fit(
                &cells,
                &weighted(weights, &req),
                Some(&with(4, sigma.clone())),
            )
            .unwrap();
            assert_eq!(fit.rows.len(), 2);
            let t = &fit.traits[0];
            near(se(t.a), (4.0 * mm - 8.0 * mf + 4.0 * ff).sqrt(), 1e-13);
            near(se(t.c), (mm - 4.0 * mf + 4.0 * ff).sqrt(), 1e-13);
            assert_eq!(fit.rows[0].variance, Some(mm));
        }
    }

    #[test]
    fn the_fit_reads_the_covariance_of_its_form() {
        let mut cells = vec![cell(&[FS], (0, 0), 0.4, 900)];
        cells[0].stratified = Some(StratifiedEstimate {
            value: Ok(0.35),
            n: 850,
            n_entries: None,
        });
        let covariance =
            Covariance::new(1, vec![0.01], Some(vec![0.04]), Some(vec![0.09])).unwrap();
        let req = request(Components::A, None);
        for (estimator, estimate, sd) in [
            (Estimator::Primary, Estimate::Pooled, 0.1),
            (Estimator::Phi, Estimate::Pooled, 0.2),
            (Estimator::Primary, Estimate::Stratified, 0.3),
        ] {
            let req = AceRequest {
                estimator,
                estimate: Some(estimate),
                ..req.clone()
            };
            let fit = super::fit(&cells, &req, Some(&covariance)).unwrap();
            near(se(fit.traits[0].a), 2.0 * sd, 1e-15);
        }
        let pooled_only = with(1, vec![0.01]);
        let req = AceRequest {
            estimate: Some(Estimate::Stratified),
            ..req
        };
        let fit = super::fit(&cells, &req, Some(&pooled_only)).unwrap();
        assert_eq!(fit.traits[0].a.unwrap().se, Err(Reason::NoCovariance));
        assert_eq!(fit.rows[0].variance, None);
        assert_eq!(
            super::fit(&cells, &weighted(Weights::Gls, &req), Some(&pooled_only))
                .unwrap_err()
                .code(),
            "missing_covariance"
        );
    }

    fn falconer() -> (Vec<AceCell>, AceRequest) {
        (
            vec![
                cell(&[MZ], (0, 0), 0.62, 911),
                cell(&[FS], (0, 0), 0.35, 5222),
                cell(&[PHS], (0, 0), 0.12, 700),
            ],
            request(Components::AC, Some(&["MZ", "FS"])),
        )
    }

    #[test]
    fn without_a_covariance_every_se_is_no_covariance() {
        let fit = fit(&noisy_cells(), &request(Components::A, Some(SHARED))).unwrap();
        for (name, x) in quantities(&fit) {
            match x {
                Ok(solved) => assert_eq!(solved.se, Err(Reason::NoCovariance), "{name}"),
                Err(reason) => assert_eq!(reason, Reason::NotRequested, "{name}"),
            }
        }
        assert!(fit.rows.iter().all(|r| r.variance.is_none()));
        let tree = fit.to_value();
        let Value::List(traits) = lookup(&tree, "traits") else {
            panic!("traits is not a list")
        };
        assert_eq!(lookup(&traits[0], "A_se"), &Value::Null);
        assert_eq!(
            lookup(&traits[0], "A_se_reason"),
            &Value::Str("no_covariance")
        );
        assert_eq!(
            lookup(&traits[0], "C_se_reason"),
            &Value::Str("not_requested")
        );
        let cross = lookup(&tree, "cross");
        assert_eq!(lookup(cross, "rC_se_reason"), &Value::Str("not_requested"));
        let Value::List(rows) = lookup(&tree, "rows") else {
            panic!("rows is not a list")
        };
        assert_eq!(lookup(&rows[0], "variance"), &Value::Null);
    }

    #[test]
    fn a_row_without_an_se_reaches_only_what_reads_it() {
        let (cells, req) = falconer();
        let nan = f64::NAN;
        let sigma = vec![
            0.0021, 0.0004, nan, //
            0.0004, 0.0009, nan, //
            nan, nan, nan,
        ];
        let covariance = with(3, sigma);
        let pairs = super::fit(&cells, &req, Some(&covariance)).unwrap();
        assert_eq!(pairs.traits[0].rows, [0, 1, 2]);
        let falconer = AceRequest {
            categories: Some(vec!["MZ".to_string(), "FS".to_string()]),
            ..req.clone()
        };
        let fs_without = with(3, vec![0.0021, nan, 0.0, nan, nan, nan, 0.0, nan, 0.0009]);
        let t = super::fit(&cells, &falconer, Some(&fs_without))
            .unwrap()
            .traits
            .remove(0);
        assert_eq!(t.a.unwrap().se, Err(Reason::RowWithoutSe));
        assert_eq!(t.c.unwrap().se, Err(Reason::RowWithoutSe));
        near(se(t.e), 0.0021f64.sqrt(), 1e-14);
        assert_eq!(pairs.rows[2].variance, None);
        assert_eq!(pairs.traits[0].a.unwrap().se, Err(Reason::RowWithoutSe));
        for weights in [Weights::InverseVariance, Weights::Gls] {
            let fit = super::fit(&cells, &weighted(weights, &req), Some(&covariance)).unwrap();
            assert_eq!(fit.rows[2].value, Err("no_se".to_string()));
            assert_eq!(fit.traits[0].rows, [0, 1]);
            near(
                se(fit.traits[0].a),
                (4.0 * 0.0021 - 8.0 * 0.0004 + 4.0 * 0.0009f64).sqrt(),
                1e-13,
            );
        }
    }

    #[test]
    fn an_indefinite_covariance_gives_negative_variance_or_no_gls() {
        let (cells, req) = falconer();
        let req = AceRequest {
            categories: Some(vec!["MZ".to_string(), "FS".to_string()]),
            ..req
        };
        let sigma = vec![
            1.0, 1.5, 0.0, //
            1.5, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let covariance = with(3, sigma);
        let pairs = super::fit(&cells, &req, Some(&covariance)).unwrap();
        let t = &pairs.traits[0];
        assert_eq!(t.a.unwrap().se, Err(Reason::NegativeVariance));
        assert_eq!(t.c.unwrap().se, Err(Reason::NegativeVariance));
        near(se(t.e), 1.0, 1e-14);
        let gls = super::fit(&cells, &weighted(Weights::Gls, &req), Some(&covariance)).unwrap();
        let t = &gls.traits[0];
        for x in [t.a, t.c, t.e] {
            assert_eq!(x, Err(Reason::CovarianceNotPositiveDefinite));
        }
        let tree = gls.to_value();
        let Value::List(traits) = lookup(&tree, "traits") else {
            panic!("traits is not a list")
        };
        assert_eq!(
            lookup(&traits[0], "A_se_reason"),
            &Value::Str("covariance_not_positive_definite")
        );
        let method = lookup(&tree, "method");
        assert!(matches!(lookup(method, "weights"), Value::Str(text) if text.starts_with("gls")));
        assert_eq!(
            lookup(lookup(&tree, "settings"), "weights"),
            &Value::Str("gls")
        );
    }

    #[test]
    fn weights_by_the_covariance_need_one_of_the_cells() {
        let (cells, req) = falconer();
        for weights in [Weights::InverseVariance, Weights::Gls] {
            let err = super::fit(&cells, &weighted(weights, &req), None).unwrap_err();
            assert_eq!(err, Error::MissingCovariance { weights });
            assert_eq!(err.code(), "missing_covariance");
        }
        let err = super::fit(&cells, &req, Some(&with(2, vec![1.0, 0.0, 0.0, 1.0]))).unwrap_err();
        assert_eq!(
            err,
            Error::CovarianceShapeMismatch {
                cells: 3,
                expected_length: 9,
                actual_length: 4
            }
        );
        assert_eq!(err.code(), "covariance_shape_mismatch");
        assert_eq!(
            Covariance::new(2, vec![1.0; 4], Some(vec![1.0; 3]), None),
            Err(Error::CovarianceShapeMismatch {
                cells: 2,
                expected_length: 4,
                actual_length: 3
            })
        );
        assert!(Covariance::new(2, vec![1.0; 4], None, Some(vec![2.0; 4])).is_ok());
    }

    fn default(req: &AceRequest) -> AceRequest {
        AceRequest {
            weights: None,
            ..req.clone()
        }
    }

    /// `noisy_cells` with a covariance like a result's whose trait-1 block
    /// has no Cholesky factor (two of its cells correlate 1.5), and one
    /// cell of each trait with no SE.
    fn one_trait_indefinite() -> (Vec<AceCell>, Covariance, [usize; 2]) {
        let cells = noisy_cells();
        let n = cells.len();
        let of = |t: usize| -> Vec<usize> {
            (0..n)
                .filter(|&i| cells[i].first_trait == t && cells[i].second_trait == t)
                .collect()
        };
        let (zero, one) = (of(0), of(1));
        let mut sigma = like_a_result(&cells);
        let (i, j) = (one[0], one[1]);
        let product = (sigma[i * n + i] * sigma[j * n + j]).sqrt();
        sigma[i * n + j] = 1.5 * product;
        sigma[j * n + i] = 1.5 * product;
        let without = [zero[3], one[3]];
        for k in without {
            for x in 0..n {
                sigma[k * n + x] = f64::NAN;
                sigma[x * n + k] = f64::NAN;
            }
        }
        (cells, with(n, sigma), without)
    }

    #[test]
    fn the_default_takes_gls_per_fit_and_pairs_where_its_block_has_no_factor() {
        let (cells, covariance, [no_se_0, no_se_1]) = one_trait_indefinite();
        let req = request(Components::AC, Some(SHARED));
        let fit = |w: Option<Weights>| {
            let req = AceRequest {
                weights: w,
                ..req.clone()
            };
            super::fit(&cells, &req, Some(&covariance)).unwrap()
        };
        let (by_default, gls, pairs) = (
            fit(None),
            fit(Some(Weights::Gls)),
            fit(Some(Weights::Pairs)),
        );
        let weights = |f: &AceFit| {
            let mut out: Vec<Weights> = f.traits.iter().map(|t| t.weights).collect();
            out.extend(f.cross.as_ref().map(|c| c.weights));
            out
        };
        assert_eq!(
            weights(&by_default),
            [Weights::Gls, Weights::Pairs, Weights::Gls]
        );
        assert_eq!(weights(&gls), [Weights::Gls; 3]);
        assert_eq!(by_default.traits[0], gls.traits[0]);
        assert_eq!(by_default.traits[1], pairs.traits[1]);
        let t = &gls.traits[1];
        for x in [t.a, t.c, t.e] {
            assert_eq!(x, Err(Reason::CovarianceNotPositiveDefinite));
        }
        let (d, g) = (
            by_default.cross.as_ref().unwrap(),
            gls.cross.as_ref().unwrap(),
        );
        assert_eq!([d.s_a, d.s_c, d.s_e], [g.s_a, g.s_c, g.s_e]);
        assert_eq!(d.rows, g.rows);
        assert_eq!(d.r_a.unwrap().se, Err(Reason::RowWithoutSe));
        assert_eq!(g.r_a, Err(Reason::CovarianceNotPositiveDefinite));
        // The pairs fit takes a row with no SE; the gls fit leaves it out.
        assert_eq!(by_default.rows[no_se_0].value, Err("no_se".to_string()));
        assert!(!by_default.traits[0].rows.contains(&no_se_0));
        assert!(by_default.rows[no_se_1].value.is_ok());
        assert!(by_default.traits[1].rows.contains(&no_se_1));
        assert_eq!(
            by_default.traits[1].a.unwrap().se,
            Err(Reason::RowWithoutSe)
        );
        assert_eq!(gls.rows[no_se_1].value, Err("no_se".to_string()));
        let tree = by_default.to_value();
        assert_eq!(
            lookup(lookup(&tree, "settings"), "weights"),
            &Value::Str("default")
        );
        let Value::List(traits) = lookup(&tree, "traits") else {
            panic!("traits is not a list")
        };
        assert_eq!(lookup(&traits[1], "weights"), &Value::Str("pairs"));
        assert_eq!(
            lookup(lookup(&tree, "cross"), "weights"),
            &Value::Str("gls")
        );
        let method = lookup(&tree, "method");
        assert!(
            matches!(lookup(method, "weights"), Value::Str(text) if text.starts_with("default"))
        );
    }

    #[test]
    fn the_default_is_pairs_without_the_covariance_read_or_for_phi() {
        let cells = noisy_cells();
        let req = request(Components::AC, Some(SHARED));
        let pairs = fit(&cells, &req).unwrap();
        let by_default = fit(&cells, &default(&req)).unwrap();
        assert_eq!(by_default.traits, pairs.traits);
        assert_eq!(by_default.cross, pairs.cross);
        assert_eq!(by_default.rows, pairs.rows);
        let covariance = with(cells.len(), like_a_result(&cells));
        let phi = AceRequest {
            estimator: Estimator::Phi,
            ..default(&req)
        };
        let phi_cov = Covariance::new(
            cells.len(),
            like_a_result(&cells),
            Some(like_a_result(&cells)),
            None,
        )
        .unwrap();
        let fit_phi = super::fit(&cells, &phi, Some(&phi_cov)).unwrap();
        assert!(fit_phi.traits.iter().all(|t| t.weights == Weights::Pairs));
        let explicit = AceRequest {
            weights: Some(Weights::Pairs),
            ..phi.clone()
        };
        assert_eq!(
            fit_phi.traits,
            super::fit(&cells, &explicit, Some(&phi_cov))
                .unwrap()
                .traits
        );
        // A covariance without the phi matrix is not the covariance read.
        let no_phi = super::fit(&cells, &phi, Some(&covariance)).unwrap();
        assert!(no_phi.traits.iter().all(|t| t.weights == Weights::Pairs));
        let gls = super::fit(&cells, &default(&req), Some(&covariance)).unwrap();
        assert!(gls.traits.iter().all(|t| t.weights == Weights::Gls));
        let explicit =
            super::fit(&cells, &weighted(Weights::Gls, &req), Some(&covariance)).unwrap();
        assert_eq!(gls.traits, explicit.traits);
        assert_eq!(gls.cross, explicit.cross);
        assert_eq!(gls.rows, explicit.rows);
    }
}
