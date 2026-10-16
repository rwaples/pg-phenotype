//! The ACE Fit: additive genetic (A), common environment (C) and unique
//! environment (E) shares solved from a Relative-Pair Correlation's cells.
//!
//! On the correlation scale (each trait's variance 1) a cell k of trait t
//! expects `r_k = 2 phi_k A_t + s_k C_t`, with `phi_k = 2^-(degree+1)` the
//! nominal kinship of its categories and `s_k = 1` for a Shared-Environment
//! Category, else 0.  Each trait's unknowns are the weighted least-squares
//! solution over its same-trait cells, weight the cell's `n`; `E = 1 - A - C`.
//! Two traits add the cross solve over the cross-trait cells,
//! `r12_k = 2 phi_k sA + s_k sC`, and the Within-Person Cross-Trait
//! Correlation, `r_wp = sA + sC + sE`; then `rA = sA / sqrt(A_1 A_2)` and so
//! on.  A cross-trait cell weighs its complete orientations, so each counts
//! once: a symmetric cell's `n_entries`, a directed cell's `n` (one
//! orientation per cell).  Values are as solved, with no constraint.
//!
//! The function is pure over cell rows as a host holds them, so a host
//! passes the cells of the result it already has.

use crate::error::{Error, Position};
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
    /// The estimator's name (`pearson`, `tetrachoric`, `phi`).
    pub estimator: String,
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
    /// The host resolves its default before the call.
    pub estimate: Estimate,
}

/// Why a fitted value is undefined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The component is not among those requested.
    NotRequested,
    /// The cells that entered do not determine the solve's unknowns.
    NotIdentified,
    /// A cross-trait correlation's component is <= 0 in a trait.
    NonpositiveComponent,
}

impl Reason {
    pub fn name(self) -> &'static str {
        match self {
            Reason::NotRequested => "not_requested",
            Reason::NotIdentified => "not_identified",
            Reason::NonpositiveComponent => "nonpositive_component",
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
    /// left out.
    pub value: Result<f64, String>,
    /// The estimate's `n_entries` where it has one (a symmetric
    /// cross-trait cell's complete orientations), else its `n`.
    pub weight: u64,
}

/// One trait's shares.
#[derive(Clone, Debug, PartialEq)]
pub struct TraitFit {
    pub a: Result<f64, Reason>,
    pub c: Result<f64, Reason>,
    pub e: Result<f64, Reason>,
    /// Indices into [`AceFit::rows`] of the rows that entered.
    pub rows: Vec<usize>,
}

/// The cross-trait covariances and correlations of two traits.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossFit {
    pub s_a: Result<f64, Reason>,
    pub s_c: Result<f64, Reason>,
    pub s_e: Result<f64, Reason>,
    pub r_a: Result<f64, Reason>,
    pub r_c: Result<f64, Reason>,
    pub r_e: Result<f64, Reason>,
    /// Indices into [`AceFit::rows`] of the rows that entered.
    pub rows: Vec<usize>,
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
}

const METHOD: [(&str, &str); 6] = [
    (
        "model",
        "correlation scale, each trait's variance 1: r_k = 2 phi_k A + s_k C over a trait's \
         same-trait cells; with two traits r12_k = 2 phi_k sA + s_k sC over the cross-trait \
         cells and r_wp = sA + sC + sE within person. phi_k = 2^-(degree+1), the nominal \
         kinship; s_k = 1 for a Shared-Environment Category, else 0. Components not requested \
         are fixed at 0",
    ),
    (
        "weights",
        "what entered the estimate read, the pooled cell or the stratified form: a same-trait \
         row's n (unordered pairs) and the within-person row's n (rows); a cross-trait row's \
         complete orientations, each once: a symmetric cell's n_entries, a directed cell's n",
    ),
    (
        "solve",
        "weighted least squares by Householder QR with column pivoting of the sqrt(weight)-scaled \
         design, rows sorted by decreasing largest entry; a requested component zero in every \
         row that entered is not_identified and left out of the solve; in the cross solve, sC \
         zero in every relative row leaves sC and sE to the within-person row alone, so both are \
         not_identified and that row is left out; any other rank deficiency is not_identified for \
         the solve's unknowns; values unconstrained",
    ),
    ("E", "the remainder: E = 1 - A - C"),
    (
        "cross",
        "rA = sA / sqrt(A_1 A_2), rC = sC / sqrt(C_1 C_2), rE = sE / sqrt(E_1 E_2); \
         nonpositive_component when a trait's component is <= 0; |r| > 1 is flagged out_of_range. \
         sE needs the within-person cell",
    ),
    ("se", "none"),
];

/// The ACE Fit of `cells`, one cell row each, as `request` asks.
///
/// A cell whose selected value is undefined (`n` = 0 included) does not
/// enter; its row names the reason.
///
/// # Errors
///
/// `trait_count` (a trait index past 1); `no_stratified_form` (stratified
/// with phi); `missing_shared_environment` (C without
/// `shared_environment`); `unknown_category` (a `shared_environment` code);
/// `unknown_entry` (a `categories` name no cell has); then over the
/// entries that enter: `empty_category_group`, `mixed_kinship`,
/// `mixed_shared_environment`, `missing_estimator`; and
/// `no_stratified_form` when stratified and none has a stratified form.
pub fn fit(cells: &[AceCell], request: &AceRequest) -> Result<AceFit, Error> {
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
    if request.estimate == Estimate::Stratified && request.estimator == Estimator::Phi {
        return Err(Error::NoStratifiedForm {
            estimator: Estimator::Phi,
        });
    }
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
    let rows = candidates
        .iter()
        .map(|&(i, c)| row(c, i, request, shared.as_ref()))
        .collect::<Result<Vec<Row>, Error>>()?;
    if request.estimate == Estimate::Stratified
        && candidates.iter().all(|(_, c)| c.stratified.is_none())
    {
        return Err(Error::NoStratifiedForm {
            estimator: Estimator::Primary,
        });
    }
    let components = request.components;
    let traits: Vec<TraitFit> = (0..=top).map(|t| trait_fit(&rows, t, components)).collect();
    let cross = (top == 1).then(|| cross_fit(&rows, &traits, components));
    Ok(AceFit {
        traits,
        cross,
        rows,
        request: request.clone(),
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

/// A cell's candidate row: its coefficients and the value it gives.
fn row(
    cell: &AceCell,
    position: usize,
    request: &AceRequest,
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
    let (value, weight) = match request.estimate {
        Estimate::Stratified => match &cell.stratified {
            Some(s) => (s.value.clone(), s.n_entries.unwrap_or(s.n)),
            None => (Err("no_stratified_form".to_string()), 0),
        },
        Estimate::Pooled => {
            let estimate = match request.estimator {
                Estimator::Primary => cell.crude.first(),
                Estimator::Phi => cell.crude.iter().find(|e| e.estimator == "phi"),
            };
            let estimate = estimate.ok_or_else(|| Error::MissingEstimator {
                estimator: request.estimator,
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
    })
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

/// The indices of the defined rows a solve `takes`.
fn entered(rows: &[Row], takes: impl Fn(&Row) -> bool) -> Vec<usize> {
    (0..rows.len())
        .filter(|&i| rows[i].value.is_ok() && takes(&rows[i]))
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

/// The WLS solution over `entered`, or `None` when it is not identified.
fn solve(rows: &[Row], entered: &[usize], columns: &Columns) -> Option<Vec<f64>> {
    let p = columns.iter().filter(|c| c.is_ok()).count();
    let mut design = vec![Vec::with_capacity(entered.len()); p];
    let mut y = Vec::with_capacity(entered.len());
    for &i in entered {
        let row = &rows[i];
        let Ok(value) = row.value else { continue };
        let scale = (row.weight as f64).sqrt();
        for (column, x) in design.iter_mut().zip(coefficients(row.kind, columns)) {
            column.push(scale * x);
        }
        y.push(scale * value);
    }
    least_squares(design, y)
}

/// The least-squares `b` of `X b = y`, `X` given by columns, or `None`
/// when `X` has rank below its columns.
///
/// Householder QR with column pivoting (each step takes the remaining
/// column of largest norm), so `|R_kk|` falls along the diagonal and its
/// first entry is the largest, over rows sorted by decreasing largest entry.
/// Without the sort, a cell of n = 3 beside one of n = 1e9 loses about
/// 5e-12 of a Falconer A; with it the error stays near 1e-15 of the
/// correlations' scale.  Full rank needs every `|R_kk|` above
/// `m * eps * |R_11|` (`m >= p` rows): Householder QR is backward stable, the
/// computed `R` being the exact factor of `X + dX` with `|dX|` of order
/// `m * eps * |X|`, so a diagonal below that bound is
/// indistinguishable from an exact zero (the bound numpy's `matrix_rank`
/// puts on singular values).  The design's entries are `sqrt(n)` times `2^-d`
/// (`d <= 5`), 0 or 1, so an identified design sits orders of magnitude
/// above the bound unless its weights differ by a factor near `1e20`.
fn least_squares(columns: Vec<Vec<f64>>, y: Vec<f64>) -> Option<Vec<f64>> {
    let (m, p) = (y.len(), columns.len());
    if m < p {
        return None;
    }
    let largest = |i: usize| columns.iter().map(|c| c[i].abs()).fold(0.0, f64::max);
    let mut by_size: Vec<usize> = (0..m).collect();
    by_size.sort_by(|&i, &j| largest(j).total_cmp(&largest(i)));
    let mut columns: Vec<Vec<f64>> = columns
        .iter()
        .map(|c| by_size.iter().map(|&i| c[i]).collect())
        .collect();
    let mut y: Vec<f64> = by_size.iter().map(|&i| y[i]).collect();
    let mut order: Vec<usize> = (0..p).collect();
    let mut diagonal = Vec::with_capacity(p);
    for k in 0..p {
        let norms: Vec<f64> = columns[k..]
            .iter()
            .map(|c| c[k..].iter().map(|x| x * x).sum())
            .collect();
        let pivot = (0..norms.len()).max_by(|&i, &j| norms[i].total_cmp(&norms[j]))? + k;
        columns.swap(k, pivot);
        order.swap(k, pivot);
        let alpha = norms[pivot - k].sqrt();
        if alpha == 0.0 {
            return None;
        }
        let r_kk = if columns[k][k] >= 0.0 { -alpha } else { alpha };
        let mut v = columns[k][k..].to_vec();
        v[0] -= r_kk;
        let vv: f64 = v.iter().map(|x| x * x).sum();
        let reflect = |target: &mut [f64]| {
            let f = 2.0 * v.iter().zip(&*target).map(|(a, b)| a * b).sum::<f64>() / vv;
            target.iter_mut().zip(&v).for_each(|(t, a)| *t -= f * a);
        };
        for column in &mut columns[k + 1..] {
            reflect(&mut column[k..]);
        }
        reflect(&mut y[k..]);
        columns[k][k] = r_kk;
        diagonal.push(r_kk);
    }
    let tolerance = m as f64 * f64::EPSILON * diagonal.first().map_or(0.0, |d| d.abs());
    if diagonal.iter().any(|d| d.abs() <= tolerance) {
        return None;
    }
    let mut b = vec![0.0; p];
    for k in (0..p).rev() {
        let tail: f64 = (k + 1..p).map(|j| columns[j][k] * b[j]).sum();
        b[k] = (y[k] - tail) / columns[k][k];
    }
    let mut solution = vec![0.0; p];
    for (k, &column) in order.iter().enumerate() {
        solution[column] = b[k];
    }
    Some(solution)
}

/// Each present column's solved value, in A, C, E order.
fn unknowns(solution: Option<Vec<f64>>, columns: Columns) -> [Result<f64, Reason>; 3] {
    let mut values = solution.unwrap_or_default().into_iter();
    columns.map(|column| column.and_then(|()| values.next().ok_or(Reason::NotIdentified)))
}

fn trait_fit(rows: &[Row], t: usize, components: Components) -> TraitFit {
    let rows_in = entered(rows, |r| {
        r.first_trait == t && r.second_trait == t && r.kind != RowKind::WithinPerson
    });
    let columns = supported(
        rows,
        &rows_in,
        [
            requested(components.a()),
            requested(components.c()),
            Err(Reason::NotRequested),
        ],
    );
    let [a, c, _] = unknowns(solve(rows, &rows_in, &columns), columns);
    let share = |x: Result<f64, Reason>| match x {
        Err(Reason::NotRequested) => Ok(0.0),
        other => other,
    };
    let e = share(a).and_then(|a| share(c).map(|c| 1.0 - a - c));
    TraitFit {
        a,
        c,
        e,
        rows: rows_in,
    }
}

/// The cross solve.  sE is in the within-person row alone, so with sC
/// zero in every relative row the two share that row and neither is
/// identified; the row is then left out and sA solved from the relatives.
fn cross_fit(rows: &[Row], traits: &[TraitFit], components: Components) -> CrossFit {
    let mut rows_in = entered(rows, |r| {
        r.first_trait != r.second_trait || r.kind == RowKind::WithinPerson
    });
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
    let [s_a, s_c, s_e] = unknowns(solve(rows, &rows_in, &columns), columns);
    let (one, two) = (&traits[0], &traits[1]);
    CrossFit {
        s_a,
        s_c,
        s_e,
        r_a: ratio(s_a, one.a, two.a),
        r_c: ratio(s_c, one.c, two.c),
        r_e: ratio(s_e, one.e, two.e),
        rows: rows_in,
    }
}

/// `s / sqrt(x1 x2)`, undefined with the first undefined operand's reason
/// or when a share is <= 0.
fn ratio(
    s: Result<f64, Reason>,
    x1: Result<f64, Reason>,
    x2: Result<f64, Reason>,
) -> Result<f64, Reason> {
    let (s, x1, x2) = (s?, x1?, x2?);
    if x1 <= 0.0 || x2 <= 0.0 {
        return Err(Reason::NonpositiveComponent);
    }
    Ok(s / (x1 * x2).sqrt())
}

fn either(out: &mut Vec<(&'static str, Value)>, keys: [&'static str; 2], x: Result<f64, Reason>) {
    let [value, reason] = keys;
    match x {
        Ok(v) => out.extend([(value, v.into()), (reason, Value::Null)]),
        Err(r) => out.extend([(value, Value::Null), (reason, r.name().into())]),
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
                either(&mut out, ["A", "A_reason"], fit.a);
                either(&mut out, ["C", "C_reason"], fit.c);
                either(&mut out, ["E", "E_reason"], fit.e);
                out.push(("rows", self.entries(&fit.rows)));
                Value::Map(out)
            })
            .collect();
        let cross = self.cross.as_ref().map_or(Value::Null, |fit| {
            let mut out = vec![];
            either(&mut out, ["sA", "sA_reason"], fit.s_a);
            either(&mut out, ["sC", "sC_reason"], fit.s_c);
            either(&mut out, ["sE", "sE_reason"], fit.s_e);
            for (keys, r) in [
                (["rA", "rA_reason", "rA_out_of_range"], fit.r_a),
                (["rC", "rC_reason", "rC_out_of_range"], fit.r_c),
                (["rE", "rE_reason", "rE_out_of_range"], fit.r_e),
            ] {
                either(&mut out, [keys[0], keys[1]], r);
                out.push((keys[2], r.ok().map(|r| r.abs() > 1.0).into()));
            }
            out.push(("rows", self.entries(&fit.rows)));
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
                    ("estimate", request.estimate.name().into()),
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
    use Category::{Av, C1, FS, MHS, MO, MZ, PHS};

    fn crude(estimator: &str, value: Result<f64, &str>) -> CrudeEstimate {
        CrudeEstimate {
            estimator: estimator.to_string(),
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
            estimate: Estimate::Pooled,
        }
    }

    fn close(actual: Result<f64, Reason>, expected: f64, relative: f64) {
        let actual = actual.unwrap();
        assert!(
            (actual - expected).abs() <= relative * expected.abs(),
            "{actual} vs {expected}: relative error {:e}",
            (actual - expected).abs() / expected.abs()
        );
    }

    fn operand_close(actual: Result<f64, Reason>, expected: f64, scale: f64) {
        let error = (actual.unwrap() - expected).abs() / scale;
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
            (refit.c.unwrap() - fit.c.unwrap()).abs() > 1e-3,
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
                    estimate,
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
        req.estimate = Estimate::Stratified;
        let fit = fit(&cells, &req).unwrap();
        assert_eq!(fit.rows.len(), 1);
        assert_eq!(fit.rows[0].weight, 850);
        close(fit.traits[0].a, 0.7, 1e-15);
        req.categories = None;
        let fit = super::fit(&cells, &req).unwrap();
        assert_eq!(fit.rows[1].value, Err("no_stratified_form".to_string()));
        assert_eq!(fit.traits[0].rows, [0]);
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
        req.estimate = Estimate::Stratified;
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
        assert!(matches!(lookup(&tree, "method"), Value::Map(m) if m.len() == METHOD.len()));
    }
}
