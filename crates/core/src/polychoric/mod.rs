//! The Polychoric Fit: the two-step maximum-likelihood latent correlation
//! of a table, its boundary flag, and each cell's two-step influence on ρ̂.
//! The Mate Correlation and the Tetrachoric both run it (ADR 0011).
//!
//! It keeps pedsum's numerics (ADR 0005): SciPy's cephes and `owens_t`
//! (`cephes`), the bivariate normal (`bvn`), AS 241 with pedsum's
//! rounding (`table::ndtri`), and Newton with SciPy's bounded Brent
//! (`fit`).

// `!(x > 0.0)` is meant: NaN fails each of these checks, as `not x > 0` does in pedsum.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub(crate) mod bvn;
pub(crate) mod cephes;
pub(crate) mod fit;
mod influence;
pub(crate) mod table;

#[cfg(test)]
mod primitives_parity;

use fit::{flag_boundary, newton_rho, Terms};
pub(crate) use influence::cell_influence;
use std::cmp::Ordering;
use table::{margin_status, thresholds, Grid, NoFit, Table, TINY};

/// Why a Polychoric Fit, its SE or its CI is missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reason {
    /// The fit is at a boundary, where the sandwich is not valid.
    Boundary,
    /// A side shows one level.
    ConstantMargin,
    /// A stratum with pairs has none at a level its side shows.
    EmptyCategory,
    /// The table is empty.
    NoCompletePairs,
    /// The sandwich is not finite.
    SandwichUndefined,
}

impl Reason {
    /// Every reason, in name order.
    pub const ALL: [Reason; 5] = [
        Reason::Boundary,
        Reason::ConstantMargin,
        Reason::EmptyCategory,
        Reason::NoCompletePairs,
        Reason::SandwichUndefined,
    ];

    /// The name hosts use.
    pub fn name(self) -> &'static str {
        match self {
            Reason::Boundary => "boundary",
            Reason::ConstantMargin => "constant_margin",
            Reason::EmptyCategory => "empty_category",
            Reason::NoCompletePairs => "no_complete_pairs",
            Reason::SandwichUndefined => "sandwich_undefined",
        }
    }
}

impl From<NoFit> for Reason {
    fn from(no_fit: NoFit) -> Reason {
        match no_fit {
            NoFit::NoPairs => Reason::NoCompletePairs,
            NoFit::ConstantMargin => Reason::ConstantMargin,
            NoFit::EmptyCategory => Reason::EmptyCategory,
        }
    }
}

/// A defined point estimate.  `boundary` is set for a latent correlation
/// only: whether ρ̂ sits at, or on a likelihood plateau reaching, a bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub value: f64,
    pub boundary: Option<bool>,
}

/// Below this a cell probability, a difference of bivariate normal CDFs near
/// 1, is mostly rounding (about 2e-15): its score and curvature, divided by
/// it, are noise, and a Newton step there can stop at once far from the
/// maximum.  A populated cell at the maximum has probability near its share
/// of the pairs, far above this.
const CELL_NOISE: f64 = 1e-12;

/// `grid[i+1, j+1] - grid[i, j+1] - grid[i+1, j] + grid[i, j]`.
#[inline]
fn corner(g: &Grid<f64>, i: usize, j: usize) -> f64 {
    g.at(i + 1, j + 1) - g.at(i, j + 1) - g.at(i + 1, j) + g.at(i, j)
}

/// The first step of a polychoric fit: weighted counts, thresholds per
/// stratum, and the populated stratum combinations' corner grids.
pub(crate) struct Tables {
    pub n: Table,
    pub a: Grid<f64>,
    pub b: Grid<f64>,
    /// Populated (row stratum, column stratum), row-major.
    pub combos: Vec<(usize, usize)>,
}

impl Tables {
    pub fn from_counts(
        n: Table,
        row_levels: &Grid<bool>,
        col_levels: &Grid<bool>,
    ) -> Result<Tables, NoFit> {
        if !n.data.iter().any(|&v| v != 0.0) {
            return Err(NoFit::NoPairs);
        }
        let (row_margin, col_margin) = (n.row_margin(), n.col_margin());
        for (margin, levels) in [(&row_margin, row_levels), (&col_margin, col_levels)] {
            if let Some(no_fit) = margin_status(margin, levels) {
                return Err(no_fit);
            }
        }
        Ok(Tables::of(n))
    }

    /// The first step of a table that passed [`Tables::from_counts`]'s checks.
    fn of(n: Table) -> Tables {
        let combos = (0..n.n_row_strata)
            .flat_map(|s| (0..n.n_col_strata).map(move |u| (s, u)))
            .filter(|&(s, u)| n.stratum_total(s, u) > 0.0)
            .collect();
        Tables {
            a: thresholds(&n.row_margin()),
            b: thresholds(&n.col_margin()),
            n,
            combos,
        }
    }

    /// ρ̂ by Newton from `start`, fitted in the table's canonical layout
    /// (issue #16).  Swapping the sides or reversing both sides' levels
    /// leaves ρ unchanged, and reversing one side's levels negates it.  At
    /// a boundary the likelihood is flat and the fit stops at a point that
    /// depends on the layout, so every layout is fitted as the greatest of
    /// its eight (by shape, then cell by cell) and the sign restored.
    ///
    /// A table equal to a sign-reversing layout of itself has a likelihood
    /// even in ρ, and gives ρ = 0 with the boundary flag taken there.  For a
    /// 2 x 2 table (equal rows or equal columns) 0 is the maximum; a larger
    /// table whose maxima are a pair ±r also gives 0, as its sign is not
    /// identified.
    pub fn fit(&self, start: f64) -> Point {
        let (layout, sign) = Layout::canonical(&self.n);
        if sign == 0.0 {
            return flag_boundary(|r| self.bound_nll(r), 0.0, self.nll(0.0));
        }
        let canonical;
        let tables = if layout == Layout::IDENTITY {
            self
        } else {
            canonical = Tables::of(layout.apply(&self.n));
            &canonical
        };
        let fit = newton_rho(
            |r| tables.terms(r),
            |r| tables.nll(r),
            |r| tables.bound_nll(r),
            if sign < 0.0 { -start } else { start },
        );
        Point {
            // `0.0 - v` negates exactly and never gives -0.0.
            value: if sign < 0.0 {
                0.0 - fit.value
            } else {
                fit.value
            },
            boundary: fit.boundary,
        }
    }

    /// The corner grid of `f(h, k)` for one stratum combination.
    pub fn grid(&self, (s, u): (usize, usize), f: impl Fn(f64, f64) -> f64) -> Grid<f64> {
        let (rows, cols) = (self.a.cols, self.b.cols);
        let mut g = Grid::filled(rows, cols, 0.0);
        for i in 0..rows {
            for j in 0..cols {
                *g.at_mut(i, j) = f(self.a.at(s, i), self.b.at(u, j));
            }
        }
        g
    }

    /// NLL and its first two ρ-derivatives (Olsson 1979 eqs 3, 4, 9), for
    /// Newton.  The Hessian is NaN, so Newton hands over to Brent, where a
    /// populated cell's probability is below [`CELL_NOISE`].
    pub fn terms(&self, rho: f64) -> Terms {
        let (terms, least) = self.accumulate(rho, true);
        if least < CELL_NOISE {
            Terms {
                hess: f64::NAN,
                ..terms
            }
        } else {
            terms
        }
    }

    /// The first two ρ-derivatives of the NLL alone (`nll` is 0), as a
    /// bootstrap draw's one Newton step needs.
    pub fn grad_hess(&self, rho: f64) -> (f64, f64) {
        let (t, _) = self.accumulate(rho, false);
        (t.grad, t.hess)
    }

    /// The terms, and the least probability of a populated cell.
    fn accumulate(&self, rho: f64, with_nll: bool) -> (Terms, f64) {
        let (mut nll, mut grad, mut hess) = (0.0, 0.0, 0.0);
        let mut least = f64::INFINITY;
        for &combo in &self.combos {
            let cdf = self.grid(combo, |h, k| bvn::cdf(h, k, rho));
            let pdf = self.grid(combo, |h, k| bvn::pdf_and_drho(h, k, rho).0);
            let drho = self.grid(combo, |h, k| bvn::pdf_and_drho(h, k, rho).1);
            for i in 0..self.n.k_row {
                for j in 0..self.n.k_col {
                    let count = self.n.at(combo.0, combo.1, i, j);
                    if count > 0.0 {
                        let pi = corner(&cdf, i, j).max(TINY);
                        least = least.min(pi);
                        let score = corner(&pdf, i, j) / pi;
                        let curvature = corner(&drho, i, j) / pi - score * score;
                        if with_nll {
                            nll += count * pi.ln();
                        }
                        grad += count * score;
                        hess += count * curvature;
                    }
                }
            }
        }
        (
            Terms {
                nll: -nll,
                grad: -grad,
                hess: -hess,
            },
            least,
        )
    }

    pub fn nll(&self, rho: f64) -> f64 {
        self.nll_at(rho, false)
    }

    /// The NLL at a bound, for the plateau test.  There a populated cell's
    /// difference of corner CDFs can be rounding (issue #8): a cell below
    /// [`CELL_NOISE`] is recomputed by [`bvn::rectangle`].  With none below,
    /// this is [`Tables::nll`] bit for bit.
    pub fn bound_nll(&self, rho: f64) -> f64 {
        self.nll_at(rho, true)
    }

    fn nll_at(&self, rho: f64, resolve_noise: bool) -> f64 {
        let mut nll = 0.0;
        for &combo in &self.combos {
            let (s, u) = combo;
            let cdf = self.grid(combo, |h, k| bvn::cdf(h, k, rho));
            for i in 0..self.n.k_row {
                for j in 0..self.n.k_col {
                    let count = self.n.at(s, u, i, j);
                    if count > 0.0 {
                        let mut pi = corner(&cdf, i, j);
                        if resolve_noise && pi < CELL_NOISE {
                            pi = bvn::rectangle(
                                (self.a.at(s, i), self.a.at(s, i + 1)),
                                (self.b.at(u, j), self.b.at(u, j + 1)),
                                rho,
                            );
                        }
                        nll += count * pi.max(TINY).ln();
                    }
                }
            }
        }
        -nll
    }
}

/// One of a table's eight layouts: optionally swap the sides (strata and
/// levels), then reverse the row and/or column levels within each stratum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    transpose: bool,
    reverse_rows: bool,
    reverse_cols: bool,
}

impl Layout {
    const IDENTITY: Layout = Layout {
        transpose: false,
        reverse_rows: false,
        reverse_cols: false,
    };

    fn all() -> impl Iterator<Item = Layout> {
        (0..8).map(|b| Layout {
            transpose: b & 4 != 0,
            reverse_rows: b & 2 != 0,
            reverse_cols: b & 1 != 0,
        })
    }

    /// Whether this layout negates ρ: one side's levels reversed.
    fn reverses(self) -> bool {
        self.reverse_rows != self.reverse_cols
    }

    /// `t`'s shape in this layout: row strata, column strata, row levels,
    /// column levels.
    fn shape(self, t: &Table) -> [usize; 4] {
        if self.transpose {
            [t.n_col_strata, t.n_row_strata, t.k_col, t.k_row]
        } else {
            [t.n_row_strata, t.n_col_strata, t.k_row, t.k_col]
        }
    }

    /// `t`'s cells in this layout, row-major, read in place.
    fn cells(self, t: &Table) -> impl Iterator<Item = f64> + '_ {
        let [n_s, n_u, k_r, k_c] = self.shape(t);
        let flip = |reverse: bool, k: usize, i: usize| if reverse { k - 1 - i } else { i };
        (0..n_s).flat_map(move |s| {
            (0..n_u).flat_map(move |u| {
                (0..k_r).flat_map(move |i| {
                    let i0 = flip(self.reverse_rows, k_r, i);
                    (0..k_c).map(move |j| {
                        let j0 = flip(self.reverse_cols, k_c, j);
                        if self.transpose {
                            t.at(u, s, j0, i0)
                        } else {
                            t.at(s, u, i0, j0)
                        }
                    })
                })
            })
        })
    }

    fn apply(self, t: &Table) -> Table {
        let [n_s, n_u, k_r, k_c] = self.shape(t);
        Table {
            data: self.cells(t).collect(),
            ..Table::zeros(n_s, n_u, k_r, k_c)
        }
    }

    /// The order of `t`'s layouts `self` and `other`: by shape, then cell
    /// by cell, without building either.
    fn order(self, other: Layout, t: &Table) -> Ordering {
        self.shape(t).cmp(&other.shape(t)).then_with(|| {
            self.cells(t)
                .zip(other.cells(t))
                .map(|(x, y)| x.total_cmp(&y))
                .find(|o| o.is_ne())
                .unwrap_or(Ordering::Equal)
        })
    }

    /// The greatest layout of `t`, the identity among equals, and ρ's sign
    /// under it: -1 when it reverses, 0 when a reversing and a
    /// non-reversing layout both give it.
    fn canonical(t: &Table) -> (Layout, f64) {
        let (mut best, mut keeps, mut reverses) = (Layout::IDENTITY, false, false);
        for layout in Layout::all() {
            match layout.order(best, t) {
                Ordering::Greater => {
                    best = layout;
                    (keeps, reverses) = (false, false);
                }
                Ordering::Less => continue,
                Ordering::Equal => {}
            }
            if layout.reverses() {
                reverses = true;
            } else {
                keeps = true;
            }
        }
        let sign = match (keeps, reverses) {
            (true, true) => 0.0,
            (false, true) => -1.0,
            _ => 1.0,
        };
        (best, sign)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fit::LATENT_BOUND;

    fn tables(counts: [f64; 4]) -> Tables {
        let mut n = Table::zeros(1, 1, 2, 2);
        n.data = counts.to_vec();
        let both = Grid {
            rows: 1,
            cols: 2,
            data: vec![true, true],
        };
        Tables::from_counts(n, &both, &both).unwrap()
    }

    #[test]
    fn a_stratified_table_fits_alike_in_every_layout() {
        // Two row strata, one column stratum, three levels by two: an
        // interior fit, and a boundary one whose levels never cross.
        let fit = |t: &Table| {
            let all = |rows, cols| Grid::filled(rows, cols, true);
            let (rows, cols) = (all(t.n_row_strata, t.k_row), all(t.n_col_strata, t.k_col));
            Tables::from_counts(t.clone(), &rows, &cols)
                .unwrap()
                .fit(0.0)
        };
        for (cells, boundary) in [
            (
                [
                    20.0, 4.0, 9.0, 8.0, 0.0, 12.0, 15.0, 6.0, 7.0, 7.0, 2.0, 11.0,
                ],
                false,
            ),
            (
                [
                    20.0, 0.0, 9.0, 8.0, 0.0, 12.0, 15.0, 0.0, 7.0, 7.0, 0.0, 11.0,
                ],
                true,
            ),
        ] {
            let mut n = Table::zeros(2, 1, 3, 2);
            n.data = cells.to_vec();
            let base = fit(&n);
            assert_eq!(base.boundary, Some(boundary));
            for layout in Layout::all() {
                let p = fit(&layout.apply(&n));
                let sign = if layout.reverses() { -1.0 } else { 1.0 };
                assert_eq!(
                    p.value.to_bits(),
                    (sign * base.value).to_bits(),
                    "{layout:?}"
                );
                assert_eq!(p.boundary, base.boundary);
            }
        }
    }

    #[test]
    fn the_bound_nll_resolves_a_cell_that_is_rounding() {
        // Issue #8: at the bound the one-pair cell's corner difference is
        // rounding.  Its probability is e^-449.0238 by SciPy's `quad` of
        // φ(x) Φ((b - ρx)/s) on a log scale, and the bound's NLL is 431.0475
        // nats above the optimum's by the same.  (#8's 672.8 floored the
        // cell at 1e-300.)
        let t = tables([14045855.0, 3530593.0, 1.0, 6012126.0]);
        let (a, b) = (t.a.at(0, 1), t.b.at(0, 1));
        let cell = bvn::rectangle((a, f64::INFINITY), (f64::NEG_INFINITY, b), LATENT_BOUND);
        assert!(
            (cell.ln() + 449.023_788).abs() < 1e-4,
            "ln p = {}",
            cell.ln()
        );
        let gap = t.bound_nll(LATENT_BOUND) - t.nll(0.995_456_868);
        assert!((gap - 431.0475).abs() < 1e-3, "gap {gap}");
        // With no cell below the noise floor it is the NLL, bit for bit.
        assert_eq!(t.bound_nll(0.99).to_bits(), t.nll(0.99).to_bits());
        // An empty corner is no term of the NLL.
        let t = tables([10.0, 0.0, 0.0, 10.0]);
        assert_eq!(
            t.bound_nll(LATENT_BOUND).to_bits(),
            t.nll(LATENT_BOUND).to_bits()
        );
    }
}
