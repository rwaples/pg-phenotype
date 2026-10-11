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

use fit::Terms;
pub(crate) use influence::cell_influence;
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
    /// Populated (mother stratum, father stratum), row-major.
    pub combos: Vec<(usize, usize)>,
}

impl Tables {
    pub fn from_counts(
        n: Table,
        m_levels: &Grid<bool>,
        f_levels: &Grid<bool>,
    ) -> Result<Tables, NoFit> {
        if !n.data.iter().any(|&v| v != 0.0) {
            return Err(NoFit::NoPairs);
        }
        let (m_margin, f_margin) = (n.mother_margin(), n.father_margin());
        for (margin, levels) in [(&m_margin, m_levels), (&f_margin, f_levels)] {
            if let Some(no_fit) = margin_status(margin, levels) {
                return Err(no_fit);
            }
        }
        let combos = (0..n.n_ms)
            .flat_map(|s| (0..n.n_fs).map(move |u| (s, u)))
            .filter(|&(s, u)| n.stratum_total(s, u) > 0.0)
            .collect();
        Ok(Tables {
            a: thresholds(&m_margin),
            b: thresholds(&f_margin),
            n,
            combos,
        })
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
            for i in 0..self.n.k_m {
                for j in 0..self.n.k_f {
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
            for i in 0..self.n.k_m {
                for j in 0..self.n.k_f {
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

#[cfg(test)]
mod tests {
    use super::*;
    use table::LATENT_BOUND;

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
