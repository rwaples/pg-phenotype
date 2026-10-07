//! The cell estimators on one weighted sample (pedsum
//! `assortative_mating.py:271-818`): the observed fit (unit weights), the
//! full refit of a nonconcave bootstrap draw, and the within-person form.

use super::bvn;
use super::fit::{newton_rho, Terms};
use super::kernels::{
    count_table, inverse_sd, margin, margin_status, pearson, stratum_moments, thresholds,
    weighted_ranks, Grid, Kernel, Polyserial, Status, Table, TINY,
};
use super::result::{Estimator, Point};
use super::sample::{n_codes, CellPairs};

/// A point estimate on one sample, or why there is none.
pub(crate) type Fitted = Result<Point, Status>;

fn closed(k: Kernel) -> Fitted {
    k.map(|value| Point {
        value,
        boundary: None,
    })
}

/// The sorting permutation of `v`, ties in index order.
pub(crate) fn argsort(v: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    order
}

/// Spearman: Pearson of the weighted average ranks.
pub(crate) fn spearman(pairs: &CellPairs, w: &[f64]) -> Kernel {
    let rm = weighted_ranks(&argsort(&pairs.m), &pairs.m, w);
    let rf = weighted_ranks(&argsort(&pairs.f), &pairs.f, w);
    pearson(&rm, &rf, w, true)
}

/// `x` standardised within each stratum, `DegenerateStratum` when one is constant.
pub(crate) fn standardise(x: &[f64], code: &[usize], w: &[f64]) -> Result<Vec<f64>, Status> {
    let m = stratum_moments(x, code, w, n_codes(code), true);
    if m.any_degenerate() {
        return Err(Status::DegenerateStratum);
    }
    let inv_sd: Vec<f64> = m
        .total
        .iter()
        .zip(&m.var)
        .map(|(&t, &v)| if t > 0.0 { 1.0 / v.sqrt() } else { 0.0 })
        .collect();
    Ok(x.iter()
        .zip(code)
        .map(|(&v, &c)| (v - m.mean[c]) * inv_sd[c])
        .collect())
}

/// Pearson of both sides standardised within their own sex x stratum.
pub(crate) fn stratified_pearson(pairs: &CellPairs, w: &[f64]) -> Kernel {
    if pairs.len() == 0 || !w.iter().any(|&v| v != 0.0) {
        return Err(Status::NoPairs);
    }
    let zm = standardise(&pairs.m, &pairs.m_stratum, w)?;
    let zf = standardise(&pairs.f, &pairs.f_stratum, w)?;
    pearson(&zm, &zf, w, true)
}

/// A cell by its sides, which decide what its primary estimator fits.
pub(crate) enum Sides<'a> {
    /// Continuous x continuous: Pearson.
    Continuous,
    /// Discrete x discrete: polychoric, from the levels each stratum shows.
    Tables {
        m: &'a Grid<bool>,
        f: &'a Grid<bool>,
    },
    /// Continuous x discrete: polyserial.
    Serial(Serial<'a>),
}

impl<'a> Sides<'a> {
    pub fn of(pairs: &'a CellPairs) -> Sides<'a> {
        match (&pairs.m_levels, &pairs.f_levels) {
            (None, None) => Sides::Continuous,
            (Some(m), Some(f)) => Sides::Tables { m, f },
            (None, Some(levels)) => Sides::Serial(Serial {
                x: &pairs.m,
                y: &pairs.f,
                x_stratum: &pairs.m_stratum,
                y_stratum: &pairs.f_stratum,
                y_levels: levels,
            }),
            (Some(levels), None) => Sides::Serial(Serial {
                x: &pairs.f,
                y: &pairs.m,
                x_stratum: &pairs.f_stratum,
                y_stratum: &pairs.m_stratum,
                y_levels: levels,
            }),
        }
    }
}

/// The table shape of a discrete x discrete cell with these levels.
pub(crate) fn table_shape(
    pairs: &CellPairs,
    m_levels: &Grid<bool>,
    f_levels: &Grid<bool>,
) -> [usize; 4] {
    let (n_m, n_f) = pairs.n_strata();
    [
        n_m.max(m_levels.rows),
        n_f.max(f_levels.rows),
        m_levels.cols,
        f_levels.cols,
    ]
}

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
    pub fn new(
        pairs: &CellPairs,
        m_levels: &Grid<bool>,
        f_levels: &Grid<bool>,
        w: &[f64],
    ) -> Result<Tables, Status> {
        let shape = table_shape(pairs, m_levels, f_levels);
        let n = count_table(
            &pairs.m_stratum,
            &pairs.f_stratum,
            &pairs.m,
            &pairs.f,
            w,
            shape,
            true,
        );
        Tables::from_counts(n, m_levels, f_levels)
    }

    pub fn from_counts(
        n: Table,
        m_levels: &Grid<bool>,
        f_levels: &Grid<bool>,
    ) -> Result<Tables, Status> {
        if !n.data.iter().any(|&v| v != 0.0) {
            return Err(Status::NoPairs);
        }
        let (m_margin, f_margin) = (n.mother_margin(), n.father_margin());
        for (margin, levels) in [(&m_margin, m_levels), (&f_margin, f_levels)] {
            if let Some(status) = margin_status(margin, levels) {
                return Err(status);
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

    /// NLL and its first two ρ-derivatives (Olsson 1979 eqs 3, 4, 9).
    pub fn terms(&self, rho: f64) -> Terms {
        let (mut nll, mut grad, mut hess) = (0.0, 0.0, 0.0);
        for &combo in &self.combos {
            let cdf = self.grid(combo, |h, k| bvn::cdf(h, k, rho));
            let pdf = self.grid(combo, |h, k| bvn::pdf_and_drho(h, k, rho).0);
            let drho = self.grid(combo, |h, k| bvn::pdf_and_drho(h, k, rho).1);
            for i in 0..self.n.k_m {
                for j in 0..self.n.k_f {
                    let count = self.n.at(combo.0, combo.1, i, j);
                    if count > 0.0 {
                        let pi = corner(&cdf, i, j).max(TINY);
                        let score = corner(&pdf, i, j) / pi;
                        let curvature = corner(&drho, i, j) / pi - score * score;
                        nll += count * pi.ln();
                        grad += count * score;
                        hess += count * curvature;
                    }
                }
            }
        }
        Terms {
            nll: -nll,
            grad: -grad,
            hess: -hess,
        }
    }

    pub fn nll(&self, rho: f64) -> f64 {
        let mut nll = 0.0;
        for &combo in &self.combos {
            let cdf = self.grid(combo, |h, k| bvn::cdf(h, k, rho));
            for i in 0..self.n.k_m {
                for j in 0..self.n.k_f {
                    let count = self.n.at(combo.0, combo.1, i, j);
                    if count > 0.0 {
                        nll += count * corner(&cdf, i, j).max(TINY).ln();
                    }
                }
            }
        }
        -nll
    }
}

/// Two-step ML polychoric (tetrachoric) ρ by Newton from `start` (0 when none).
pub(crate) fn polychoric(
    pairs: &CellPairs,
    m_levels: &Grid<bool>,
    f_levels: &Grid<bool>,
    w: &[f64],
    start: Option<f64>,
) -> Fitted {
    let tables = Tables::new(pairs, m_levels, f_levels, w)?;
    Ok(newton_rho(
        |r| tables.terms(r),
        |r| tables.nll(r),
        start.unwrap_or(0.0),
    ))
}

/// The first step of a polyserial fit: `x` moments per stratum, the `y`
/// margin and thresholds per stratum.
pub(crate) struct FirstStep {
    pub mean: Vec<f64>,
    pub var: Vec<f64>,
    pub margin: Grid<f64>,
    pub tau: Grid<f64>,
}

/// The arrays of a polyserial fit, `x` continuous and `y` discrete.
#[derive(Clone, Copy)]
pub(crate) struct Serial<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub x_stratum: &'a [usize],
    pub y_stratum: &'a [usize],
    pub y_levels: &'a Grid<bool>,
}

impl Serial<'_> {
    pub fn first_step(&self, w: &[f64]) -> Result<FirstStep, Status> {
        let moments = stratum_moments(self.x, self.x_stratum, w, n_codes(self.x_stratum), true);
        let present: Vec<usize> = (0..moments.total.len())
            .filter(|&s| moments.total[s] > 0.0)
            .collect();
        if present.is_empty() {
            return Err(Status::NoPairs);
        }
        let lo = present
            .iter()
            .map(|&s| moments.lo[s])
            .fold(f64::INFINITY, f64::min);
        let hi = present
            .iter()
            .map(|&s| moments.hi[s])
            .fold(f64::NEG_INFINITY, f64::max);
        if lo == hi {
            return Err(Status::ConstantMargin);
        }
        if moments.any_degenerate() {
            return Err(Status::DegenerateStratum);
        }
        let n_strata = self.y_levels.rows.max(n_codes(self.y_stratum));
        let m = margin(
            self.y,
            self.y_stratum,
            w,
            n_strata,
            self.y_levels.cols,
            true,
        );
        if let Some(status) = margin_status(&m, self.y_levels) {
            return Err(status);
        }
        Ok(FirstStep {
            mean: moments.mean,
            var: moments.var,
            tau: thresholds(&m),
            margin: m,
        })
    }

    pub fn kernel<'b>(
        &'b self,
        first: &'b FirstStep,
        inv_sd: &'b [f64],
        w: &'b [f64],
    ) -> Polyserial<'b> {
        Polyserial {
            x: self.x,
            x_stratum: self.x_stratum,
            mean: &first.mean,
            inv_sd,
            y: self.y,
            y_stratum: self.y_stratum,
            tau: &first.tau,
            w,
        }
    }
}

/// Olsson, Drasgow & Dorans (1982) eq 38, the Newton start of an observed fit.
fn ad_hoc_polyserial(z: &[f64], y: &[f64], w: &[f64], margin: &Grid<f64>) -> f64 {
    let r_xy = pearson(z, y, w, true).unwrap_or(0.0);
    let var = stratum_moments(y, &vec![0; y.len()], w, 1, true).var[0];
    let mut pooled = Grid::filled(1, margin.cols, 0.0);
    for c in 0..margin.cols {
        *pooled.at_mut(0, c) = (0..margin.rows).fold(0.0, |a, s| a + margin.at(s, c));
    }
    let tau = thresholds(&pooled);
    let density: f64 = (1..margin.cols).fold(0.0, |a, j| a + bvn::phi(tau.at(0, j)));
    r_xy * var.sqrt() / density
}

/// Two-step ML polyserial (biserial) ρ by Newton from `start`, or from the
/// eq 38 estimate.
pub(crate) fn polyserial(serial: Serial<'_>, w: &[f64], start: Option<f64>) -> Fitted {
    let first = serial.first_step(w)?;
    let inv_sd = inverse_sd(&first.var);
    let kernel = serial.kernel(&first, &inv_sd, w);
    let start = start.unwrap_or_else(|| {
        let z: Vec<f64> = serial
            .x
            .iter()
            .zip(serial.x_stratum)
            .map(|(&x, &s)| (x - first.mean[s]) * inv_sd[s])
            .collect();
        ad_hoc_polyserial(&z, serial.y, w, &first.margin)
    });
    Ok(newton_rho(
        |r| kernel.terms(r, true),
        |r| kernel.terms(r, true).nll,
        start,
    ))
}

/// The pooled 2 x 2 table of a binary x binary cell.
pub(crate) fn two_by_two(pairs: &CellPairs, w: &[f64]) -> [[f64; 2]; 2] {
    let zeros = vec![0; pairs.len()];
    let t = count_table(&zeros, &zeros, &pairs.m, &pairs.f, w, [1, 1, 2, 2], true);
    [[t.data[0], t.data[1]], [t.data[2], t.data[3]]]
}

/// Cross-product ratio `ad / (bc)`, `inf` when `bc = 0`.
pub(crate) fn odds_ratio(pairs: &CellPairs, w: &[f64]) -> Kernel {
    let [[a, b], [c, d]] = two_by_two(pairs, w);
    if a == 0.0 && b == 0.0 && c == 0.0 && d == 0.0 {
        return Err(Status::NoPairs);
    }
    if a + b == 0.0 || c + d == 0.0 || a + c == 0.0 || b + d == 0.0 {
        return Err(Status::ConstantMargin);
    }
    Ok(a * d / (b * c))
}

/// Estimator `est` of a cell, crude (pooled) or stratified, on weights `w`
/// from `start` (latent fits only; the observed fit passes none).
pub(crate) fn point(
    est: Estimator,
    stratified: bool,
    pairs: &CellPairs,
    w: &[f64],
    start: Option<f64>,
) -> Fitted {
    match est {
        Estimator::Phi | Estimator::PointBiserial => closed(pearson(&pairs.m, &pairs.f, w, true)),
        Estimator::Spearman => closed(spearman(pairs, w)),
        Estimator::OddsRatio => closed(odds_ratio(pairs, w)),
        Estimator::Pearson
        | Estimator::Tetrachoric
        | Estimator::Polychoric
        | Estimator::Biserial
        | Estimator::Polyserial => primary(stratified, pairs, w, start),
    }
}

/// The cell's primary estimate, by its sides: crude on the pooled pairs, or stratified.
fn primary(stratified: bool, pairs: &CellPairs, w: &[f64], start: Option<f64>) -> Fitted {
    let pooled;
    let pairs = if stratified {
        pairs
    } else {
        pooled = pairs.pooled();
        &pooled
    };
    match Sides::of(pairs) {
        Sides::Continuous if stratified => closed(stratified_pearson(pairs, w)),
        Sides::Continuous => closed(pearson(&pairs.m, &pairs.f, w, true)),
        Sides::Tables { m, f } => polychoric(pairs, m, f, w, start),
        Sides::Serial(serial) => polyserial(serial, w, start),
    }
}
