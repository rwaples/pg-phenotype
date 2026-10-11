//! The two-step influence of one pair on a Polychoric Fit's ρ̂, by its
//! cell: the ρ score corrected for the estimated thresholds (Olsson 1979
//! eqs 10, 12), as the Mate Network sandwich and the Tetrachoric SE use it.

use super::bvn;
use super::cephes::ndtr;
use super::table::{Grid, TINY};
use super::{Reason, Tables};

/// `A_ρτ A_ττ⁻¹ ψ_τ` per (stratum, level) of one discrete side.
fn threshold_correction(a_rho_tau: &Grid<f64>, tau: &Grid<f64>, n_stratum: &[f64]) -> Grid<f64> {
    let k = tau.cols - 1;
    let mut out = Grid::filled(tau.rows, k, 0.0);
    for (s, &n_s) in n_stratum.iter().enumerate().take(tau.rows) {
        for j in 0..k.saturating_sub(1) {
            let t = tau.at(s, j + 1);
            if !t.is_finite() {
                continue;
            }
            let cdf = ndtr(t);
            let scale = -a_rho_tau.at(s, j + 1) / (n_s * bvn::phi(t));
            for l in 0..k {
                let below = if l <= j { 1.0 } else { 0.0 };
                *out.at_mut(s, l) += scale * (below - cdf);
            }
        }
    }
    out
}

/// Influence on the two-step polychoric ρ̂ of one pair, by its cell.
pub(crate) struct CellInfluence {
    /// Per (mother stratum, father stratum), the cells' scores; `None` when unpopulated.
    score: Vec<Option<Grid<f64>>>,
    n_fs: usize,
    correction_a: Grid<f64>,
    correction_b: Grid<f64>,
    a_rr: f64,
}

impl CellInfluence {
    /// The influence of a pair in cell `(i, j)` of stratum combination `(s, u)`.
    pub fn at(&self, s: usize, u: usize, i: usize, j: usize) -> f64 {
        let score = self.score[s * self.n_fs + u]
            .as_ref()
            .map_or(0.0, |g| g.at(i, j));
        (score - (self.correction_a.at(s, i) + self.correction_b.at(u, j))) / -self.a_rr
    }
}

/// The two-step polychoric influence of each cell of `tables` at `rho`.
pub(crate) fn cell_influence(tables: &Tables, rho: f64) -> Result<CellInfluence, Reason> {
    let (k_m, k_f) = (tables.n.k_m, tables.n.k_f);
    let q = (1.0 - rho) * (1.0 + rho);
    struct Combo {
        score: Grid<f64>,
        cross_a: Vec<f64>,
        cross_b: Vec<f64>,
    }
    let mut a_rr = 0.0;
    let mut combos = Vec::with_capacity(tables.combos.len());
    for &(s, u) in &tables.combos {
        let cdf = tables.grid((s, u), |h, k| bvn::cdf(h, k, rho));
        let pdf = tables.grid((s, u), |h, k| bvn::pdf_and_drho(h, k, rho).0);
        let drho = tables.grid((s, u), |h, k| bvn::pdf_and_drho(h, k, rho).1);
        let d_cdf_h = tables.grid((s, u), |h, k| {
            let phi_h = if h.is_finite() { bvn::phi(h) } else { 0.0 };
            let hh = if h.is_finite() { h } else { 0.0 };
            phi_h
                * if k.is_finite() {
                    ndtr((k - rho * hh) / q.sqrt())
                } else {
                    f64::from(u8::from(k > 0.0))
                }
        });
        let d_cdf_k = tables.grid((s, u), |h, k| {
            let phi_k = if k.is_finite() { bvn::phi(k) } else { 0.0 };
            let kk = if k.is_finite() { k } else { 0.0 };
            phi_k
                * if h.is_finite() {
                    ndtr((h - rho * kk) / q.sqrt())
                } else {
                    f64::from(u8::from(h > 0.0))
                }
        });
        let fin = |v: f64| if v.is_finite() { v } else { 0.0 };
        let d_pdf_h = tables.grid((s, u), |h, k| {
            -bvn::pdf_and_drho(h, k, rho).0 * (fin(h) - rho * fin(k)) / q
        });
        let d_pdf_k = tables.grid((s, u), |h, k| {
            -bvn::pdf_and_drho(h, k, rho).0 * (fin(k) - rho * fin(h)) / q
        });
        let mut pi = Grid::filled(k_m, k_f, 0.0);
        let mut score = Grid::filled(k_m, k_f, 0.0);
        for i in 0..k_m {
            for j in 0..k_f {
                let c = |g: &Grid<f64>| {
                    g.at(i + 1, j + 1) - g.at(i, j + 1) - g.at(i + 1, j) + g.at(i, j)
                };
                let p = c(&cdf).max(TINY);
                let sc = c(&pdf) / p;
                *pi.at_mut(i, j) = p;
                *score.at_mut(i, j) = sc;
                // Empty cells are skipped where pedsum sums them (ADR 0005).
                let n = tables.n.at(s, u, i, j);
                if n > 0.0 {
                    a_rr += n * (c(&drho) / p - sc * sc);
                }
            }
        }
        // Σ_cells n ∂score/∂τ per threshold of each side (Olsson 1979 eqs 10, 12).
        let mut cross_a = vec![0.0; k_m + 1];
        let mut cross_b = vec![0.0; k_f + 1];
        for i in 0..k_m {
            let (mut upper, mut lower) = (0.0, 0.0);
            for j in 0..k_f {
                let n = tables.n.at(s, u, i, j);
                if n == 0.0 {
                    continue;
                }
                let (p, sc) = (pi.at(i, j), score.at(i, j));
                let d_pi_upper = d_cdf_h.at(i + 1, j + 1) - d_cdf_h.at(i + 1, j);
                let d_pi_lower = d_cdf_h.at(i, j) - d_cdf_h.at(i, j + 1);
                let d_num_upper = d_pdf_h.at(i + 1, j + 1) - d_pdf_h.at(i + 1, j);
                let d_num_lower = d_pdf_h.at(i, j) - d_pdf_h.at(i, j + 1);
                upper += n * (d_num_upper - sc * d_pi_upper) / p;
                lower += n * (d_num_lower - sc * d_pi_lower) / p;
            }
            cross_a[i + 1] += upper;
            cross_a[i] += lower;
        }
        for j in 0..k_f {
            let (mut upper, mut lower) = (0.0, 0.0);
            for i in 0..k_m {
                let n = tables.n.at(s, u, i, j);
                if n == 0.0 {
                    continue;
                }
                let (p, sc) = (pi.at(i, j), score.at(i, j));
                let d_pi_upper = d_cdf_k.at(i + 1, j + 1) - d_cdf_k.at(i, j + 1);
                let d_pi_lower = d_cdf_k.at(i, j) - d_cdf_k.at(i + 1, j);
                let d_num_upper = d_pdf_k.at(i + 1, j + 1) - d_pdf_k.at(i, j + 1);
                let d_num_lower = d_pdf_k.at(i, j) - d_pdf_k.at(i + 1, j);
                upper += n * (d_num_upper - sc * d_pi_upper) / p;
                lower += n * (d_num_lower - sc * d_pi_lower) / p;
            }
            cross_b[j + 1] += upper;
            cross_b[j] += lower;
        }
        combos.push(Combo {
            score,
            cross_a,
            cross_b,
        });
    }
    if !(a_rr < 0.0) {
        return Err(Reason::SandwichUndefined);
    }
    let n = &tables.n;
    let mut a_rho_a = Grid::filled(tables.a.rows, tables.a.cols, 0.0);
    let mut a_rho_b = Grid::filled(tables.b.rows, tables.b.cols, 0.0);
    for (&(s, u), combo) in tables.combos.iter().zip(&combos) {
        for (t, v) in combo.cross_a.iter().enumerate() {
            *a_rho_a.at_mut(s, t) += v;
        }
        for (t, v) in combo.cross_b.iter().enumerate() {
            *a_rho_b.at_mut(u, t) += v;
        }
    }
    let n_m: Vec<f64> = (0..n.n_ms)
        .map(|s| (0..n.n_fs).map(|u| n.stratum_total(s, u)).sum())
        .collect();
    let n_f: Vec<f64> = (0..n.n_fs)
        .map(|u| (0..n.n_ms).map(|s| n.stratum_total(s, u)).sum())
        .collect();
    let correction_a = threshold_correction(&a_rho_a, &tables.a, &n_m);
    let correction_b = threshold_correction(&a_rho_b, &tables.b, &n_f);
    let mut score = vec![None; n.n_ms * n.n_fs];
    for (&(s, u), combo) in tables.combos.iter().zip(combos) {
        score[s * n.n_fs + u] = Some(combo.score);
    }
    Ok(CellInfluence {
        score,
        n_fs: n.n_fs,
        correction_a,
        correction_b,
        a_rr,
    })
}
