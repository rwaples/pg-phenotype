//! A Polychoric Fit's table: weighted counts by stratum and level, each
//! side's thresholds, and the checks a table must pass to be fitted.

use super::cephes::{c_erfc, kernel_ndtr};
use std::f64::consts::SQRT_2;

/// The floor of a cell probability before its log.
pub(crate) const TINY: f64 = 1e-300;
/// ρ is searched, and a one-step draw clipped, to `(-LATENT_BOUND, LATENT_BOUND)`.
pub(crate) const LATENT_BOUND: f64 = 0.9999;

/// pedsum's kernel `ndtri` (Wichura 1988, AS 241 PPND16), `±inf` at `p = 0, 1`:
/// R's coefficients with pedsum's rounding.
pub(crate) fn ndtri(p: f64) -> f64 {
    crate::normal::as241(p, crate::normal::As241Rounding::Pedsum)
}

/// Why a table has no Polychoric Fit: the first step's checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NoFit {
    NoPairs,
    ConstantMargin,
    EmptyCategory,
}

/// A dense row-major matrix.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Grid<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T: Copy> Grid<T> {
    pub fn filled(rows: usize, cols: usize, value: T) -> Grid<T> {
        Grid {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> T {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn at_mut(&mut self, r: usize, c: usize) -> &mut T {
        &mut self.data[r * self.cols + c]
    }

    pub fn row(&self, r: usize) -> &[T] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

/// Weighted pair counts by (mother stratum, father stratum, mother level, father level).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Table {
    pub n_ms: usize,
    pub n_fs: usize,
    pub k_m: usize,
    pub k_f: usize,
    pub data: Vec<f64>,
}

impl Table {
    pub fn zeros(n_ms: usize, n_fs: usize, k_m: usize, k_f: usize) -> Table {
        Table {
            n_ms,
            n_fs,
            k_m,
            k_f,
            data: vec![0.0; n_ms * n_fs * k_m * k_f],
        }
    }

    #[inline]
    pub fn index(&self, s: usize, u: usize, i: usize, j: usize) -> usize {
        ((s * self.n_fs + u) * self.k_m + i) * self.k_f + j
    }

    #[inline]
    pub fn at(&self, s: usize, u: usize, i: usize, j: usize) -> f64 {
        self.data[self.index(s, u, i, j)]
    }

    /// Mother margin by (mother stratum, mother level).
    pub fn mother_margin(&self) -> Grid<f64> {
        let mut out = Grid::filled(self.n_ms, self.k_m, 0.0);
        for s in 0..self.n_ms {
            for i in 0..self.k_m {
                let mut v = 0.0;
                for u in 0..self.n_fs {
                    for j in 0..self.k_f {
                        v += self.at(s, u, i, j);
                    }
                }
                *out.at_mut(s, i) = v;
            }
        }
        out
    }

    /// Father margin by (father stratum, father level).
    pub fn father_margin(&self) -> Grid<f64> {
        let mut out = Grid::filled(self.n_fs, self.k_f, 0.0);
        for u in 0..self.n_fs {
            for j in 0..self.k_f {
                let mut v = 0.0;
                for s in 0..self.n_ms {
                    for i in 0..self.k_m {
                        v += self.at(s, u, i, j);
                    }
                }
                *out.at_mut(u, j) = v;
            }
        }
        out
    }

    /// The same counts as one mother and one father stratum.
    pub fn pooled(&self) -> Table {
        let mut out = Table::zeros(1, 1, self.k_m, self.k_f);
        for s in 0..self.n_ms {
            for u in 0..self.n_fs {
                for i in 0..self.k_m {
                    for j in 0..self.k_f {
                        out.data[i * self.k_f + j] += self.at(s, u, i, j);
                    }
                }
            }
        }
        out
    }

    /// Total weight of one (mother stratum, father stratum) table.
    pub fn stratum_total(&self, s: usize, u: usize) -> f64 {
        let start = self.index(s, u, 0, 0);
        self.data[start..start + self.k_m * self.k_f].iter().sum()
    }
}

/// Per row of `margin` (strata x levels), Φ⁻¹ of the cumulative
/// proportions framed by -inf and +inf; a row without counts is all -inf
/// inside.
pub(crate) fn thresholds(margin: &Grid<f64>) -> Grid<f64> {
    let k = margin.cols;
    let mut out = Grid::filled(margin.rows, k + 1, 0.0);
    for s in 0..margin.rows {
        let total: f64 = margin.row(s).iter().fold(0.0, |a, &v| a + v);
        *out.at_mut(s, 0) = f64::NEG_INFINITY;
        *out.at_mut(s, k) = f64::INFINITY;
        let mut cum = 0.0;
        for j in 0..k.saturating_sub(1) {
            cum += margin.at(s, j);
            *out.at_mut(s, j + 1) = if total > 0.0 {
                ndtri(cum / total)
            } else {
                f64::NEG_INFINITY
            };
        }
    }
    out
}

/// `Φ(upper) − Φ(lower)` without cancellation in the upper tail.
#[inline]
pub(crate) fn interval_mass(upper: f64, lower: f64) -> f64 {
    if lower > 0.0 {
        return 0.5 * (c_erfc(lower / SQRT_2) - c_erfc(upper / SQRT_2));
    }
    kernel_ndtr(upper) - kernel_ndtr(lower)
}

/// pedsum `_check_levels`: a level a stratum shows that this margin lacks
/// fails before a constant margin does.
pub(crate) fn margin_status(margin: &Grid<f64>, levels: &Grid<bool>) -> Option<NoFit> {
    for s in 0..margin.rows {
        if margin.row(s).iter().any(|&v| v > 0.0)
            && (0..margin.cols).any(|c| levels.at(s, c) && margin.at(s, c) == 0.0)
        {
            return Some(NoFit::EmptyCategory);
        }
    }
    let n_levels = (0..margin.cols)
        .filter(|&c| (0..margin.rows).any(|s| margin.at(s, c) > 0.0))
        .count();
    (n_levels < 2).then_some(NoFit::ConstantMargin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_frame_and_invert_the_margin() {
        let m = Grid {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        };
        let t = thresholds(&m);
        assert_eq!(t.at(0, 0), f64::NEG_INFINITY);
        assert_eq!(t.at(0, 3), f64::INFINITY);
        assert!((t.at(0, 1) - ndtri(0.25)).abs() < 1e-16);
        assert_eq!(t.at(0, 2), ndtri(0.75));
        assert_eq!(
            t.row(1),
            &[
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY
            ]
        );
    }

    #[test]
    fn margin_status_order() {
        let levels = Grid::filled(1, 2, true);
        let one = Grid {
            rows: 1,
            cols: 2,
            data: vec![3.0, 0.0],
        };
        assert_eq!(margin_status(&one, &levels), Some(NoFit::EmptyCategory));
        assert_eq!(
            margin_status(
                &one,
                &Grid {
                    rows: 1,
                    cols: 2,
                    data: vec![true, false]
                }
            ),
            Some(NoFit::ConstantMargin)
        );
        let both = Grid {
            rows: 1,
            cols: 2,
            data: vec![3.0, 1.0],
        };
        assert_eq!(margin_status(&both, &levels), None);
    }
}
