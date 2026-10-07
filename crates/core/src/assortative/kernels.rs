//! One pass over a cell's Mating Pairs per kernel, ported from pedsum
//! `assortative_kernels.py`.
//!
//! Every kernel takes frequency weights `w` (a pair with `w = 0` is absent),
//! so a bootstrap draw is a weight vector.  Each O(n) pass sums fixed blocks
//! of [`BLOCK`] pairs into partials and adds the partials in block order, so
//! a result is the same under any thread count, and the serial form a draw
//! uses equals the parallel form a point fit uses.

use super::cephes::{c_erfc, kernel_ndtr};
use super::fit::Terms;
use super::result::Reason;
use rayon::prelude::*;
use std::f64::consts::{PI, SQRT_2};
use std::ops::Range;

/// Pairs per reduction block.
pub(crate) const BLOCK: usize = 1 << 14;
/// The floor of a cell probability before its log.
pub(crate) const TINY: f64 = 1e-300;
/// ρ is searched, and a one-step draw clipped, to `(-LATENT_BOUND, LATENT_BOUND)`.
pub(crate) const LATENT_BOUND: f64 = 0.9999;

#[inline]
pub(crate) fn sqrt_2pi() -> f64 {
    (2.0 * PI).sqrt()
}

// Wichura (1988) AS 241 PPND16, coefficients in ascending powers, as pedsum's kernel lists them.
#[allow(clippy::excessive_precision)]
const PPND16_A: [f64; 8] = [
    3.3871328727963666080e0,
    1.3314166789178437745e2,
    1.9715909503065514427e3,
    1.3731693765509461125e4,
    4.5921953931549871457e4,
    6.7265770927008700853e4,
    3.3430575583588128105e4,
    2.5090809287301226727e3,
];
#[allow(clippy::excessive_precision)]
const PPND16_B: [f64; 8] = [
    1.0,
    4.2313330701600911252e1,
    6.8718700749205790830e2,
    5.3941960214247511077e3,
    2.1213794301586595867e4,
    3.9307895800092710610e4,
    2.8729085735721942674e4,
    5.2264952788528545610e3,
];
#[allow(clippy::excessive_precision)]
const PPND16_C: [f64; 8] = [
    1.42343711074968357734e0,
    4.63033784615654529590e0,
    5.76949722146069140550e0,
    3.64784832476320460504e0,
    1.27045825245236838258e0,
    2.41780725177450611770e-1,
    2.27238449892691845833e-2,
    7.74545014278341407640e-4,
];
#[allow(clippy::excessive_precision)]
const PPND16_D: [f64; 8] = [
    1.0,
    2.05319162663775882187e0,
    1.67638483018380384940e0,
    6.89767334985100004550e-1,
    1.48103976427480074590e-1,
    1.51986665636164571966e-2,
    5.47593808499534494600e-4,
    1.05075007164441684324e-9,
];
#[allow(clippy::excessive_precision)]
const PPND16_E: [f64; 8] = [
    6.65790464350110377720e0,
    5.46378491116411436990e0,
    1.78482653991729133580e0,
    2.96560571828504891230e-1,
    2.65321895265761230930e-2,
    1.24266094738807843860e-3,
    2.71155556874348757815e-5,
    2.01033439929228813265e-7,
];
#[allow(clippy::excessive_precision)]
const PPND16_F: [f64; 8] = [
    1.0,
    5.99832206555887937690e-1,
    1.36929880922735805310e-1,
    1.48753612908506148525e-2,
    7.86869131145613259100e-4,
    1.84631831751005468180e-5,
    1.42151175831644588870e-7,
    2.04426310338993978564e-15,
];

fn horner(coef: &[f64; 8], r: f64) -> f64 {
    coef.iter().rev().fold(0.0, |acc, &c| acc * r + c)
}

/// pedsum's kernel `ndtri` (AS 241), `±inf` at `p = 0, 1`.
pub(crate) fn ndtri(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q * horner(&PPND16_A, r) / horner(&PPND16_B, r);
    }
    let r = (-(if q < 0.0 { p } else { 1.0 - p }).ln()).sqrt();
    let value = if r <= 5.0 {
        let r = r - 1.6;
        horner(&PPND16_C, r) / horner(&PPND16_D, r)
    } else {
        let r = r - 5.0;
        horner(&PPND16_E, r) / horner(&PPND16_F, r)
    };
    if q < 0.0 {
        -value
    } else {
        value
    }
}

/// Why a kernel found no estimate on a sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    ConstantMargin,
    EmptyCategory,
    DegenerateStratum,
    NoPairs,
}

impl Status {
    /// The name a result gives it.
    pub fn reason(self) -> Reason {
        match self {
            Status::ConstantMargin => Reason::ConstantMargin,
            Status::EmptyCategory => Reason::EmptyCategory,
            Status::DegenerateStratum => Reason::DegenerateStratum,
            Status::NoPairs => Reason::NoCompletePairs,
        }
    }
}

/// An estimate on one sample, or why there is none.
pub(crate) type Kernel = Result<f64, Status>;

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

pub(crate) fn n_blocks(n: usize) -> usize {
    n.div_ceil(BLOCK).max(1)
}

fn block(n: usize, b: usize) -> Range<usize> {
    b * BLOCK..n.min((b + 1) * BLOCK)
}

/// `f` of each block of `0..n`, in block order, on the pool when `par`.
pub(crate) fn per_block<T: Send>(
    n: usize,
    par: bool,
    f: impl Fn(Range<usize>) -> T + Sync,
) -> Vec<T> {
    let nb = n_blocks(n);
    if par {
        (0..nb).into_par_iter().map(|b| f(block(n, b))).collect()
    } else {
        (0..nb).map(|b| f(block(n, b))).collect()
    }
}

/// Elementwise sum of `parts` in order, from zeros (numba `_block_sum`).
pub(crate) fn block_sum(parts: &[Vec<f64>]) -> Vec<f64> {
    let mut out = vec![0.0; parts.first().map_or(0, Vec::len)];
    for p in parts {
        for (o, v) in out.iter_mut().zip(p) {
            *o += v;
        }
    }
    out
}

/// Per-stratum moments of one side.  An absent stratum has weight 0,
/// `lo = +inf` and `hi = -inf`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Moments {
    pub total: Vec<f64>,
    pub mean: Vec<f64>,
    /// The weighted `1/N` variance.
    pub var: Vec<f64>,
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
}

impl Moments {
    /// Whether some present stratum is constant.
    pub fn any_degenerate(&self) -> bool {
        self.lo.iter().zip(&self.hi).any(|(l, h)| l == h)
    }
}

/// Per stratum `code`: total weight, mean, `1/N` variance, min and max of `x`.
pub(crate) fn stratum_moments(
    x: &[f64],
    code: &[usize],
    w: &[f64],
    n_codes: usize,
    par: bool,
) -> Moments {
    let parts = per_block(x.len(), par, |range| {
        let mut total = vec![0.0; n_codes];
        let mut s = vec![0.0; n_codes];
        let mut lo = vec![f64::INFINITY; n_codes];
        let mut hi = vec![f64::NEG_INFINITY; n_codes];
        for i in range {
            if w[i] > 0.0 {
                let c = code[i];
                total[c] += w[i];
                s[c] += w[i] * x[i];
                lo[c] = lo[c].min(x[i]);
                hi[c] = hi[c].max(x[i]);
            }
        }
        (total, s, lo, hi)
    });
    let total = block_sum(&parts.iter().map(|p| p.0.clone()).collect::<Vec<_>>());
    let mut mean = block_sum(&parts.iter().map(|p| p.1.clone()).collect::<Vec<_>>());
    let mut lo = vec![f64::INFINITY; n_codes];
    let mut hi = vec![f64::NEG_INFINITY; n_codes];
    for p in &parts {
        for c in 0..n_codes {
            lo[c] = lo[c].min(p.2[c]);
            hi[c] = hi[c].max(p.3[c]);
        }
    }
    for c in 0..n_codes {
        if total[c] > 0.0 {
            mean[c] /= total[c];
        }
    }
    let var_parts = per_block(x.len(), par, |range| {
        let mut var = vec![0.0; n_codes];
        for i in range {
            if w[i] > 0.0 {
                let c = code[i];
                let d = x[i] - mean[c];
                var[c] += w[i] * d * d;
            }
        }
        var
    });
    let mut var = block_sum(&var_parts);
    for c in 0..n_codes {
        if total[c] > 0.0 {
            var[c] /= total[c];
        }
    }
    Moments {
        total,
        mean,
        var,
        lo,
        hi,
    }
}

/// `1/sd` per stratum, 0 where the variance is not positive.
pub(crate) fn inverse_sd(var: &[f64]) -> Vec<f64> {
    var.iter()
        .map(|&v| if v > 0.0 { 1.0 / v.sqrt() } else { 0.0 })
        .collect()
}

/// Weighted Pearson correlation; `NoPairs` without weight, `ConstantMargin`
/// for a constant side.
pub(crate) fn pearson(a: &[f64], b: &[f64], w: &[f64], par: bool) -> Kernel {
    let sums = per_block(a.len(), par, |range| {
        let mut out = [
            0.0,
            0.0,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for i in range {
            if w[i] > 0.0 {
                out[0] += w[i];
                out[1] += w[i] * a[i];
                out[2] += w[i] * b[i];
                out[3] = out[3].min(a[i]);
                out[4] = out[4].max(a[i]);
                out[5] = out[5].min(b[i]);
                out[6] = out[6].max(b[i]);
            }
        }
        out
    });
    let (mut total, mut sa, mut sb) = (0.0, 0.0, 0.0);
    let (mut lo_a, mut hi_a, mut lo_b, mut hi_b) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for s in &sums {
        total += s[0];
        sa += s[1];
        sb += s[2];
        lo_a = lo_a.min(s[3]);
        hi_a = hi_a.max(s[4]);
        lo_b = lo_b.min(s[5]);
        hi_b = hi_b.max(s[6]);
    }
    if total == 0.0 {
        return Err(Status::NoPairs);
    }
    if lo_a == hi_a || lo_b == hi_b {
        return Err(Status::ConstantMargin);
    }
    let (ma, mb) = (sa / total, sb / total);
    let cross = per_block(a.len(), par, |range| {
        let mut out = [0.0; 3];
        for i in range {
            if w[i] > 0.0 {
                let da = a[i] - ma;
                let db = b[i] - mb;
                out[0] += w[i] * da * db;
                out[1] += w[i] * da * da;
                out[2] += w[i] * db * db;
            }
        }
        out
    });
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for c in &cross {
        sab += c[0];
        saa += c[1];
        sbb += c[2];
    }
    Ok((sab / (saa * sbb).sqrt()).clamp(-1.0, 1.0))
}

/// Average ranks of the multiset in which pair `i` occurs `w[i]` times,
/// `order` sorting `v`; a pair of weight 0 gets rank 0.
pub(crate) fn weighted_ranks(order: &[usize], v: &[f64], w: &[f64]) -> Vec<f64> {
    let v_sorted: Vec<f64> = order.iter().map(|&i| v[i]).collect();
    let w_sorted: Vec<f64> = order.iter().map(|&i| w[i]).collect();
    let mut rank_sorted = vec![0.0; v.len()];
    sorted_ranks_into(&mut rank_sorted, &v_sorted, &w_sorted);
    let mut rank = vec![0.0; v.len()];
    for (t, &i) in order.iter().enumerate() {
        rank[i] = rank_sorted[t];
    }
    rank
}

/// [`weighted_ranks`] of values already in sorted order, in that order.
pub(crate) fn sorted_ranks_into(rank: &mut [f64], v_sorted: &[f64], w_sorted: &[f64]) {
    let n = v_sorted.len();
    rank.fill(0.0);
    let mut cum = 0.0;
    let mut i = 0;
    while i < n {
        let value = v_sorted[i];
        let mut j = i;
        let mut group = 0.0;
        while j < n && v_sorted[j] == value {
            group += w_sorted[j];
            j += 1;
        }
        if group > 0.0 {
            let avg = cum + (group + 1.0) / 2.0;
            for t in i..j {
                if w_sorted[t] > 0.0 {
                    rank[t] = avg;
                }
            }
            cum += group;
        }
        i = j;
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

/// Weighted count table of a discrete x discrete cell.
#[allow(clippy::too_many_arguments)]
pub(crate) fn count_table(
    m_stratum: &[usize],
    f_stratum: &[usize],
    m: &[f64],
    f: &[f64],
    w: &[f64],
    shape: [usize; 4],
    par: bool,
) -> Table {
    let [n_ms, n_fs, k_m, k_f] = shape;
    let parts = per_block(m.len(), par, |range| {
        let mut t = Table::zeros(n_ms, n_fs, k_m, k_f);
        for i in range {
            let at = t.index(m_stratum[i], f_stratum[i], m[i] as usize, f[i] as usize);
            t.data[at] += w[i];
        }
        t.data
    });
    let mut out = Table::zeros(n_ms, n_fs, k_m, k_f);
    out.data = block_sum(&parts);
    out
}

/// Weighted level counts of one discrete side by (stratum, level).
pub(crate) fn margin(
    code: &[f64],
    stratum: &[usize],
    w: &[f64],
    n_strata: usize,
    k: usize,
    par: bool,
) -> Grid<f64> {
    let parts = per_block(code.len(), par, |range| {
        let mut out = vec![0.0; n_strata * k];
        for i in range {
            out[stratum[i] * k + code[i] as usize] += w[i];
        }
        out
    });
    Grid {
        rows: n_strata,
        cols: k,
        data: block_sum(&parts),
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
fn interval_mass(upper: f64, lower: f64) -> f64 {
    if lower > 0.0 {
        return 0.5 * (c_erfc(lower / SQRT_2) - c_erfc(upper / SQRT_2));
    }
    kernel_ndtr(upper) - kernel_ndtr(lower)
}

/// A polyserial cell's arrays: `x` standardised within `x_stratum` by
/// `mean` and `inv_sd`, `y` levels with `tau` thresholds per `y_stratum`.
#[derive(Clone, Copy)]
pub(crate) struct Polyserial<'a> {
    pub x: &'a [f64],
    pub x_stratum: &'a [usize],
    pub mean: &'a [f64],
    pub inv_sd: &'a [f64],
    pub y: &'a [f64],
    pub y_stratum: &'a [usize],
    pub tau: &'a Grid<f64>,
    pub w: &'a [f64],
}

/// ρ and the powers of `s = √(1−ρ²)` every pair's terms use.
#[derive(Clone, Copy)]
struct At {
    rho: f64,
    s: f64,
    s3: f64,
    s5: f64,
}

impl At {
    fn new(rho: f64) -> At {
        let s = ((1.0 - rho) * (1.0 + rho)).sqrt();
        let s3 = s * s * s;
        At {
            rho,
            s,
            s3,
            s5: s3 * s * s,
        }
    }
}

/// One threshold `t` of a pair's level: `ts = (t − ρz)/s`, and for a finite
/// `t` the density `φ(ts)` and `d = (tρ − z)/s³`, so that `pdf · d` is the
/// threshold's share of `∂P/∂ρ` (eq 26).  An infinite `t` has `pdf = d = 0`,
/// so both its shares are 0.
#[derive(Clone, Copy)]
struct Edge {
    t: f64,
    ts: f64,
    pdf: f64,
    d: f64,
}

impl Edge {
    #[inline(always)]
    fn new(t: f64, z: f64, at: At) -> Edge {
        let ts = (t - at.rho * z) / at.s;
        if t.is_infinite() {
            return Edge {
                t,
                ts,
                pdf: 0.0,
                d: 0.0,
            };
        }
        Edge {
            t,
            ts,
            pdf: (-0.5 * ts * ts).exp() / sqrt_2pi(),
            d: (t * at.rho - z) / at.s3,
        }
    }

    /// The threshold's share of `∂P/∂ρ`.
    #[inline(always)]
    fn g(&self) -> f64 {
        self.pdf * self.d
    }

    /// The threshold's share of `∂²P/∂ρ²`.
    #[inline(always)]
    fn h(&self, z: f64, at: At) -> f64 {
        if self.t.is_infinite() {
            return 0.0;
        }
        let d = self.d;
        self.pdf
            * (-self.ts * d * d + self.t / at.s3 + (self.t * at.rho - z) * 3.0 * at.rho / at.s5)
    }
}

/// One pair of weight `w`: its standardised `z`, its level's upper and lower
/// thresholds, and the level's probability `P = Φ(ts_u) − Φ(ts_l)`, floored.
struct Pair {
    w: f64,
    z: f64,
    upper: Edge,
    lower: Edge,
    p: f64,
}

impl Pair {
    /// `∂ log P / ∂ρ`.
    #[inline(always)]
    fn score(&self) -> f64 {
        (self.upper.g() - self.lower.g()) / self.p
    }
}

impl Polyserial<'_> {
    /// The NLL in ρ of the conditional likelihood and its first two
    /// ρ-derivatives (Olsson, Drasgow & Dorans 1982 eqs 19-20, 26).
    pub fn terms(&self, rho: f64, par: bool) -> Terms {
        let parts = per_block(self.x.len(), par, |range| self.sums(rho, range));
        let s = block_sum(&parts.iter().map(|p| p.to_vec()).collect::<Vec<_>>());
        Terms {
            nll: s[0],
            grad: s[1],
            hess: s[2],
        }
    }

    /// Pair `i` at `at`, or `None` at weight 0.
    #[inline(always)]
    fn pair(&self, i: usize, at: At) -> Option<Pair> {
        let w = self.w[i];
        if w == 0.0 {
            return None;
        }
        let xs = self.x_stratum[i];
        let z = (self.x[i] - self.mean[xs]) * self.inv_sd[xs];
        let c = self.y[i] as usize;
        let upper = Edge::new(self.tau.at(self.y_stratum[i], c + 1), z, at);
        let lower = Edge::new(self.tau.at(self.y_stratum[i], c), z, at);
        let p = interval_mass(upper.ts, lower.ts).max(TINY);
        Some(Pair {
            w,
            z,
            upper,
            lower,
            p,
        })
    }

    /// The `grad` of [`Polyserial::terms`] alone, with the same arithmetic.
    pub fn grad(&self, rho: f64, par: bool) -> f64 {
        let parts = per_block(self.x.len(), par, |range| self.grad_sum(rho, range));
        parts.iter().fold(0.0, |a, &g| a + g)
    }

    fn grad_sum(&self, rho: f64, range: Range<usize>) -> f64 {
        let at = At::new(rho);
        let mut grad = 0.0;
        for i in range {
            if let Some(pair) = self.pair(i, at) {
                grad -= pair.w * pair.score();
            }
        }
        grad
    }

    fn sums(&self, rho: f64, range: Range<usize>) -> [f64; 3] {
        let at = At::new(rho);
        let (mut nll, mut grad, mut hess) = (0.0, 0.0, 0.0);
        for i in range {
            let Some(pair) = self.pair(i, at) else {
                continue;
            };
            let score = pair.score();
            let curvature =
                (pair.upper.h(pair.z, at) - pair.lower.h(pair.z, at)) / pair.p - score * score;
            nll -= pair.w * pair.p.ln();
            grad -= pair.w * score;
            hess -= pair.w * curvature;
        }
        [nll, grad, hess]
    }
}

/// One pair's eq 26 score and its derivatives in `z`, `τ_u`, `τ_l` and ρ.
fn polyserial_pair(rho: f64, z: f64, tau_u: f64, tau_l: f64) -> [f64; 5] {
    let s = ((1.0 - rho) * (1.0 + rho)).sqrt();
    let s3 = s * s * s;
    let s5 = s3 * s * s;
    let side = |tau: f64| -> [f64; 5] {
        if tau.is_infinite() {
            return [0.0; 5];
        }
        let a = (tau - rho * z) / s;
        let pdf = (-0.5 * a * a).exp() / sqrt_2pi();
        let d = (tau * rho - z) / s3;
        [
            pdf,
            pdf * d,
            pdf * (-a * d * d + tau / s3 + (tau * rho - z) * 3.0 * rho / s5),
            pdf * (a * rho / s * d - 1.0 / s3),
            pdf * (-a / s * d + rho / s3),
        ]
    };
    let [pdf_u, g_u, h_u, gz_u, gt_u] = side(tau_u);
    let [pdf_l, g_l, h_l, gz_l, gt_l] = side(tau_l);
    let a_u = (tau_u - rho * z) / s;
    let a_l = (tau_l - rho * z) / s;
    let p = interval_mass(a_u, a_l).max(TINY);
    let score = (g_u - g_l) / p;
    let dp_dz = -(rho / s) * (pdf_u - pdf_l);
    let ds_dz = (gz_u - gz_l) / p - score * dp_dz / p;
    let ds_dtu = gt_u / p - score * (pdf_u / s) / p;
    let ds_dtl = -gt_l / p + score * (pdf_l / s) / p;
    let ds_drho = (h_u - h_l) / p - score * score;
    [score, ds_dz, ds_dtu, ds_dtl, ds_drho]
}

/// Per-pair influence on the two-step polyserial ρ̂, and `A_ρρ` (negative
/// at a maximum).  `var` is the `x` strata's `1/N` variance.
pub(crate) fn polyserial_influence(p: &Polyserial<'_>, var: &[f64], rho: f64) -> (Vec<f64>, f64) {
    let n_xs = p.mean.len();
    let (n_ys, k1) = (p.tau.rows, p.tau.cols);
    let mut cdf_tau = Grid::filled(n_ys, k1, 0.0);
    let mut pdf_tau = Grid::filled(n_ys, k1, 0.0);
    for t in 0..n_ys {
        for j in 1..k1 - 1 {
            let tau = p.tau.at(t, j);
            if !tau.is_infinite() {
                *cdf_tau.at_mut(t, j) = kernel_ndtr(tau);
                *pdf_tau.at_mut(t, j) = (-0.5 * tau * tau).exp() / sqrt_2pi();
            }
        }
    }
    let width = 2 * n_ys + 2 * n_xs + n_ys * k1 + 1;
    let parts = per_block(p.x.len(), true, |range| {
        let mut out = vec![0.0; width];
        let (n_x, rest) = out.split_at_mut(n_xs);
        let (n_y, rest) = rest.split_at_mut(n_ys);
        let (a_mu, rest) = rest.split_at_mut(n_xs);
        let (a_var, rest) = rest.split_at_mut(n_xs);
        let (a_tau, a_rr) = rest.split_at_mut(n_ys * k1);
        for i in range {
            let w = p.w[i];
            if w == 0.0 {
                continue;
            }
            let s = p.x_stratum[i];
            let t = p.y_stratum[i];
            let c = p.y[i] as usize;
            let z = (p.x[i] - p.mean[s]) * p.inv_sd[s];
            let [_, ds_dz, ds_dtu, ds_dtl, ds_drho] =
                polyserial_pair(rho, z, p.tau.at(t, c + 1), p.tau.at(t, c));
            n_x[s] += w;
            n_y[t] += w;
            a_mu[s] -= w * ds_dz * p.inv_sd[s];
            a_var[s] -= w * ds_dz * z * p.inv_sd[s] * p.inv_sd[s] / 2.0;
            a_tau[t * k1 + c + 1] += w * ds_dtu;
            a_tau[t * k1 + c] += w * ds_dtl;
            a_rr[0] += w * ds_drho;
        }
        out
    });
    // numba sums each partial array on its own; the elementwise order is the same.
    let sums = block_sum(&parts);
    let (n_x, rest) = sums.split_at(n_xs);
    let (n_y, rest) = rest.split_at(n_ys);
    let (a_mu, rest) = rest.split_at(n_xs);
    let (a_var, rest) = rest.split_at(n_xs);
    let (a_tau, a_rr) = rest.split_at(n_ys * k1);
    let a_rr = a_rr[0];
    if !(a_rr < 0.0) {
        return (vec![0.0; p.x.len()], a_rr);
    }
    let out = per_block(p.x.len(), true, |range| {
        range
            .map(|i| {
                let w = p.w[i];
                if w == 0.0 {
                    return 0.0;
                }
                let s = p.x_stratum[i];
                let t = p.y_stratum[i];
                let c = p.y[i] as usize;
                let dev = p.x[i] - p.mean[s];
                let z = dev * p.inv_sd[s];
                let [score, ..] = polyserial_pair(rho, z, p.tau.at(t, c + 1), p.tau.at(t, c));
                let mut correction = -(a_mu[s] * dev + a_var[s] * (dev * dev - var[s])) / n_x[s];
                for j in 1..k1 - 1 {
                    if pdf_tau.at(t, j) > 0.0 {
                        let below = if c < j { 1.0 } else { 0.0 };
                        correction -= a_tau[t * k1 + j] * (below - cdf_tau.at(t, j))
                            / (n_y[t] * pdf_tau.at(t, j));
                    }
                }
                -(score - correction) / a_rr
            })
            .collect::<Vec<f64>>()
    });
    (out.concat(), a_rr)
}

/// One side of a Pearson influence: values, strata, moments.
#[derive(Clone, Copy)]
pub(crate) struct Side<'a> {
    pub x: &'a [f64],
    pub stratum: &'a [usize],
    pub mean: &'a [f64],
    pub var: &'a [f64],
}

/// Per-pair influence on the Pearson correlation of `a` and `b`, each
/// standardised within its own stratum.
pub(crate) fn pearson_influence(a: Side<'_>, b: Side<'_>, w: &[f64]) -> Vec<f64> {
    let (n_a, n_b) = (a.mean.len(), b.mean.len());
    let inv_a = inverse_sd(a.var);
    let inv_b = inverse_sd(b.var);
    let width = 2 + 3 * n_a + 3 * n_b;
    let parts = per_block(a.x.len(), true, |range| {
        let mut out = vec![0.0; width];
        let (scalars, rest) = out.split_at_mut(2);
        let (count_a, rest) = rest.split_at_mut(n_a);
        let (count_b, rest) = rest.split_at_mut(n_b);
        let (a_mu_a, rest) = rest.split_at_mut(n_a);
        let (a_var_a, rest) = rest.split_at_mut(n_a);
        let (a_mu_b, a_var_b) = rest.split_at_mut(n_b);
        for i in range {
            if w[i] == 0.0 {
                continue;
            }
            let s = a.stratum[i];
            let t = b.stratum[i];
            let za = (a.x[i] - a.mean[s]) * inv_a[s];
            let zb = (b.x[i] - b.mean[t]) * inv_b[t];
            scalars[0] += w[i];
            scalars[1] += w[i] * za * zb;
            count_a[s] += w[i];
            count_b[t] += w[i];
            a_mu_a[s] -= w[i] * zb * inv_a[s];
            a_var_a[s] -= w[i] * za * zb * inv_a[s] * inv_a[s] / 2.0;
            a_mu_b[t] -= w[i] * za * inv_b[t];
            a_var_b[t] -= w[i] * za * zb * inv_b[t] * inv_b[t] / 2.0;
        }
        out
    });
    let sums = block_sum(&parts);
    let (scalars, rest) = sums.split_at(2);
    let (count_a, rest) = rest.split_at(n_a);
    let (count_b, rest) = rest.split_at(n_b);
    let (a_mu_a, rest) = rest.split_at(n_a);
    let (a_var_a, rest) = rest.split_at(n_a);
    let (a_mu_b, a_var_b) = rest.split_at(n_b);
    let (total, cross) = (scalars[0], scalars[1]);
    let rho = cross / total;
    per_block(a.x.len(), true, |range| {
        range
            .map(|i| {
                if w[i] == 0.0 {
                    return 0.0;
                }
                let s = a.stratum[i];
                let t = b.stratum[i];
                let dev_a = a.x[i] - a.mean[s];
                let dev_b = b.x[i] - b.mean[t];
                let psi = dev_a * inv_a[s] * dev_b * inv_b[t] - rho;
                let mut correction =
                    -(a_mu_a[s] * dev_a + a_var_a[s] * (dev_a * dev_a - a.var[s])) / count_a[s];
                correction -=
                    (a_mu_b[t] * dev_b + a_var_b[t] * (dev_b * dev_b - b.var[t])) / count_b[t];
                (psi - correction) / total
            })
            .collect::<Vec<f64>>()
    })
    .concat()
}

/// Per (stratum, level) of `margin`: `E[η | level]` under that stratum's
/// thresholds; 0 at an empty level.
pub(crate) fn latent_means(margin: &Grid<f64>) -> Grid<f64> {
    let tau = thresholds(margin);
    let mut out = Grid::filled(margin.rows, margin.cols, 0.0);
    for s in 0..margin.rows {
        let total = margin.row(s).iter().fold(0.0, |a, &v| a + v);
        for c in 0..margin.cols {
            let count = margin.at(s, c);
            if count > 0.0 {
                let lo = tau.at(s, c);
                let hi = tau.at(s, c + 1);
                let pdf_lo = (-0.5 * lo * lo).exp() / sqrt_2pi();
                let pdf_hi = (-0.5 * hi * hi).exp() / sqrt_2pi();
                *out.at_mut(s, c) = (pdf_lo - pdf_hi) * total / count;
            }
        }
    }
    out
}

/// pedsum `_check_levels`: a level a stratum shows that this margin lacks
/// fails before a constant margin does.
pub(crate) fn margin_status(margin: &Grid<f64>, levels: &Grid<bool>) -> Option<Status> {
    for s in 0..margin.rows {
        if margin.row(s).iter().any(|&v| v > 0.0)
            && (0..margin.cols).any(|c| levels.at(s, c) && margin.at(s, c) == 0.0)
        {
            return Some(Status::EmptyCategory);
        }
    }
    let n_levels = (0..margin.cols)
        .filter(|&c| (0..margin.rows).any(|s| margin.at(s, c) > 0.0))
        .count();
    (n_levels < 2).then_some(Status::ConstantMargin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_cover_every_pair_once() {
        let n = 3 * BLOCK + 17;
        let lens: Vec<usize> = per_block(n, true, |r| r.len());
        assert_eq!(lens, vec![BLOCK, BLOCK, BLOCK, 17]);
        assert_eq!(per_block(0, false, |r| r.len()), vec![0]);
    }

    #[test]
    fn parallel_and_serial_kernels_agree_bit_for_bit() {
        let n = 2 * BLOCK + 5;
        let a: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1000) as f64 / 37.0).collect();
        let b: Vec<f64> = (0..n)
            .map(|i| ((i * 104_729) % 997) as f64 / 13.0 + a[i])
            .collect();
        let w: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let code: Vec<usize> = (0..n).map(|i| i % 4).collect();
        assert_eq!(pearson(&a, &b, &w, true), pearson(&a, &b, &w, false));
        assert_eq!(
            stratum_moments(&a, &code, &w, 4, true),
            stratum_moments(&a, &code, &w, 4, false)
        );
    }

    #[test]
    fn ranks_average_ties_by_weight() {
        let v = [3.0, 1.0, 3.0, 2.0];
        let w = [1.0, 2.0, 1.0, 0.0];
        let order = [1, 3, 0, 2];
        assert_eq!(weighted_ranks(&order, &v, &w), vec![3.5, 1.5, 3.5, 0.0]);
    }

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
        assert_eq!(margin_status(&one, &levels), Some(Status::EmptyCategory));
        assert_eq!(
            margin_status(
                &one,
                &Grid {
                    rows: 1,
                    cols: 2,
                    data: vec![true, false]
                }
            ),
            Some(Status::ConstantMargin)
        );
        let both = Grid {
            rows: 1,
            cols: 2,
            data: vec![3.0, 1.0],
        };
        assert_eq!(margin_status(&both, &levels), None);
    }
}
