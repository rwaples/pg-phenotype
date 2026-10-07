//! The standard bivariate normal: Φ2 by Owen's T, φ2 and ∂φ2/∂ρ
//! (pedsum `assortative_mating.py:352-409`).

use super::cephes::{ndtr, owens_t};
use std::f64::consts::PI;

/// Owen's `T(h, a)` with `T(h, ±inf) = ±Φ(-|h|) / 2`.
fn owens_t_ext(h: f64, a: f64) -> f64 {
    if a.is_infinite() {
        a.signum() * ndtr(-h.abs()) / 2.0
    } else {
        owens_t(h, a)
    }
}

/// Φ2 at finite `h`, `k` (Owen 1956).
fn cdf_finite(h: f64, k: f64, rho: f64) -> f64 {
    let s = ((1.0 - rho) * (1.0 + rho)).sqrt();
    let sign = |v: f64| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    let a_h = if h == 0.0 {
        if k == 0.0 {
            -rho / s
        } else {
            sign(k) * f64::INFINITY
        }
    } else {
        (k - rho * h) / (h * s)
    };
    let a_k = if k == 0.0 {
        if h == 0.0 {
            f64::INFINITY
        } else {
            sign(h) * f64::INFINITY
        }
    } else {
        (h - rho * k) / (k * s)
    };
    let hk = h * k;
    let beta = if hk < 0.0 || (hk == 0.0 && h + k < 0.0) {
        0.5
    } else {
        0.0
    };
    0.5 * (ndtr(h) + ndtr(k)) - owens_t_ext(h, a_h) - owens_t_ext(k, a_k) - beta
}

/// `P(X < h, Y < k)` at correlation `rho`; `h`, `k` may be infinite.
pub(crate) fn cdf(h: f64, k: f64, rho: f64) -> f64 {
    if h.is_finite() && k.is_finite() {
        return cdf_finite(h, k, rho);
    }
    if h == f64::INFINITY {
        ndtr(k)
    } else if k == f64::INFINITY {
        ndtr(h)
    } else {
        0.0
    }
}

/// `φ2(h, k; ρ)` and `∂φ2/∂ρ`, both 0 where an argument is infinite.
pub(crate) fn pdf_and_drho(h: f64, k: f64, rho: f64) -> (f64, f64) {
    if !(h.is_finite() && k.is_finite()) {
        return (0.0, 0.0);
    }
    let q = (1.0 - rho) * (1.0 + rho);
    let quad = h * h - 2.0 * rho * h * k + k * k;
    let pdf = (-quad / (2.0 * q)).exp() / (2.0 * PI * q.sqrt());
    let drho = pdf * (h * k * q - rho * quad + rho * q) / (q * q);
    (pdf, drho)
}

/// `φ(x)`, the standard normal density, as NumPy evaluates `exp(-x²/2)/√(2π)`.
#[inline]
pub(crate) fn phi(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}
