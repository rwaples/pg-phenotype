//! The standard bivariate normal: Φ2 by Owen's T, φ2 and ∂φ2/∂ρ
//! (pedsum `assortative_mating.py:352-409`).

use super::cephes::{ndtr, owens_t};
use super::kernels::interval_mass;
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

/// Past this `|x|`, `φ(x)` underflows to 0.
const X_MAX: f64 = 40.0;
/// The relative error [`rectangle`] stops at.
const RECTANGLE_RTOL: f64 = 1e-10;
/// The most bisections [`rectangle`] makes.
const MAX_SPLITS: usize = 500;
/// Offsets, in step widths, of [`rectangle`]'s cuts around a step.
const STEP_CUTS: [f64; 6] = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0];
/// [`rectangle`]'s cuts across φ's bulk.
const BULK_CUTS: [f64; 9] = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0];

/// `P(h.0 < X <= h.1, k.0 < Y <= k.1)` at correlation `rho`, to a relative
/// error near [`RECTANGLE_RTOL`] however small, for a cell whose difference
/// of corner CDFs is rounding (issue #8).  It integrates
/// `φ(x) P(k.0 < Y <= k.1 | x)` over `x`, where `Y | x ~ N(ρx, 1 - ρ²)`;
/// both factors are tail-safe.
pub(crate) fn rectangle(h: (f64, f64), k: (f64, f64), rho: f64) -> f64 {
    let s = ((1.0 - rho) * (1.0 + rho)).sqrt();
    let f = |x: f64| phi(x) * interval_mass((k.1 - rho * x) / s, (k.0 - rho * x) / s);
    let (lo, hi) = (h.0.max(-X_MAX), h.1.min(X_MAX));
    if !(lo < hi) {
        return 0.0;
    }
    // The conditional mass switches on or off within a few s/|ρ| of x = k/ρ,
    // a step narrower than a 15-point rule over the whole range resolves:
    // cut across each step, and across φ's bulk, so none falls between nodes.
    let mut cuts = vec![lo, hi];
    cuts.extend(BULK_CUTS);
    for edge in [k.0, k.1]
        .into_iter()
        .filter(|e| e.is_finite() && rho != 0.0)
    {
        let (at, width) = (edge / rho, s / rho.abs());
        cuts.extend(
            STEP_CUTS
                .iter()
                .flat_map(|&c| [at - c * width, at + c * width]),
        );
    }
    cuts.retain(|&x| lo <= x && x <= hi);
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    integrate(f, &cuts)
}

/// `∫ f` over the intervals between `cuts`, bisecting the interval with the
/// largest error estimate until the total is within [`RECTANGLE_RTOL`].
fn integrate(f: impl Fn(f64) -> f64, cuts: &[f64]) -> f64 {
    let piece = |a: f64, b: f64| {
        let (value, error) = gauss_kronrod(&f, a, b);
        (a, b, value, error)
    };
    let mut pieces: Vec<_> = cuts.windows(2).map(|w| piece(w[0], w[1])).collect();
    for _ in 0..MAX_SPLITS {
        let total: f64 = pieces.iter().map(|p| p.2).sum();
        let error: f64 = pieces.iter().map(|p| p.3).sum();
        if error <= RECTANGLE_RTOL * total.abs() {
            break;
        }
        let Some(worst) = (0..pieces.len()).max_by(|&i, &j| pieces[i].3.total_cmp(&pieces[j].3))
        else {
            break;
        };
        let (a, b, ..) = pieces.swap_remove(worst);
        let mid = 0.5 * (a + b);
        pieces.extend([piece(a, mid), piece(mid, b)]);
    }
    pieces.iter().map(|p| p.2).sum()
}

/// The 15-point Kronrod abscissae on `[0, 1]`, descending, then their weights,
/// and the weights of the 7-point Gauss rule on every other one
/// (QUADPACK `qk15`).
#[allow(clippy::excessive_precision)]
const XGK: [f64; 8] = [
    0.991455371120812639206854697526329,
    0.949107912342758524526189684047851,
    0.864864423359769072789712788640926,
    0.741531185599394439863864773280788,
    0.586087235467691130294144845693013,
    0.405845151377397166906606412076961,
    0.207784955007898467600689403773245,
    0.0,
];
#[allow(clippy::excessive_precision)]
const WGK: [f64; 8] = [
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714,
];
#[allow(clippy::excessive_precision)]
const WG: [f64; 4] = [
    0.129484966168869693270611432679082,
    0.279705391489276667901467771423780,
    0.381830050505118944950369775488975,
    0.417959183673469387755102040816327,
];

/// `∫ f` over `[a, b]` by the 15-point Kronrod rule, and its distance from
/// the embedded 7-point Gauss rule as the error estimate.
fn gauss_kronrod(f: impl Fn(f64) -> f64, a: f64, b: f64) -> (f64, f64) {
    let (centre, half) = (0.5 * (a + b), 0.5 * (b - a));
    let fc = f(centre);
    let (mut kronrod, mut gauss) = (WGK[7] * fc, WG[3] * fc);
    for j in 0..7 {
        let dx = half * XGK[j];
        let pair = f(centre - dx) + f(centre + dx);
        kronrod += WGK[j] * pair;
        if j % 2 == 1 {
            gauss += WG[j / 2] * pair;
        }
    }
    (kronrod * half, ((kronrod - gauss) * half).abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_kronrod_is_exact_to_its_degrees() {
        // Kronrod-15 is exact to degree 22 and Gauss-7 to degree 13, so the
        // error estimate vanishes through degree 13.
        for d in 0..=22 {
            let (value, error) = gauss_kronrod(|x: f64| x.powi(d), 0.0, 1.0);
            let exact = 1.0 / f64::from(d + 1);
            assert!(
                (value - exact).abs() < 1e-15,
                "degree {d}: {value} vs {exact}"
            );
            if d <= 13 {
                assert!(error < 1e-15, "degree {d}: error {error}");
            } else {
                assert!(error > 1e-12, "degree {d}: Gauss-7 should miss it");
            }
        }
    }

    #[test]
    fn a_rectangle_at_zero_correlation_is_a_product_of_tails() {
        let mass = |a: f64, b: f64| interval_mass(b, a);
        for (h, k) in [
            ((8.0, 9.0), (8.5, f64::INFINITY)),
            ((-1.0, 0.5), (f64::NEG_INFINITY, -12.0)),
            ((f64::NEG_INFINITY, f64::INFINITY), (0.3, 0.4)),
        ] {
            let want = mass(h.0, h.1) * mass(k.0, k.1);
            let got = rectangle(h, k, 0.0);
            assert!(
                (got / want - 1.0).abs() < 1e-9,
                "{h:?} {k:?}: {got:e} vs {want:e}"
            );
        }
    }

    #[test]
    fn a_rectangle_matches_the_corner_cdfs_where_they_are_exact() {
        let corner = |h: (f64, f64), k: (f64, f64), rho: f64| {
            cdf(h.1, k.1, rho) - cdf(h.0, k.1, rho) - cdf(h.1, k.0, rho) + cdf(h.0, k.0, rho)
        };
        let inf = f64::INFINITY;
        for rho in [-0.9, -0.3, 0.2, 0.7, 0.95, 0.9999] {
            for (h, k) in [
                ((-inf, -0.4), (-inf, 0.1)),
                ((-0.4, 0.8), (0.1, inf)),
                ((-1.5, -0.2), (-0.7, 0.6)),
                ((0.8, inf), (-inf, 1.2)),
            ] {
                let (got, want) = (rectangle(h, k, rho), corner(h, k, rho));
                if want > 1e-6 {
                    assert!(
                        (got - want).abs() < 1e-12,
                        "{rho} {h:?} {k:?}: {got} vs {want}"
                    );
                }
            }
        }
    }
}
