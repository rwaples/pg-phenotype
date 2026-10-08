//! Standard normal tails, quantile, and truncated-normal moments.
//!
//! `cdf` and `sf` take Cephes `ndtr`'s branches (`erf` near the centre,
//! `erfc` in the tails) on libm's `erf` and `erfc`, so they agree with
//! SciPy's `ndtr` to a few ulp, not bit for bit; assortative mating, which
//! must match SciPy's bits, has its own Cephes port (`assortative::cephes`).
//! `quantile` is Wichura's AS241, the algorithm and coefficients of R's
//! `qnorm`, so a threshold here is the one PAFGRS computes; assortative
//! mating evaluates the same coefficients with pedsum's rounding
//! ([`as241`] with [`As241Rounding::Pedsum`]).  The truncated moments port
//! fitACE's PA-FGRS kernels line for line, guards included.

use std::f64::consts::FRAC_1_SQRT_2;

const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Density of N(0, 1), as fitACE writes it; assortative mating's
/// `bvn::phi` divides by `sqrt(2 pi)` instead, as pedsum does.
#[inline]
pub(crate) fn pdf(x: f64) -> f64 {
    FRAC_1_SQRT_2PI * (-0.5 * x * x).exp()
}

/// P(Z <= x).
#[inline]
pub(crate) fn cdf(x: f64) -> f64 {
    let t = x * FRAC_1_SQRT_2;
    let z = t.abs();
    if z < FRAC_1_SQRT_2 {
        0.5 + 0.5 * libm::erf(t)
    } else {
        let y = 0.5 * libm::erfc(z);
        if t > 0.0 {
            1.0 - y
        } else {
            y
        }
    }
}

/// P(Z > x).
#[inline]
pub(crate) fn sf(x: f64) -> f64 {
    cdf(-x)
}

/// AS241's three rational approximations, numerator then denominator
/// coefficients from the highest power down, verbatim from R's `qnorm.c`.
#[allow(clippy::excessive_precision)]
const CENTRAL: [f64; 15] = [
    2509.0809287301226727,
    33430.575583588128105,
    67265.770927008700853,
    45921.953931549871457,
    13731.693765509461125,
    1971.5909503065514427,
    133.14166789178437745,
    3.387132872796366608,
    5226.495278852854561,
    28729.085735721942674,
    39307.89580009271061,
    21213.794301586595867,
    5394.1960214247511077,
    687.1870074920579083,
    42.313330701600911252,
];
#[allow(clippy::excessive_precision)]
const NEAR_TAIL: [f64; 15] = [
    7.7454501427834140764e-4,
    0.0227238449892691845833,
    0.24178072517745061177,
    1.27045825245236838258,
    3.64784832476320460504,
    5.7694972214606914055,
    4.6303378461565452959,
    1.42343711074968357734,
    1.05075007164441684324e-9,
    5.475938084995344946e-4,
    0.0151986665636164571966,
    0.14810397642748007459,
    0.68976733498510000455,
    1.6763848301838038494,
    2.05319162663775882187,
];
#[allow(clippy::excessive_precision)]
const FAR_TAIL: [f64; 15] = [
    2.01033439929228813265e-7,
    2.71155556874348757815e-5,
    0.0012426609473880784386,
    0.026532189526576123093,
    0.29656057182850489123,
    1.7848265399172913358,
    5.4637849111641143699,
    6.6579046435011037772,
    2.04426310338993978564e-15,
    1.4215117583164458887e-7,
    1.8463183175100546818e-5,
    7.868691311456132591e-4,
    0.0148753612908506148525,
    0.13692988092273580531,
    0.59983220655588793769,
];

/// `(num(r), den(r))` for the eight numerator and seven denominator
/// coefficients of `c`, Horner order as `qnorm.c` writes it, `den` ending
/// in `+ 1`.
#[inline]
fn rational(c: &[f64; 15], r: f64) -> (f64, f64) {
    let num = c[1..8].iter().fold(c[0], |acc, &k| acc * r + k);
    let den = c[9..15].iter().fold(c[8], |acc, &k| acc * r + k) * r + 1.0;
    (num, den)
}

/// Where the two AS241 ports this crate reproduces round differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum As241Rounding {
    /// R's `qnorm`: `q * (num / den)`, upper tail `0.5 - p + 0.5`.
    R,
    /// pedsum's kernel `ndtri`: `q * num / den`, upper tail `1 - p`.
    Pedsum,
}

/// The `p` quantile of N(0, 1) for `0 < p < 1` (AS241, as R's `qnorm`).
/// `p` of 0 or 1 gives an infinity, and NaN passes through.  Every `p` a
/// CIP table can produce has `min(p, 1 - p) > exp(-27^2)`, so R's
/// asymptotic branch beyond is not needed.
pub(crate) fn quantile(p: f64) -> f64 {
    as241(p, As241Rounding::R)
}

/// AS241 with `rounding`'s arithmetic; see [`quantile`].
pub(crate) fn as241(p: f64, rounding: As241Rounding) -> f64 {
    if p.is_nan() {
        return p;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let (num, den) = rational(&CENTRAL, 0.180625 - q * q);
        return match rounding {
            As241Rounding::R => q * (num / den),
            As241Rounding::Pedsum => q * num / den,
        };
    }
    let tail = match (q > 0.0, rounding) {
        (false, _) => p,
        (true, As241Rounding::R) => 0.5 - p + 0.5,
        (true, As241Rounding::Pedsum) => 1.0 - p,
    };
    let r = (-tail.ln()).sqrt();
    let (num, den) = if r <= 5.0 {
        rational(&NEAR_TAIL, r - 1.6)
    } else {
        rational(&FAR_TAIL, r - 5.0)
    };
    let val = num / den;
    if q < 0.0 {
        -val
    } else {
        val
    }
}

/// Mean and variance of N(mu, var) truncated to (-inf, upper].
fn below(mu: f64, var: f64, upper: f64) -> (f64, f64) {
    let sd = var.sqrt();
    if sd < 1e-15 {
        return (mu, 0.0);
    }
    let beta = (upper - mu) / sd;
    let cdf_b = cdf(beta);
    if cdf_b < 1e-15 {
        return (upper, 0.0);
    }
    let r = pdf(beta) / cdf_b;
    (mu - sd * r, (var * (1.0 - beta * r - r * r)).max(0.0))
}

/// Mean and variance of N(mu, var) truncated to [lower, inf).
fn above(mu: f64, var: f64, lower: f64) -> (f64, f64) {
    let sd = var.sqrt();
    if sd < 1e-15 {
        return (mu, 0.0);
    }
    let alpha = (lower - mu) / sd;
    let sf_a = sf(alpha);
    if sf_a < 1e-15 {
        return (lower, 0.0);
    }
    let r = pdf(alpha) / sf_a;
    (mu + sd * r, (var * (1.0 + alpha * r - r * r)).max(0.0))
}

/// Mean and variance of N(mu, var) truncated to [lower, upper].
pub(crate) fn truncated(mu: f64, var: f64, lower: f64, upper: f64) -> (f64, f64) {
    if lower == upper {
        return (if lower.is_infinite() { 1e10 } else { lower }, 0.0);
    }
    if lower == f64::NEG_INFINITY {
        return below(mu, var, upper);
    }
    if upper == f64::INFINITY {
        return above(mu, var, lower);
    }
    let sd = var.sqrt();
    if sd < 1e-15 {
        return (mu, 0.0);
    }
    let a = (lower - mu) / sd;
    let b = (upper - mu) / sd;
    let mass = cdf(b) - cdf(a);
    if mass < 1e-15 {
        return ((lower + upper) / 2.0, 0.0);
    }
    let (pa, pb) = (pdf(a), pdf(b));
    let ratio = (pb - pa) / mass;
    let m = mu - sd * ratio;
    let v = var * (1.0 - (b * pb - a * pa) / mass - ratio * ratio);
    (m, v.max(0.0))
}

/// Moments of one observation under PA-FGRS[mix].
///
/// A case (`upper` infinite) is truncated above its threshold.  A control
/// whose lifetime risk is partly observed (`w < 1`) is a two-component
/// mixture: truncated below the threshold (truly unaffected) or above it (a
/// future case), weighted by the conditional probability of each given the
/// remaining risk `kp = w * P(Z > upper)`.
pub(crate) fn observation(mu: f64, var: f64, lower: f64, upper: f64, w: f64) -> (f64, f64) {
    let sf_marg = if upper == f64::INFINITY {
        0.0
    } else {
        sf(upper)
    };
    let kp = w * sf_marg;
    if kp <= 0.0 || upper == f64::INFINITY {
        return truncated(mu, var, lower, upper);
    }
    let sd = var.sqrt();
    let cdf_cond = cdf((upper - mu) / sd);
    let sf_cond = 1.0 - cdf_cond;
    let w_below = if sf_marg < 1e-15 {
        1.0
    } else {
        let denom = 1.0 - sf_cond * kp / sf_marg;
        if denom.abs() > 1e-15 {
            cdf_cond / denom
        } else {
            1.0
        }
    }
    .clamp(0.0, 1.0);
    let w_above = 1.0 - w_below;
    let (m0, v0) = truncated(mu, var, lower, upper);
    let (m1, v1) = truncated(mu, var, upper, f64::INFINITY);
    let mean = w_below * m0 + w_above * m1;
    let var = w_below * (m0 * m0 + v0) + w_above * (m1 * m1 + v1) - mean * mean;
    (mean, var.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn tails_are_complementary_and_symmetric() {
        for &x in &[-8.0, -3.1, -0.7, 0.0, 0.3, 0.71, 2.2, 6.0] {
            assert!((cdf(x) + sf(x) - 1.0).abs() < 1e-15);
            assert_eq!(cdf(x), sf(-x));
        }
        assert_eq!(cdf(0.0), 0.5);
        assert!((pdf(0.0) - 1.0 / (2.0 * PI).sqrt()).abs() < 1e-16);
    }

    #[test]
    fn quantile_inverts_cdf() {
        for &p in &[
            1e-15,
            1e-9,
            0.001,
            0.05,
            0.075,
            0.3,
            0.5,
            0.9,
            0.925,
            0.99,
            1.0 - 1e-12,
        ] {
            let z = quantile(p);
            let back = cdf(z);
            assert!(((back - p) / p).abs() < 1e-12, "p {p} z {z} back {back}");
        }
        assert_eq!(quantile(0.5), 0.0);
        assert_eq!(quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(quantile(1.0), f64::INFINITY);
    }

    #[test]
    fn truncated_moments_match_closed_forms() {
        // Half normal: mean sqrt(2/pi), variance 1 - 2/pi.
        let (m, v) = truncated(0.0, 1.0, 0.0, f64::INFINITY);
        assert!((m - (2.0 / PI).sqrt()).abs() < 1e-15);
        assert!((v - (1.0 - 2.0 / PI)).abs() < 1e-15);
        let (m, _) = truncated(0.0, 1.0, f64::NEG_INFINITY, 0.0);
        assert!((m + (2.0 / PI).sqrt()).abs() < 1e-15);
        let (m, v) = truncated(0.0, 4.0, -1.0, 1.0);
        assert!(m.abs() < 1e-15 && v < 4.0 / 3.0 && v > 0.0);
    }

    #[test]
    fn fully_observed_control_is_plain_truncation() {
        let t = quantile(0.95);
        let mix = observation(0.2, 0.9, f64::NEG_INFINITY, t, 1.0);
        let plain = truncated(0.2, 0.9, f64::NEG_INFINITY, t);
        assert!((mix.0 - plain.0).abs() < 1e-14 && (mix.1 - plain.1).abs() < 1e-14);
    }
}
