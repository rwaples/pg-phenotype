//! Fitting ρ of a latent correlation: Newton on the analytic score, with
//! SciPy's bounded Brent as the fallback, and the boundary flag.

use super::kernels::LATENT_BOUND;
use super::result::Point;

/// A fit within this distance of a bound is flagged (pedsum decision 9).
pub(crate) const BOUNDARY_MARGIN: f64 = 1e-3;
/// So is one whose NLL the nearer bound matches to this relative tolerance.
pub(crate) const BOUNDARY_NLL_TOL: f64 = 1e-6;
const NEWTON_TOL: f64 = 1e-8;
const NEWTON_MAX_ITER: usize = 12;
/// Newton starts are clipped to this.
const START_CLIP: f64 = 0.99;
/// `minimize_scalar(..., method="bounded", options={"xatol": 1e-7})`.
const BRENT_XATOL: f64 = 1e-7;

/// SciPy's `_minimize_scalar_bounded` (`scipy/optimize/_optimize.py`,
/// 1.18.1) with `maxiter = 500`: `(x, f(x))` at the minimum on `[x1, x2]`.
pub(crate) fn minimize_bounded(
    mut func: impl FnMut(f64) -> f64,
    x1: f64,
    x2: f64,
    xatol: f64,
) -> (f64, f64) {
    let maxfun = 500;
    let sqrt_eps = 2.2e-16_f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());
    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let (mut rat, mut e) = (0.0_f64, 0.0_f64);
    let mut fx = func(xf);
    let mut num = 1;
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;
    let sign = |v: f64| -> f64 {
        if v.is_nan() {
            v
        } else if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    let indicator = |c: bool| if c { 1.0 } else { 0.0 };
    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = (p + 0.0) / q;
                let x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = sign(xm - xf) + indicator((xm - xf) == 0.0);
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let si = sign(rat) + indicator(rat == 0.0);
        // np.maximum propagates NaN; f64::max would not.
        let step = if rat.is_nan() {
            rat
        } else {
            rat.abs().max(tol1)
        };
        let x = xf + si * step;
        let fu = func(x);
        num += 1;
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            (fulc, ffulc) = (nfc, fnfc);
            (nfc, fnfc) = (xf, fx);
            (xf, fx) = (x, fu);
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                (fulc, ffulc) = (nfc, fnfc);
                (nfc, fnfc) = (x, fu);
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                (fulc, ffulc) = (x, fu);
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= maxfun {
            break;
        }
    }
    (xf, fx)
}

/// `rho` with pedsum's boundary flag: at a bound, or on a plateau reaching one.
fn flag_boundary(nll: &mut impl FnMut(f64) -> f64, rho: f64, best: f64) -> Point {
    let at_bound = rho.abs() >= LATENT_BOUND - BOUNDARY_MARGIN;
    let plateau = nll(LATENT_BOUND.copysign(rho)) - best <= BOUNDARY_NLL_TOL * best.abs().max(1.0);
    Point {
        value: rho,
        boundary: Some(at_bound || plateau),
    }
}

/// ρ̂ on `(-LATENT_BOUND, LATENT_BOUND)` by bounded Brent.
pub(crate) fn maximise_rho(mut nll: impl FnMut(f64) -> f64) -> Point {
    let (rho, best) = minimize_bounded(&mut nll, -LATENT_BOUND, LATENT_BOUND, BRENT_XATOL);
    flag_boundary(&mut nll, rho, best)
}

/// ρ̂ by Newton on the analytic score from `start`; `terms(ρ)` is the NLL
/// and its first two ρ-derivatives.  Hands over to [`maximise_rho`] when the
/// Hessian is not positive, a step leaves the bound, or 12 steps do not
/// converge.
pub(crate) fn newton_rho(
    mut terms: impl FnMut(f64) -> (f64, f64, f64),
    mut nll: impl FnMut(f64) -> f64,
    start: f64,
) -> Point {
    let mut rho = start.clamp(-START_CLIP, START_CLIP);
    for _ in 0..NEWTON_MAX_ITER {
        let (best, grad, hess) = terms(rho);
        if !(hess > 0.0) {
            return maximise_rho(nll);
        }
        let step = -grad / hess;
        rho += step;
        if !(-LATENT_BOUND < rho && rho < LATENT_BOUND) {
            return maximise_rho(nll);
        }
        if step.abs() < NEWTON_TOL {
            return flag_boundary(&mut nll, rho, best);
        }
    }
    maximise_rho(nll)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brent_finds_an_interior_minimum() {
        let (x, f) = minimize_bounded(|x| (x - 0.3) * (x - 0.3) + 1.0, -1.0, 1.0, 1e-7);
        assert!((x - 0.3).abs() < 1e-6 && (f - 1.0).abs() < 1e-12);
    }

    #[test]
    fn newton_hands_over_and_flags_bounds() {
        let nll = |r: f64| -10.0 * r;
        let fit = newton_rho(|r| (nll(r), -10.0, 0.0), nll, 0.0);
        assert!(fit.value > LATENT_BOUND - BOUNDARY_MARGIN);
        assert_eq!(fit.boundary, Some(true));
        let q = |r: f64| (r - 0.2) * (r - 0.2);
        let fit = newton_rho(|r| (q(r), 2.0 * (r - 0.2), 2.0), q, 0.0);
        assert!((fit.value - 0.2).abs() < 1e-12);
        assert_eq!(fit.boundary, Some(false));
    }
}
