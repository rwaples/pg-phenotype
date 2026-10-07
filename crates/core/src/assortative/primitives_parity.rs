//! The primitives against SciPy and pedsum's kernels at the oracle pin
//! (`tools/make_pedsum_am_primitives.py`, values as IEEE-754 bits).

use super::{bvn, cephes, fit, kernels};

const PRIMITIVES: &str =
    include_str!("../../../../tests/golden/pedsum_am_142adf300d5b/primitives.txt");

fn rows(name: &str) -> Vec<Vec<f64>> {
    PRIMITIVES
        .lines()
        .filter_map(|line| {
            let mut parts = line.split(' ');
            (parts.next() == Some(name)).then(|| {
                parts
                    .map(|h| f64::from_bits(u64::from_str_radix(h, 16).expect("hex")))
                    .collect()
            })
        })
        .collect()
}

/// Distance in units in the last place; 0 for equal values, infinities and NaNs included.
fn ulps(a: f64, b: f64) -> u64 {
    if a == b || (a.is_nan() && b.is_nan()) {
        return 0;
    }
    if a.is_nan() || b.is_nan() || a.signum() != b.signum() {
        return u64::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

fn worst(name: &str, got: impl Fn(&[f64]) -> Vec<f64>, n_args: usize) -> (u64, f64) {
    let mut worst = (0u64, 0.0f64);
    for row in rows(name) {
        let want = &row[n_args..];
        for (g, w) in got(&row[..n_args]).iter().zip(want) {
            worst.0 = worst.0.max(ulps(*g, *w));
            if g.is_finite() && w.is_finite() {
                worst.1 = worst.1.max((g - w).abs());
            }
        }
    }
    println!("{name}: max {} ulp, max abs {:e}", worst.0, worst.1);
    worst
}

// Measured 2026-10-07 (Linux, glibc 2.39): every port bit-identical on the grid.

#[test]
fn cephes_ports_are_bit_identical_to_scipy() {
    assert_eq!(worst("ndtr", |a| vec![cephes::ndtr(a[0])], 1).0, 0);
    assert_eq!(
        worst("owens_t", |a| vec![cephes::owens_t(a[0], a[1])], 2).0,
        0
    );
}

#[test]
fn kernel_normal_functions_match_numba() {
    assert_eq!(
        worst("kernel_ndtr", |a| vec![cephes::kernel_ndtr(a[0])], 1).0,
        0
    );
    assert_eq!(worst("ndtri", |a| vec![kernels::ndtri(a[0])], 1).0, 0);
}

#[test]
fn bivariate_normal_matches_pedsum() {
    let (cdf_ulps, cdf_abs) = worst("bvn", |a| vec![bvn::cdf(a[0], a[1], a[2])], 3);
    assert_eq!(cdf_ulps, 0, "cdf {cdf_abs:e}");
    let (density_ulps, _) = worst(
        "bvn",
        |a| {
            let (pdf, drho) = bvn::pdf_and_drho(a[0], a[1], a[2]);
            vec![bvn::cdf(a[0], a[1], a[2]), pdf, drho]
        },
        3,
    );
    assert_eq!(density_ulps, 0);
}

#[test]
fn bounded_brent_retraces_scipy() {
    for row in rows("brent") {
        let (c, amp) = (row[0], row[1]);
        let mut nfev = 0.0;
        let (x, fun) = fit::minimize_bounded(
            |x| {
                nfev += 1.0;
                (x - c) * (x - c) + amp * (5.0 * x).sin()
            },
            -kernels::LATENT_BOUND,
            kernels::LATENT_BOUND,
            1e-7,
        );
        assert_eq!((x, fun, nfev), (row[2], row[3], row[4]), "c {c} amp {amp}");
    }
}
