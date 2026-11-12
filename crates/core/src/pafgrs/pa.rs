//! Pearson-Aitken conditioning on a sequence of truncated observations.

use crate::normal;

/// One observation's truncation interval and risk weight.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Obs {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) w: f64,
}

/// Upper triangle of a symmetric covariance, columns back to back: column
/// `b` holds `(0..=b, b)`.  Conditioning step `j` reads and updates only the
/// leading `j` columns, a contiguous prefix, and column `j` is one slice.
#[derive(Debug, Default)]
pub(crate) struct UpperCov {
    size: usize,
    v: Vec<f64>,
}

impl UpperCov {
    /// `size` dimensions, every entry zero.  Keeps the allocation.
    pub(crate) fn reset(&mut self, size: usize) {
        self.size = size;
        self.v.clear();
        self.v.resize(col_start(size), 0.0);
    }

    /// Set entry `(a, b)`, `a <= b < size`.
    #[inline]
    pub(crate) fn set(&mut self, a: usize, b: usize, x: f64) {
        debug_assert!(a <= b && b < self.size);
        self.v[col_start(b) + a] = x;
    }

    /// Entry `(a, b)`, `a <= b < size`.
    #[inline]
    pub(crate) fn get(&self, a: usize, b: usize) -> f64 {
        debug_assert!(a <= b && b < self.size);
        self.v[col_start(b) + a]
    }
}

/// How far below zero a variance may fall while conditioning and still be
/// rounding; a covariance's entries are at most of order 1.
pub(crate) const VARIANCE_SLACK: f64 = 1e-9;

/// Whether the targets' block [`condition`] left is a posterior: every
/// mean (and covariance) in `finite` is finite, and every variance is
/// finite and not negative beyond [`VARIANCE_SLACK`].
///
/// Each step of [`condition`] keeps a positive semidefinite covariance one,
/// so a negative variance proves the covariance was not (ADR 0015).
pub(crate) fn is_posterior(finite: &[f64], variances: &[f64]) -> bool {
    finite.iter().all(|x| x.is_finite())
        && variances
            .iter()
            .all(|v| (-VARIANCE_SLACK..f64::INFINITY).contains(v))
}

/// Where column `b` starts: the `b` columns before it hold `1 + .. + b`.
#[inline]
fn col_start(b: usize) -> usize {
    b * (b + 1) / 2
}

/// Condition the leading `targets` dimensions of a joint normal on `obs`.
///
/// `cov.size` is `targets + obs.len()`.  Observation `i` is dimension
/// `targets + i`.  Conditioning runs from the last dimension to the first
/// observation, so `obs` is ordered most informative first.  On return
/// `mu[..targets]` and the leading upper block of `cov` hold the posterior.
///
/// The arithmetic is fitACE's numba kernel operation for operation: each
/// step updates the mean, then the upper triangle by the rank-one term
/// `cov[a, j] * cov[b, j] * (1/v - u/v^2)`.
///
/// Returns `false`, leaving `cov` and `mu` half conditioned, when an
/// observation's variance given the ones before it is below
/// `-`[`VARIANCE_SLACK`]: the covariance was not positive semidefinite
/// (ADR 0015).  A variance that rounding put below zero is taken as zero.
#[must_use]
pub(crate) fn condition(
    cov: &mut UpperCov,
    mu: &mut [f64],
    targets: usize,
    obs: &[Obs],
    col: &mut Vec<f64>,
) -> bool {
    let size = targets + obs.len();
    debug_assert_eq!(cov.size, size);
    let v = &mut cov.v;
    mu[..size].fill(0.0);
    for j in (targets..size).rev() {
        let o = obs[j - targets];
        let cj = col_start(j);
        let vj = v[cj + j];
        if vj < -VARIANCE_SLACK {
            return false;
        }
        // Its square root would be NaN; a variance not below zero is kept
        // as it is, a negative zero included.
        let vj = if vj < 0.0 { 0.0 } else { vj };
        let (m, u) = normal::observation(mu[j], vj, o.lower, o.upper, o.w);
        let inv = if vj > 1e-30 { 1.0 / vj } else { 0.0 };
        let delta = m - mu[j];
        let factor = inv - inv * u * inv;
        // Column j, copied once so the update below can write the prefix.
        col.clear();
        col.extend_from_slice(&v[cj..cj + j]);
        for (mu_a, &c_aj) in mu[..j].iter_mut().zip(col.iter()) {
            *mu_a += c_aj * inv * delta;
        }
        for b in 0..j {
            let c_bj = col[b];
            let start = col_start(b);
            let column = &mut v[start..=start + b];
            for (x, &c_aj) in column.iter_mut().zip(&col[..=b]) {
                *x -= c_aj * c_bj * factor;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The square row-major kernel this module replaced, kept as the oracle.
    fn condition_square(
        cov: &mut [f64],
        mu: &mut [f64],
        targets: usize,
        obs: &[Obs],
        col: &mut Vec<f64>,
    ) {
        let size = targets + obs.len();
        debug_assert_eq!(cov.len(), size * size);
        mu[..size].fill(0.0);
        for j in (targets..size).rev() {
            let o = obs[j - targets];
            let vj = cov[j * size + j];
            let (m, v) = normal::observation(mu[j], vj, o.lower, o.upper, o.w);
            let inv = if vj > 1e-30 { 1.0 / vj } else { 0.0 };
            let delta = m - mu[j];
            let factor = inv - inv * v * inv;
            // Column j, gathered once so the update below runs along rows.
            col.clear();
            col.extend((0..j).map(|a| cov[a * size + j]));
            for (mu_a, &c_aj) in mu[..j].iter_mut().zip(col.iter()) {
                *mu_a += c_aj * inv * delta;
            }
            for a in 0..j {
                let c_aj = col[a];
                let row = &mut cov[a * size + a..a * size + j];
                for (x, &c_bj) in row.iter_mut().zip(&col[a..j]) {
                    *x -= c_aj * c_bj * factor;
                }
            }
        }
    }

    /// The packed kernel against the square one, bit for bit, on reused
    /// buffers that shrink and grow and start out stale.
    #[test]
    fn matches_square_kernel_bit_for_bit() {
        for targets in [1, 2] {
            let mut col = Vec::new();
            let mut square_col = Vec::new();
            let mut cov = UpperCov::default();
            let mut mu = Vec::new();
            for m in [0, 1, 2, 3, 4, 5, 17, 48, 76, 184, 402, 1, 96, 0] {
                let size = targets + m;
                let entry = |a: usize, b: usize| {
                    if a == b {
                        if a < targets {
                            0.3 + a as f64 * 0.3
                        } else {
                            1.0
                        }
                    } else {
                        0.01 / (a + b + 1) as f64
                    }
                };
                let obs: Vec<_> = (0..m)
                    .map(|i| {
                        if i % 10 == 0 {
                            Obs {
                                lower: 1.2815515655446004,
                                upper: f64::INFINITY,
                                w: 1.0,
                            }
                        } else {
                            Obs {
                                lower: f64::NEG_INFINITY,
                                upper: 1.2815515655446004,
                                w: (i % 8 + 1) as f64 / 8.0,
                            }
                        }
                    })
                    .collect();
                let mut expected = vec![f64::NAN; size * size];
                for a in 0..size {
                    for b in a..size {
                        expected[a * size + b] = entry(a, b);
                    }
                }
                let mut expected_mu = vec![-7.0; size];
                condition_square(
                    &mut expected,
                    &mut expected_mu,
                    targets,
                    &obs,
                    &mut square_col,
                );

                cov.reset(size);
                for a in 0..size {
                    for b in a..size {
                        cov.set(a, b, entry(a, b));
                    }
                }
                mu.resize(size, -7.0);
                assert!(condition(&mut cov, &mut mu, targets, &obs, &mut col));
                for a in 0..size {
                    for b in a..size {
                        assert_eq!(
                            cov.get(a, b).to_bits(),
                            expected[a * size + b].to_bits(),
                            "targets={targets} m={m} ({a}, {b})"
                        );
                    }
                    assert_eq!(
                        mu[a].to_bits(),
                        expected_mu[a].to_bits(),
                        "targets={targets} m={m} mu[{a}]"
                    );
                }
            }
        }
    }

    /// A symmetric matrix from its rows.
    fn upper(rows: &[&[f64]]) -> UpperCov {
        let mut cov = UpperCov::default();
        cov.reset(rows.len());
        for (a, row) in rows.iter().enumerate() {
            for b in a..rows.len() {
                cov.set(a, b, row[b]);
            }
        }
        cov
    }

    /// Whether conditioning went through, and the posterior mean and
    /// variance of dimension 0, given cases above `qnorm(0.9)` in every
    /// other dimension.
    fn given_cases(rows: &[&[f64]]) -> (bool, f64, f64) {
        let mut cov = upper(rows);
        let case = Obs {
            lower: 1.2815515655446004,
            upper: f64::INFINITY,
            w: 1.0,
        };
        let mut mu = vec![0.0; rows.len()];
        let done = condition(
            &mut cov,
            &mut mu,
            1,
            &vec![case; rows.len() - 1],
            &mut Vec::new(),
        );
        (done, mu[0], cov.get(0, 0))
    }

    /// Issue #32's covariances, inbred MZ twins with kinship 11/16: a
    /// negative target variance where a twin is the proband, and where the
    /// twins are the relatives a second twin whose variance given the
    /// first is negative, which stops the conditioning.
    #[test]
    fn an_indefinite_covariance_leaves_no_posterior() {
        let (done, est, var) = given_cases(&[&[0.5, 0.6875], &[0.6875, 1.0]]);
        assert!(
            done && is_posterior(&[est], &[var]) && var > 0.1,
            "{est} {var}"
        );
        let (done, est, var) = given_cases(&[&[0.7, 0.9625], &[0.9625, 1.0]]);
        assert!(done && est.is_finite() && var < -0.06, "{est} {var}");
        assert!(!is_posterior(&[est], &[var]));
        let (done, ..) = given_cases(&[&[0.8, 0.8, 0.8], &[0.8, 1.0, 1.1], &[0.8, 1.1, 1.0]]);
        assert!(!done);
    }

    /// An observation's variance that rounding put below zero is zero: the
    /// step changes nothing, and no NaN reaches the target.
    #[test]
    fn a_variance_rounded_below_zero_is_zero() {
        let (done, est, var) = given_cases(&[&[0.5, 0.25], &[0.25, -1e-17]]);
        assert!(done);
        assert_eq!(
            (est.to_bits(), var.to_bits()),
            (0f64.to_bits(), 0.5f64.to_bits())
        );
    }

    #[test]
    fn a_posterior_is_finite_with_no_negative_variance() {
        assert!(is_posterior(&[0.3, -2.0], &[0.0, 1.5]));
        assert!(is_posterior(&[0.0], &[-1e-12]));
        assert!(!is_posterior(&[0.0], &[-1e-6]));
        assert!(!is_posterior(&[f64::NAN], &[0.1]));
        assert!(!is_posterior(&[f64::INFINITY], &[0.1]));
        assert!(!is_posterior(&[0.0], &[f64::NAN]));
        assert!(!is_posterior(&[0.0], &[f64::INFINITY]));
    }

    #[test]
    fn packed_index_covers_triangle_once() {
        for size in [0, 1, 2, 3, 7, 40] {
            let mut seen = vec![0u32; col_start(size)];
            for b in 0..size {
                for a in 0..=b {
                    seen[col_start(b) + a] += 1;
                }
            }
            assert!(seen.iter().all(|&n| n == 1), "size={size}");
        }
    }

    /// The differential test overwrites every entry, so it cannot see a
    /// reset that leaves stale values or the wrong length.
    #[test]
    fn reset_zeroes_and_sizes() {
        let mut cov = UpperCov::default();
        cov.reset(9);
        for size in [9, 4, 13] {
            for b in 0..cov.size {
                for a in 0..=b {
                    cov.set(a, b, 1.5);
                }
            }
            cov.reset(size);
            assert_eq!(cov.v.len(), size * (size + 1) / 2, "size={size}");
            for b in 0..size {
                for a in 0..=b {
                    assert_eq!(cov.get(a, b), 0.0, "size={size} ({a}, {b})");
                }
            }
        }
        for b in 0..cov.size {
            for a in 0..=b {
                cov.set(a, b, (100 * b + a) as f64);
            }
        }
        for b in 0..cov.size {
            for a in 0..=b {
                assert_eq!(cov.get(a, b), (100 * b + a) as f64, "({a}, {b})");
            }
        }
    }
}
