//! Pearson-Aitken conditioning on a sequence of truncated observations.

use crate::normal;

/// One observation's truncation interval and risk weight.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Obs {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) w: f64,
}

/// Condition the leading `targets` dimensions of a joint normal on `obs`.
///
/// `cov` is `size * size` row-major, `size = targets + obs.len()`, and only
/// its upper triangle is read or written.  Observation `i` is dimension
/// `targets + i`.  Conditioning runs from the last dimension to the first
/// observation, so `obs` is ordered most informative first.  On return
/// `mu[..targets]` and the leading upper block of `cov` hold the posterior.
///
/// The arithmetic is fitACE's numba kernel operation for operation: each
/// step updates the mean, then the upper triangle by the rank-one term
/// `cov[a, j] * cov[b, j] * (1/v - u/v^2)`.
pub(crate) fn condition(
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
