//! The canonical conditioning order (ADR 0004).
//!
//! Observations are sorted most informative first and conditioned from the
//! least informative end.  The key is R PAFGRS's lexicographic one with a
//! unique final key, so no two observations tie and the order never
//! depends on input order or sort stability.
//!
//! Univariate, for relative `r` of proband `p` among the valid relatives:
//! `w` descending, `C[r, p]` descending, `sum_j C[r, j]` descending (the
//! proband column and the diagonal included, as R's
//! `rowSums(covmat[-1, ])`), row ascending.  Off the diagonal `C = 2 h2
//! phi` with `h2 > 0`, so the second and third keys order exactly as
//! `phi(r, p)` and `phi(r, p) + sum_{j != r} phi(r, j)`, which is what is
//! compared: dyadic float32 kinship summed in float64 is exact, so a tie in
//! exact arithmetic is a tie here, broken by row.

#[derive(Clone, Copy, Debug)]
pub(crate) struct UniKey {
    pub(crate) w: f64,
    pub(crate) to_proband: f64,
    pub(crate) row_sum: f64,
    pub(crate) row: u32,
    /// Position in the caller's valid list.
    pub(crate) index: usize,
}

pub(crate) fn sort_univariate(keys: &mut [UniKey]) {
    keys.sort_unstable_by(|a, b| {
        b.w.total_cmp(&a.w)
            .then_with(|| b.to_proband.total_cmp(&a.to_proband))
            .then_with(|| b.row_sum.total_cmp(&a.row_sum))
            .then_with(|| a.row.cmp(&b.row))
    });
}

/// Bivariate observation `o = (r, t)`: `w` descending, `|C[o, p1]| +
/// |C[o, p2]|` descending, `sum_j |C[o, j]|` descending, row ascending,
/// trait ascending.
///
/// With `G` the proband's genetic covariance and `t'` the other trait,
/// `|C[o, p1]| + |C[o, p2]| = 2 phi(r, p) (h2_t + |cov_g|)` and the row sum
/// is that plus `1`, plus `|rho_within|` when `(r, t')` is observed, plus
/// `2 (h2_t A + |cov_g| B)`, where `A` and `B` sum `phi(r, s)` over the
/// other relatives `s` observed on `t` and on `t'`.  `A` and `B` are exact
/// and the rest is one fixed expression, so exact ties stay ties.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BivKey {
    pub(crate) w: f64,
    pub(crate) to_proband: f64,
    pub(crate) row_sum: f64,
    pub(crate) row: u32,
    pub(crate) trait_index: u8,
    pub(crate) index: usize,
}

pub(crate) fn sort_bivariate(keys: &mut [BivKey]) {
    keys.sort_unstable_by(|a, b| {
        b.w.total_cmp(&a.w)
            .then_with(|| b.to_proband.total_cmp(&a.to_proband))
            .then_with(|| b.row_sum.total_cmp(&a.row_sum))
            .then_with(|| a.row.cmp(&b.row))
            .then_with(|| a.trait_index.cmp(&b.trait_index))
    });
}
