"""NumPy reference cores of PA-FGRS, the oracle the Rust kernels answer to.

The truncated-normal helpers and the two conditioning cores are fitACE's
Python fallbacks (``fitace_pafgrs/pafgrs.py`` and ``pafgrs_bivariate.py`` at
fitACE ``cf9a374``), copied verbatim except that the conditioning order is
an argument instead of the R/NumPy lexsort written inline, so a test can
hand the oracle the order it is checking (ADR 0004).  Do not "fix" them:
they are the reference.

:func:`canonical_univariate_order` and :func:`canonical_bivariate_order` are
the #16 keys written independently of the Rust code; :func:`numba_order` is
the scalar key fitACE's numba scorer used, for the gate's order step.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Truncated normal helpers (verbatim)
# ---------------------------------------------------------------------------


def _trunc_norm_below_py(mu: float, var: float, trunc: float) -> tuple[float, float]:
    """E[X] and Var[X] for X ~ N(mu, var) truncated to (-inf, trunc]."""
    sd = np.sqrt(var)
    if sd < 1e-15:
        return mu, 0.0
    beta = (trunc - mu) / sd
    phi_b = norm.pdf(beta)
    cdf_b = norm.cdf(beta)
    if cdf_b < 1e-15:
        return trunc, 0.0
    r = phi_b / cdf_b
    return mu - sd * r, max(var * (1 - beta * r - r * r), 0.0)


def _trunc_norm_above_py(mu: float, var: float, trunc: float) -> tuple[float, float]:
    """E[X] and Var[X] for X ~ N(mu, var) truncated to [trunc, +inf)."""
    sd = np.sqrt(var)
    if sd < 1e-15:
        return mu, 0.0
    alpha = (trunc - mu) / sd
    phi_a = norm.pdf(alpha)
    sf_a = norm.sf(alpha)
    if sf_a < 1e-15:
        return trunc, 0.0
    r = phi_a / sf_a
    return mu + sd * r, max(var * (1 + alpha * r - r * r), 0.0)


def _trunc_norm_py(mu: float, var: float, lower: float, upper: float) -> tuple[float, float]:
    """E[X] and Var[X] for X ~ N(mu, var) truncated to [lower, upper]."""
    if lower == upper:
        return (1e10 if np.isinf(lower) else lower), 0.0
    if np.isneginf(lower):
        return _trunc_norm_below_py(mu, var, upper)
    if np.isposinf(upper):
        return _trunc_norm_above_py(mu, var, lower)
    sd = np.sqrt(var)
    if sd < 1e-15:
        return mu, 0.0
    a = (lower - mu) / sd
    b = (upper - mu) / sd
    Phi_diff = norm.cdf(b) - norm.cdf(a)
    if Phi_diff < 1e-15:
        return (lower + upper) / 2.0, 0.0
    phi_a, phi_b = norm.pdf(a), norm.pdf(b)
    ratio = (phi_b - phi_a) / Phi_diff
    m = mu - sd * ratio
    v = var * (1 - (b * phi_b - a * phi_a) / Phi_diff - ratio * ratio)
    return m, max(v, 0.0)


def _trunc_norm_mixture_py(
    mu: float,
    var: float,
    lower: float,
    upper: float,
    kp: float,
) -> tuple[float, float]:
    """Mixture model moments for partially-observed controls."""
    if kp <= 0 or np.isposinf(upper):
        return _trunc_norm_py(mu, var, lower, upper)

    sd = np.sqrt(var)
    cdf_u_cond = norm.cdf(upper, loc=mu, scale=sd)
    sf_u_cond = 1.0 - cdf_u_cond
    sf_u_marg = norm.sf(upper)

    if sf_u_marg < 1e-15:
        w_below = 1.0
    else:
        denom = 1.0 - sf_u_cond * kp / sf_u_marg
        w_below = cdf_u_cond / denom if abs(denom) > 1e-15 else 1.0

    w_below = np.clip(w_below, 0.0, 1.0)
    w_above = 1.0 - w_below

    m0, v0 = _trunc_norm_py(mu, var, lower, upper)
    m1, v1 = _trunc_norm_py(mu, var, upper, np.inf)

    new_mean = w_below * m0 + w_above * m1
    new_var = w_below * (m0 * m0 + v0) + w_above * (m1 * m1 + v1) - new_mean * new_mean
    return new_mean, max(new_var, 0.0)


# ---------------------------------------------------------------------------
# Conditioning cores (verbatim but for the `order` argument)
# ---------------------------------------------------------------------------


def pa_fgrs_core(
    rel_t1: np.ndarray, rel_t2: np.ndarray, rel_w: np.ndarray, covmat: np.ndarray, order: np.ndarray
) -> tuple[float, float]:
    """Univariate conditioning; *order* sorts the valid relatives most informative first."""
    valid = (np.isfinite(rel_t1) | np.isfinite(rel_t2)) & (rel_w > 0)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return 0.0, float(covmat[0, 0])

    rel_t1 = rel_t1[valid]
    rel_t2 = rel_t2[valid]
    rel_w = rel_w[valid]
    keep = np.concatenate([[0], np.where(valid)[0] + 1])
    cm = covmat[np.ix_(keep, keep)].copy()

    rel_t1 = rel_t1[order]
    rel_t2 = rel_t2[order]
    rel_w = rel_w[order]
    reorder = np.concatenate([[0], order + 1])
    cm = cm[np.ix_(reorder, reorder)]

    n = cm.shape[0]
    mu = np.zeros(n)
    cov = cm

    for _ in range(n_valid):
        j = cov.shape[0] - 1
        ri = j - 1

        kp = rel_w[ri] * norm.sf(rel_t2[ri])
        upd_m, upd_v = _trunc_norm_mixture_py(mu[j], cov[j, j], rel_t1[ri], rel_t2[ri], kp)

        c_j = cov[:j, j].copy()
        inv_vj = 1.0 / cov[j, j] if cov[j, j] > 1e-30 else 0.0

        mu[:j] += c_j * inv_vj * (upd_m - mu[j])
        cov[:j, :j] -= np.outer(c_j, c_j) * (inv_vj - inv_vj * upd_v * inv_vj)

        mu = mu[:j]
        cov = cov[:j, :j]

    return float(mu[0]), float(max(cov[0, 0], 0.0))


def pa_fgrs_bivariate_core(
    obs_lower: np.ndarray, obs_upper: np.ndarray, obs_w: np.ndarray, covmat: np.ndarray, order: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Bivariate conditioning; *order* sorts the valid observations most informative first."""
    valid = (np.isfinite(obs_lower) | np.isfinite(obs_upper)) & (obs_w > 0)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return 0.0, 0.0, float(covmat[0, 0]), float(covmat[1, 1]), float(covmat[0, 1])

    obs_lower = obs_lower[valid]
    obs_upper = obs_upper[valid]
    obs_w = obs_w[valid]

    obs_cm_idx = np.where(valid)[0] + 2
    keep = np.concatenate([[0, 1], obs_cm_idx])
    cm = covmat[np.ix_(keep, keep)].copy()

    obs_lower = obs_lower[order]
    obs_upper = obs_upper[order]
    obs_w = obs_w[order]
    reorder = np.concatenate([[0, 1], order + 2])
    cm = cm[np.ix_(reorder, reorder)]

    mu = np.zeros(cm.shape[0])
    cov = cm

    for _ in range(n_valid):
        j = cov.shape[0] - 1
        ri = j - 2

        kp = obs_w[ri] * norm.sf(obs_upper[ri])
        upd_m, upd_v = _trunc_norm_mixture_py(mu[j], cov[j, j], obs_lower[ri], obs_upper[ri], kp)

        c_j = cov[:j, j].copy()
        inv_vj = 1.0 / cov[j, j] if cov[j, j] > 1e-30 else 0.0

        mu[:j] += c_j * inv_vj * (upd_m - mu[j])
        cov[:j, :j] -= np.outer(c_j, c_j) * (inv_vj - inv_vj * upd_v * inv_vj)

        mu = mu[:j]
        cov = cov[:j, :j]

    return (
        float(mu[0]),
        float(mu[1]),
        float(max(cov[0, 0], 0.0)),
        float(max(cov[1, 1], 0.0)),
        float(cov[0, 1]),
    )


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def canonical_univariate_order(w: np.ndarray, kin_p: np.ndarray, kin: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """#16 univariate: w desc, phi(r, p) desc, phi(r, p) + sum_j phi(r, j) desc, row asc.

    *kin* is the symmetric kinship among the valid relatives with a zero
    diagonal, float64 of the float32 values (sums of dyadics: exact).
    """
    row_sum = kin_p + kin.sum(axis=1)
    return np.lexsort((rows, -row_sum, -kin_p, -w))


def canonical_bivariate_order(
    w: np.ndarray,
    kin_p: np.ndarray,
    kin: np.ndarray,
    rows: np.ndarray,
    traits: np.ndarray,
    person: np.ndarray,
    h2: tuple[float, float],
    cov_g: float,
    rho_within: float,
) -> np.ndarray:
    """#16 bivariate over observations ``o = (r, t)``.

    ``|C[o, p1]| + |C[o, p2]| = 2 phi_rp (h2_t + |cov_g|)`` and
    ``sum_j |C[o, j]| = that + 1 + |rho| [(r, t') observed] + 2 (h2_t A + |cov_g| B)``
    with ``A``/``B`` the kinship sums over other people observed on ``t``/``t'``.
    *kin* is indexed by observation; *person* maps each observation to its person.
    """
    m = len(w)
    h2v = np.array(h2)[traits]
    abs_cov = abs(cov_g)
    to_proband = 2.0 * kin_p * (h2v + abs_cov)
    other_person = person[:, None] != person[None, :]
    same_trait = traits[:, None] == traits[None, :]
    a = np.where(other_person & same_trait, kin, 0.0).sum(axis=1)
    b = np.where(other_person & ~same_trait, kin, 0.0).sum(axis=1)
    paired = np.array([np.any((person == person[i]) & (np.arange(m) != i)) for i in range(m)])
    rho_term = np.where(paired, abs(rho_within), 0.0)
    row_sum = to_proband + 1.0 + rho_term + 2.0 * (h2v * a + abs_cov * b)
    return np.lexsort((traits, rows, -row_sum, -to_proband, -w))


def numba_order(w: np.ndarray, kin_p: np.ndarray, h2: float) -> np.ndarray:
    """fitACE numba's univariate key ``w*1e12 + 2 phi h2 * 1e6``, argsort descending."""
    return np.argsort(-(w * 1e12 + (2.0 * kin_p * h2) * 1e6))


def numba_bivariate_order(w: np.ndarray, kin_p: np.ndarray, traits: np.ndarray, g: np.ndarray) -> np.ndarray:
    """fitACE numba's bivariate key ``w*1e12 + (|C[o,p1]| + |C[o,p2]|)*1e6``, argsort descending."""
    c = 2.0 * kin_p[:, None] * g[:, traits].T
    return np.argsort(-(w * 1e12 + (np.abs(c[:, 0]) + np.abs(c[:, 1])) * 1e6))
