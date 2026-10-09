"""A SciPy reference for the tetrachoric correlation of a 2 x 2 table.

Independent of the Rust fit: on a 2 x 2 table the thresholds come from the
margins and the two-step MLE of rho matches the table exactly,
``Phi2(tau_x, tau_y; rho) = n00 / n``, so rho is that equation's root, found
by ``brentq`` to 1e-15.  The SEs are closed forms at that fit:

- ``se_two_step``: the delta-method SE of rho with the thresholds estimated,
  ``sqrt([I^-1]_rr)`` for the multinomial information
  ``I = n J' diag(1/p) J`` of (tau_x, tau_y, rho).  On a saturated table this
  is the two-step sandwich without its ``n/(n-1)`` factor.
- ``se_known_thresholds``: ``1 / sqrt(n phi2^2 sum(1/p))``, the thresholds
  treated as known (what simACE's ``tetrachoric_from_table`` intends).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from tests.oracle.assortative_reference import bvn_cdf

LATENT_BOUND = 0.9999


@dataclass(frozen=True)
class Reference:
    """``rho`` is ``None`` when the table has no interior fit (an empty
    cell, or a root outside the latent bound)."""

    rho: float | None
    se_two_step: float | None
    se_known_thresholds: float | None


def _phi2(h: float, k: float, rho: float) -> float:
    q = (1 - rho) * (1 + rho)
    return math.exp(-(h * h - 2 * rho * h * k + k * k) / (2 * q)) / (2 * math.pi * math.sqrt(q))


def reference(table: list[list[int]]) -> Reference:
    """The reference fit of ``table``, rows by the ``x`` level."""
    (n00, n01), (n10, n11) = table
    n = n00 + n01 + n10 + n11
    none = Reference(None, None, None)
    # An empty cell needs |rho| = 1 to fit exactly: no interior fit.
    if 0 in (n00, n01, n10, n11):
        return none
    h, k = float(ndtri((n00 + n01) / n)), float(ndtri((n00 + n10) / n))
    target = n00 / n

    def gap(r: float) -> float:
        return float(bvn_cdf(h, k, r)) - target

    if gap(-LATENT_BOUND) * gap(LATENT_BOUND) >= 0:
        return none
    rho = brentq(gap, -LATENT_BOUND, LATENT_BOUND, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)
    p00 = float(bvn_cdf(h, k, rho))
    p = np.array([p00, float(ndtr(h)) - p00, float(ndtr(k)) - p00, 1 - float(ndtr(h)) - float(ndtr(k)) + p00])
    s = math.sqrt((1 - rho) * (1 + rho))
    phi_h, phi_k = (math.exp(-v * v / 2) / math.sqrt(2 * math.pi) for v in (h, k))
    # dp00 / d tau_x, d tau_y, d rho.
    d_h, d_k, d_r = phi_h * float(ndtr((k - rho * h) / s)), phi_k * float(ndtr((h - rho * k) / s)), _phi2(h, k, rho)
    # Rows: cells 00, 01 (Phi(h) - p00), 10 (Phi(k) - p00), 11 (the rest).
    jac = np.array([[d_h, d_k, d_r], [phi_h - d_h, -d_k, -d_r], [-d_h, phi_k - d_k, -d_r], [0.0, 0.0, 0.0]])
    jac[3] = -(jac[0] + jac[1] + jac[2])
    info = n * jac.T @ (jac / p[:, None])
    two_step = math.sqrt(np.linalg.inv(info)[2, 2])
    known = 1 / math.sqrt(n * d_r * d_r * float(np.sum(1 / p)))
    return Reference(float(rho), two_step, known)
