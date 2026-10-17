"""A NumPy reference for the Relative-Pair Correlation's cluster sandwich SE (ADR 0013).

Independent of the Rust store and sweep: each entry (one complete
orientation of a pair) gets its influence on r̂ from the closed forms here,
divided by the fit's entries, and its pair's MRCA from :func:`top_keys`, a
vectorised walk up the pedigree.  :func:`meats` then sums the influences
over each cluster design; ``mrca_person`` is the SE's
``Σ_m Φ_m² + Σ_i S_i² − Σ_(i,m) S_(i,m)²``.  Entries rather than pairs make
a symmetric cross-trait pair complete in one orientation count once, as the
point estimate counts it; a pair with both orientations is the pair-level
formula.

:func:`cross_meat` joins two estimates' entries over the same three keys,
the covariance of their estimates (ADR 0014); :func:`brute_cross` is its
definition, every two entries that share an MRCA or a member.
:func:`kin_meat` is the Within-Person cell's meat, the kinship kernel over
its rows' relative pairs of degree at most 2 (:data:`KERNEL`), and
:func:`within_person_psi` its estimates' influences per row.

``tools/relative_pairs_se_calibration.py`` imports these too.
"""

from __future__ import annotations

import itertools
import math
from typing import NamedTuple

import numpy as np
from pedigree_graph import RELATIONSHIPS
from scipy.special import ndtr, ndtri

#: The Within-Person kernel's categories, those of degree at most 2: the WP prototype's ``kin2``.
KERNEL = tuple(code for code, category in RELATIONSHIPS.items() if category.degree <= 2)


def degenerate(entries: int) -> float:
    """The fraction of its scale within which a meat over *entries* entries is rounding (``degenerate_meat``).

    Rust's ``degenerate``: ``max(2^10, 4 √entries)`` double epsilons, in the same float operations.
    """
    return max(4.0 * math.sqrt(entries), 1024.0) * float(np.finfo(np.float64).eps)


def pearson_if(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray] | None:
    """r̂ and each entry's influence.  On a same-trait symmetric cell's doubled entries the two sides'
    means and spreads coincide, so this is the intraclass Pearson."""
    if len(x) == 0:
        return None
    mx, my = x.mean(), y.mean()
    sx, sy = math.sqrt(((x - mx) ** 2).mean()), math.sqrt(((y - my) ** 2).mean())
    if sx == 0 or sy == 0:
        return None
    u, v = (x - mx) / sx, (y - my) / sy
    r = float((u * v).mean())
    return r, u * v - r / 2 * (u * u + v * v)


def bvn_pdf(h: float, k: float, rho: float) -> float:
    q = 1 - rho * rho
    return math.exp(-(h * h - 2 * rho * h * k + k * k) / (2 * q)) / (2 * math.pi * math.sqrt(q))


def tetrachoric_if(table: np.ndarray, rho: float) -> np.ndarray:
    """Influence of one entry in cell (i, j) of a directed 2 x 2 two-step tetrachoric at ρ̂.

    dρ = (dp11 - Qa dpa - Qb dpb) / φ2(h, k; ρ), with Qa = P(Y > k | X = h).
    """
    n = table.sum()
    p11, pa, pb = table[1, 1] / n, table[1].sum() / n, table[:, 1].sum() / n
    h, k = -float(ndtri(pa)), -float(ndtri(pb))
    s = math.sqrt(1 - rho * rho)
    qa, qb = float(ndtr(-(k - rho * h) / s)), float(ndtr(-(h - rho * k) / s))
    phi2 = bvn_pdf(h, k, rho)
    i, j = np.meshgrid([0.0, 1.0], [0.0, 1.0], indexing="ij")
    return ((i * j - p11) - qa * (i - pa) - qb * (j - pb)) / phi2


def _group_sq(keys: np.ndarray, w: np.ndarray) -> float:
    """Σ over distinct keys of (Σ w at that key)², by sort (NumPy's hash unique is slow on int64)."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    sums = np.add.reduceat(w[order], starts)
    return float(sums @ sums)


def _group_sums(keys: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each distinct key, ascending, and the sum of w at it, by sort."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    return k[starts], np.add.reduceat(w[order], starts)


def _group_dot(keys_j: np.ndarray, w_j: np.ndarray, keys_k: np.ndarray, w_k: np.ndarray) -> float:
    """Σ over the keys both hold of (Σ w_j at it)(Σ w_k at it), by sort."""
    if len(keys_j) == 0 or len(keys_k) == 0:
        return 0.0
    (kj, sj), (kk, sk) = _group_sums(keys_j, w_j), _group_sums(keys_k, w_k)
    _, at_j, at_k = np.intersect1d(kj, kk, assume_unique=True, return_indices=True)
    return float(sj[at_j] @ sk[at_k])


class Units(NamedTuple):
    """One estimate's entries: each one's influence (already over the entries), its members (both of a
    relative pair's, as two arrays; a within-person entry's one), and its pair's MRCA (``None`` within
    person, an entry its own cluster)."""

    psi: np.ndarray
    members: tuple[np.ndarray, ...]
    mrca: np.ndarray | None


def cross_meat(j: Units, k: Units) -> float:
    """Cov of two estimates over their entries: ``Σ_m Φ_jm Φ_km + Σ_i S_ji S_ki − Σ_(i,m) S_j(i,m) S_k(i,m)``.

    The SE's ``mrca_person`` meat (:func:`meats`) when *j* is *k*: an entry of each adds ψ_e ψ_f once when
    they share an MRCA or a member.  A within-person estimate has no MRCA, so it joins on members alone.
    """

    def by_member(u: Units, key) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate([key(i) for i in u.members]), np.tile(u.psi, len(u.members))

    person = _group_dot(*by_member(j, lambda i: i), *by_member(k, lambda i: i))
    if j.mrca is None or k.mrca is None:
        return person
    stride = int(max(j.mrca.max(initial=0), k.mrca.max(initial=0))) + 1
    cluster = _group_dot(j.mrca, j.psi, k.mrca, k.psi)
    group = _group_dot(*by_member(j, lambda i: i * stride + j.mrca), *by_member(k, lambda i: i * stride + k.mrca))
    return cluster + person - group


def brute_cross(j: Units, k: Units) -> float:
    """:func:`cross_meat` by its definition: Σ ψ_e ψ_f over every entry e of *j* and f of *k* that share an
    MRCA or a member, each two once."""
    total = 0.0
    for e in range(len(j.psi)):
        hit = np.zeros(len(k.psi), dtype=bool)
        if j.mrca is not None and k.mrca is not None:
            hit |= k.mrca == j.mrca[e]
        for i in j.members:
            for other in k.members:
                hit |= other == i[e]
        total += j.psi[e] * float(k.psi[hit].sum())
    return total


def meats(phi: np.ndarray, a: np.ndarray, b: np.ndarray, g: np.ndarray, m: np.ndarray) -> dict[str, float]:
    """Each design's meat for influences φ of units (a, b), pairs or entries, with Mate Network g and MRCA m.

    ``pair`` treats each unit as its own cluster; the designs that join a
    shared member or key do not depend on whether a pair's two entries are
    one unit or two.  ``mn_person_scale`` and ``mrca_person_scale`` add up
    their meat's three sums of squares had each unit the influence
    ``1/len(phi)``: the scale of the meat's rounding (``degenerate_meat``).
    """
    unit = np.full(len(phi), 1 / len(phi))
    twice_unit = np.concatenate([unit, unit])
    pair = float(phi @ phi)
    twice = np.concatenate([phi, phi])
    person = _group_sq(np.concatenate([a, b]), twice) - pair
    out = {"pair": pair, "person": person}
    for name, key in (("mn", g), ("mrca", m)):
        stride = int(key.max()) + 1
        by_person = _group_sq(np.concatenate([a * stride + key, b * stride + key]), twice) - pair
        out[name] = _group_sq(key, phi)
        out[f"{name}_person"] = out[name] + person - by_person
        out[f"{name}_person_scale"] = (
            _group_sq(key, unit)
            + _group_sq(np.concatenate([a, b]), twice_unit)
            + _group_sq(np.concatenate([a * stride + key, b * stride + key]), twice_unit)
        )
    return out


def parent_indices(ped: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mother and father per row as indices: a row's parent its row, an external parent (an id with no
    row) ``n + its rank`` among the external ids, ``-1`` when missing; external parents get rows of
    their own with no parents, so the walks in :func:`top_keys` stop there."""
    ids = np.asarray(ped["id"], dtype=np.int64)
    row = {int(i): r for r, i in enumerate(ids)}
    parents = [np.asarray(ped[k], dtype=np.int64) for k in ("mother", "father")]
    external = sorted({int(p) for col in parents for p in col if p >= 0 and int(p) not in row})
    rank = {p: len(ids) + i for i, p in enumerate(external)}
    pad = np.full(len(external), -1, dtype=np.int64)
    return tuple(
        np.concatenate([np.array([row.get(int(p), rank.get(int(p), -1)) for p in col], dtype=np.int64), pad])
        for col in parents
    )


def top_keys(
    code: str, a: np.ndarray, b: np.ndarray, mother: np.ndarray, father: np.ndarray, net: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each pair's top-mating Mate Network and MRCA; x is a's ancestor up-1 meioses up the path."""
    c = RELATIONSHIPS[code]
    parents = (mother, father)

    def ancestors(start: np.ndarray, depth: int) -> list[np.ndarray]:
        out = []
        for path in itertools.product((0, 1), repeat=depth):
            x = start.copy()
            for step in path:
                x = np.where(x >= 0, parents[step][np.maximum(x, 0)], -1)
            out.append(x)
        return out

    if c.up == 0:
        x = a
        mrca = np.where(mother[x] >= 0, mother[x], father[x])
    else:
        x = np.full(len(a), -1)
        mrca = np.full(len(a), -1)
        mine = ancestors(a, c.up - 1)
        theirs = [b] if c.down == 0 else ancestors(b, c.down - 1)
        for xa in mine:
            ok = xa >= 0
            ma, fa = np.where(ok, mother[np.maximum(xa, 0)], -1), np.where(ok, father[np.maximum(xa, 0)], -1)
            for y in theirs:
                if c.down == 0:
                    hit, shared = ok & ((ma == y) | (fa == y)), y
                else:
                    yy = np.maximum(y, 0)
                    same_m = ok & (y >= 0) & (y != xa) & (ma >= 0) & (ma == mother[yy])
                    same_f = ok & (y >= 0) & (y != xa) & (fa >= 0) & (fa == father[yy])
                    hit, shared = same_m | same_f, np.where(same_m, ma, fa)
                new = (x < 0) & hit
                x, mrca = np.where(new, xa, x), np.where(new, shared, mrca)
        if (x < 0).any():
            raise RuntimeError(f"{code}: {int((x < 0).sum())} pairs have no top mating")
    top = np.where(mother[x] >= 0, mother[x], father[x])
    # Founder MZ twins have no parents: such a pair is its own cluster under both keys.
    lone = top < 0
    return np.where(lone, net.max() + 1 + a, net[top]), np.where(lone, len(mother) + a, mrca)


def kin_meat(psi: np.ndarray, on: np.ndarray, pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, int]:
    """The Within-Person meat ``Σ_i ψ_i² + 2 Σ_(a<b) ψ_a ψ_b``, its scale and its entries.

    *psi* is each row's influence over the entries, the rows *on*, 0 off them; *pairs* each kernel
    category's relative pairs (rows a, b), each listed once.  The scale is the meat had each of the n
    entries the influence 1/n: ``(n + 2P) / n²`` over the P pairs whose rows are both entries.
    """
    n = int(on.sum())
    meat = float(psi @ psi) + 2 * sum(float(psi[a] @ psi[b]) for a, b in pairs)
    both = sum(int((on[a] & on[b]).sum()) for a, b in pairs)
    return meat, (n + 2 * both) / n**2, n


def within_person_psi(
    x: np.ndarray, y: np.ndarray, kind: str, stratum: np.ndarray | None, tetrachoric
) -> dict[tuple[object, str], tuple[float, np.ndarray, np.ndarray] | None]:
    """Each Within-Person estimate, its influence per row over its entries (0 off them) and its rows;
    ``None`` at a tetrachoric bound.

    Keyed ``(scope, estimator)``: scope ``"pooled"`` (every row both traits show), each known stratum label
    (its rows), and with continuous traits ``"stratified"`` (the rows of known strata, each trait
    standardised within its stratum: ψ = z₁z₂ − r_s/2 (z₁² + z₂²) + r_s − r).  *tetrachoric* maps a 2 x 2
    table to its fit's ``(value, boundary)``.
    """
    n = len(x)
    both = ~np.isnan(x) & ~np.isnan(y)
    scopes: list[tuple[object, np.ndarray]] = [("pooled", both)]
    if stratum is not None:
        scopes += [(float(s), both & (stratum == s)) for s in np.unique(stratum[~np.isnan(stratum)])]
    out: dict[tuple[object, str], tuple[float, np.ndarray, np.ndarray] | None] = {}
    for scope, rows in scopes:
        a, b = x[rows], y[rows]
        fits: dict[str, tuple[float, np.ndarray] | None] = {}
        if kind == "continuous":
            fits["pearson"] = pearson_if(a, b)
        else:
            ai, bi = a.astype(np.int64), b.astype(np.int64)
            t = np.zeros((2, 2), dtype=np.int64)
            np.add.at(t, (ai, bi), 1)
            value, boundary = tetrachoric(t)
            defined = value is not None and not boundary
            fits["tetrachoric"] = (value, tetrachoric_if(t.astype(float), value)[ai, bi]) if defined else None
            fits["phi"] = pearson_if(a, b)
        for est, fit in fits.items():
            if fit is None:
                out[scope, est] = None
                continue
            psi = np.zeros(n)
            psi[rows] = fit[1] / rows.sum()
            out[scope, est] = (fit[0], psi, rows)
    if kind == "continuous" and stratum is not None:
        rows = both & ~np.isnan(stratum)
        z = [np.zeros(n), np.zeros(n)]
        r_s = np.zeros(n)
        for s in np.unique(stratum[rows]):
            at = rows & (stratum == s)
            for side, v in enumerate((x, y)):
                z[side][at] = (v[at] - v[at].mean()) / v[at].std()
            r_s[at] = (z[0][at] @ z[1][at]) / at.sum()
        r = float(z[0][rows] @ z[1][rows]) / rows.sum()
        psi = np.zeros(n)
        psi[rows] = (z[0] * z[1] - r_s / 2 * (z[0] ** 2 + z[1] ** 2) + r_s - r)[rows] / rows.sum()
        out["stratified", "pearson"] = (r, psi, rows)
    return out
