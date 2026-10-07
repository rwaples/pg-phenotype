"""Rust scores against the NumPy reference cores, same covariance and order (#11a)."""

import numpy as np
import pytest

from pg_phenotype import pafgrs
from tests.oracle import reference
from tests.pedigrees import random_pedigree
from tests.scoring_fixtures import observe, proband_view, random_trait

TOL = 1e-10


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ndegree", [1, 2, 3])
def test_univariate_matches_reference(seed, ndegree):
    ped = random_pedigree(seed, generations=5)
    n = len(ped["id"])
    prep = pafgrs.prepare(ped, ndegree=ndegree)
    trait = random_trait(n, seed)
    h2 = 0.2 + 0.15 * seed
    got = pafgrs.score_univariate(prep, trait.trait, age=trait.age, cip=trait.cip, h2=h2)
    affected, w, thr = observe(trait)
    assert got.metadata["threshold"] == pytest.approx(thr, abs=1e-15)
    for i in range(n):
        rows, kin_p, kin = proband_view(prep, i)
        valid = w[rows] > 0
        assert got.n_relatives[i] == valid.sum()
        rows, kin_p, kin = rows[valid], kin_p[valid], kin[np.ix_(valid, valid)]
        d = len(rows)
        cov = np.empty((d + 1, d + 1))
        cov[0, 0] = h2
        cov[0, 1:] = cov[1:, 0] = 2.0 * kin_p * h2
        cov[1:, 1:] = 2.0 * kin * h2
        np.fill_diagonal(cov[1:, 1:], 1.0)
        t1 = np.where(affected[rows], thr, -np.inf)
        t2 = np.where(affected[rows], np.inf, thr)
        order = reference.canonical_univariate_order(w[rows], kin_p, kin, rows)
        est, var = reference.pa_fgrs_core(t1, t2, w[rows], cov, order)
        assert got.est[i] == pytest.approx(est, abs=TOL), i
        assert got.var[i] == pytest.approx(var, abs=TOL), i


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("ndegree", [1, 2, 3])
def test_bivariate_matches_reference(seed, ndegree):
    ped = random_pedigree(seed, generations=5)
    n = len(ped["id"])
    prep = pafgrs.prepare(ped, ndegree=ndegree)
    traits = (random_trait(n, 100 + seed, prevalence=0.1), random_trait(n, 200 + seed, prevalence=0.2))
    h2 = (0.3 + 0.1 * seed, 0.6)
    rg = [-0.4, 0.0, 0.5, 0.9][seed]
    rho = None if seed % 2 else rg * np.sqrt(h2[0] * h2[1]) + 0.05
    got = pafgrs.score_bivariate(
        prep,
        tuple(t.trait for t in traits),
        age=tuple(t.age for t in traits),
        cip=tuple(t.cip for t in traits),
        h2=h2,
        rg=rg,
        rho_within=rho,
    )
    cov_g = rg * np.sqrt(h2[0] * h2[1])
    rho_within = got.metadata["rho_within"]
    assert rho_within == (cov_g if rho is None else rho)
    observed = [observe(t) for t in traits]
    g = np.array([[h2[0], cov_g], [cov_g, h2[1]]])
    for i in range(n):
        rows, kin_p, kin = proband_view(prep, i)
        obs = [(x, t) for x in range(len(rows)) for t in (0, 1) if observed[t][1][rows[x]] > 0]
        people = sorted({x for x, _ in obs})
        assert got.n_relatives[i] == len(people)
        assert got.n_obs1[i] == sum(t == 0 for _, t in obs)
        assert got.n_obs2[i] == sum(t == 1 for _, t in obs)
        m = len(obs)
        person = np.array([x for x, _ in obs], dtype=np.int64)
        trait_of = np.array([t for _, t in obs], dtype=np.int64)
        cov = np.zeros((m + 2, m + 2))
        cov[:2, :2] = g
        for a, (x, t) in enumerate(obs):
            cov[:2, 2 + a] = cov[2 + a, :2] = 2.0 * kin_p[x] * g[:, t]
            for b, (y, u) in enumerate(obs):
                if a == b:
                    cov[2 + a, 2 + b] = 1.0
                elif x == y:
                    cov[2 + a, 2 + b] = rho_within
                else:
                    cov[2 + a, 2 + b] = 2.0 * kin[x, y] * g[t, u]
        w = np.array([observed[t][1][rows[x]] for x, t in obs])
        aff = np.array([observed[t][0][rows[x]] for x, t in obs], dtype=bool)
        thr = np.array([observed[t][2] for _, t in obs])
        lower = np.where(aff, thr, -np.inf)
        upper = np.where(aff, np.inf, thr)
        kin_obs = kin[np.ix_(person, person)] if m else np.zeros((0, 0))
        order = reference.canonical_bivariate_order(
            w, kin_p[person], kin_obs, rows[person], trait_of, person, h2, cov_g, rho_within
        )
        want = reference.pa_fgrs_bivariate_core(lower, upper, w, cov, order)
        have = (got.est1[i], got.est2[i], got.var1[i], got.var2[i], got.cov12[i])
        assert have == pytest.approx(want, abs=TOL), i
