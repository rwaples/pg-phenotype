"""Random traits on the random pedigrees, and the oracle's view of a proband."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pg_phenotype import Trait, pafgrs


@dataclass(frozen=True)
class Case:
    """A binary trait's statuses, ages and CIP table, as PA-FGRS takes them."""

    affected: object
    age: object
    cip: pafgrs.Cip

    @property
    def trait(self) -> Trait:
        return Trait(self.affected, kind="binary")


def random_trait(n: int, seed: int, *, prevalence: float = 0.15) -> Case:
    """Cases, censored controls, missing statuses, and controls without age."""
    rng = np.random.default_rng(seed)
    ages = np.linspace(0.0, 80.0, 41)
    cip = prevalence * (1.0 - np.exp(-((ages / 45.0) ** 2.5)))
    cip = cip / cip[-1] * prevalence
    affected = (rng.random(n) < 0.25).astype(np.float64)
    age = rng.uniform(0.0, 95.0, n)
    affected[rng.random(n) < 0.15] = np.nan
    age[(affected == 0) & (rng.random(n) < 0.1)] = np.nan
    age[(affected == 1) & (rng.random(n) < 0.1)] = np.nan
    return Case(affected=affected, age=age, cip=pafgrs.Cip(ages, cip))


def observe(trait: Case) -> tuple[np.ndarray, np.ndarray, float]:
    """Per row ``(affected, w)`` and the threshold, written from the contract."""
    affected = trait.affected == 1
    k = trait.cip.prevalence
    cip_at = np.interp(np.nan_to_num(trait.age, nan=0.0), trait.cip.ages, trait.cip.cip, left=0.0, right=k)
    w = np.where(affected, 1.0, np.clip(cip_at / k, 0.0, 1.0))
    w[np.isnan(trait.affected)] = 0.0
    w[(trait.affected == 0) & np.isnan(trait.age)] = 0.0
    return affected, w, trait.cip.threshold


def proband_view(prep, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative rows, kinship to the proband, and the dense relative kinship."""
    rows, kin_p = prep.relatives(i)
    d = len(rows)
    kin = np.zeros((d, d))
    j, k = np.triu_indices(d, 1)
    kin[j, k] = prep.triangle(i)
    kin[k, j] = kin[j, k]
    return rows, kin_p.astype(np.float64), kin
