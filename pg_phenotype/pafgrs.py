"""PA-FGRS: Pearson-Aitken family genetic risk scores.

``prepare`` builds a pedigree's relatives once; ``score_univariate`` and
``score_bivariate`` score every proband per trait and parameter variant.
A trait must be binary (1 affected, 0 unaffected); its ages and CIP table
are passed beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pg_phenotype import _native
from pg_phenotype._input import floats, pedigree_arrays
from pg_phenotype._threads import thread_budget

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pg_phenotype._trait import Trait

__all__ = ["BivariateScores", "Cip", "UnivariateScores", "prepare", "score_bivariate", "score_univariate"]


def prepare(pedigree: object, *, ndegree: int = 2, probands: Sequence[int] | np.ndarray | None = None) -> _native.Prep:
    """Validate *pedigree* and build its relatives up to *ndegree*.

    Call once per pedigree and degree, then score each trait and parameter
    variant against the result.  A relative of a proband is a row whose
    closest relationship category is at most *ndegree* and whose exact
    kinship to the proband is at least ``0.5**(ndegree + 1) - 1e-6``.

    Args:
        pedigree: Columns ``id``, ``mother``, ``father`` (``-1`` or NA when
            missing) and optional ``twin``, ``sex`` (``0`` female, ``1`` male,
            ``-1`` unknown), as a polars or pandas frame or a mapping of arrays.
        ndegree: The deepest relationship degree that counts, 1 to 5.
        probands: Ids to score; ``None`` scores every row.

    Returns:
        An opaque in-memory handle.

    Raises:
        ValidationError: The pedigree, *ndegree*, or *probands* is invalid.
        ResourceError: A capacity or allocation limit was reached.
    """
    cols = pedigree_arrays(pedigree)
    proband_ids = None if probands is None else np.ascontiguousarray(probands, dtype=np.int64)
    return _native.prepare(
        cols.id,
        cols.mother,
        cols.father,
        cols.twin,
        cols.sex,
        ndegree=ndegree,
        probands=proband_ids,
        threads=thread_budget(),
    )


@dataclass(frozen=True)
class Cip:
    """Cumulative incidence proportion by age.

    ``ages`` strictly increasing and finite; ``cip`` non-decreasing in
    ``[0, 1)`` with a positive last value, the lifetime prevalence ``K``.
    CIP at an age is linear interpolation: 0 below the first age, ``K`` at
    and above the last.

    Raises:
        ParameterError: ``invalid_cip`` naming the first offending position.
    """

    ages: np.ndarray
    cip: np.ndarray
    prevalence: float = field(init=False)
    threshold: float = field(init=False)

    def __post_init__(self) -> None:
        ages = np.ascontiguousarray(self.ages, dtype=np.float64)
        cip = np.ascontiguousarray(self.cip, dtype=np.float64)
        prevalence, threshold = _native.check_cip(ages.tolist(), cip.tolist())
        object.__setattr__(self, "ages", ages)
        object.__setattr__(self, "cip", cip)
        object.__setattr__(self, "prevalence", prevalence)
        object.__setattr__(self, "threshold", threshold)


@dataclass(frozen=True)
class UnivariateScores:
    """One record per proband, in pedigree input-row order."""

    id: np.ndarray
    est: np.ndarray
    var: np.ndarray
    n_relatives: np.ndarray
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, np.ndarray]:
        """The per-proband columns."""
        return {"id": self.id, "est": self.est, "var": self.var, "n_relatives": self.n_relatives}


@dataclass(frozen=True)
class BivariateScores:
    """One record per proband, in pedigree input-row order."""

    id: np.ndarray
    est1: np.ndarray
    est2: np.ndarray
    var1: np.ndarray
    var2: np.ndarray
    cov12: np.ndarray
    n_relatives: np.ndarray
    n_obs1: np.ndarray
    n_obs2: np.ndarray
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, np.ndarray]:
        """The per-proband columns."""
        names = ("id", "est1", "est2", "var1", "var2", "cov12", "n_relatives", "n_obs1", "n_obs2")
        return {name: getattr(self, name) for name in names}


def _metadata(prep: _native.Prep, n_probands: int) -> dict[str, object]:
    from pg_phenotype import __version__

    return {
        "n_probands": n_probands,
        "ndegree": prep.ndegree,
        "pg_phenotype_version": __version__,
        "pedigree_graph_core_rev": _native.pg_core_rev(),
    }


def score_univariate(prep: _native.Prep, trait: Trait, *, age: object, cip: Cip, h2: float) -> UnivariateScores:
    """Score every proband of *prep* on one binary trait.

    Args:
        prep: From :func:`prepare`.
        trait: A binary :class:`~pg_phenotype.Trait`: 1 affected, 0
            unaffected, missing unknown (``w = 0``).
        age: Per pedigree row, the age at onset for a case and at last
            observation for a control; NA where unknown.  A control with no
            age is unobserved and counted; a case needs none.
        cip: The trait's CIP table.
        h2: Liability-scale heritability, in ``(0, 1]``.

    Returns:
        Posterior mean and variance of each proband's genetic liability.  A
        proband with no informative relative gets ``est = 0``, ``var = h2``.

    Raises:
        ParameterError: *h2* out of range (checked before any work).
        ValidationError: A trait that is not binary, a length that is not
            the pedigree's, a value outside {0, 1}, or a negative or
            infinite age.
    """
    raw = _native.score_univariate(
        prep,
        trait.values,
        trait.kind,
        floats(age, "age"),
        cip.ages.tolist(),
        cip.cip.tolist(),
        h2=float(h2),
        threads=thread_budget(),
    )
    metadata = _metadata(prep, len(raw["id"]))
    metadata |= {
        "h2": float(h2),
        "prevalence": cip.prevalence,
        "threshold": raw["threshold"],
        "controls_without_age": raw["controls_without_age"],
    }
    return UnivariateScores(
        id=raw["id"], est=raw["est"], var=raw["var"], n_relatives=raw["n_relatives"], metadata=metadata
    )


def score_bivariate(
    prep: _native.Prep,
    traits: Sequence[Trait],
    *,
    age: Sequence[object],
    cip: Sequence[Cip],
    h2: Sequence[float],
    rg: float,
    rho_within: float | None = None,
) -> BivariateScores:
    """Score every proband of *prep* on two binary traits jointly.

    Args:
        prep: From :func:`prepare`.
        traits: The two binary traits.
        age: Each trait's ages, as :func:`score_univariate` takes them.
        cip: Each trait's CIP table.
        h2: The two liability-scale heritabilities, each in ``(0, 1]``.
        rg: Genetic correlation, in ``[-1, 1]``.
        rho_within: A person's cross-trait liability correlation; defaults
            to ``rg * sqrt(h2_1 * h2_2)``.

    Returns:
        The joint posterior per proband.  With no informative relative:
        ``est = 0``, ``var = h2``, ``cov12 = rg * sqrt(h2_1 * h2_2)``.

    Raises:
        ParameterError: A parameter out of range, or a *rho_within* that
            leaves the non-genetic covariance indefinite.
        ValidationError: As :func:`score_univariate`, per trait.
    """
    if not len(traits) == len(age) == len(cip) == len(h2) == 2:
        raise ValueError("score_bivariate takes exactly two traits, ages, CIP tables and h2 values")
    t1, t2 = traits
    raw = _native.score_bivariate(
        prep,
        t1.values,
        t1.kind,
        floats(age[0], "age1"),
        t2.values,
        t2.kind,
        floats(age[1], "age2"),
        (cip[0].ages.tolist(), cip[0].cip.tolist()),
        (cip[1].ages.tolist(), cip[1].cip.tolist()),
        h2=(float(h2[0]), float(h2[1])),
        rg=float(rg),
        rho_within=None if rho_within is None else float(rho_within),
        threads=thread_budget(),
    )
    metadata = _metadata(prep, len(raw["id"]))
    metadata |= {
        "h2": (float(h2[0]), float(h2[1])),
        "rg": float(rg),
        "rho_within": raw["rho_within"],
        "prevalence": (cip[0].prevalence, cip[1].prevalence),
        "threshold": tuple(raw["threshold"]),
        "controls_without_age": tuple(raw["controls_without_age"]),
    }
    columns = ("id", "est1", "est2", "var1", "var2", "cov12", "n_relatives", "n_obs1", "n_obs2")
    return BivariateScores(**{c: raw[c] for c in columns}, metadata=metadata)
