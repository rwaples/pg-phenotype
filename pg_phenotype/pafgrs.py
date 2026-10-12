"""PA-FGRS: Pearson-Aitken family genetic risk scores.

``prepare`` builds a pedigree's relatives once; ``score_univariate`` and
``score_bivariate`` score every proband per trait and parameter variant.
A trait must be binary (1 affected, 0 unaffected); its ages and CIP table
are passed beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np

from pg_phenotype import _native
from pg_phenotype._input import floats, id_array
from pg_phenotype._pedigree import native_pedigree
from pg_phenotype._trait import checked

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pg_phenotype._trait import Trait

__all__ = ["BivariateScores", "Cip", "UnivariateScores", "prepare", "score_bivariate", "score_univariate"]


def prepare(pedigree: object, *, ndegree: int = 2, probands: Sequence[int] | np.ndarray | None = None) -> _native.Prep:
    """Build *pedigree*'s relatives up to *ndegree*.

    Call once per pedigree and degree, then score each trait and parameter
    variant against the result; scoring many traits needs only the result.
    A relative of a proband is a row whose closest relationship category is
    at most *ndegree* and whose exact kinship to the proband is at least
    ``0.5**(ndegree + 1) - 1e-6``.

    Args:
        pedigree: A :class:`~pg_phenotype.Pedigree`, or its columns ``id``,
            ``mother``, ``father`` (``-1`` or NA when missing) and optional
            ``twin``, ``sex`` (``0`` female, ``1`` male, ``-1`` unknown), as a
            polars or pandas frame or a mapping of arrays.  Columns are read
            first and checked by pedigree-graph's rules after *ndegree*, for
            this call alone.  A Pedigree helps when preparing
            again (another *ndegree* or set of probands) or running other
            methods on the same pedigree.
        ndegree: The deepest relationship degree that counts, 1 to 5.
        probands: Ids to score; ``None`` scores every row.

    Returns:
        An opaque in-memory handle.  It does not hold *pedigree*.

    Raises:
        ValidationError: The pedigree, *ndegree*, or *probands* is invalid.
        ResourceError: A capacity or allocation limit was reached.
    """
    native = native_pedigree(pedigree)
    proband_ids = None if probands is None else id_array(probands, "probands")
    return _native.prepare(native, ndegree=ndegree, probands=proband_ids)


@dataclass(frozen=True)
class Cip:
    """Cumulative incidence proportion by age.

    ``ages`` strictly increasing and finite; ``cip`` non-decreasing in
    ``[0, 1)`` with a positive last value, the lifetime prevalence ``K``.
    CIP at an age is 0 below the first age and ``K`` at and above the last.
    Between ages, ``interpolation`` ``"linear"`` draws a straight line
    (a smooth table) and ``"step"`` holds the value at the last age at or
    below (an estimator's step function).

    Raises:
        ParameterError: ``invalid_cip`` naming the first offending position.
        ValueError: An ``interpolation`` other than ``"linear"`` or ``"step"``.
    """

    ages: np.ndarray
    cip: np.ndarray
    interpolation: Literal["linear", "step"] = "linear"
    prevalence: float = field(init=False)
    threshold: float = field(init=False)

    def __post_init__(self) -> None:
        ages = np.ascontiguousarray(self.ages, dtype=np.float64)
        cip = np.ascontiguousarray(self.cip, dtype=np.float64)
        prevalence, threshold = _native.check_cip((ages.tolist(), cip.tolist(), self.interpolation))
        object.__setattr__(self, "ages", ages)
        object.__setattr__(self, "cip", cip)
        object.__setattr__(self, "prevalence", prevalence)
        object.__setattr__(self, "threshold", threshold)

    def _native(self) -> tuple[list[float], list[float], str]:
        return self.ages.tolist(), self.cip.tolist(), self.interpolation

    @classmethod
    def from_aalen_johansen(cls, age: object, status: object, *, entry: object = None) -> Cip:
        """Estimate a CIP from ages and survival status by Aalen-Johansen.

        Death competes with the trait: a person who dies can no longer
        become a case.  The estimate is a step function, rising at each age
        where a case occurs; the result is a ``"step"`` table holding it at
        those ages.

        Args:
            age: Per person, the age at onset for a case, at death, or at
                last observation; finite and ``>= 0``.
            status: Per person, ``0`` censored, ``1`` case, ``2`` died
                without becoming a case.
            entry: Per person, the age they enter observation (late
                entry), before *age*; a censored person may leave at entry
                and is then never at risk.  A person is at risk at age
                ``t`` when ``entry < t <= age``.  ``None``: everyone is at
                risk from birth, ``t <= age``.

        Returns:
            A ``"step"`` :class:`Cip`, its last value ``K`` the estimated
            lifetime prevalence.

        Raises:
            ValidationError: ``pair_length_mismatch`` when *status* or
                *entry* is not *age*'s length.  Then person by person, in
                this order: ``invalid_age`` for an age that is missing,
                negative or infinite, ``invalid_status``, ``invalid_age``
                for such an entry, and ``entry_not_before_age``.  Last,
                ``cip_not_estimable`` with
                ``reason`` ``zero_prevalence`` (no case while anyone was at
                risk) or ``prevalence_one`` (everyone at risk became a
                case).
        """
        entry_values = None if entry is None else floats(entry, "entry")
        ages, cip = _native.cip_aalen_johansen(floats(age, "age"), floats(status, "status"), entry_values)
        return cls(np.asarray(ages), np.asarray(cip), interpolation="step")


@dataclass(frozen=True)
class UnivariateScores:
    """One record per proband, in pedigree input-row order."""

    id: np.ndarray
    est: np.ndarray
    var: np.ndarray
    n_relatives: np.ndarray
    metadata: Mapping[str, object]

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
    metadata: Mapping[str, object]

    def to_dict(self) -> dict[str, np.ndarray]:
        """The per-proband columns."""
        names = ("id", "est1", "est2", "var1", "var2", "cov12", "n_relatives", "n_obs1", "n_obs2")
        return {name: getattr(self, name) for name in names}


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
        ValidationError: In order: a trait whose length is not the
            pedigree's, a trait that is not binary, a value outside {0, 1},
            an age whose length is not the pedigree's, or a negative or
            infinite age.
    """
    raw = _native.score_univariate(prep, checked(trait), floats(age, "age"), cip._native(), h2=float(h2))
    return UnivariateScores(
        id=raw["id"],
        est=raw["est"],
        var=raw["var"],
        n_relatives=raw["n_relatives"],
        metadata=MappingProxyType(raw["metadata"]),
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
    raw = _native.score_bivariate(
        prep,
        (checked(traits[0], "traits[0]"), checked(traits[1], "traits[1]")),
        (floats(age[0], "age1"), floats(age[1], "age2")),
        (cip[0]._native(), cip[1]._native()),
        h2=(float(h2[0]), float(h2[1])),
        rg=float(rg),
        rho_within=None if rho_within is None else float(rho_within),
    )
    metadata = raw["metadata"]
    metadata["controls_without_age"] = tuple(metadata["controls_without_age"])
    columns = ("id", "est1", "est2", "var1", "var2", "cov12", "n_relatives", "n_obs1", "n_obs2")
    return BivariateScores(**{c: raw[c] for c in columns}, metadata=MappingProxyType(metadata))
