"""Assortative mating: the Mate Correlation of one or two traits over a pedigree's Mating Pairs.

A Mating Pair is a mother and father with a child in the pedigree; a Mate
Network is a connected set of pairs through shared mates.  Each cell (mother
trait x father trait) gets crude estimators on its pairs and, with strata,
the primary estimator standardised within each sex x stratum.  Inference is a
Mate Network cluster-robust sandwich SE, an optional one-step Mate Network
bootstrap, and a father-permutation test of each primary estimate;
``inference=False`` skips it all for the point estimates and counts.  The
method follows pedsum #13; :attr:`MateCorrelation.method` describes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from pg_phenotype import _native
from pg_phenotype._input import count, strata
from pg_phenotype._pedigree import native_pedigree
from pg_phenotype._trait import Trait, checked

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "Cell",
    "Draws",
    "Dropped",
    "EstimatorResult",
    "MateCorrelation",
    "Permutation",
    "Sample",
    "Stratified",
    "WithinPerson",
    "mate_correlation",
]


@dataclass(frozen=True)
class Draws:
    """Draws requested, how many gave a value, and why the others did not (counts by reason)."""

    requested: int
    valid: int
    failed: int
    failure_reasons: Mapping[str, int]


@dataclass(frozen=True)
class Permutation:
    """The permutation test of a primary estimate: Besag & Clifford's closed sequential scheme.

    ``draws`` counts the draws read (``draws_used``); ``p`` is ``None`` with
    ``p_unavailable_reason`` set (``not_requested``,
    ``no_informative_permutations``, ``no_valid_permutations``).
    """

    statistic: str
    p: float | None
    p_unavailable_reason: str | None
    draws: Draws
    seed: int
    n_fixed_fathers: int
    stopped_early: bool
    draws_used: int
    sequential_h: int


@dataclass(frozen=True)
class EstimatorResult:
    """One estimator of one cell.

    ``value`` is ``None`` exactly when ``reason`` says why there is no estimate.
    ``boundary`` is set for a latent correlation only.  ``se``, ``ci`` and the
    permutation's ``p`` are each ``None`` with a ``*_unavailable_reason``;
    without inference ``se`` and ``ci`` are ``not_requested`` and
    ``permutation`` is ``None``.
    """

    estimator: str
    primary: bool
    value: float | None
    reason: str | None
    boundary: bool | None
    se: float | None
    se_unavailable_reason: str | None
    ci: tuple[float, float] | None
    ci_method: str | None
    ci_unavailable_reason: str | None
    bootstrap: Draws | None
    permutation: Permutation | None


@dataclass(frozen=True)
class Stratified:
    """The stratified form of a cell's primary estimator, with the strata its pairs span."""

    result: EstimatorResult
    n_strata_mothers: int
    n_strata_fathers: int


@dataclass(frozen=True)
class Dropped:
    """Mating Pairs a cell leaves out, by why."""

    mother_missing: int
    father_missing: int
    both_missing: int
    small_stratum: int
    degenerate_stratum: int


@dataclass(frozen=True)
class Cell:
    """The mother's trait ``mother_trait`` against the father's ``father_trait`` (indices into the traits).

    ``n_mate_networks`` and ``largest_mate_network_share`` are ``None``
    without inference; the share is also ``None`` when the cell has no pairs.
    """

    mother_trait: int
    father_trait: int
    n: int
    n_dropped: Dropped
    n_mate_networks: int | None
    largest_mate_network_share: float | None
    table: tuple[tuple[int, int], tuple[int, int]] | None
    crude: tuple[EstimatorResult, ...]
    stratified: Stratified | None

    @property
    def primary(self) -> EstimatorResult:
        """The crude primary estimator (the first)."""
        return self.crude[0]

    def __getitem__(self, estimator: str) -> EstimatorResult:
        """The crude estimator named ``estimator``."""
        for r in self.crude:
            if r.estimator == estimator:
                return r
        raise KeyError(estimator)


@dataclass(frozen=True)
class Sample:
    """The Mating Pairs before any cell's filtering.

    The Mate Network counts are ``None`` without inference.
    """

    n_total: int
    n_dropped_unknown_stratum: int
    n_mate_networks: int | None
    largest_mate_network_share: float | None
    n_mothers_multiple_mates: int
    n_fathers_multiple_mates: int


@dataclass(frozen=True)
class WithinPerson:
    """The Within-Person Cross-Trait Correlation of one sex, once per distinct person."""

    estimator: str
    n: int
    value: float | None
    boundary: bool | None
    reason: str | None


@dataclass(frozen=True)
class MateCorrelation:
    """The result of :func:`mate_correlation`.

    Attributes:
        sample: Counts over the Mating Pairs.
        cells: Mother trait 0 x father trait 0, then (0, 1), (1, 0), (1, 1).
        within_person: ``{"mothers": ..., "fathers": ...}`` with two traits
            and inference.
        settings: The settings used, the thread count, and the CI level.
        method: The method description (SE, CI scale, bootstrap and
            permutation design).
        metadata: ``pg_phenotype_version`` and ``pedigree_graph_core_rev``.
    """

    sample: Sample
    cells: tuple[Cell, ...]
    within_person: Mapping[str, WithinPerson] | None
    settings: Mapping[str, object]
    method: Mapping[str, object]
    metadata: Mapping[str, object]

    def cell(self, mother_trait: int, father_trait: int) -> Cell:
        """The cell of the mothers' trait ``mother_trait`` and the fathers' ``father_trait``."""
        for c in self.cells:
            if (c.mother_trait, c.father_trait) == (mother_trait, father_trait):
                return c
        raise KeyError((mother_trait, father_trait))


def _draws(raw: dict | None) -> Draws | None:
    if raw is None:
        return None
    return Draws(raw["requested"], raw["valid"], raw["failed"], MappingProxyType(dict(raw["failure_reasons"])))


def _permutation(raw: dict | None) -> Permutation | None:
    if raw is None:
        return None
    draws = _draws(raw["draws"])
    assert draws is not None
    return Permutation(
        statistic=raw["statistic"],
        p=raw["p"],
        p_unavailable_reason=raw["p_unavailable_reason"],
        draws=draws,
        seed=raw["seed"],
        n_fixed_fathers=raw["n_fixed_fathers"],
        stopped_early=raw["stopped_early"],
        draws_used=raw["draws_used"],
        sequential_h=raw["sequential_h"],
    )


def _result(raw: dict) -> EstimatorResult:
    defined = raw["reason"] is None
    return EstimatorResult(
        estimator=raw["estimator"],
        primary=raw["primary"],
        value=raw.get("value"),
        reason=raw["reason"],
        boundary=raw.get("boundary"),
        se=raw.get("se"),
        se_unavailable_reason=raw.get("se_unavailable_reason"),
        ci=raw.get("ci"),
        ci_method=raw.get("ci_method"),
        ci_unavailable_reason=raw.get("ci_unavailable_reason"),
        bootstrap=_draws(raw["bootstrap"]) if defined else None,
        permutation=_permutation(raw["permutation"]) if defined else None,
    )


def _cell(raw: dict) -> Cell:
    stratified = raw["stratified"]
    table = raw["table"]
    return Cell(
        mother_trait=raw["mother_trait"],
        father_trait=raw["father_trait"],
        n=raw["n"],
        n_dropped=Dropped(**raw["n_dropped"]),
        n_mate_networks=raw["n_mate_networks"],
        largest_mate_network_share=raw["largest_mate_network_share"],
        table=None if table is None else (tuple(table[0]), tuple(table[1])),
        crude=tuple(_result(r) for r in raw["crude"]),
        stratified=None
        if stratified is None
        else Stratified(_result(stratified), stratified["n_strata_mothers"], stratified["n_strata_fathers"]),
    )


def mate_correlation(
    pedigree: object,
    traits: Trait | Sequence[Trait],
    *,
    stratum: object = None,
    permutations: int | None = None,
    bootstrap: int = 0,
    seed: int = 0,
    min_stratum_networks: int = 10,
    spearman: bool = False,
    inference: bool = True,
) -> MateCorrelation:
    """The Mate Correlation of one or two traits over *pedigree*'s Mating Pairs.

    A Mating Pair is a distinct mother and father, both pedigree rows, of at
    least one child.  With two traits there are four cells (mother trait x
    father trait) and the Within-Person Cross-Trait Correlation per sex.

    Args:
        pedigree: A :class:`~pg_phenotype.Pedigree`, which keeps its Mating
            Pairs, and the Mate Networks a call with inference builds, for
            later calls, or its columns ``id``, ``mother``, ``father``
            (``-1`` or NA when missing) and optional ``twin``, ``sex``, as a
            polars or pandas frame or a mapping of arrays.  Columns are read
            first and checked by pedigree-graph's rules after the counts, for
            this call alone.
        traits: One or two :class:`~pg_phenotype.Trait` (continuous, binary
            or ordinal).  Binary and ordinal values are level codes
            ``0..k-1``; every level is taken by some row.
        stratum: One integer label per row (a Depth, a birth-year bin), NA
            where unknown.  Pairs with a mate of unknown stratum are dropped;
            each cell then drops its pairs in any sex x stratum spanning
            fewer than *min_stratum_networks* Mate Networks or constant there,
            and adds the stratified primary estimator.
        permutations: Father permutations per primary estimate; 0 turns them
            off.  ``None`` is 999 with inference and 0 without.
        bootstrap: Mate Network bootstrap draws; 0 gives each estimate a Wald
            CI from its sandwich SE.
        seed: Keys every permutation and bootstrap draw (int64).
        min_stratum_networks: The thin-stratum rule, used only with *stratum*.
        spearman: Add Spearman to the crude estimators of a continuous x
            continuous cell.  Off by default: it ranks every pair, which costs
            more than the rest of such a cell.
        inference: ``False`` returns the point estimates and the pair counts
            alone: no Mate Networks (their counts are ``None``), no SE or CI
            (``not_requested``), no permutation record, and no
            Within-Person Cross-Trait Correlation (the default computes
            it).  Cell estimates and counts are the default's.  It takes no
            draws: *permutations* and *bootstrap* must be 0.  A
            :class:`~pg_phenotype.Pedigree` builds no Mate Networks for such
            a call.

    Returns:
        A :class:`MateCorrelation`.

    Raises:
        ValidationError: The pedigree, a trait (``trait_count``,
            ``trait_length_mismatch``, ``unsupported_trait_kind``,
            ``invalid_trait_value``, ``all_missing_trait``, ``constant_trait``,
            ``sparse_ordinal_codes``, ``unused_level``) or the strata
            (``stratum_length_mismatch``) is invalid.
        ParameterError: A negative draw count, a nonzero one without
            inference, *min_stratum_networks* below 1, or an integer outside
            int64.
    """
    native = native_pedigree(pedigree)
    if isinstance(traits, Trait):
        trait_list = [traits]
    else:
        try:
            given = list(traits)
        except TypeError:
            name = type(traits).__name__
            raise TypeError(f"traits must be a pg_phenotype.Trait or a sequence of them, got {name}") from None
        trait_list = [checked(t, f"traits[{i}]") for i, t in enumerate(given)]
    stratum_arrays = None if stratum is None else strata(stratum)
    for name, flag in (("spearman", spearman), ("inference", inference)):
        if not isinstance(flag, (bool, np.bool_)):
            raise TypeError(f"{name} must be a bool, got {flag!r}")
    raw = _native.mate_correlation(
        native,
        trait_list,
        stratum_arrays,
        permutations=None if permutations is None else count("permutations", permutations),
        bootstrap=count("bootstrap", bootstrap),
        seed=count("seed", seed),
        min_stratum_networks=count("min_stratum_networks", min_stratum_networks),
        spearman=bool(spearman),
        inference=bool(inference),
    )
    s = raw["sample"]
    within = raw["within_person"]
    return MateCorrelation(
        sample=Sample(**s),
        cells=tuple(_cell(c) for c in raw["cells"]),
        within_person=None
        if within is None
        else MappingProxyType({sex: WithinPerson(**w) for sex, w in within.items()}),
        settings=MappingProxyType(dict(raw["settings"])),
        method=MappingProxyType(dict(raw["method"])),
        metadata=MappingProxyType(raw["metadata"]),
    )
