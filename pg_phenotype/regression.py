"""Parent-offspring regression: each child's trait on its midparent's, its mother's or its father's.

:func:`parent_offspring` fits three ordinary least-squares lines over the
pedigree's children, pooled and within strata of the child.  The slope's SE
is the cluster-robust sandwich over Mate Networks (a parent in no Mating Pair
is a network of its own), with a Wald p-value and interval, beside the plain
OLS SE.  Slopes are reported raw: under the usual assumptions (no environment
shared by parent and child, random mating) the midparent slope estimates h2
and a single-parent slope h2/2, on the observed scale for a 0/1 trait.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from pg_phenotype import _native
from pg_phenotype._input import strata
from pg_phenotype._memory import release_free_memory
from pg_phenotype._pedigree import native_pedigree
from pg_phenotype._threads import thread_budget
from pg_phenotype._trait import Trait

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["Dropped", "Fit", "ParentOffspring", "Subset", "parent_offspring"]


@dataclass(frozen=True)
class Dropped:
    """Pedigree rows a regression leaves out, each under the first reason that applies."""

    parent_not_in_pedigree: int
    child_value_missing: int
    parent_value_missing: int


@dataclass(frozen=True)
class Fit:
    """One regression line over one subset of children.

    ``slope`` and ``intercept`` are ``None`` exactly when ``reason`` says why
    (``no_complete_children``, ``constant_parent_value``).  ``r``, ``se``,
    ``se_ols``, ``p`` and ``ci`` are each ``None`` with a
    ``*_unavailable_reason``: the line's reason, or ``constant_child_value``
    (r), ``too_few_children`` (``se_ols``, under three children),
    ``single_mate_network`` or ``sandwich_undefined`` (``se``), and
    ``zero_se`` (``p``, ``ci``) for a perfect fit.  ``se`` is the Mate
    Network sandwich SE of the slope with a G/(G-1) factor; ``p`` and ``ci``
    are its two-sided Wald test and interval at ``ci_level``.
    ``n + sum(dropped)`` is the number of rows in the subset.
    """

    n: int
    n_mate_networks: int
    dropped: Dropped
    slope: float | None
    reason: str | None
    intercept: float | None
    r: float | None
    r_unavailable_reason: str | None
    r2: float | None
    se: float | None
    se_unavailable_reason: str | None
    se_ols: float | None
    se_ols_unavailable_reason: str | None
    p: float | None
    p_unavailable_reason: str | None
    ci: tuple[float, float] | None
    ci_unavailable_reason: str | None


@dataclass(frozen=True)
class Subset:
    """The three regressions of one set of children."""

    midparent: Fit
    mother: Fit
    father: Fit


@dataclass(frozen=True)
class ParentOffspring:
    """The result of :func:`parent_offspring`.

    Attributes:
        pooled: Every child, whatever its stratum.
        strata: Each known stratum label of any row, ascending, to its
            children's regressions; ``None`` without *stratum*.
        n_unknown_stratum: Rows whose stratum is unknown, which are in
            ``pooled`` only.
        settings: ``ci_level``.
        method: How the SEs, the p-value and the clusters are computed.
        metadata: ``pg_phenotype_version`` and ``pedigree_graph_core_rev``.
    """

    pooled: Subset
    strata: Mapping[int, Subset] | None
    n_unknown_stratum: int
    settings: Mapping[str, object]
    method: Mapping[str, object]
    metadata: Mapping[str, object]


def _fit(raw: dict) -> Fit:
    ci = raw.pop("ci")
    return Fit(dropped=Dropped(**raw.pop("dropped")), ci=None if ci is None else tuple(ci), **raw)


def _subset(raw: dict) -> Subset:
    return Subset(_fit(raw["midparent"]), _fit(raw["mother"]), _fit(raw["father"]))


def parent_offspring(pedigree: object, trait: Trait, *, stratum: object = None) -> ParentOffspring:
    """The midparent-, mother- and father-offspring regressions of *trait*.

    Each child (a pedigree row) is regressed on the mean of its parents'
    values, on its mother's, and on its father's.  A regression uses the
    children whose parents it needs are pedigree rows and whose own and parent
    values are observed; the other parent may be missing for a single-parent
    regression.

    Args:
        pedigree: A :class:`~pg_phenotype.Pedigree` or its columns ``id``,
            ``mother``, ``father`` (``-1`` or NA when missing) and optional
            ``twin``, ``sex``, as a polars or pandas frame or a mapping of
            arrays, checked for this call alone.
        trait: A continuous, binary or ordinal :class:`~pg_phenotype.Trait`
            on the pedigree's rows; codes are regressed as numbers.  An
            all-missing or constant trait gives fits with a reason.
        stratum: One integer label per row (a generation, a sex x generation
            code), NA where unknown, read at the child.  Each known label gets
            its own regressions; children of unknown stratum are pooled only.

    Returns:
        A :class:`ParentOffspring`.

    Raises:
        ValidationError: The pedigree, the trait (``trait_length_mismatch``,
            ``unsupported_trait_kind`` for a categorical trait,
            ``invalid_trait_value``) or the strata
            (``stratum_length_mismatch``) is invalid.
        TypeError: *trait* is not a :class:`~pg_phenotype.Trait`.
    """
    from pg_phenotype import __version__

    if not isinstance(trait, Trait):
        raise TypeError(f"trait must be a pg_phenotype.Trait, got {type(trait).__name__}")
    native = native_pedigree(pedigree)
    labels, known = (None, None) if stratum is None else strata(stratum)
    release_free_memory()
    raw = _native.parent_offspring(
        native,
        np.ascontiguousarray(trait.values),
        trait.kind,
        None if trait.levels is None else len(trait.levels),
        labels,
        known,
        threads=thread_budget(),
    )
    by_stratum = raw["strata"]
    return ParentOffspring(
        pooled=_subset(raw["pooled"]),
        strata=None if by_stratum is None else MappingProxyType({s["stratum"]: _subset(s) for s in by_stratum}),
        n_unknown_stratum=raw["n_unknown_stratum"],
        settings=MappingProxyType(dict(raw["settings"])),
        method=MappingProxyType(dict(raw["method"])),
        metadata=MappingProxyType(
            {"pg_phenotype_version": __version__, "pedigree_graph_core_rev": _native.pg_core_rev()}
        ),
    )
