"""Relative-Pair Correlation: one trait's correlation between relatives, per relationship category.

:func:`correlation` finds every Relative Pair of the requested Relationship
Categories (pedigree-graph's MZ, MO, FO, FS, MHS, ..., 1C) and Category
Groups (such as PO for MO and FO) and correlates the trait between the two
members.  A continuous trait gets Pearson's r; a binary trait gets the
tetrachoric correlation (primary) and phi.

A symmetric category's pairs enter in both orders, so its Pearson is
intraclass (one mean and variance) and its tetrachoric fits one threshold;
the result does not depend on row order.  A directed category's pairs enter
once, in its roles' order (offspring then parent).  Pairs are reduced to
exact sums as pedigree-graph-core finds them and never stored.  There is no
SE yet: relative pairs share people, and the cluster design waits on a
calibration study (ADR 0009).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from pg_phenotype import _native
from pg_phenotype._memory import release_free_memory
from pg_phenotype._pedigree import native_pedigree
from pg_phenotype._threads import thread_budget
from pg_phenotype._trait import Trait
from pg_phenotype.assortative import _count

__all__ = ["Cell", "Estimate", "RelativePairCorrelation", "correlation"]

#: The default bound on the engine's accumulators (1 GiB).
DEFAULT_MEMORY_BUDGET_BYTES = 2**30


@dataclass(frozen=True)
class Estimate:
    """One estimator's value in a cell.

    ``value`` is ``None`` exactly when ``reason`` says why:
    ``no_complete_pairs`` (no pair has both values), ``constant_margin`` (a
    binary side shows one level) or ``constant_value`` (a continuous side is
    constant at the quantization step).  ``boundary`` is the tetrachoric
    fit's bound flag, ``None`` for other estimators.
    """

    estimator: str
    primary: bool
    value: float | None
    reason: str | None
    boundary: bool | None


@dataclass(frozen=True)
class Cell:
    """The correlation of one category or Category Group.

    Attributes:
        category: The category's code, or the group's name.
        members: The pedigree-graph codes it covers.
        symmetric: Whether its pairs enter in both orders.
        roles: A directed cell's roles, first then second (a group reads
            mother and father as parent); ``None`` when symmetric.
        n: Unordered pairs with both values observed.
        n_dropped: Unordered pairs with either value missing.
        concordance: A binary symmetric cell's pairs as (both 0, discordant,
            both 1); ``None`` otherwise.
        table: A binary directed cell's 2 x 2 counts, rows by the first
            role's value; ``None`` otherwise.
        crude: Each estimator's value, the primary first.
    """

    category: str
    members: tuple[str, ...]
    symmetric: bool
    roles: tuple[str, str] | None
    n: int
    n_dropped: int
    concordance: tuple[int, int, int] | None
    table: tuple[tuple[int, int], tuple[int, int]] | None
    crude: tuple[Estimate, ...]

    @property
    def primary(self) -> Estimate:
        """The primary estimate: Pearson for a continuous trait, tetrachoric for a binary one."""
        return self.crude[0]

    def __getitem__(self, estimator: str) -> Estimate:
        """The estimate of *estimator* (``pearson``, ``tetrachoric``, ``phi``)."""
        for estimate in self.crude:
            if estimate.estimator == estimator:
                return estimate
        raise KeyError(estimator)


@dataclass(frozen=True)
class RelativePairCorrelation:
    """The result of :func:`correlation`.

    Attributes:
        cells: One per requested entry, in request order.
        settings: ``categories`` (the entries' names), ``trait_kind``,
            ``memory_budget_bytes``, ``threads``, and a continuous trait's
            ``quantization_exponent`` e and ``quantization_step`` 2^-e.
        method: How pairs enter, the estimators, the quantization, and the
            absence of an SE.
        metadata: ``pg_phenotype_version`` and ``pedigree_graph_core_rev``.
    """

    cells: tuple[Cell, ...]
    settings: Mapping[str, object]
    method: Mapping[str, object]
    metadata: Mapping[str, object]

    def cell(self, category: str) -> Cell:
        """The cell of *category*, a code or a group's name."""
        for cell in self.cells:
            if cell.category == category:
                return cell
        raise KeyError(category)


def _codes(name: str, members: object) -> list[str]:
    if isinstance(members, str) or not isinstance(members, Sequence):
        raise TypeError(f"category group {name!r} must be a sequence of codes, got {members!r}")
    codes = list(members)
    for code in codes:
        if not isinstance(code, str):
            raise TypeError(f"category group {name!r} holds {code!r}, which is not a code")
    return codes


def _entries(categories: object) -> list[tuple[str, list[str] | None]] | None:
    if categories is None:
        return None
    if isinstance(categories, str | Mapping) or not isinstance(categories, Sequence):
        raise TypeError(f"categories must be a sequence of codes and group mappings, got {categories!r}")
    entries: list[tuple[str, list[str] | None]] = []
    for item in categories:
        if isinstance(item, str):
            entries.append((item, None))
        elif isinstance(item, Mapping):
            for name, members in item.items():
                if not isinstance(name, str):
                    raise TypeError(f"a category group's name must be a str, got {name!r}")
                entries.append((name, _codes(name, members)))
        else:
            raise TypeError(f"a categories entry must be a code or a group mapping, got {item!r}")
    return entries


def _estimate(raw: dict) -> Estimate:
    return Estimate(**raw)


def _cell(raw: dict) -> Cell:
    roles, concordance, table = raw["roles"], raw["concordance"], raw["table"]
    return Cell(
        category=raw["category"],
        members=tuple(raw["members"]),
        symmetric=raw["symmetric"],
        roles=None if roles is None else (roles[0], roles[1]),
        n=raw["n"],
        n_dropped=raw["n_dropped"],
        concordance=None if concordance is None else (concordance[0], concordance[1], concordance[2]),
        table=None if table is None else ((table[0][0], table[0][1]), (table[1][0], table[1][1])),
        crude=tuple(_estimate(e) for e in raw["crude"]),
    )


def correlation(
    pedigree: object,
    trait: Trait,
    *,
    categories: Sequence[str | Mapping[str, Sequence[str]]] | None = None,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> RelativePairCorrelation:
    """The Relative-Pair Correlation of *trait* in each requested category.

    Args:
        pedigree: A :class:`~pg_phenotype.Pedigree` or its columns ``id``,
            ``mother``, ``father`` (``-1`` or NA when missing) and optional
            ``twin``, ``sex``, as a polars or pandas frame or a mapping of
            arrays, checked for this call alone.
        trait: A continuous or binary :class:`~pg_phenotype.Trait` on the
            pedigree's rows.  An all-missing or constant trait gives cells
            with a reason.
        categories: The cells, in order: pedigree-graph codes (``"FS"``,
            ``"MO"``, ``"1C"``, ...) and mappings of a group's name to its
            codes (``{"PO": ["MO", "FO"]}``).  A group's codes are all
            symmetric, or all directed with the same roles (mother and father
            read as parent); no code is in two entries.  ``None`` gives every
            category up to degree 3, one cell each.  A category with no pairs
            still gets a cell, with ``n = 0``.
        memory_budget_bytes: The bound on the engine's accumulators.

    Returns:
        A :class:`RelativePairCorrelation`.

    Raises:
        ValidationError: The pedigree, the trait (``trait_length_mismatch``,
            ``unsupported_trait_kind`` for an ordinal or categorical trait,
            ``invalid_trait_value``), the request (``unknown_category``,
            ``duplicate_category``, ``mixed_category_group``,
            ``duplicate_entry_name``, ``empty_category_group``), or a
            continuous trait whose range is too wide for its spread to keep
            2^-20 of its interquartile range (``insufficient_precision``).
        ParameterError: *memory_budget_bytes* is below 1
            (``parameter_out_of_range``).
        ResourceError: The accumulators do not fit the budget
            (``memory_budget_exceeded``).
        TypeError: *trait* is not a :class:`~pg_phenotype.Trait`, or an
            argument has the wrong type.
    """
    from pg_phenotype import __version__

    if not isinstance(trait, Trait):
        raise TypeError(f"trait must be a pg_phenotype.Trait, got {type(trait).__name__}")
    entries = _entries(categories)
    budget = _count("memory_budget_bytes", memory_budget_bytes)
    native = native_pedigree(pedigree)
    release_free_memory()
    raw = _native.relative_pairs_correlation(
        native,
        np.ascontiguousarray(trait.values),
        trait.kind,
        None if trait.levels is None else len(trait.levels),
        entries,
        budget,
        threads=thread_budget(),
    )
    return RelativePairCorrelation(
        cells=tuple(_cell(c) for c in raw["cells"]),
        settings=MappingProxyType(dict(raw["settings"])),
        method=MappingProxyType(dict(raw["method"])),
        metadata=MappingProxyType(
            {"pg_phenotype_version": __version__, "pedigree_graph_core_rev": _native.pg_core_rev()}
        ),
    )
