"""Relative-Pair Correlation: one or two traits' correlation between relatives, per relationship category.

:func:`correlation` finds every Relative Pair of the requested Relationship
Categories (pedigree-graph's MZ, MO, FO, FS, MHS, ..., 1C) and Category
Groups (such as PO for MO and FO) and correlates the trait between the two
members.  A continuous trait gets Pearson's r; a binary trait gets the
tetrachoric correlation (primary) and phi.

A symmetric category's pairs enter in both orders, so its Pearson is
intraclass (one mean and variance) and its tetrachoric fits one threshold;
the result does not depend on row order.  A directed category's pairs enter
once, in its roles' order (offspring then parent).

With two traits of one kind, each entry gets a cell per trait combination:
trait ``first_trait`` of the first member against ``second_trait`` of the
second.  A symmetric category's cross-trait cell (0, 1) reads each pair in
both orientations, trait 0 of one member against trait 1 of the other, with
a threshold (or mean and spread) per trait; a directed category's (0, 1) and
(1, 0) read its roles apart.  The result's ``within_person`` is the
Within-Person Cross-Trait Correlation: each row's trait 0 against its trait 1.  Pairs are reduced to
exact sums as pedigree-graph-core finds them and never stored.  There is no
SE yet: relative pairs share people, and the cluster design waits on a
calibration study (ADR 0009).

With ``stratum=`` (a generation, a birth cohort), each cell also gets one
:class:`StratumCell` per known stratum and a :class:`Stratified` fit.  A
directed pair is in its first role's stratum (the offspring's, the
descendant's), whatever the second member's; a symmetric pair is in a
stratum only when both members are, so a pair across two strata is counted
in ``n_cross_stratum`` and stays in the pooled cell alone.

:func:`ace` solves a result's cells for the additive genetic (A), common
environment (C) and unique environment (E) shares, and with two traits their
cross-trait correlations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from pg_phenotype import _native
from pg_phenotype._input import count, strata
from pg_phenotype._pedigree import native_pedigree
from pg_phenotype._trait import as_traits

if TYPE_CHECKING:
    from pg_phenotype._trait import Trait

__all__ = [
    "AceFit",
    "AceRow",
    "Cell",
    "CrossShares",
    "Estimate",
    "RelativePairCorrelation",
    "Stratified",
    "StratumCell",
    "TraitShares",
    "ace",
    "correlation",
]

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


class _Crude:
    """By-name access to the ``crude`` estimates of a cell or a stratum's cell."""

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
class StratumCell(_Crude):
    """One known stratum's pairs of a cell, with the cell's pooled fields.

    A directed cell's stratum holds the pairs whose first member (offspring,
    descendant) is in it, the second member's stratum known or not; a
    symmetric cell's holds the pairs whose members are both in it.

    Attributes:
        n: Unordered pairs with a complete orientation.
        n_dropped: The other unordered pairs.
        n_entries: A symmetric cross-trait cell's complete orientations;
            ``None`` otherwise.
        concordance: A binary symmetric same-trait cell's pairs as (both 0,
            discordant, both 1); ``None`` otherwise.
        table: Any other binary cell's 2 x 2 counts, as the cell's;
            ``None`` otherwise.
        crude: Each estimator's value, the primary first.
    """

    n: int
    n_dropped: int
    n_entries: int | None
    concordance: tuple[int, int, int] | None
    table: tuple[tuple[int, int], tuple[int, int]] | None
    crude: tuple[Estimate, ...]


@dataclass(frozen=True)
class Stratified:
    """The primary estimator fitted over every pair of two known strata at once.

    Rows are the first member's stratum and columns the second's (both
    orders for a symmetric cell, so a stratum's members share one threshold
    or one mean and SD).  A binary side gets a threshold per stratum, a
    continuous side is standardised within each stratum, and one correlation
    is fitted.  A pair across two strata enters, though it is in no
    :class:`StratumCell`; a pair with either stratum unknown does not.  A
    stratum whose side is constant over the pairs kept carries no
    information, so its pairs are left out, until no stratum's side is
    constant.

    Attributes:
        n: Unordered pairs with a complete orientation fitted.
        n_entries: A symmetric cross-trait cell's complete orientations
            fitted; ``None`` otherwise.
        n_degenerate_stratum: Unordered pairs with a complete orientation
            and both strata known that a constant stratum left out.
        estimate: The fit; ``reason`` is ``no_complete_pairs`` when no pair
            has both strata known, ``constant_margin`` or ``constant_value``
            when every stratum was left out.
    """

    n: int
    n_entries: int | None
    n_degenerate_stratum: int
    estimate: Estimate


@dataclass(frozen=True)
class Cell(_Crude):
    """The correlation of one category or Category Group, for one trait combination.

    The result's ``within_person`` is a Cell too: category
    ``"within_person"``, no members or roles, each row paired with itself.

    Attributes:
        category: The category's code, or the group's name.
        members: The pedigree-graph codes it covers.
        symmetric: Whether its pairs enter in both orders.
        roles: A directed cell's roles, first then second (a group reads
            mother and father as parent); ``None`` when symmetric.
        first_trait: The trait (0-based) read in each pair's first member,
            a directed cell's first role.
        second_trait: The trait read in the second member.
        n: Unordered pairs with a complete orientation: trait
            ``first_trait`` of one member and ``second_trait`` of the other
            both observed.
        n_dropped: The category's other unordered pairs.
        n_entries: A symmetric cross-trait cell's complete orientations,
            between ``n`` and ``2 n``: the entries its estimates read.
            ``None`` otherwise.
        concordance: A binary symmetric same-trait cell's pairs as (both 0,
            discordant, both 1); ``None`` otherwise.
        table: Any other binary cell's 2 x 2 counts, rows by trait
            ``first_trait`` of one member (a directed cell's first role) and
            columns by ``second_trait`` of the other, a symmetric cell's two
            orientations summed; ``None`` otherwise.
        crude: Each estimator's value, the primary first.
        strata: Each known stratum label of any row, ascending, to its
            :class:`StratumCell` (``n = 0`` when it has no pairs); ``None``
            without *stratum*.
        n_cross_stratum: Symmetric pairs with a complete orientation whose
            members are in two known strata, in no stratum's cell.
        n_unknown_stratum: Pairs with a complete orientation in no stratum's
            cell for an unknown stratum: a directed pair's first member's,
            either member's of a symmetric pair.  ``n`` is the strata's ``n``
            plus ``n_cross_stratum`` plus ``n_unknown_stratum``.
        stratified: The fit over the strata; ``None`` without *stratum*.
    """

    category: str
    members: tuple[str, ...]
    symmetric: bool
    roles: tuple[str, str] | None
    first_trait: int
    second_trait: int
    n: int
    n_dropped: int
    n_entries: int | None
    concordance: tuple[int, int, int] | None
    table: tuple[tuple[int, int], tuple[int, int]] | None
    crude: tuple[Estimate, ...]
    strata: Mapping[int, StratumCell] | None
    n_cross_stratum: int
    n_unknown_stratum: int
    stratified: Stratified | None


@dataclass(frozen=True)
class RelativePairCorrelation:
    """The result of :func:`correlation`.

    Attributes:
        cells: Each requested entry's cells, in request order: one with one
            trait; with two, a symmetric entry's (0, 0), (1, 1), (0, 1) and
            a directed entry's (0, 0), (0, 1), (1, 0), (1, 1), as
            ``(first_trait, second_trait)``.
        within_person: With two traits, the Within-Person Cross-Trait
            Correlation (a :class:`Cell` over the rows, each in its own
            stratum); ``None`` with one.
        settings: ``categories`` (the entries' names), ``trait_kind``,
            ``memory_budget_bytes``, ``threads``, and continuous traits'
            ``quantization_exponents`` e, one per trait, and
            ``quantization_steps`` 2^-e (``None`` for binary traits).
        method: How pairs and traits enter, the estimators, the
            quantization, the strata, and the absence of an SE.
        metadata: ``pg_phenotype_version`` and ``pedigree_graph_core_rev``.
    """

    cells: tuple[Cell, ...]
    within_person: Cell | None
    settings: Mapping[str, object]
    method: Mapping[str, object]
    metadata: Mapping[str, object]

    def cell(self, category: str, *, first_trait: int = 0, second_trait: int = 0) -> Cell:
        """The cell of *category*, a code or a group's name, reading *first_trait* against *second_trait*."""
        for cell in self.cells:
            if (cell.category, cell.first_trait, cell.second_trait) == (category, first_trait, second_trait):
                return cell
        raise KeyError((category, first_trait, second_trait))


def _codes(name: str, members: object) -> list[str]:
    if isinstance(members, str) or not isinstance(members, Sequence):
        raise TypeError(f"category group {name!r} must be a sequence of codes, got {members!r}")
    codes = list(members)
    for code in codes:
        if not isinstance(code, str):
            raise TypeError(f"category group {name!r} holds {code!r}, which is not a code")
    return codes


def _entries(categories: object) -> list[tuple[int, str, list[str] | None]] | None:
    """Each entry as ``(position, name, members)``, *position* its index in *categories*.

    The groups of one mapping share its position, so an error names the
    caller's ``categories[i]``.
    """
    if categories is None:
        return None
    if isinstance(categories, str | Mapping) or not isinstance(categories, Sequence):
        raise TypeError(f"categories must be a sequence of codes and group mappings, got {categories!r}")
    entries: list[tuple[int, str, list[str] | None]] = []
    for position, item in enumerate(categories):
        if isinstance(item, str):
            entries.append((position, item, None))
        elif isinstance(item, Mapping):
            if not item:
                raise ValueError(f"categories[{position}] is an empty mapping: map each group's name to its codes")
            for name, members in item.items():
                if not isinstance(name, str):
                    raise TypeError(f"a category group's name must be a str, got {name!r}")
                entries.append((position, name, _codes(name, members)))
        else:
            raise TypeError(f"a categories entry must be a code or a group mapping, got {item!r}")
    return entries


def _stratum_cell(raw: dict) -> StratumCell:
    concordance, table = raw["concordance"], raw["table"]
    return StratumCell(
        n=raw["n"],
        n_dropped=raw["n_dropped"],
        n_entries=raw["n_entries"],
        concordance=None if concordance is None else (concordance[0], concordance[1], concordance[2]),
        table=None if table is None else ((table[0][0], table[0][1]), (table[1][0], table[1][1])),
        crude=tuple(Estimate(**e) for e in raw["crude"]),
    )


def _cell(raw: dict) -> Cell:
    roles, by_stratum, stratified = raw["roles"], raw["strata"], raw["stratified"]
    pooled = _stratum_cell(raw)
    return Cell(
        category=raw["category"],
        members=tuple(raw["members"]),
        symmetric=raw["symmetric"],
        roles=None if roles is None else (roles[0], roles[1]),
        first_trait=raw["first_trait"],
        second_trait=raw["second_trait"],
        n=pooled.n,
        n_dropped=pooled.n_dropped,
        n_entries=pooled.n_entries,
        concordance=pooled.concordance,
        table=pooled.table,
        crude=pooled.crude,
        strata=None if by_stratum is None else MappingProxyType({s["stratum"]: _stratum_cell(s) for s in by_stratum}),
        n_cross_stratum=raw["n_cross_stratum"],
        n_unknown_stratum=raw["n_unknown_stratum"],
        stratified=None
        if stratified is None
        else Stratified(
            n=stratified["n"],
            n_entries=stratified["n_entries"],
            n_degenerate_stratum=stratified["n_degenerate_stratum"],
            estimate=Estimate(**stratified["estimate"]),
        ),
    )


def correlation(
    pedigree: object,
    traits: Trait | Sequence[Trait],
    *,
    categories: Sequence[str | Mapping[str, Sequence[str]]] | None = None,
    stratum: object = None,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> RelativePairCorrelation:
    """The Relative-Pair Correlation of *traits* in each requested category.

    Args:
        pedigree: A :class:`~pg_phenotype.Pedigree` or its columns ``id``,
            ``mother``, ``father`` (``-1`` or NA when missing) and optional
            ``twin``, ``sex``, as a polars or pandas frame or a mapping of
            arrays, checked for this call alone.
        traits: A continuous or binary :class:`~pg_phenotype.Trait` on the
            pedigree's rows, or a sequence of one or two of one kind.  An
            all-missing or constant trait gives cells with a reason.
        categories: The cells, in order: pedigree-graph codes (``"FS"``,
            ``"MO"``, ``"1C"``, ...) and mappings of a group's name to its
            codes (``{"PO": ["MO", "FO"]}``).  A group's codes are all
            symmetric, or all directed with the same roles (mother and father
            read as parent); no code is in two entries, and no group is named
            like a code or ``"within_person"``.  Errors give positions in this sequence; the groups of
            one mapping share its position.  ``None`` gives every
            category up to degree 3.  A category with no pairs still gets its
            cells, with ``n = 0``.
        stratum: One integer label per row (a generation, a birth cohort),
            NA where unknown.  Each cell gets a :class:`StratumCell` per
            known label and a :class:`Stratified` fit; a directed pair is
            in its first role's stratum, a symmetric pair in a stratum only
            when both members are.  ``within_person`` puts each row in its
            own stratum.  The pooled fields do not change.
        memory_budget_bytes: The bound on the engine's accumulators and,
            with two traits, the within-person accumulator.

    Returns:
        A :class:`RelativePairCorrelation`.

    Raises:
        ValidationError: The pedigree, the traits (``trait_count`` for none
            or more than two; each one's ``trait_length_mismatch``,
            ``unsupported_trait_kind`` for an ordinal or categorical trait,
            ``invalid_trait_value``; then ``unsupported_trait_kind`` for two
            of different kinds), the strata
            (``stratum_length_mismatch``), the request (``unknown_category``,
            ``duplicate_category``, ``mixed_category_group``,
            ``duplicate_entry_name``, ``reserved_group_name``,
            ``empty_category_group``), or a
            continuous trait whose range is too wide for its spread to keep
            2^-20 of its interquartile range (``insufficient_precision``).
        ParameterError: *memory_budget_bytes* is below 1
            (``parameter_out_of_range``).
        ResourceError: The accumulators do not fit the budget
            (``memory_budget_exceeded``).
        TypeError: *traits* is not a :class:`~pg_phenotype.Trait` or a
            sequence of them, or an argument has the wrong type.
        ValueError: A group mapping in *categories* is empty.
    """
    trait_list = as_traits(traits)
    entries = _entries(categories)
    budget = count("memory_budget_bytes", memory_budget_bytes)
    native = native_pedigree(pedigree)
    by_stratum = None if stratum is None else strata(stratum)
    raw = _native.relative_pairs_correlation(native, trait_list, by_stratum, entries, budget)
    within = raw["within_person"]
    return RelativePairCorrelation(
        cells=tuple(_cell(c) for c in raw["cells"]),
        within_person=None if within is None else _cell(within),
        settings=MappingProxyType(dict(raw["settings"])),
        method=MappingProxyType(dict(raw["method"])),
        metadata=MappingProxyType(raw["metadata"]),
    )


@dataclass(frozen=True)
class AceRow:
    """One cell :func:`ace` considered, entered or not.

    Attributes:
        entry: The cell's category code, group name, or ``"within_person"``.
        first_trait: The trait (0-based) of the first member.
        second_trait: The trait of the second member.
        within_person: Whether it is the Within-Person Cross-Trait
            Correlation.
        kinship: The nominal kinship ``2^-(degree+1)`` of its categories;
            ``None`` within person.
        shared: Whether it is a Shared-Environment Category; ``None`` within
            person or when the call declared none.
        value: The correlation the fit reads; ``None`` when undefined.
        weight: What entered that correlation, each counted once: a
            symmetric cross-trait cell's complete orientations
            (``n_entries``), else its ``n`` (unordered pairs, or rows
            within person).
        used: Whether it entered the fit (its value is defined).
        reason: Why ``value`` is ``None``: the estimate's own reason,
            ``no_stratified_form``, or ``no_complete_pairs`` for a weight of 0.
    """

    entry: str
    first_trait: int
    second_trait: int
    within_person: bool
    kinship: float | None
    shared: bool | None
    value: float | None
    weight: int
    used: bool
    reason: str | None


@dataclass(frozen=True)
class TraitShares:
    """One trait's variance shares on the correlation scale.

    Each share is ``None`` exactly when its ``_reason`` says why:
    ``not_requested`` (a component left out of the fit, though its share is
    0 in ``E``) or ``not_identified`` (the cells that entered do not
    determine it).

    Attributes:
        trait: The trait (0-based).
        A: The additive genetic share.
        A_reason: Why ``A`` is ``None``.
        C: The common environment share.
        C_reason: Why ``C`` is ``None``.
        E: The remainder, ``1 - A - C``.
        E_reason: Why ``E`` is ``None``.
        rows: The entries of the rows that entered.
    """

    trait: int
    A: float | None
    A_reason: str | None
    C: float | None
    C_reason: str | None
    E: float | None
    E_reason: str | None
    rows: tuple[str, ...]


@dataclass(frozen=True)
class CrossShares:
    """Two traits' cross-trait covariance shares and correlations.

    A value is ``None`` exactly when its ``_reason`` says why:
    ``not_requested``, ``not_identified`` (``sE`` and ``rE`` without the
    within-person cell), or ``nonpositive_component`` (a correlation whose
    component is <= 0 in a trait).

    Attributes:
        sA: The additive genetic covariance.
        sA_reason: Why ``sA`` is ``None``.
        sC: The common environment covariance.
        sC_reason: Why ``sC`` is ``None``.
        sE: The unique environment covariance.
        sE_reason: Why ``sE`` is ``None``.
        rA: ``sA / sqrt(A_1 A_2)``, unconstrained.
        rA_reason: Why ``rA`` is ``None``.
        rA_out_of_range: Whether ``|rA| > 1``; ``None`` when ``rA`` is.
        rC: ``sC / sqrt(C_1 C_2)``.
        rC_reason: Why ``rC`` is ``None``.
        rC_out_of_range: Whether ``|rC| > 1``.
        rE: ``sE / sqrt(E_1 E_2)``.
        rE_reason: Why ``rE`` is ``None``.
        rE_out_of_range: Whether ``|rE| > 1``.
        rows: The entries of the rows that entered: cross-trait cells and
            ``"within_person"``.
    """

    sA: float | None
    sA_reason: str | None
    sC: float | None
    sC_reason: str | None
    sE: float | None
    sE_reason: str | None
    rA: float | None
    rA_reason: str | None
    rA_out_of_range: bool | None
    rC: float | None
    rC_reason: str | None
    rC_out_of_range: bool | None
    rE: float | None
    rE_reason: str | None
    rE_out_of_range: bool | None
    rows: tuple[str, ...]


@dataclass(frozen=True)
class AceFit:
    """The result of :func:`ace`.

    Attributes:
        traits: One :class:`TraitShares` per trait.
        cross: With two traits, their :class:`CrossShares`; ``None`` with one.
        rows: Every cell considered, in the result's order, the
            within-person cell last.
        settings: The request: ``categories``, ``components``,
            ``shared_environment``, ``estimator`` and ``estimate`` (resolved).
        method: The model, weights, solve, ``E``, the cross-trait
            correlations and the absence of an SE.
        metadata: ``pg_phenotype_version`` and ``pedigree_graph_core_rev``.
    """

    traits: tuple[TraitShares, ...]
    cross: CrossShares | None
    rows: tuple[AceRow, ...]
    settings: Mapping[str, object]
    method: Mapping[str, object]
    metadata: Mapping[str, object]


_COMPONENTS = ("A", "C")
_ESTIMATORS = ("primary", "phi")
_ESTIMATES = ("stratified", "pooled")


def _names(name: str, value: object) -> list[str] | None:
    """A sequence of str argument as a list, ``None`` kept."""
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence) or not all(isinstance(v, str) for v in value):
        raise TypeError(f"{name} must be a sequence of str, got {value!r}")
    return list(value)


def _components(components: object) -> list[str]:
    names = [components] if isinstance(components, str) else _names("components", components)
    if not names or len(set(names)) != len(names) or not set(names) <= set(_COMPONENTS):
        raise ValueError(f"components must be 'A', 'C' or both, each once, got {components!r}")
    return names


def _rows_as_tuple(raw: dict) -> dict:
    return {**raw, "rows": tuple(raw["rows"])}


def _ace_cell(cell: Cell, *, within_person: bool) -> tuple:
    stratified = cell.stratified
    return (
        cell.category,
        None if within_person else list(cell.members),
        cell.first_trait,
        cell.second_trait,
        [(e.estimator, e.value, e.reason) for e in cell.crude],
        cell.n,
        cell.n_entries,
        None
        if stratified is None
        else (stratified.estimate.value, stratified.estimate.reason, stratified.n, stratified.n_entries),
    )


def ace(
    result: RelativePairCorrelation,
    *,
    categories: Sequence[str] | None = None,
    components: str | Sequence[str] = _COMPONENTS,
    shared_environment: Sequence[str] | None = None,
    estimator: str = "primary",
    estimate: str | None = None,
) -> AceFit:
    """The ACE Fit of *result*'s cells: A, C and E shares, and cross-trait correlations.

    On the correlation scale (each trait's variance 1) a cell of trait t
    expects ``r = 2 phi A_t + s C_t``, ``phi = 2^-(degree+1)`` the nominal
    kinship of its categories and ``s`` 1 for a Shared-Environment Category,
    else 0; a Category Group's members must share both.  Each trait's
    components are the weighted least-squares solution over its same-trait
    cells, each weighted by the unordered pairs that entered the estimate
    read (the pooled cell's ``n``, or its ``stratified.n``), with
    ``E = 1 - A - C``.  A component not requested is fixed at 0.  With two
    traits the cross-trait cells give ``r12 = 2 phi sA + s sC`` and the
    within-person cell ``r_wp = sA + sC + sE``, solved together, a
    cross-trait cell weighted by its complete orientations so each counts
    once (a symmetric cell's ``n_entries`` or ``stratified.n_entries``, a
    directed cell's ``n``) and the within-person cell by its ``n``; then
    ``rA = sA / sqrt(A_1 A_2)``, ``rC`` and ``rE`` alike.  Values are as
    solved, unconstrained: a correlation is ``None`` with
    ``nonpositive_component`` when its component is <= 0 in a trait, and
    ``|r| > 1`` is kept and flagged ``out_of_range``.  A component no
    entered cell informs is ``not_identified`` and left out, so the others
    are still solved: without a shared cell, C is ``not_identified`` and A
    is solved alone; without a shared cross-trait cell, sC and sE meet only
    in the within-person cell, which is left out, and sA is solved alone.
    Otherwise fewer independent cells than unknowns gives
    ``not_identified``.  There is no SE.

    The binary primary estimate, the tetrachoric, is on the liability scale,
    so the shares are too, with no Dempster-Lerner step;
    ``estimator="phi", estimate="pooled"`` reads phi, on the observed scale.
    Parent-offspring regression stays
    :func:`pg_phenotype.regression.parent_offspring`.

    An exactly identified call is a closed form (``r`` a cell's primary
    estimate)::

        ace(res, categories=["FS"], components="A")  # A = 2 r_FS
        ace(res, categories=["MZ", "FS"], shared_environment=["MZ", "FS"])
        # A = 2 (r_MZ - r_FS), C = 2 r_FS - r_MZ (Falconer)
        ace(res, categories=["PHS"], components="A")  # A = 4 r_PHS
        ace(res, categories=["1C"], components="A")  # A = 8 r_1C

    Args:
        result: A :class:`RelativePairCorrelation`.
        categories: The entries to fit, by category code or group name;
            ``None`` is every entry of *result*.  The within-person cell
            enters the cross solve either way.
        components: ``"A"``, ``"C"``, or both; E is the remainder.
        shared_environment: The codes of the Shared-Environment Categories,
            whose pairs share the common environment (such as MZ, FS, MHS
            when the household follows the mother); required with C.
        estimator: ``"primary"`` (Pearson or tetrachoric) or ``"phi"``.
        estimate: ``"stratified"`` reads each cell's stratified fit,
            ``"pooled"`` its pooled estimate; ``None`` is ``"stratified"``
            when *result* has strata, else ``"pooled"``.

    Returns:
        An :class:`AceFit`.

    Raises:
        ValidationError: A *shared_environment* code that is not a
            Relationship Category's (``unknown_category``), a *categories*
            name that is no entry of *result* (``unknown_entry``), or a
            Category Group fitted whose categories differ in kinship degree
            (``mixed_kinship``) or straddle *shared_environment*
            (``mixed_shared_environment``).
        ValueError: *components*, *estimator* or *estimate* is not one of
            its values; C without *shared_environment*
            (``missing_shared_environment``); stratified with phi, or with
            no cell carrying a stratified form (``no_stratified_form``); phi
            of a continuous cell (``missing_estimator``).
        TypeError: *result* is not a :class:`RelativePairCorrelation`, or
            *categories* or *shared_environment* is not a sequence of str.
    """
    if not isinstance(result, RelativePairCorrelation):
        raise TypeError(f"result must be a RelativePairCorrelation, got {type(result).__name__}")
    names = _names("categories", categories)
    parts = _components(components)
    shared = _names("shared_environment", shared_environment)
    if estimator not in _ESTIMATORS:
        raise ValueError(f"estimator must be one of {_ESTIMATORS}, got {estimator!r}")
    if estimate is None:
        estimate = "stratified" if any(c.stratified is not None for c in result.cells) else "pooled"
    elif estimate not in _ESTIMATES:
        raise ValueError(f"estimate must be one of {_ESTIMATES}, got {estimate!r}")
    cells = [_ace_cell(c, within_person=False) for c in result.cells]
    if result.within_person is not None:
        cells.append(_ace_cell(result.within_person, within_person=True))
    raw = _native.relative_pairs_ace(cells, names, parts, shared, estimator, estimate)
    cross = raw["cross"]
    return AceFit(
        traits=tuple(TraitShares(**_rows_as_tuple(t)) for t in raw["traits"]),
        cross=None if cross is None else CrossShares(**_rows_as_tuple(cross)),
        rows=tuple(AceRow(**r) for r in raw["rows"]),
        settings=MappingProxyType(dict(raw["settings"])),
        method=MappingProxyType(dict(raw["method"])),
        metadata=MappingProxyType(raw["metadata"]),
    )
