"""The Relative-Pair Correlation against a NumPy reference, and its contract."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from pg_phenotype import ParameterError, ResourceError, Trait, ValidationError
from pg_phenotype import relative_pairs as rp
from tests.oracle.relative_pairs_reference import (
    SYMMETRIC,
    reference,
    stratified_reference,
    within_person_reference,
    within_person_stratified_reference,
)
from tests.pedigrees import random_pedigree

DEFAULT = ["MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "Av", "GGP", "HAv", "GAv", "1C"]
DIRECTED = [c for c in DEFAULT if c not in SYMMETRIC]


def _values(seed: int, n: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    v = rng.normal(size=n) if kind == "continuous" else (rng.random(n) < 0.4).astype(float)
    v[rng.random(n) < 0.15] = np.nan
    return v


def _family() -> dict[str, np.ndarray]:
    """Grandparents 1, 2; their daughter 3 with founders 4 and 9; full sibs 5-7; 8 a maternal half-sib."""
    return {
        "id": np.arange(1, 10),
        "mother": np.array([-1, -1, 1, -1, 3, 3, 3, 3, -1]),
        "father": np.array([-1, -1, 2, -1, 4, 4, 4, 9, -1]),
    }


def _outbred(seed: int, *, generations: int = 3) -> dict[str, np.ndarray]:
    """Founder couples whose descendants marry founders only: no loops, so no pair fits a category both ways."""
    rng = np.random.default_rng(seed)
    mother, father, sex = [], [], []

    def add(m: int, f: int, s: int) -> int:
        mother.append(m)
        father.append(f)
        sex.append(s)
        return len(mother) - 1

    current = []
    for _ in range(6):
        m, f = add(-1, -1, 0), add(-1, -1, 1)
        current += [add(m, f, int(rng.integers(0, 2))) for _ in range(int(rng.integers(1, 5)))]
    for _ in range(generations - 1):
        nxt = []
        for r in current:
            if rng.random() < 0.7:
                spouse = add(-1, -1, 1 - sex[r])
                m, f = (r, spouse) if sex[r] == 0 else (spouse, r)
                nxt += [add(m, f, int(rng.integers(0, 2))) for _ in range(int(rng.integers(1, 4)))]
        current = nxt
    ids = np.arange(1, len(mother) + 1)
    return {
        "id": ids,
        "mother": np.array([ids[m] if m >= 0 else -1 for m in mother]),
        "father": np.array([ids[f] if f >= 0 else -1 for f in father]),
        "sex": np.array(sex),
    }


def _shuffled(ped: dict[str, np.ndarray], values: np.ndarray, seed: int):
    order = np.random.default_rng(seed).permutation(len(values))
    return {k: v[order] for k, v in ped.items()}, values[order]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_every_default_cell_matches_the_reference(seed, kind):
    ped = random_pedigree(seed, generations=5)
    values = _values(seed, len(ped["id"]), kind)
    res = rp.correlation(ped, Trait(values, kind=kind))
    assert [c.category for c in res.cells] == DEFAULT
    for cell in res.cells:
        ref = reference(ped, values, kind, [cell.category])
        assert (cell.n, cell.n_dropped) == (ref.n, ref.n_dropped), cell.category
        assert cell.symmetric == (cell.category in SYMMETRIC)
        if kind == "continuous":
            assert cell.primary.estimator == "pearson"
            _close(cell["pearson"].value, ref.pearson)
            continue
        assert [e.estimator for e in cell.crude] == ["tetrachoric", "phi"]
        assert cell.primary is cell.crude[0]
        assert cell["tetrachoric"].value == ref.tetrachoric, cell.category
        _close(cell["phi"].value, ref.phi)
        t = ref.table
        if cell.symmetric:
            assert cell.concordance == (t[0][0] // 2, t[0][1], t[1][1] // 2)
            assert cell.table is None
        else:
            assert cell.table == t
            assert cell.concordance is None


def _close(value: float | None, expected: float | None) -> None:
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_groups_match_the_reference_on_their_pooled_pairs(kind):
    ped = random_pedigree(3, generations=5)
    values = _values(3, len(ped["id"]), kind)
    request = [{"PO": ["MO", "FO"], "HS": ["MHS", "PHS"]}, "FS", {"GPs": ["GP", "GGP"]}]
    res = rp.correlation(ped, Trait(values, kind=kind), categories=request)
    assert [c.category for c in res.cells] == ["PO", "HS", "FS", "GPs"]
    assert list(res.settings["categories"]) == ["PO", "HS", "FS", "GPs"]
    po = res.cell("PO")
    assert po.members == ("MO", "FO")
    assert po.roles == ("offspring", "parent")
    assert res.cell("GPs").roles == ("descendant", "ancestor")
    assert res.cell("HS").roles is None
    for cell in res.cells:
        ref = reference(ped, values, kind, list(cell.members))
        assert (cell.n, cell.n_dropped) == (ref.n, ref.n_dropped)
        if kind == "continuous":
            _close(cell.primary.value, ref.pearson)
        else:
            assert cell.primary.value == ref.tetrachoric


def test_single_code_roles_are_pedigree_graphs():
    res = rp.correlation(_family(), Trait([0.0, 1, 1, 0, 1, 1, 0, 1, 0], kind="binary"), categories=["MO", "FO"])
    assert res.cell("MO").roles == ("offspring", "mother")
    assert res.cell("FO").roles == ("offspring", "father")
    with pytest.raises(KeyError):
        res.cell("PO")
    with pytest.raises(KeyError):
        res.cell("MO")["pearson"]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_symmetric_cells_do_not_depend_on_row_order(seed, kind):
    ped = random_pedigree(seed, generations=5)
    values = _values(seed, len(ped["id"]), kind)
    a = rp.correlation(ped, Trait(values, kind=kind))
    b = rp.correlation(*_shuffled_trait(ped, values, kind, seed))
    for x, y in zip(a.cells, b.cells, strict=True):
        if x.symmetric:
            assert x == y, x.category


def _shuffled_trait(ped, values, kind, seed):
    ped, values = _shuffled(ped, values, seed)
    return ped, Trait(values, kind=kind)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_every_cell_is_row_order_free_without_loops(seed, kind):
    ped = _outbred(seed)
    values = _values(seed, len(ped["id"]), kind)
    a = rp.correlation(ped, Trait(values, kind=kind))
    b = rp.correlation(*_shuffled_trait(ped, values, kind, seed))
    assert a.cells == b.cells
    assert min(a.cell(c).n for c in ("MO", "GP", "1C")) > 0


def _loop() -> dict[str, np.ndarray]:
    """x and y are each other's half-aunt: x's father p is y's maternal half-sib, y's father q is x's.

    Rows: M1 1, M2 2, F2 3, F4 4, p 5 (M2, F4), q 6 (M1, F2), x 7 (M1, p), y 8 (M2, q).
    """
    return {
        "id": np.arange(1, 9),
        "mother": np.array([-1, -1, -1, -1, 2, 1, 1, 2]),
        "father": np.array([-1, -1, -1, -1, 4, 3, 5, 6]),
    }


def test_a_pair_directed_both_ways_enters_in_the_lower_rows_orientation():
    ped = _loop()
    values = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, 0.0])
    x_first = rp.correlation(ped, Trait(values, kind="binary"), categories=["HAv"]).cells[0]
    swap = np.array([0, 1, 2, 3, 4, 5, 7, 6])
    y_first = rp.correlation(
        {k: v[swap] for k, v in ped.items()}, Trait(values[swap], kind="binary"), categories=["HAv"]
    ).cells[0]
    # x (1) first: the pair is (niece x, aunt y), row 1 column 0.  y first: (y, x).
    assert x_first.table == ((0, 0), (1, 0))
    assert y_first.table == ((0, 1), (0, 0))
    assert x_first.n == y_first.n == 1


def test_close_values_near_a_large_offset_stay_distinct():
    ped = {"id": np.arange(1, 7), "mother": np.array([-1, -1, 1, 1, 1, 1]), "father": np.array([-1, -1, 2, 2, 2, 2])}
    values = np.array([np.nan, np.nan] + [1e12 + 0.01 * k for k in range(4)])
    res = rp.correlation(ped, Trait(values, kind="continuous"), categories=["FS"])
    assert res.cells[0].primary.value == pytest.approx(-1 / 3, abs=1e-9)
    (e,) = res.settings["quantization_exponents"]
    assert res.settings["quantization_steps"] == (2.0**-e,)
    assert res.settings["quantization_steps"][0] < 1e-12


def test_an_outlier_past_the_spread_raises_insufficient_precision():
    ped = random_pedigree(1, generations=5)
    values = np.random.default_rng(1).normal(size=len(ped["id"]))
    values[0] = 1e12
    with pytest.raises(ValidationError) as err:
        rp.correlation(ped, Trait(values, kind="continuous"))
    assert err.value.code == "insufficient_precision"
    assert err.value.fields["field"] == "trait"
    assert err.value.fields["step"] > 2.0**-20 * err.value.fields["spread"]


@pytest.mark.parametrize("scale", [1e300, 1e-300])
def test_extreme_scales_follow_the_guard(scale):
    ped = random_pedigree(2, generations=5)
    base = np.random.default_rng(2).uniform(-1, 1, size=len(ped["id"]))
    res = rp.correlation(ped, Trait(base * scale, kind="continuous"), categories=["FS", "MO"])
    ref = rp.correlation(ped, Trait(base, kind="continuous"), categories=["FS", "MO"])
    for cell, expected in zip(res.cells, ref.cells, strict=True):
        assert cell.primary.value == pytest.approx(expected.primary.value, abs=1e-12)


@pytest.mark.parametrize(
    ("values", "kind", "code"),
    [
        ([0.0] * 8, "binary", "trait_length_mismatch"),
        ([0.0] * 9, "categorical", "unsupported_trait_kind"),
        ([0.0] * 9, "ordinal", "unsupported_trait_kind"),
        ([0.0, 2.0] + [0.0] * 7, "binary", "invalid_trait_value"),
        ([0.0, 0.5] + [0.0] * 7, "binary", "invalid_trait_value"),
        ([0.0, -1.0] + [0.0] * 7, "binary", "invalid_trait_value"),
        ([0.0, np.inf] + [0.0] * 7, "continuous", "invalid_trait_value"),
        ([0.0, -np.inf] + [0.0] * 7, "continuous", "invalid_trait_value"),
    ],
)
def test_the_trait_is_checked_before_the_request(values, kind, code):
    trait = Trait(np.array(values), kind=kind)
    with pytest.raises(ValidationError) as err:
        rp.correlation(_family(), trait, categories=["nope"])
    assert err.value.code == code


@pytest.mark.parametrize(
    ("categories", "code", "fields"),
    [
        (["XX"], "unknown_category", {"category": "XX", "position": 0}),
        (["FS", {"G": ["MHS", "nope"]}], "unknown_category", {"category": "nope", "position": 1}),
        # The groups of one mapping share its position.
        ([{"A": ["MO"], "B": ["XX"]}], "unknown_category", {"category": "XX", "position": 0}),
        (["FS", "FS"], "duplicate_category", {"category": "FS", "positions": (0, 1), "entries": ("FS", "FS")}),
        ([{"G": ["FS", "FS"]}], "duplicate_category", {"category": "FS", "positions": (0, 0), "entries": ("G", "G")}),
        (
            ["MO", {"PO": ["MO", "FO"]}],
            "duplicate_category",
            {"category": "MO", "positions": (0, 1), "entries": ("MO", "PO")},
        ),
        (
            [{"A": ["FS"], "B": ["FS"]}],
            "duplicate_category",
            {"category": "FS", "positions": (0, 0), "entries": ("A", "B")},
        ),
        ([{"G": ["FS"]}, {"G": ["MHS"]}], "duplicate_entry_name", {"name": "G", "positions": (0, 1)}),
        (["FS", {"FS": ["MHS"]}], "reserved_group_name", {"group": "FS", "position": 1}),
        ([{"MO": ["FO"]}], "reserved_group_name", {"group": "MO", "position": 0}),
        (
            ["FS", {"within_person": ["MHS"]}],
            "reserved_group_name",
            {"group": "within_person", "position": 1},
        ),
        ([{"G": []}], "empty_category_group", {"group": "G", "position": 0}),
        ([{"G": ["MO", "GP"]}], "mixed_category_group", {"group": "G", "categories": ("MO", "GP")}),
        ([{"G": ["FS", "MO"]}], "mixed_category_group", {"group": "G", "categories": ("FS", "MO")}),
    ],
)
def test_category_requests_are_checked(categories, code, fields):
    with pytest.raises(ValidationError) as err:
        rp.correlation(_family(), Trait(np.zeros(9), kind="binary"), categories=categories)
    assert err.value.code == code
    assert dict(err.value.fields) == fields


def test_duplicate_messages_name_the_entries():
    trait = Trait(np.zeros(9), kind="binary")
    with pytest.raises(ValidationError, match=r'listed twice in group "G" \(categories\[0\]\)'):
        rp.correlation(_family(), trait, categories=[{"G": ["FS", "FS"]}])
    with pytest.raises(ValidationError, match=r'in both "MO" \(categories\[0\]\) and "PO" \(categories\[1\]\)'):
        rp.correlation(_family(), trait, categories=["MO", {"PO": ["MO", "FO"]}])


def test_an_empty_group_mapping_is_rejected():
    with pytest.raises(ValueError, match=r"categories\[1\] is an empty mapping"):
        rp.correlation(_family(), Trait(np.zeros(9), kind="binary"), categories=["FS", {}])


@pytest.mark.parametrize(
    "categories",
    ["FS", {"PO": ["MO", "FO"]}, [1], [{"G": "FS"}], [{"G": [1]}], [{1: ["FS"]}]],
)
def test_malformed_requests_raise_type_error(categories):
    with pytest.raises(TypeError):
        rp.correlation(_family(), Trait(np.zeros(9), kind="binary"), categories=categories)


def test_the_budget_is_checked_and_bounds_the_engine():
    trait = Trait(np.zeros(9), kind="binary")
    with pytest.raises(ParameterError) as err:
        rp.correlation(_family(), trait, memory_budget_bytes=0)
    assert err.value.code == "parameter_out_of_range"
    with pytest.raises(ResourceError) as err:
        rp.correlation(_family(), trait, memory_budget_bytes=16)
    assert err.value.code == "memory_budget_exceeded"
    with pytest.raises(TypeError):
        rp.correlation(_family(), trait, memory_budget_bytes=True)


def test_empty_categories_missing_and_constant_traits_carry_reasons():
    ped = _family()
    res = rp.correlation(ped, Trait(np.full(9, 2.5), kind="continuous"))
    assert res.cell("FS").primary.reason == "constant_value"
    assert (res.cell("MZ").n, res.cell("MZ").primary.reason) == (0, "no_complete_pairs")
    res = rp.correlation(ped, Trait(np.full(9, np.nan), kind="continuous"))
    fs = res.cell("FS")
    assert (fs.n, fs.n_dropped, fs.primary.value, fs.primary.reason) == (0, 3, None, "no_complete_pairs")
    res = rp.correlation(ped, Trait(np.ones(9), kind="binary"))
    assert [e.reason for e in res.cell("FS").crude] == ["constant_margin", "constant_margin"]
    assert res.cell("FS").concordance == (0, 0, 3)
    assert rp.correlation(ped, Trait(np.ones(9), kind="binary"), categories=[]).cells == ()


def test_settings_method_and_metadata():
    import pg_phenotype

    res = rp.correlation(_family(), Trait(np.zeros(9), kind="binary"), categories=["FS", {"PO": ["MO", "FO"]}])
    s = res.settings
    assert list(s["categories"]) == ["FS", "PO"]
    assert s["trait_kind"] == "binary"
    assert s["memory_budget_bytes"] == 2**30
    assert s["threads"] == pg_phenotype.thread_budget()
    assert s["se"] is False
    assert (s["quantization_exponents"], s["quantization_steps"]) == (None, None)
    assert res.method["se"].startswith("with se, a cluster sandwich (ADR 0013)")
    assert res.metadata["pedigree_graph_core_rev"] == pg_phenotype.pg_core_rev()
    assert pg_phenotype.relative_pairs is rp


def test_the_simace_gate_agrees():
    """docs/gates/relative-pairs: the reference everywhere, simACE on directed categories (tools/relative_pairs_gate.py)."""
    from tests import relative_pairs_golden as g

    oracle = g.oracle_rows()
    assert len(oracle) == 3 * len(DEFAULT)
    failed = [name for name, ok in g.checks(oracle, g.simace_rows()).items() if not ok]
    assert failed == []


def test_the_gate_withholds_comparison_below_simaces_floor():
    """A directed category under ten pairs: simACE gives None by design, which is checked, not a gap."""
    from tests import relative_pairs_golden as g

    row = {
        "symmetric": False,
        "simace_floor": True,
        "n": 4,
        "simace_n_pairs": 4,
        "counts_match": True,
        "pearson_gap": None,
        "phi_gap": None,
        "tetrachoric_gap": None,
        "simace_liability": None,
        "simace_tetrachoric": None,
        "simace_phi": None,
    }
    assert all(g.checks([], [row]).values())
    assert not g.checks([], [{**row, "simace_phi": 0.1}])["simace_floor_withheld"]
    assert not g.checks([], [{**row, "simace_floor": False}])["simace_directed_pearson_within"]


def _generation(ped: dict[str, np.ndarray], seed: int) -> np.ndarray:
    """A cohort-like stratum: each row's depth (founders 0, else one more than its deeper parent).

    A fifth of rows are one later, so some sibs and cousins span two strata, and about a tenth are unknown.
    """
    row = {int(i): r for r, i in enumerate(ped["id"])}
    depth = np.zeros(len(ped["id"]))
    for _ in range(len(depth)):
        before = depth.copy()
        for r, (m, f) in enumerate(zip(ped["mother"], ped["father"], strict=True)):
            parents = [depth[row[int(p)]] for p in (m, f) if int(p) in row]
            depth[r] = 1 + max(parents) if parents else 0
        if (depth == before).all():
            break
    rng = np.random.default_rng(seed + 7)
    depth[rng.random(len(depth)) < 0.2] += 1
    depth[rng.random(len(depth)) < 0.1] = np.nan
    return depth


def _value_close(value: float | None, expected: float | None, tol: float) -> None:
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected, abs=tol)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_stratum_cells_counts_and_stratified_fit_match_the_reference(seed, kind):
    ped = random_pedigree(seed, generations=5)
    values = _values(seed, len(ped["id"]), kind)
    stratum = _generation(ped, seed)
    res = rp.correlation(ped, Trait(values, kind=kind), stratum=stratum)
    plain = rp.correlation(ped, Trait(values, kind=kind))
    fitted = 0
    for cell, pooled in zip(res.cells, plain.cells, strict=True):
        ref = stratified_reference(ped, values, kind, [cell.category], stratum)
        assert cell.crude == pooled.crude
        assert (cell.n, cell.n_dropped) == (pooled.n, pooled.n_dropped)
        assert list(cell.strata) == sorted(ref.strata), cell.category
        for label, got in cell.strata.items():
            expected = ref.strata[label]
            assert (got.n, got.n_dropped) == (expected.n, expected.n_dropped), (cell.category, label)
            if kind == "continuous":
                _close(got.primary.value, expected.pearson)
            else:
                assert got.primary.value == expected.tetrachoric
                _close(got["phi"].value, expected.phi)
        assert (cell.n_cross_stratum, cell.n_unknown_stratum) == (ref.n_cross_stratum, ref.n_unknown_stratum)
        assert sum(s.n for s in cell.strata.values()) + cell.n_cross_stratum + cell.n_unknown_stratum == cell.n
        stratified = cell.stratified
        assert (stratified.n, stratified.n_degenerate_stratum) == (ref.n, ref.n_degenerate_stratum), cell.category
        assert stratified.estimate.reason == ref.reason, cell.category
        assert stratified.estimate.estimator == cell.primary.estimator
        # The oracle's polychoric stops at Brent's 1e-7.
        _value_close(stratified.estimate.value, ref.value, 1e-12 if kind == "continuous" else 1e-6)
        fitted += stratified.estimate.value is not None
    assert fitted >= 4


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_symmetric_stratum_cells_do_not_depend_on_row_order(seed, kind):
    ped = random_pedigree(seed, generations=5)
    values = _values(seed, len(ped["id"]), kind)
    stratum = _generation(ped, seed)
    a = rp.correlation(ped, Trait(values, kind=kind), stratum=stratum)
    order = np.random.default_rng(seed).permutation(len(values))
    shuffled = {k: v[order] for k, v in ped.items()}
    b = rp.correlation(shuffled, Trait(values[order], kind=kind), stratum=stratum[order])
    for x, y in zip(a.cells, b.cells, strict=True):
        if x.symmetric:
            assert x == y, x.category


def test_stratified_cells_have_their_shape():
    ped = _family()
    stratum = [10, 10, 20, None, 30, np.nan, 31, 30, 20]
    trait = Trait([0.0, 1, 1, 0, 1, 1, 0, 1, 0], kind="binary")
    res = rp.correlation(ped, trait, categories=["MO", "FS"], stratum=stratum)
    mo, fs = res.cells
    assert isinstance(mo.strata, MappingProxyType)
    assert list(mo.strata) == [10, 20, 30, 31]
    s30 = mo.strata[30]
    assert isinstance(s30, rp.StratumCell)
    # MO offspring 4, 7 at 30, both with mother 2 (1): offspring values 1, 1.
    assert (s30.n, s30.n_dropped, s30.table, s30.concordance) == (2, 0, ((0, 0), (0, 2)), None)
    assert s30.primary is s30.crude[0] is s30["tetrachoric"]
    assert s30["phi"].reason == "constant_margin"
    with pytest.raises(KeyError):
        s30["pearson"]
    assert (mo.n_cross_stratum, mo.n_unknown_stratum) == (0, 1)
    assert fs.strata[30].concordance == (0, 0, 0)
    assert (fs.n_cross_stratum, fs.n_unknown_stratum) == (1, 2)
    assert isinstance(fs.stratified, rp.Stratified)
    assert fs.stratified.estimate.estimator == "tetrachoric"
    assert fs.stratified.estimate.primary
    for obj in (s30, mo.stratified, mo):
        with pytest.raises(AttributeError):
            obj.n = 0
    assert {"StratumCell", "Stratified"} <= set(rp.__all__)
    plain = rp.correlation(ped, trait, categories=["MO"]).cells[0]
    assert (plain.strata, plain.n_cross_stratum, plain.n_unknown_stratum, plain.stratified) == (None, 0, 0, None)
    assert "strata" in res.method


def test_strata_are_checked_after_the_trait_and_before_the_request():
    with pytest.raises(ValidationError) as err:
        rp.correlation(_family(), Trait(np.zeros(8), kind="binary"), categories=["XX"], stratum=np.zeros(7))
    assert err.value.code == "trait_length_mismatch"
    with pytest.raises(ValidationError) as err:
        rp.correlation(_family(), Trait(np.zeros(9), kind="binary"), categories=["XX"], stratum=np.zeros(8))
    assert err.value.code == "stratum_length_mismatch"
    assert dict(err.value.fields) == {"expected_length": 9, "actual_length": 8}
    with pytest.raises(ValidationError) as err:
        rp.correlation(_family(), Trait(np.zeros(9), kind="binary"), categories=["XX"], stratum=np.zeros(9))
    assert err.value.code == "unknown_category"


def _second(seed: int, values: np.ndarray, kind: str) -> np.ndarray:
    """A second trait of *kind* sharing *values*' signal, with its own missing rows."""
    rng = np.random.default_rng(seed + 200)
    signal = np.nan_to_num(values, nan=0.0)
    liability = 0.5 * signal + rng.normal(size=len(values))
    v = liability if kind == "continuous" else (liability > 0.3).astype(float)
    v[rng.random(len(v)) < 0.15] = np.nan
    return v


def _two(seed: int, kind: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ped = random_pedigree(seed, generations=5)
    x = _values(seed, len(ped["id"]), kind)
    return ped, x, _second(seed, x, kind)


def _matches(cell: rp.Cell | rp.StratumCell, ref, kind: str, symmetric: bool, cross: bool, where: object) -> None:
    assert (cell.n, cell.n_dropped, cell.n_entries) == (ref.n, ref.n_dropped, ref.n_entries), where
    if kind == "continuous":
        _close(cell["pearson"].value, ref.pearson)
        return
    assert cell["tetrachoric"].value == ref.tetrachoric, where
    _close(cell["phi"].value, ref.phi)
    t = ref.table
    if symmetric and not cross:
        assert (cell.concordance, cell.table) == ((t[0][0] // 2, t[0][1], t[1][1] // 2), None), where
    else:
        assert (cell.concordance, cell.table) == (None, t), where


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_every_two_trait_cell_matches_the_reference(seed, kind):
    ped, x, y = _two(seed, kind)
    res = rp.correlation(ped, [Trait(x, kind=kind), Trait(y, kind=kind)])
    combos = {True: [(0, 0), (1, 1), (0, 1)], False: [(0, 0), (0, 1), (1, 0), (1, 1)]}
    expected = [(c, *t) for c in DEFAULT for t in combos[c in SYMMETRIC]]
    assert [(c.category, c.first_trait, c.second_trait) for c in res.cells] == expected
    traits = (x, y)
    cross_entries = 0
    for cell in res.cells:
        cross = cell.first_trait != cell.second_trait
        other = traits[cell.second_trait] if cross else None
        ref = reference(ped, traits[cell.first_trait], kind, [cell.category], other)
        where = (cell.category, cell.first_trait, cell.second_trait)
        _matches(cell, ref, kind, cell.symmetric, cross, where)
        if cell.symmetric and cross:
            assert cell.n <= cell.n_entries <= 2 * cell.n, where
            cross_entries += cell.n_entries - cell.n
    # Some symmetric pairs are complete one way only, and some both.
    assert cross_entries > 0
    assert res.settings["trait_kind"] == kind
    assert list(res.settings["categories"]) == DEFAULT
    exponents = res.settings["quantization_exponents"]
    assert (exponents is None) == (kind == "binary")
    if exponents is not None:
        assert len(exponents) == 2
        assert res.settings["quantization_steps"] == tuple(2.0**-e for e in exponents)


@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_same_trait_cells_of_two_traits_are_the_single_trait_cells_bit_for_bit(kind):
    ped, x, y = _two(1, kind)
    stratum = _generation(ped, 1)
    for by in (None, stratum):
        both = rp.correlation(ped, (Trait(x, kind=kind), Trait(y, kind=kind)), stratum=by)
        for t, values in enumerate((x, y)):
            one = rp.correlation(ped, Trait(values, kind=kind), stratum=by)
            assert one.within_person is None
            same = [
                replace(c, first_trait=0, second_trait=0) for c in both.cells if c.first_trait == c.second_trait == t
            ]
            assert tuple(same) == one.cells
            assert both.cell("FS", first_trait=t, second_trait=t).crude == one.cell("FS").crude
        fs = both.cell("FS")
        assert (fs.first_trait, fs.second_trait) == (0, 0)


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_two_trait_strata_match_the_reference(seed, kind):
    ped, x, y = _two(seed, kind)
    stratum = _generation(ped, seed)
    res = rp.correlation(ped, [Trait(x, kind=kind), Trait(y, kind=kind)], stratum=stratum)
    traits = (x, y)
    fitted = 0
    for cell in res.cells:
        cross = cell.first_trait != cell.second_trait
        other = traits[cell.second_trait] if cross else None
        ref = stratified_reference(ped, traits[cell.first_trait], kind, [cell.category], stratum, other)
        where = (cell.category, cell.first_trait, cell.second_trait)
        assert list(cell.strata) == sorted(ref.strata), where
        for label, got in cell.strata.items():
            _matches(got, ref.strata[label], kind, cell.symmetric, cross, (*where, label))
        assert (cell.n_cross_stratum, cell.n_unknown_stratum) == (ref.n_cross_stratum, ref.n_unknown_stratum), where
        assert sum(s.n for s in cell.strata.values()) + cell.n_cross_stratum + cell.n_unknown_stratum == cell.n, where
        stratified = cell.stratified
        assert (stratified.n, stratified.n_entries, stratified.n_degenerate_stratum) == (
            ref.n,
            ref.n_entries,
            ref.n_degenerate_stratum,
        ), where
        assert stratified.estimate.reason == ref.reason, where
        _value_close(stratified.estimate.value, ref.value, 1e-12 if kind == "continuous" else 1e-6)
        fitted += cross and stratified.estimate.value is not None
    assert fitted >= 4


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_within_person_matches_the_reference(seed, kind):
    ped, x, y = _two(seed, kind)
    stratum = _generation(ped, seed)
    traits = [Trait(x, kind=kind), Trait(y, kind=kind)]
    w = rp.correlation(ped, traits, categories=[]).within_person
    assert isinstance(w, rp.Cell)
    assert (w.category, w.members, w.symmetric, w.roles, w.first_trait, w.second_trait) == (
        "within_person",
        (),
        False,
        None,
        0,
        1,
    )
    assert (w.strata, w.stratified) == (None, None)
    _matches(w, within_person_reference(x, y, kind), kind, False, True, "pooled")
    w = rp.correlation(ped, traits, stratum=stratum, categories=["FS"]).within_person
    ref = within_person_stratified_reference(x, y, kind, stratum)
    assert list(w.strata) == sorted(ref.strata)
    for label, got in w.strata.items():
        _matches(got, ref.strata[label], kind, False, True, label)
    assert (w.n_cross_stratum, w.n_unknown_stratum) == (0, ref.n_unknown_stratum)
    assert sum(s.n for s in w.strata.values()) + w.n_unknown_stratum == w.n
    assert (w.stratified.n, w.stratified.n_degenerate_stratum) == (ref.n, ref.n_degenerate_stratum)
    assert w.stratified.estimate.reason == ref.reason
    _value_close(w.stratified.estimate.value, ref.value, 1e-12 if kind == "continuous" else 1e-6)
    assert w.stratified.estimate.value is not None


def test_traits_take_one_or_two_of_one_kind():
    ped = _family()
    x = Trait([0.3, -1.2, 0.8, 2.0, 1.5, -0.4, 0.9, 0.1, 1.1])
    b = Trait([0.0, 1, 1, 0, 1, 1, 0, 1, 0], kind="binary")
    assert rp.correlation(ped, x, categories=["FS"]) == rp.correlation(ped, [x], categories=["FS"])
    res = rp.correlation(ped, (x, x), categories=["FS", "MO"])
    assert res.cell("FS", first_trait=0, second_trait=1).n_entries == 6
    assert res.cell("MO", first_trait=1, second_trait=0).roles == ("offspring", "mother")
    with pytest.raises(KeyError):
        res.cell("FS", first_trait=1, second_trait=0)
    with pytest.raises(ValidationError) as err:
        rp.correlation(ped, [], categories=["FS"])
    assert (err.value.code, dict(err.value.fields)) == ("trait_count", {"actual": 0, "minimum": 1, "maximum": 2})
    with pytest.raises(ValidationError) as err:
        rp.correlation(ped, [x, x, x], categories=["FS"])
    assert err.value.code == "trait_count"
    with pytest.raises(ValidationError) as err:
        rp.correlation(ped, [x, b], categories=["FS"])
    assert (err.value.code, dict(err.value.fields)) == ("unsupported_trait_kind", {"kinds": ("continuous", "binary")})
    assert str(err.value) == "the traits are continuous and binary: pass two traits of one kind"
    with pytest.raises(ValidationError) as err:
        rp.correlation(ped, [x, Trait(np.zeros(8))], categories=["FS"])
    assert (err.value.code, err.value.fields["field"]) == ("trait_length_mismatch", "traits[1]")
    with pytest.raises(TypeError, match=r"traits\[1\]"):
        rp.correlation(ped, [x, np.zeros(9)])
    with pytest.raises(TypeError):
        rp.correlation(ped, 1.0)
    with pytest.raises(TypeError):
        rp.correlation(ped, trait=x)
