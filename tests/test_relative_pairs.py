"""The Relative-Pair Correlation against a NumPy reference, and its contract."""

from __future__ import annotations

import numpy as np
import pytest

from pg_phenotype import ParameterError, ResourceError, Trait, ValidationError
from pg_phenotype import relative_pairs as rp
from tests.oracle.relative_pairs_reference import SYMMETRIC, reference
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
    e = res.settings["quantization_exponent"]
    assert res.settings["quantization_step"] == 2.0**-e
    assert res.settings["quantization_step"] < 1e-12


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
    assert (s["quantization_exponent"], s["quantization_step"]) == (None, None)
    assert res.method["se"] == "none (ADR 0009)"
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
