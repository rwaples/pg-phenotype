"""The ACE Fit of a Relative-Pair Correlation: recipes, two traits, the host's checks."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from pedigree_graph import RELATIONSHIPS

from pg_phenotype import Trait, ValidationError
from pg_phenotype import relative_pairs as rp
from tests.pedigrees import random_pedigree

CATEGORIES = ["MZ", "FS", "MHS", "PHS", "MO", "1C"]
KINSHIP = {"MZ": 0.5, "FS": 0.25, "MO": 0.25, "MHS": 0.125, "PHS": 0.125, "1C": 0.0625}
SHARED = ["MZ", "FS", "MHS"]


def _familial(ped: dict[str, np.ndarray], seed: int, kind: str) -> np.ndarray:
    """A trait with a family signal: each row's parents' mean plus noise, founders noise alone."""
    rng = np.random.default_rng(seed)
    row = {int(i): r for r, i in enumerate(ped["id"])}
    values = np.full(len(ped["id"]), np.nan)
    for _ in range(len(values)):
        for r, (m, f) in enumerate(zip(ped["mother"], ped["father"], strict=True)):
            parents = [values[row[int(p)]] for p in (m, f) if int(p) in row]
            if np.isnan(values[r]) and not any(np.isnan(parents)):
                values[r] = (np.mean(parents) if parents else 0.0) + rng.normal()
    twin = ped["twin"]
    for r, t in enumerate(twin):
        if t >= 0:
            values[r] = values[row[int(t)]] + 0.3 * rng.normal()
    return values if kind == "continuous" else (values > 0.2).astype(float)


@pytest.fixture(scope="module")
def ped() -> dict[str, np.ndarray]:
    return random_pedigree(3, generations=5)


@pytest.fixture(scope="module")
def one(ped) -> rp.RelativePairCorrelation:
    return rp.correlation(ped, Trait(_familial(ped, 1, "continuous")), categories=CATEGORIES)


@pytest.fixture(scope="module")
def binary_two(ped) -> rp.RelativePairCorrelation:
    traits = [Trait(_familial(ped, s, "binary"), kind="binary") for s in (1, 2)]
    return rp.correlation(ped, traits, categories=CATEGORIES, stratum=_depth(ped))


def _depth(ped: dict[str, np.ndarray]) -> np.ndarray:
    row = {int(i): r for r, i in enumerate(ped["id"])}
    depth = np.zeros(len(ped["id"]))
    for _ in range(len(depth)):
        for r, (m, f) in enumerate(zip(ped["mother"], ped["father"], strict=True)):
            parents = [depth[row[int(p)]] for p in (m, f) if int(p) in row]
            depth[r] = 1 + max(parents) if parents else 0
    return depth


def _r(res: rp.RelativePairCorrelation, category: str) -> float:
    value = res.cell(category).primary.value
    assert value is not None
    return value


def _operand_close(actual: float | None, expected: float, scale: float) -> None:
    assert actual is not None
    assert abs(actual - expected) <= 1e-15 * abs(scale), (actual, expected)


def test_kinship_is_pedigree_graphs_nominal_kinship(ped):
    codes = list(RELATIONSHIPS)
    res = rp.correlation(ped, Trait(_familial(ped, 1, "continuous")), categories=codes)
    fit = rp.ace(res, components="A")
    assert {r.entry: r.kinship for r in fit.rows} == {c: RELATIONSHIPS[c].nominal_kinship for c in codes}


def test_recipes_are_their_closed_forms(one):
    sibs = rp.ace(one, categories=["FS"], components="A").traits[0]
    _operand_close(sibs.A, 2 * _r(one, "FS"), 2 * _r(one, "FS"))
    assert (sibs.C, sibs.C_reason, sibs.rows) == (None, "not_requested", ("FS",))
    _operand_close(sibs.E, 1 - 2 * _r(one, "FS"), 1 + abs(2 * _r(one, "FS")))

    r_mz, r_fs = _r(one, "MZ"), _r(one, "FS")
    falconer = rp.ace(one, categories=["MZ", "FS"], shared_environment=["MZ", "FS"]).traits[0]
    _operand_close(falconer.A, 2 * (r_mz - r_fs), 2 * (abs(r_mz) + abs(r_fs)))
    _operand_close(falconer.C, 2 * r_fs - r_mz, 2 * abs(r_fs) + abs(r_mz))

    half_sibs = rp.ace(one, categories=["PHS"], components="A").traits[0]
    _operand_close(half_sibs.A, 4 * _r(one, "PHS"), 4 * _r(one, "PHS"))
    cousins = rp.ace(one, categories=["1C"], components="A").traits[0]
    _operand_close(cousins.A, 8 * _r(one, "1C"), 8 * _r(one, "1C"))


def test_one_trait_fit_shape(one):
    fit = rp.ace(one, shared_environment=SHARED)
    assert isinstance(fit, rp.AceFit)
    assert fit.cross is None
    assert [row.entry for row in fit.rows] == CATEGORIES
    for row, cell in zip(fit.rows, one.cells, strict=True):
        assert (row.value, row.weight, row.kinship) == (cell.primary.value, cell.n, KINSHIP[cell.category])
        assert row.shared == (cell.category in SHARED)
        assert (row.used, row.reason, row.within_person) == (True, None, False)
    assert dict(fit.settings) == {
        "categories": None,
        "components": ["A", "C"],
        "shared_environment": SHARED,
        "estimator": "primary",
        "estimate": "pooled",
    }
    assert set(fit.method) == {"model", "weights", "solve", "E", "cross", "se"}
    assert set(fit.metadata) == {"pg_phenotype_version", "pedigree_graph_core_rev"}
    assert rp.ace(one, components=("C", "A"), shared_environment=SHARED) == fit
    a_only = rp.ace(one, components=["A"])
    assert a_only.rows[0].shared is None


def test_two_traits_add_cross_cells_and_the_within_person_cell(binary_two):
    fit = rp.ace(binary_two, shared_environment=SHARED)
    assert fit.settings["estimate"] == "stratified"
    cells = [*binary_two.cells, binary_two.within_person]
    assert [(r.entry, r.first_trait, r.second_trait) for r in fit.rows] == [
        (c.category, c.first_trait, c.second_trait) for c in cells
    ]
    assert fit.rows[-1].within_person
    assert (fit.rows[-1].kinship, fit.rows[-1].shared) == (None, None)
    for row, cell in zip(fit.rows, cells, strict=True):
        assert cell.stratified is not None
        assert row.value == cell.stratified.estimate.value
    assert [t.trait for t in fit.traits] == [0, 1]
    assert fit.cross is not None
    assert "within_person" in fit.cross.rows
    assert set(fit.cross.rows) <= {*CATEGORIES, "within_person"}
    for t, shares in enumerate(fit.traits):
        assert shares.rows == tuple(
            r.entry for r in fit.rows if r.used and (r.first_trait, r.second_trait) == (t, t) and not r.within_person
        )
    assert fit.cross.rows == tuple(
        r.entry for r in fit.rows if r.used and (r.first_trait != r.second_trait or r.within_person)
    )
    assert any(not r.used for r in fit.rows)
    with_only = rp.ace(binary_two, categories=["FS"], components="A")
    assert [r.entry for r in with_only.rows] == ["FS", "FS", "FS", "within_person"]
    assert with_only.cross is not None
    assert with_only.cross.rows == ("FS", "within_person")


@pytest.mark.parametrize("estimate", ["pooled", "stratified"])
def test_cross_rows_weigh_each_complete_orientation_once(binary_two, estimate):
    fit = rp.ace(binary_two, components="A", estimate=estimate)
    cells = [*binary_two.cells, binary_two.within_person]
    kinds = set()
    for row, cell in zip(fit.rows, cells, strict=True):
        counts = cell if estimate == "pooled" else cell.stratified
        assert counts is not None
        cross = cell.first_trait != cell.second_trait and not row.within_person
        if cross and cell.symmetric:
            assert counts.n_entries is not None
            assert row.weight == counts.n_entries, row
            kinds.add("symmetric" if counts.n_entries != counts.n else "complete both ways")
        else:
            assert counts.n_entries is None
            assert row.weight == counts.n, row
            kinds.add("directed" if cross else "within_person" if row.within_person else "same-trait")
    assert kinds >= {"symmetric", "directed", "within_person", "same-trait"}


def test_pooled_reads_each_cells_primary_or_phi(binary_two):
    pooled = rp.ace(binary_two, components="A", estimate="pooled")
    phi = rp.ace(binary_two, components="A", estimator="phi", estimate="pooled")
    cells = [*binary_two.cells, binary_two.within_person]
    assert [r.value for r in pooled.rows] == [c.primary.value for c in cells]
    assert [r.value for r in phi.rows] == [c["phi"].value for c in cells]
    assert [r.weight for r in phi.rows] == [r.weight for r in pooled.rows]
    assert phi.settings["estimator"] == "phi"
    with pytest.raises(ValueError, match="phi has no stratified form"):
        rp.ace(binary_two, components="A", estimator="phi")


def test_the_default_estimate_follows_the_result(ped, one):
    assert rp.ace(one, components="A").settings["estimate"] == "pooled"
    stratified = rp.correlation(ped, Trait(_familial(ped, 1, "continuous")), categories=["FS"], stratum=_depth(ped))
    fit = rp.ace(stratified, components="A")
    assert fit.settings["estimate"] == "stratified"
    assert fit.rows[0].value == stratified.cells[0].stratified.estimate.value
    with pytest.raises(ValueError, match="no entry has a stratified form"):
        rp.ace(one, components="A", estimate="stratified")


def _model(res: rp.RelativePairCorrelation, truth: dict[str, float]) -> rp.RelativePairCorrelation:
    """*res* with each cell's primary value the model's exact correlation."""
    a, c = (truth["A1"], truth["A2"]), (truth["C1"], truth["C2"])
    e = (1 - a[0] - c[0], 1 - a[1] - c[1])
    s_a = truth["rA"] * math.sqrt(a[0] * a[1])
    s_c = truth["rC"] * math.sqrt(c[0] * c[1])
    s_e = truth["rE"] * math.sqrt(e[0] * e[1])

    def exact(cell: rp.Cell) -> rp.Cell:
        if cell.category == "within_person":
            r = s_a + s_c + s_e
        else:
            shared = cell.category in SHARED
            t1, t2 = cell.first_trait, cell.second_trait
            ga, gc = (a[t1], c[t1]) if t1 == t2 else (s_a, s_c)
            r = 2 * KINSHIP[cell.category] * ga + shared * gc
        return replace(cell, crude=(replace(cell.primary, value=r, reason=None), *cell.crude[1:]))

    within = res.within_person
    return replace(
        res, cells=tuple(exact(c) for c in res.cells), within_person=None if within is None else exact(within)
    )


def test_exact_model_correlations_recover_every_parameter(ped):
    traits = [Trait(_familial(ped, s, "continuous")) for s in (4, 5)]
    res = rp.correlation(ped, traits, categories=CATEGORIES)
    truth = {"A1": 0.5, "C1": 0.2, "A2": 0.3, "C2": 0.1, "rA": 0.6, "rC": 0.5, "rE": 0.2}
    fit = rp.ace(_model(res, truth), shared_environment=SHARED)
    got = {
        "A1": fit.traits[0].A,
        "C1": fit.traits[0].C,
        "A2": fit.traits[1].A,
        "C2": fit.traits[1].C,
        "rA": fit.cross.rA,
        "rC": fit.cross.rC,
        "rE": fit.cross.rE,
    }
    for key, expected in truth.items():
        assert got[key] == pytest.approx(expected, abs=1e-12), key
    assert fit.cross.rA_out_of_range is False


def test_without_the_within_person_cell_se_is_not_identified(one):
    fit = rp.ace(one, components="A")
    assert fit.cross is None
    res = replace(one, within_person=None)
    assert rp.ace(res, components="A") == fit


def test_undefined_cells_stay_out_with_their_reason(ped):
    res = rp.correlation(ped, Trait(np.zeros(len(ped["id"])), kind="continuous"), categories=["FS", "PHS"])
    fit = rp.ace(res, components="A")
    assert [(r.used, r.reason) for r in fit.rows] == [(False, "constant_value")] * 2
    assert (fit.traits[0].A, fit.traits[0].A_reason) == (None, "not_identified")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"components": ()}, "components"),
        ({"components": "E"}, "components"),
        ({"components": "AC"}, "components"),
        ({"components": ["A", "A"]}, "components"),
        ({"components": ["A", "E"]}, "components"),
        ({"estimator": "tetrachoric"}, "estimator"),
        ({"estimate": "both"}, "estimate"),
    ],
)
def test_host_value_errors(one, kwargs, match):
    with pytest.raises(ValueError, match=match):
        rp.ace(one, **{"shared_environment": SHARED, **kwargs})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"categories": "FS"},
        {"categories": [1]},
        {"shared_environment": "FS"},
        {"components": [1]},
    ],
)
def test_host_type_errors(one, kwargs):
    with pytest.raises(TypeError):
        rp.ace(one, **{"shared_environment": SHARED, **kwargs})


def test_a_result_is_required(one):
    with pytest.raises(TypeError, match="RelativePairCorrelation"):
        rp.ace(one.cells)


@pytest.mark.parametrize(
    ("kwargs", "code", "fields"),
    [
        ({"shared_environment": ["MZ", "XX"]}, "unknown_category", {"category": "XX", "position": 1}),
        (
            {"categories": ["FS", "sibs"], "components": "A"},
            "unknown_entry",
            {"name": "sibs", "position": 1},
        ),
    ],
)
def test_core_validation_errors_carry_their_codes(one, kwargs, code, fields):
    with pytest.raises(ValidationError) as err:
        rp.ace(one, **kwargs)
    assert (err.value.code, dict(err.value.fields)) == (code, fields)


def test_group_errors_carry_their_codes(ped):
    x = Trait(_familial(ped, 1, "continuous"))
    half_sibs = rp.correlation(ped, x, categories=[{"HS": ["MHS", "PHS"]}])
    with pytest.raises(ValidationError) as err:
        rp.ace(half_sibs, shared_environment=["MHS"])
    assert (err.value.code, dict(err.value.fields)) == (
        "mixed_shared_environment",
        {"group": "HS", "categories": ("MHS", "PHS")},
    )
    sibs = rp.correlation(ped, x, categories=[{"sibs": ["FS", "MHS"]}])
    with pytest.raises(ValidationError) as err:
        rp.ace(sibs, components="A")
    assert err.value.code == "mixed_kinship"


def test_core_usage_errors_are_value_errors(one):
    with pytest.raises(ValueError, match="shared_environment"):
        rp.ace(one)
    with pytest.raises(ValueError, match='"MZ" has no phi estimate'):
        rp.ace(one, components="A", estimator="phi")
