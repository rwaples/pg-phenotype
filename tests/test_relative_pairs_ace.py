"""The ACE Fit of a Relative-Pair Correlation: recipes, two traits, SEs, the host's checks."""

from __future__ import annotations

import math
import sys
from dataclasses import replace

import numpy as np
import pytest
from pedigree_graph import RELATIONSHIPS

from pg_phenotype import Trait, ValidationError
from pg_phenotype import relative_pairs as rp
from tests.pedigrees import depth, random_pedigree

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
    return rp.correlation(ped, traits, categories=CATEGORIES, stratum=depth(ped))


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
        assert (row.used, row.reason, row.within_person, row.variance) == (True, None, False, None)
    assert dict(fit.settings) == {
        "categories": None,
        "components": ["A", "C"],
        "shared_environment": SHARED,
        "estimator": "primary",
        "estimate": "pooled",
        "weights": "default",
    }
    assert fit.traits[0].weights == "pairs"
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
    stratified = rp.correlation(ped, Trait(_familial(ped, 1, "continuous")), categories=["FS"], stratum=depth(ped))
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
        ({"weights": "iv"}, "weights"),
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


# The fit's values before it had an SE, as float.hex: pairs weights keep their bits.
# Recorded on Linux; another platform's libm can move the tetrachoric cells by an
# ulp, so elsewhere the values only have to agree to rounding.
_BEFORE_THE_SE = {
    "one_ac": [
        "0x1.0c7aff82ba42ap-1",
        "0x1.c9a169bc81d56p-5",
        "0x1.add5d3c2fb401p-2",
    ],
    "one_a": [
        "0x1.24f58cd6b77e6p-1",
        None,
        "0x1.b614e65291034p-2",
    ],
    "two_stratified": [
        "-0x1.72d7d3859073bp-5",
        "0x1.9374ffc8fc3c2p-3",
        "0x1.b2503d4619f84p-1",
        "0x1.176a858eb6824p-3",
        "-0x1.8fe54d9b98997p-4",
        "0x1.ec22084fc572ap-1",
        "0x1.00c3b7caf5499p-2",
        "-0x1.28168460bf339p-2",
        "-0x1.173f9c4833868p-4",
        None,
        None,
        "-0x1.35417f8eda35ap-4",
    ],
    "two_pooled": [
        "0x1.7d890b383f13dp-3",
        "0x1.59fe117d1e8dap-2",
        "0x1.e73d68e6c1e88p-2",
        "0x1.185a382ce86aap-2",
        "0x1.28bd04acc3a1cp-2",
        "0x1.bee8c32653f3ap-2",
        "0x1.fe38912f9aed6p-4",
        "-0x1.498b60a276a7cp-2",
        "0x1.a9e4936e4685ap-4",
        "0x1.1a663d987bc53p-1",
        "-0x1.074a272b498a8p+0",
        "0x1.d34b00edab520p-3",
    ],
    "two_phi": [
        "0x1.7e484a4b50892p-2",
        None,
        "0x1.40dbdada57bb7p-1",
        "0x1.966d2d4fbd931p-2",
        None,
        "0x1.34c9695821368p-1",
        "-0x1.9711065cf896cp-4",
        None,
        "0x1.4968b1a6fcc4ap-5",
        "-0x1.0860483bb2ed2p-2",
        None,
        "0x1.0be8faf93c4a7p-4",
    ],
    "two_subset": [
        "-0x1.c9982d2ec1868p+0",
        "0x1.d4d1b5c77e0a0p-1",
        "0x1.df2f524b02818p+0",
        "0x1.5b27a632c8797p-3",
        "0x1.556cefc9c9e13p-5",
        "0x1.93df4776b1439p-1",
        "-0x1.c2599f5372899p-2",
        "0x1.ecff02ae3a86ep-3",
        "0x1.7adda951f9e9ep-4",
        None,
        "0x1.3b739ef55163cp+0",
        "0x1.37cb1cd8df93ap-4",
    ],
}


def test_without_a_covariance_the_default_keeps_the_values_of_the_fit_before_the_se(one, binary_two):
    calls = {
        "one_ac": (one, {"shared_environment": SHARED}),
        "one_a": (one, {"components": "A"}),
        "two_stratified": (binary_two, {"shared_environment": SHARED}),
        "two_pooled": (binary_two, {"shared_environment": SHARED, "estimate": "pooled"}),
        "two_phi": (binary_two, {"components": "A", "estimator": "phi", "estimate": "pooled"}),
        "two_subset": (binary_two, {"categories": ["FS", "MZ", "PHS"], "shared_environment": SHARED}),
    }
    for name, (res, kwargs) in calls.items():
        fit = rp.ace(res, **kwargs)
        assert fit.settings["weights"] == "default", name
        assert set(_weights(fit)) == {"pairs"}, name
        assert _solves(fit) == _solves(rp.ace(res, weights="pairs", **kwargs)), name
        values = _values(fit)
        got = [None if v is None else v.hex() for v, _, _ in values.values()]
        if sys.platform == "linux":
            assert got == _BEFORE_THE_SE[name], name
        else:
            expected = _BEFORE_THE_SE[name]
            assert [h is None for h in got] == [h is None for h in expected], name
            values_got = [float.fromhex(h) for h in got if h is not None]
            values_expected = [float.fromhex(h) for h in expected if h is not None]
            assert values_got == pytest.approx(values_expected, rel=1e-12), name
        assert all(reason == "no_covariance" for value, _, reason in values.values() if value is not None)


def _solves(fit: rp.AceFit) -> tuple:
    """What *fit* solved, without its settings and method, which echo the request."""
    return fit.traits, fit.cross, fit.rows


def _weights(fit: rp.AceFit) -> list[str | None]:
    """Each trait's solve's weighting, then the cross solve's."""
    return [t.weights for t in fit.traits] + ([] if fit.cross is None else [fit.cross.weights])


def _values(fit: rp.AceFit) -> dict[str, tuple[float | None, float | None, str | None]]:
    """Every value of *fit* by name, with its SE and the SE's reason."""
    out = {}
    for t in fit.traits:
        for k in "ACE":
            out[f"{k}{t.trait}"] = (getattr(t, k), getattr(t, f"{k}_se"), getattr(t, f"{k}_se_reason"))
    if fit.cross is not None:
        for k in ("sA", "sC", "sE", "rA", "rC", "rE"):
            out[k] = (getattr(fit.cross, k), getattr(fit.cross, f"{k}_se"), getattr(fit.cross, f"{k}_se_reason"))
    return out


WEIGHTS = ("pairs", "inverse_variance", "gls")


def _like_a_result(res: rp.RelativePairCorrelation) -> np.ndarray:
    """A positive definite covariance over *res*'s cells: variances 1 / n, correlations 0.3^|i-j|."""
    cells = [*res.cells, *([] if res.within_person is None else [res.within_person])]
    sd = 1 / np.sqrt([c.n for c in cells])
    i = np.arange(len(cells))
    return np.outer(sd, sd) * 0.3 ** np.abs(i[:, None] - i[None, :])


def _with(res: rp.RelativePairCorrelation, pooled: np.ndarray) -> rp.RelativePairCorrelation:
    return replace(res, covariance=rp.CellCovariance(pooled=pooled, phi=None, stratified=None))


@pytest.fixture(scope="module")
def one_cov(ped) -> rp.RelativePairCorrelation:
    trait = Trait(_familial(ped, 1, "continuous"))
    return rp.correlation(ped, trait, categories=CATEGORIES, se=True, covariance=True)


def test_falconer_ses_are_the_closed_forms_under_every_weighting(one_cov):
    sigma = one_cov.covariance.pooled
    mz, fs = (CATEGORIES.index(c) for c in ("MZ", "FS"))
    mm, mf, ff = sigma[mz, mz], sigma[mz, fs], sigma[fs, fs]
    # Read in the other order: each row indexes the covariance by its cell.
    for weights in WEIGHTS:
        fit = rp.ace(one_cov, categories=["FS", "MZ"], shared_environment=["MZ", "FS"], weights=weights)
        t = fit.traits[0]
        assert t.A_se == pytest.approx(math.sqrt(4 * mm - 8 * mf + 4 * ff), rel=1e-12)
        assert t.C_se == pytest.approx(math.sqrt(mm - 4 * mf + 4 * ff), rel=1e-12)
        assert t.E_se == pytest.approx(math.sqrt(mm), rel=1e-12)
        assert [r.variance for r in fit.rows] == [mm, ff]
        assert fit.settings["weights"] == t.weights == weights
        assert fit.method["weights"].startswith(weights)


def _design(res: rp.RelativePairCorrelation) -> tuple[np.ndarray, np.ndarray]:
    """The one-trait A and C design of *res*'s cells, and their primary values."""
    x = np.array([[2 * KINSHIP[c.category], float(c.category in SHARED)] for c in res.cells])
    return x, np.array([_r(res, c.category) for c in res.cells])


def test_inverse_variance_is_weighted_least_squares_with_a_sandwich(one):
    res = _with(one, _like_a_result(one))
    sigma = res.covariance.pooled
    x, y = _design(res)
    w = np.diag(1 / np.diag(sigma))
    g = np.linalg.solve(x.T @ w @ x, x.T @ w)
    fit = rp.ace(res, shared_environment=SHARED, weights="inverse_variance").traits[0]
    np.testing.assert_allclose([fit.A, fit.C], g @ y, rtol=1e-12)
    np.testing.assert_allclose([fit.A_se, fit.C_se], np.sqrt(np.diag(g @ sigma @ g.T)), rtol=1e-12)
    g_e = -g.sum(axis=0)
    assert fit.E_se == pytest.approx(math.sqrt(g_e @ sigma @ g_e), rel=1e-12)


def test_gls_is_generalized_least_squares_and_its_variance(one):
    res = _with(one, _like_a_result(one))
    sigma = res.covariance.pooled
    x, y = _design(res)
    precision = np.linalg.inv(sigma)
    variance = np.linalg.inv(x.T @ precision @ x)
    g = variance @ x.T @ precision
    fit = rp.ace(res, shared_environment=SHARED, weights="gls").traits[0]
    np.testing.assert_allclose([fit.A, fit.C], g @ y, rtol=1e-12)
    np.testing.assert_allclose([fit.A_se, fit.C_se], np.sqrt(np.diag(variance)), rtol=1e-12)
    np.testing.assert_allclose(g @ sigma @ g.T, variance, rtol=1e-10)
    assert fit.E_se == pytest.approx(math.sqrt(variance.sum()), rel=1e-12)


def _shifted(res: rp.RelativePairCorrelation, k: int, by: float) -> rp.RelativePairCorrelation:
    """*res* with cell *k*'s primary value moved by *by*; the within-person cell is last."""

    def move(cell: rp.Cell) -> rp.Cell:
        return replace(cell, crude=(replace(cell.primary, value=cell.primary.value + by), *cell.crude[1:]))

    if k == len(res.cells):
        return replace(res, within_person=move(res.within_person))
    return replace(res, cells=tuple(move(c) if i == k else c for i, c in enumerate(res.cells)))


@pytest.mark.parametrize("weights", WEIGHTS)
def test_every_se_is_the_finite_difference_gradients_variance(ped, weights):
    traits = [Trait(_familial(ped, s, "continuous")) for s in (4, 5)]
    res = rp.correlation(ped, traits, categories=CATEGORIES)
    truth = {"A1": 0.5, "C1": 0.2, "A2": 0.3, "C2": 0.1, "rA": 0.6, "rC": 0.5, "rE": 0.2}
    res = _model(res, truth)
    res = _with(res, _like_a_result(res))
    sigma = res.covariance.pooled
    base = _values(rp.ace(res, shared_environment=SHARED, weights=weights))
    h = 1e-6
    steps = [
        (
            _values(rp.ace(_shifted(res, k, h), shared_environment=SHARED, weights=weights)),
            _values(rp.ace(_shifted(res, k, -h), shared_environment=SHARED, weights=weights)),
        )
        for k in range(len(sigma))
    ]
    for name, (value, se, reason) in base.items():
        assert value is not None, name
        assert reason is None, name
        g = np.array([(up[name][0] - down[name][0]) / (2 * h) for up, down in steps])
        assert se == pytest.approx(math.sqrt(g @ sigma @ g), rel=1e-6), name


def test_the_within_person_row_carries_se_and_re(binary_two):
    sigma = _like_a_result(binary_two)
    base = rp.ace(_with(binary_two, sigma), shared_environment=SHARED, estimate="pooled", weights="pairs")
    wider = sigma.copy()
    wider[-1, -1] *= 4
    fit = rp.ace(_with(binary_two, wider), shared_environment=SHARED, estimate="pooled", weights="pairs")
    assert fit.cross.sE_se > base.cross.sE_se
    assert fit.cross.rE_se > base.cross.rE_se
    assert fit.cross.sA_se == base.cross.sA_se
    assert fit.traits == base.traits
    assert fit.rows[-1].variance == wider[-1, -1]


def test_without_a_covariance_every_se_is_no_covariance(one):
    fit = rp.ace(one, shared_environment=SHARED)
    ses = {name: (se, reason) for name, (_, se, reason) in _values(fit).items()}
    assert ses == dict.fromkeys(("A0", "C0", "E0"), (None, "no_covariance"))
    assert (fit.traits[0].C_se, rp.ace(one, components="A").traits[0].C_se_reason) == (None, "not_requested")
    for weights in ("inverse_variance", "gls"):
        with pytest.raises(ValueError, match="needs the covariance"):
            rp.ace(one, shared_environment=SHARED, weights=weights)


def test_a_cell_without_an_se_reaches_only_what_reads_it(one):
    sigma = _like_a_result(one)
    phs = CATEGORIES.index("PHS")
    sigma[phs, :] = sigma[:, phs] = np.nan
    res = _with(one, sigma)
    pairs = rp.ace(res, shared_environment=SHARED, weights="pairs")
    assert pairs.rows[phs].variance is None
    assert pairs.rows[phs].used
    assert (pairs.traits[0].A_se, pairs.traits[0].A_se_reason) == (None, "row_without_se")
    falconer = rp.ace(res, categories=["MZ", "FS"], shared_environment=["MZ", "FS"], weights="pairs")
    assert falconer.traits[0].A_se is not None
    for weights in ("inverse_variance", "gls"):
        fit = rp.ace(res, shared_environment=SHARED, weights=weights)
        assert (fit.rows[phs].used, fit.rows[phs].value, fit.rows[phs].reason) == (False, None, "no_se")
        assert "PHS" not in fit.traits[0].rows
        assert fit.traits[0].A_se_reason is None


def test_an_indefinite_covariance_gives_negative_variance_or_no_gls(one):
    mz, fs = (CATEGORIES.index(c) for c in ("MZ", "FS"))
    sigma = np.eye(len(CATEGORIES))
    sigma[mz, fs] = sigma[fs, mz] = 1.5
    res = _with(one, sigma)
    kwargs = {"categories": ["MZ", "FS"], "shared_environment": ["MZ", "FS"]}
    t = rp.ace(res, weights="pairs", **kwargs).traits[0]
    assert (t.A_se_reason, t.C_se_reason) == ("negative_variance", "negative_variance")
    assert t.E_se == pytest.approx(1.0, rel=1e-14)
    gls = rp.ace(res, weights="gls", **kwargs)
    for k in "ACE":
        t = gls.traits[0]
        assert (getattr(t, k), getattr(t, f"{k}_reason")) == (None, "covariance_not_positive_definite")
        assert (getattr(t, f"{k}_se"), getattr(t, f"{k}_se_reason")) == (None, "covariance_not_positive_definite")
    # The default falls back to pairs where gls asked for refuses.
    by_default = rp.ace(res, **kwargs)
    assert _weights(by_default) == ["pairs"]
    assert _solves(by_default) == _solves(rp.ace(res, weights="pairs", **kwargs))


def test_the_default_weights_are_gls_where_the_result_has_the_covariance_read(one, binary_two, generated):
    assert generated.covariance is not None
    for estimate in ("stratified", "pooled"):
        fit = rp.ace(generated, shared_environment=SHARED, estimate=estimate)
        assert _weights(fit) == ["gls"] * 3, estimate
        gls = rp.ace(generated, shared_environment=SHARED, estimate=estimate, weights="gls")
        assert _solves(fit) == _solves(gls), estimate
    assert rp.ace(generated, shared_environment=SHARED).settings["estimate"] == "stratified"
    sigma = _like_a_result(binary_two)
    full = replace(binary_two, covariance=rp.CellCovariance(pooled=sigma, phi=sigma, stratified=None))
    cases = {
        # The primary estimator, its form's matrix present.
        ("primary", "pooled"): "gls",
        # Phi stays on pairs though the result has its matrix.
        ("phi", "pooled"): "pairs",
        # A covariance without the stratified matrix is not the covariance read.
        ("primary", "stratified"): "pairs",
    }
    for (estimator, estimate), expected in cases.items():
        fit = rp.ace(full, components="A", estimator=estimator, estimate=estimate)
        assert set(_weights(fit)) == {expected}, (estimator, estimate)
        explicit = rp.ace(full, components="A", estimator=estimator, estimate=estimate, weights=expected)
        assert _solves(fit) == _solves(explicit)
    assert _weights(rp.ace(one, components="A")) == ["pairs"]


def _small(categories: list[str] | None) -> rp.RelativePairCorrelation:
    """Two continuous traits on a small pedigree, whose covariance blocks are often not positive definite."""
    ped = random_pedigree(3, n_founders=40, generations=4)
    rng = np.random.default_rng(1)
    x = rng.normal(size=len(ped["id"]))
    traits = [Trait(x), Trait(0.5 * x + rng.normal(size=len(x)))]
    return rp.correlation(ped, traits, categories=categories, se=True, covariance=True)


@pytest.fixture(scope="module")
def small() -> rp.RelativePairCorrelation:
    return _small(None)


def test_the_default_falls_back_to_pairs_for_a_fit_whose_covariance_has_no_factor(small):
    sigma = small.covariance.pooled
    zero = [i for i, c in enumerate(small.cells) if (c.first_trait, c.second_trait) == (0, 0) and sigma[i, i] > 0]
    assert np.linalg.eigvalsh(sigma[np.ix_(zero, zero)]).min() < 0
    fit = rp.ace(small, components="A")
    pairs = rp.ace(small, components="A", weights="pairs")
    assert _weights(fit) == ["pairs"] * 3
    assert _solves(fit) == _solves(pairs)
    assert all(t.A is not None for t in fit.traits)
    gls = rp.ace(small, components="A", weights="gls")
    for t in gls.traits:
        assert (t.A, t.A_reason, t.A_se_reason) == (None, *["covariance_not_positive_definite"] * 2)
    assert gls.cross.sA_reason == "covariance_not_positive_definite"


def test_the_default_weighs_each_fit_on_its_own():
    res = _small(["FS"])
    fit = rp.ace(res, components="A")
    assert _weights(fit) == ["gls", "gls", "pairs"]
    gls = rp.ace(res, components="A", weights="gls")
    pairs = rp.ace(res, components="A", weights="pairs")
    assert fit.traits == gls.traits
    assert (fit.cross.sA, fit.cross.sA_se, fit.cross.rows) == (pairs.cross.sA, pairs.cross.sA_se, pairs.cross.rows)
    assert fit.cross.rE_se is not None
    assert gls.cross.sA_reason == "covariance_not_positive_definite"


def test_a_covariance_of_other_cells_is_a_validation_error(one_cov):
    res = replace(one_cov, cells=one_cov.cells[:-1])
    with pytest.raises(ValidationError) as err:
        rp.ace(res, components="A")
    n = len(CATEGORIES) - 1
    assert (err.value.code, dict(err.value.fields)) == (
        "covariance_shape_mismatch",
        {"cells": n, "expected_length": n * n, "actual_length": (n + 1) ** 2},
    )


@pytest.fixture(scope="module")
def generated() -> rp.RelativePairCorrelation:
    ped = random_pedigree(7, n_founders=60, generations=5, size=150)
    traits = [Trait(_familial(ped, s, "continuous")) for s in (4, 5)]
    return rp.correlation(ped, traits, categories=CATEGORIES, stratum=depth(ped), se=True, covariance=True)


@pytest.mark.parametrize("estimate", ["pooled", "stratified"])
def test_a_generated_pedigree_gives_every_se_under_every_weighting(generated, estimate):
    for weights in WEIGHTS:
        fit = rp.ace(generated, shared_environment=SHARED, estimate=estimate, weights=weights)
        values = _values(fit)
        for name, (value, se, reason) in values.items():
            if value is not None:
                assert reason is None, (weights, name, reason)
                assert se is not None, (weights, name)
                assert math.isfinite(se), (weights, name)
                assert se > 0, (weights, name)
        assert all(values[k][0] is not None for k in ("A0", "C0", "E0", "A1", "C1", "E1", "sA", "sC", "sE", "rA"))
        assert all(r.used and r.variance > 0 for r in fit.rows)
