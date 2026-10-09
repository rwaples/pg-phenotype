"""The public tetrachoric correlation (:mod:`pg_phenotype.correlation`) and its gate against simACE (issue #2)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pg_phenotype import Trait, ValidationError
from pg_phenotype.assortative import mate_correlation
from pg_phenotype.correlation import tetrachoric
from tests import assortative_builders as b
from tests.oracle.tetrachoric_reference import reference
from tests.tetrachoric_golden import load

if TYPE_CHECKING:
    from pg_phenotype.correlation import Tetrachoric

#: The widest rho gap to simACE inside the bound, 1.13e-7 observed (docs/gates/tetrachoric).
SIMACE_TOL = 2e-7
#: Bounded Brent's xatol, which a fallback fit stops within.
ORACLE_TOL = 1e-7


def assert_matches_oracle(got: Tetrachoric, table: list[list[int]]) -> None:
    """rho within Brent's tolerance; the SE to 1e-9, plus its move with rho, about 1 / (1 - |rho|)."""
    want = reference(table)
    assert got.value is not None
    assert want.rho is not None
    assert want.se_two_step is not None
    gap = abs(got.value - want.rho)
    assert gap < ORACLE_TOL, table
    n = sum(map(sum, table))
    assert got.se is not None
    rel = abs(got.se / (want.se_two_step * math.sqrt(n / (n - 1))) - 1)
    assert rel < 1e-9 + 10 * gap / (1 - abs(want.rho)), table


def test_a_table_and_its_pairs_give_the_same_result():
    table = [[40, 10], [15, 35]]
    x = np.repeat([0.0, 0.0, 1.0, 1.0], [40, 10, 15, 35])
    y = np.repeat([0.0, 1.0, 0.0, 1.0], [40, 10, 15, 35])
    from_table = tetrachoric(table=table)
    from_pairs = tetrachoric(x, y)
    assert from_pairs == from_table
    assert from_table.table == ((40, 10), (15, 35))
    assert (from_table.n, from_table.n_dropped) == (100, 0)
    assert from_table.reason is None
    assert from_table.boundary is False
    assert from_table.ci_method == "sandwich"
    assert from_table.ci_level == 0.95
    ci = from_table.ci
    assert ci is not None
    assert ci[0] < from_table.value < ci[1]  # ty: ignore[unsupported-operator]


def test_missing_values_drop_their_pair():
    x = [1, 0, 1, None, 1, 0, 1, 0]
    y = [1.0, 0.0, 0.0, 1.0, np.nan, 0.0, 1.0, 1.0]
    got = tetrachoric(x, y)
    assert got.table == ((2, 1), (1, 2))
    assert (got.n, got.n_dropped) == (6, 2)
    assert tetrachoric(pd.Series(x, dtype="Int64"), pl.Series(y)) == got
    assert tetrachoric(np.array([True, False, True, False]), np.array([True, False, False, False])).n == 4


@pytest.mark.parametrize(
    ("table", "reason"),
    [
        ([[0, 0], [0, 0]], "no_complete_pairs"),
        ([[3, 4], [0, 0]], "constant_margin"),
        ([[3, 0], [4, 0]], "constant_margin"),
    ],
)
def test_an_undefined_estimate_carries_its_reason_everywhere(table, reason):
    got = tetrachoric(table=table)
    assert (got.value, got.boundary, got.se, got.ci, got.ci_method) == (None, None, None, None, None)
    assert got.reason == got.se_unavailable_reason == got.ci_unavailable_reason == reason


def test_an_empty_cell_is_a_boundary_fit_without_an_se():
    got = tetrachoric(table=[[10, 0], [0, 10]])
    assert got.boundary is True
    assert got.value is not None
    assert got.value > 0.998
    assert (got.se, got.ci) == (None, None)
    assert got.se_unavailable_reason == got.ci_unavailable_reason == "boundary"


@pytest.mark.parametrize(
    "table",
    [
        # Newton from 0 overshoots to where a small cell's probability is rounding noise or
        # floored: the score and Hessian blow up, the step is about 0, and Newton stopped there,
        # flagged as a boundary, until `Tables::terms` handed such points to Brent.
        [[9891, 104], [1, 4]],  # stopped at 0.9915, NLL above the start's; MLE 0.7969
        [[7637907, 54], [832877, 25795]],  # stopped at 0.9863, NLL below the start's; MLE 0.8875
        [[17191, 1100], [74523, 1]],  # -0.9901; MLE -0.9009
        [[363660, 204808], [13, 40361]],  # 0.9903; MLE 0.9219
        [[6272522, 1655325], [1703, 170718]],  # Hessian 1e225, finite; MLE 0.8595
        [[12059874, 10], [7909006, 2501135]],  # NLL fell on the overshoot; MLE 0.9618
    ],
)
def test_newton_overshooting_into_a_noisy_cell_falls_back_to_brent(table):
    got = tetrachoric(table=table)
    assert got.boundary is False
    assert_matches_oracle(got, table)


def test_skewed_random_tables_fit_their_interior_maximum():
    """No interior MLE is missed on tables with near-empty cells and n up to 3e7.

    The development sweep (10 seeds, about 140,000 interior tables) found none either.
    """
    rng = np.random.default_rng(20261009)
    checked = 0
    for _ in range(4000):
        n = int(10 ** rng.uniform(1, 7.5))
        t = rng.multinomial(n, rng.dirichlet(rng.choice([0.05, 0.2, 1.0], size=4))).reshape(2, 2).tolist()
        want = reference(t).rho
        if want is None or abs(want) >= 0.998:
            continue
        got = tetrachoric(table=t).value
        assert got is not None
        # At n near 3e7 rounding in the NLL alone moves its minimum by up to about 1e-6;
        # the missed maxima were 0.01 or more away.
        assert abs(got - want) < 1e-6, t
        checked += 1
    assert checked > 1000


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"table": [[1, 2]]}, "invalid_shape"),
        ({"table": [[1, 2], [3]]}, "invalid_shape"),
        ({"table": [[2**53 + 1, 1], [1, 1]]}, "invalid_table"),
        ({"table": [[1, -1], [2, 3]]}, "invalid_table"),
        ({"table": [[1.5, 1], [1, 1]]}, "invalid_table"),
        ({"table": [["a", 1], [1, 1]]}, "invalid_table"),
        ({"x": [0, 1, 2], "y": [0, 1, 1]}, "invalid_trait_value"),
        ({"x": [0, 1], "y": [0, 1, 1]}, "pair_length_mismatch"),
        ({"x": [[0, 1]], "y": [[0, 1]]}, "invalid_shape"),
    ],
)
def test_invalid_input_is_a_validation_error(kwargs, code):
    with pytest.raises(ValidationError) as err:
        tetrachoric(**kwargs)
    assert err.value.code == code


def test_invalid_entries_name_their_position():
    with pytest.raises(ValidationError) as err:
        tetrachoric(table=[[1, 2], [-3, 4]])
    assert err.value.fields == {"field": "table", "row": 1, "column": 0, "value": -3}
    with pytest.raises(ValidationError) as err:
        tetrachoric([0, 1, 1], [0, 1, 7])
    assert err.value.fields == {"field": "y", "kind": "binary", "position": 2, "value": 7.0}


@pytest.mark.parametrize("kwargs", [{}, {"x": [0, 1]}, {"x": [0, 1], "y": [0, 1], "table": [[1, 1], [1, 1]]}])
def test_pass_pairs_or_a_table(kwargs):
    with pytest.raises(TypeError):
        tetrachoric(**kwargs)


def test_a_binary_mate_correlation_cell_of_lone_couples_is_the_tetrachoric():
    """Every Mating Pair its own Mate Network: the cell's fit and sandwich are the standalone ones."""
    rng = np.random.default_rng(3)
    n = 400
    latent = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], size=n)
    pairs = [(2 * i, 2 * i + 1) for i in range(n)]
    ped = b.pedigree(pairs)
    by_id = {m: latent[i, 0] for i, (m, _) in enumerate(pairs)} | {f: latent[i, 1] for i, (_, f) in enumerate(pairs)}
    cell = mate_correlation(ped, Trait(b.binary(ped, by_id, cut=0.5), "binary"), permutations=0).cell(0, 0)
    assert cell.n_mate_networks == n
    assert cell.table is not None
    primary = cell.primary
    got = tetrachoric(table=cell.table)
    assert primary.estimator == "tetrachoric"
    assert got.value == primary.value
    assert got.boundary == primary.boundary
    assert got.se is not None
    assert primary.se is not None
    assert math.isclose(got.se, primary.se, rel_tol=1e-12)


def test_known_values_against_the_oracle():
    for table in ([[250, 250], [250, 250]], [[40, 10], [15, 35]], [[980, 5], [5, 10]], [[3, 7], [8, 2]]):
        assert_matches_oracle(tetrachoric(table=table), table)


@pytest.fixture(scope="module")
def gate() -> list[tuple[object, Tetrachoric]]:
    return [(case, tetrachoric(table=case.table)) for case in load().cases]


def test_gate_undefined_estimates_are_simaces_nans(gate):
    for case, got in gate:
        assert (got.reason is not None) == (case.simace_r is None), case
        assert got.reason in (None, "no_complete_pairs", "constant_margin")


def test_gate_interior_rho_matches_simace(gate):
    gaps = [abs(got.value - case.simace_r) for case, got in gate if got.reason is None and not got.boundary]
    assert len(gaps) == 806
    assert max(gaps) < SIMACE_TOL


def test_gate_matches_the_oracle(gate):
    checked = 0
    for case, got in gate:
        if got.reason is None and not got.boundary and reference(case.table).rho is not None:
            assert_matches_oracle(got, case.table)
            checked += 1
    assert checked == 806


def test_gate_boundary_fits_are_empty_cells_or_within_the_margin(gate):
    for case, got in gate:
        if got.boundary:
            assert 0 in (case.table[0] + case.table[1]) or abs(got.value) >= 0.9999 - 1e-3, case
            assert got.se is None
            assert got.se_unavailable_reason == "boundary"
