"""``Cip.from_aalen_johansen`` and its gate against simACE and survfit (issue #4, ADR 0010)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pg_phenotype import Trait, ValidationError, pafgrs
from pg_phenotype.pafgrs import Cip
from tests.aj_golden import load

if TYPE_CHECKING:
    from tests.aj_golden import Case

GOLDEN = load()
#: survfit's widest relative gap, 4.5e-14 observed (docs/gates/aalen-johansen).
SURVFIT_RTOL = 1e-12


def estimate(case: Case, *, entry: np.ndarray | None = None) -> Cip:
    return Cip.from_aalen_johansen(case.age, case.status, entry=case.entry if entry is None else entry)


@pytest.mark.parametrize("case", GOLDEN.cases, ids=lambda c: c.key)
def test_matches_survfit(case):
    cip = estimate(case)
    assert cip.interpolation == "step"
    np.testing.assert_array_equal(cip.ages, case.survfit_ages)
    np.testing.assert_allclose(cip.cip, case.survfit_cif, rtol=SURVFIT_RTOL, atol=0)


@pytest.mark.parametrize("case", [c for c in GOLDEN.cases if c.scenario != "integer_ages"], ids=lambda c: c.key)
def test_matches_simace_bit_for_bit_on_continuous_ages(case):
    cip = estimate(case)
    np.testing.assert_array_equal(cip.ages, case.simace_ages)
    np.testing.assert_array_equal(cip.cip, case.simace_cif)
    if case.simace_wrapper_k is not None:
        assert cip.prevalence == case.simace_wrapper_k


@pytest.mark.parametrize("case", [c for c in GOLDEN.cases if c.scenario == "integer_ages"], ids=lambda c: c.key)
def test_differs_from_simace_only_by_its_entry_tie_rule(case):
    """simACE counts a row entering at t at risk at t; one ulp earlier entries reproduce it."""
    assert case.entry is not None
    late = case.entry[case.entry > 0]
    if np.isin(case.age[case.status != 0], late).any():
        assert not np.array_equal(estimate(case).cip, case.simace_cif)
    shifted = np.nextafter(case.entry, -np.inf).clip(0)
    np.testing.assert_array_equal(estimate(case, entry=shifted).cip, case.simace_cif)


def test_zero_entries_match_no_entry():
    case = next(c for c in GOLDEN.cases if c.scenario == "no_entry")
    zeros = np.zeros(len(case.age))
    assert np.array_equal(estimate(case, entry=zeros).cip, estimate(case).cip)


def test_corrects_the_naive_cip_under_late_entry():
    """Cases over all rows, fitACE's ``compute_empirical_cip``, understates K."""
    case = next(c for c in GOLDEN.cases if c.key == "late_entry/trait1/pooled")
    assert case.n_dropped is not None
    naive = case.n_cases / (len(case.age) + case.n_dropped)
    assert estimate(case).prevalence > 1.4 * naive


def test_hand_example():
    # t=1: Y=5, a case: 1/5.  t=2: a death.  t=4: Y=2: += 3/5 * 1/2.  t=5: Y=1: += 3/10.
    cip = Cip.from_aalen_johansen([1, 2, 3, 4, 5], [1, 2, 0, 1, 1])
    assert cip.interpolation == "step"
    np.testing.assert_array_equal(cip.ages, [1.0, 4.0, 5.0])
    np.testing.assert_allclose(cip.cip, [0.2, 0.5, 0.8], rtol=1e-15)
    assert cip.prevalence == cip.cip[-1]


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"age": [1.0, 2.0, 3.0], "status": [1, 0, 0]}),
        pl.DataFrame({"age": [1.0, 2.0, 3.0], "status": [1, 0, 0]}),
    ],
    ids=["pandas", "polars"],
)
def test_takes_frame_columns(frame):
    assert Cip.from_aalen_johansen(frame["age"], frame["status"]).cip.tolist() == [1 / 3]


@pytest.mark.parametrize(
    ("age", "status", "entry", "code", "fields"),
    [
        ([1.0], [1, 0], None, "pair_length_mismatch", {"columns": ("age", "status")}),
        ([1.0], [1], [0.0, 0.0], "pair_length_mismatch", {"columns": ("age", "entry")}),
        ([1.0, None], [1, 0], None, "invalid_age", {"field": "age", "position": 1}),
        ([1.0, -1.0], [1, 0], None, "invalid_age", {"field": "age", "position": 1}),
        ([1.0, 2.0], [1, 3], None, "invalid_status", {"position": 1, "value": 3.0}),
        ([1.0, 2.0], [1, None], None, "invalid_status", {"position": 1}),
        ([1.0, 2.0], [1, 0], [0.0, np.inf], "invalid_age", {"field": "entry", "position": 1}),
        ([1.0, 2.0], [1, 0], [0.0, 2.5], "entry_not_before_age", {"position": 1, "entry": 2.5, "age": 2.0}),
        ([1.0, 2.0], [1, 2], [0.0, 2.0], "entry_not_before_age", {"position": 1}),
        ([1.0, 2.0], [0, 2], None, "cip_not_estimable", {"reason": "zero_prevalence"}),
        ([], [], None, "cip_not_estimable", {"reason": "zero_prevalence"}),
        ([1.0, 2.0], [1, 1], None, "cip_not_estimable", {"reason": "prevalence_one"}),
    ],
)
def test_rejects(age, status, entry, code, fields):
    with pytest.raises(ValidationError) as err:
        Cip.from_aalen_johansen(age, status, entry=entry)
    assert err.value.code == code
    assert {k: err.value.fields[k] for k in fields} == fields


def test_a_censored_row_may_leave_at_its_entry_age():
    with_row = Cip.from_aalen_johansen([1, 2, 3, 4, 3], [1, 0, 1, 0, 0], entry=[0, 0, 0, 0, 3])
    without = Cip.from_aalen_johansen([1, 2, 3, 4], [1, 0, 1, 0], entry=[0, 0, 0, 0])
    np.testing.assert_array_equal(with_row.ages, without.ages)
    np.testing.assert_array_equal(with_row.cip, without.cip)


def test_a_step_table_scores_a_control_by_the_last_step():
    """The mode reaches scoring: a control at 55 on a step table [20, 60] carries CIP(20)."""
    pedigree = {"id": [1, 2, 3], "mother": [-1, -1, 1], "father": [-1, -1, 2]}
    prep = pafgrs.prepare(pedigree, ndegree=1, probands=[3])
    trait = Trait([0.0, 1.0, np.nan], kind="binary")
    age = [55.0, 40.0, np.nan]

    def est(cip: Cip) -> float:
        return float(pafgrs.score_univariate(prep, trait, age=age, cip=cip, h2=0.5).est[0])

    step = est(Cip([20.0, 60.0], [0.05, 0.1], interpolation="step"))
    # A linear table flat from 20 to 55 gives the mother the same w (0.5) and K.
    assert step == est(Cip([20.0, 55.0, 60.0], [0.05, 0.05, 0.1]))
    # Linear between 20 and 60 gives her w = 0.9375.
    assert step != est(Cip([20.0, 60.0], [0.05, 0.1]))


def test_cip_interpolation_is_checked():
    assert Cip([0.0, 1.0], [0.0, 0.1]).interpolation == "linear"
    with pytest.raises(ValueError, match="interpolation"):
        Cip([0.0, 1.0], [0.0, 0.1], interpolation="cubic")  # ty: ignore[invalid-argument-type]
