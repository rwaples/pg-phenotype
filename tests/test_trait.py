"""The shared Trait input: kinds, inference, and what PA-FGRS accepts."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

import pg_phenotype
from pg_phenotype import Trait, pafgrs


@pytest.mark.parametrize(
    ("values", "kind"),
    [
        ([0, 1, None, 1], "binary"),
        ([True, False, True], "binary"),
        (pd.array([True, None, False], dtype="boolean"), "binary"),
        (pl.Series([0.0, 1.0, None]), "binary"),
        ([0.5, 1.2, np.nan], "continuous"),
        (["low", "high", None, "mid"], "categorical"),
    ],
)
def test_kind_is_inferred(values, kind):
    assert Trait(values).kind == kind


def test_values_are_float_with_nan_for_missing():
    t = Trait(pd.array([1, None, 0], dtype="Int64"))
    assert t.values.dtype == np.float64
    assert np.isnan(t.values[1])
    assert t.values[[0, 2]].tolist() == [1.0, 0.0]


def test_strings_are_coded_in_sorted_order():
    t = Trait(["b", "a", None, "c", "a"])
    assert t.levels == ("a", "b", "c")
    np.testing.assert_array_equal(t.values, [1.0, 0.0, np.nan, 2.0, 0.0])


def test_integers_beyond_binary_need_a_kind():
    with pytest.raises(pg_phenotype.ValidationError) as err:
        Trait([0, 1, 2, 3])
    assert err.value.code == "ambiguous_trait_kind"
    assert Trait([0, 1, 2, 3], kind="ordinal").kind == "ordinal"
    assert Trait([0, 1, 2, 3], kind="continuous").kind == "continuous"


def test_codes_must_be_integers():
    with pytest.raises(pg_phenotype.ValidationError) as err:
        Trait([0, 1.5, 2], kind="categorical")
    assert err.value.code == "invalid_trait_value"
    assert dict(err.value.fields)["position"] == 1


def test_unknown_kind_is_an_error():
    with pytest.raises(pg_phenotype.ValidationError) as err:
        Trait([0, 1], kind="nominal")
    assert err.value.code == "invalid_trait_kind"


def test_pafgrs_needs_a_binary_trait():
    ped = {"id": [1, 2, 3], "mother": [-1, -1, 1], "father": [-1, -1, 2]}
    cip = pafgrs.Cip([0.0, 50.0], [0.0, 0.1])
    trait = Trait([0.2, 1.4, 0.9])
    with pytest.raises(pg_phenotype.ValidationError) as err:
        pafgrs.score_univariate(pafgrs.prepare(ped), trait, age=[10.0, 20.0, 30.0], cip=cip, h2=0.4)
    assert err.value.code == "trait_kind_mismatch"
    assert dict(err.value.fields) == {"field": "trait", "expected": "binary", "actual": "continuous"}


def test_pandas_categoricals_keep_category_order():
    ordered = pd.Categorical(["severe", "mild", None], categories=["mild", "moderate", "severe"], ordered=True)
    t = Trait(pd.Series(ordered))
    assert t.kind == "ordinal"
    assert t.levels == ("mild", "moderate", "severe")
    np.testing.assert_array_equal(t.values, [2.0, 0.0, np.nan])
    unordered = Trait(pd.Categorical(["b", "a", "b"], categories=["b", "a"]))
    assert unordered.kind == "categorical"
    np.testing.assert_array_equal(unordered.values, [0.0, 1.0, 0.0])
