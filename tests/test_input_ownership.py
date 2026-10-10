"""Native calls read their inputs with the GIL released, so no input may be
memory another Python thread can write: a writeable caller array is copied,
and a read-only view (polars, copy-on-write pandas) is read in place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pg_phenotype import Trait
from pg_phenotype._input import floats, id_array, pedigree_arrays


def _read_only(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    out.flags.writeable = False
    return out


def test_pedigree_columns_copy_writeable_arrays_only():
    ids = np.arange(1, 5, dtype=np.int64)
    mother = np.array([-1, -1, 1, 1], dtype=np.int64)
    father = _read_only(np.array([-1, -1, 2, 2], dtype=np.int64))
    arrays = pedigree_arrays({"id": ids, "mother": mother, "father": father})
    assert not np.shares_memory(arrays.id, ids)
    assert not np.shares_memory(arrays.mother, mother)
    assert np.shares_memory(arrays.father, father)
    frame = pl.DataFrame({"id": ids, "mother": mother, "father": father})
    assert np.shares_memory(pedigree_arrays(frame).id, frame["id"].to_numpy())


@pytest.mark.parametrize("read", [floats, id_array])
def test_a_writeable_array_is_copied_and_a_read_only_one_is_not(read):
    values = np.arange(1, 6, dtype=np.float64 if read is floats else np.int64)
    assert not np.shares_memory(read(values, "x"), values)
    view = values[::-1][::-1]
    assert not np.shares_memory(read(view, "x"), values)
    frozen = _read_only(values)
    assert np.shares_memory(read(frozen, "x"), frozen)
    series = pl.Series(values)
    assert np.shares_memory(read(series, "x"), series.to_numpy())


def test_a_trait_keeps_a_read_only_array_of_its_own():
    values = np.array([0.1, 0.5, np.nan])
    trait = Trait(values, kind="continuous")
    values[0] = 9.0
    assert trait.values[0] == 0.1
    assert not trait.values.flags.writeable
    for source in ([0.0, 1.0], pd.Series([0.0, 1.0]), pl.Series([0.0, 1.0]), ["a", "b"]):
        assert not Trait(source).values.flags.writeable
