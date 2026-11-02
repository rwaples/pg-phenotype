"""The error classes survive pickling, so a worker's error reaches its caller (issue #28)."""

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from pg_phenotype import ParameterError, PgPhenotypeError, ResourceError, Trait, ValidationError
from pg_phenotype import relative_pairs as rp


@pytest.mark.parametrize(
    ("cls", "base"),
    [
        (PgPhenotypeError, Exception),
        (ValidationError, ValueError),
        (ParameterError, ValueError),
        (ResourceError, MemoryError),
    ],
)
def test_every_error_class_round_trips_through_pickle(cls, base):
    err = cls("some_code", "a message", value=6, names=("a", "b"))
    err.add_note("while scoring rep 3")
    got = pickle.loads(pickle.dumps(err))
    assert type(got) is cls
    assert isinstance(got, base)
    assert got.code == "some_code"
    assert dict(got.fields) == {"value": 6, "names": ("a", "b")}
    assert str(got) == "a message"
    assert got.args == err.args
    assert got.__notes__ == ["while scoring rep 3"]
    with pytest.raises(TypeError):
        got.fields["value"] = 7


def _exceed_the_budget():
    ped = {"id": np.arange(4), "mother": np.array([-1, -1, 0, 0]), "father": np.array([-1, -1, 1, 1])}
    rp.correlation(ped, Trait(np.array([0.1, 0.2, 0.3, 0.4]), "continuous"), memory_budget_bytes=1)


def test_a_native_error_raised_in_a_worker_reaches_the_parent():
    with pytest.raises(ResourceError) as local:
        _exceed_the_budget()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=context) as pool, pytest.raises(ResourceError) as remote:
        pool.submit(_exceed_the_budget).result()
    assert remote.value.code == local.value.code == "memory_budget_exceeded"
    assert dict(remote.value.fields) == dict(local.value.fields)
    assert str(remote.value) == str(local.value)
