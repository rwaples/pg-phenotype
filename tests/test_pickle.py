"""Every public result survives pickling, so a worker can return one (issue #29)."""

import dataclasses
import multiprocessing
import pickle
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from types import MappingProxyType

import numpy as np
import pytest

from pg_phenotype import Trait, assortative, correlation, pafgrs, regression
from pg_phenotype import relative_pairs as rp
from pg_phenotype._mapping import ReadOnlyMapping
from tests import assortative_builders as b
from tests.pedigrees import depth, random_pedigree
from tests.scoring_fixtures import random_trait


def _fixture():
    ped = random_pedigree(7, generations=5)
    n = len(ped["id"])
    rng = np.random.default_rng(0)
    x, y = Trait(rng.normal(size=n), "continuous"), Trait(rng.normal(size=n), "continuous")
    return ped, x, y, depth(ped)


def _results():
    ped, x, y, stratum = _fixture()
    t1, t2 = random_trait(len(ped["id"]), 1), random_trait(len(ped["id"]), 2, prevalence=0.3)
    prep = pafgrs.prepare(ped)
    # random_pedigree's depths leave every stratified mate cell empty, so draws need this frame.
    mates, mx, mb, decades = b.frame(0, 120)
    mate_traits = [Trait(mx, "continuous"), Trait(mb, "binary")]
    pairs = rp.correlation(ped, [x, y], stratum=stratum, se=True, covariance=True, influence=True)
    return {
        "univariate": pafgrs.score_univariate(prep, t1.trait, age=t1.age, cip=t1.cip, h2=0.4),
        "bivariate": pafgrs.score_bivariate(
            prep, (t1.trait, t2.trait), age=(t1.age, t2.age), cip=(t1.cip, t2.cip), h2=(0.4, 0.5), rg=0.3
        ),
        "mate": assortative.mate_correlation(
            mates, mate_traits, stratum=decades, permutations=19, bootstrap=19, min_stratum_networks=2
        ),
        "regression": regression.parent_offspring(ped, x, stratum=stratum),
        "pairs": pairs,
        "ace": rp.ace(pairs, components="A"),
        "tetrachoric": correlation.tetrachoric(table=[[20, 10], [5, 15]]),
    }


def _assert_same(got, want, path="result", *, rtol=0.0, mappings=None):
    """Equal values and types, and every mapping still read-only, all the way down.

    Floats match exactly unless *rtol* is set; *mappings*, when given, collects each mapping's path and size.
    """

    def same(g, w, p):
        _assert_same(g, w, p, rtol=rtol, mappings=mappings)

    assert type(got) is type(want), path
    if dataclasses.is_dataclass(want):
        for f in dataclasses.fields(want):
            same(getattr(got, f.name), getattr(want, f.name), f"{path}.{f.name}")
    elif isinstance(want, Mapping):
        assert list(got) == list(want), path
        with pytest.raises(TypeError):
            got["__new__"] = 1  # type: ignore[index]
        if mappings is not None:
            mappings[path] = len(want)
        for k in want:
            same(got[k], want[k], f"{path}[{k!r}]")
    elif isinstance(want, (tuple, list)):
        assert len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            same(g, w, f"{path}[{i}]")
    elif isinstance(want, np.ndarray):
        assert got.dtype == want.dtype, path
        if rtol and np.issubdtype(want.dtype, np.floating):
            np.testing.assert_allclose(got, want, rtol=rtol, equal_nan=True, err_msg=path)
        else:
            np.testing.assert_array_equal(got, want, err_msg=path)
    elif isinstance(want, float):
        assert got == pytest.approx(want, rel=rtol, abs=0, nan_ok=True), path
    else:
        assert got == want, path


@pytest.fixture(scope="module")
def results():
    return _results()


@pytest.mark.parametrize("name", ["univariate", "bivariate", "mate", "regression", "pairs", "ace", "tetrachoric"])
def test_every_result_round_trips_through_pickle(results, name):
    want = results[name]
    _assert_same(pickle.loads(pickle.dumps(want)), want)


def test_the_mate_result_exercises_draws_and_its_failure_reasons(results):
    mappings = {}
    _assert_same(results["mate"], results["mate"], mappings=mappings)
    reasons = [size for path, size in mappings.items() if path.endswith(".failure_reasons")]
    assert reasons, "no Draws reached"
    assert max(reasons) > 0, "every Draws.failure_reasons is empty"


def _relative_pairs():
    ped, x, y, stratum = _fixture()
    return rp.correlation(ped, [x, y], stratum=stratum, se=True)


def test_a_result_computed_in_a_worker_reaches_the_parent():
    local = _relative_pairs()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=context) as pool:
        remote = pool.submit(_relative_pairs).result()
    # A spawned child may size its thread pool differently, so SE floats may differ in the last bits.
    _assert_same(remote, local, rtol=1e-12)


def test_the_mapping_reads_like_the_mappingproxy_it_replaced():
    data = {"a": 1, "b": [1]}
    new, old = ReadOnlyMapping(data), MappingProxyType(dict(data))
    assert new == old == data
    assert list(reversed(new)) == list(reversed(old))
    assert new | {"z": 1} == old | {"z": 1}
    assert {"z": 1} | new == {"z": 1} | old
    assert new.copy() == old.copy()
    assert type(new.copy()) is dict
    for operand in ([("z", 1)], 5):
        with pytest.raises(TypeError):
            new | operand
        with pytest.raises(TypeError):
            operand | new
    data["a"] = 2
    assert new["a"] == 1
