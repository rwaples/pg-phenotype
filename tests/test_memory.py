"""Every native call that computes in the pool hands free heap pages back first (``crossing.rs``)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pg_phenotype import Pedigree, Trait, _native, assortative, pafgrs, regression, relative_pairs

PEDIGREE = {"id": np.arange(1, 7), "mother": [-1, -1, -1, -1, 1, 1], "father": [-1, -1, -1, -1, 3, 3]}
VALUES = Trait([0.1, 0.9, 0.4, 0.2, 0.3, 0.6])
STATUS = Trait([1.0, 0.0, 0.0, 1.0, np.nan, 0.0])
AGE = [50.0, 60.0, 40.0, 70.0, np.nan, 20.0]
CIP = pafgrs.Cip([0.0, 80.0], [0.0, 0.1])


def _score_univariate(prep):
    return pafgrs.score_univariate(prep, STATUS, age=AGE, cip=CIP, h2=0.5)


def _score_bivariate(prep):
    return pafgrs.score_bivariate(prep, (STATUS, STATUS), age=(AGE, AGE), cip=(CIP, CIP), h2=(0.5, 0.4), rg=0.3)


#: Each crossing native callable, as a public call of it.  A score is called
#: with a Prep built outside the counted call.
CROSSING = {
    "Pedigree": lambda prep: Pedigree(PEDIGREE),
    "prepare": lambda prep: pafgrs.prepare(PEDIGREE),
    "score_univariate": _score_univariate,
    "score_bivariate": _score_bivariate,
    "mate_correlation": lambda prep: assortative.mate_correlation(PEDIGREE, VALUES, permutations=0),
    "parent_offspring": lambda prep: regression.parent_offspring(PEDIGREE, VALUES),
    "relative_pairs_correlation": lambda prep: relative_pairs.correlation(PEDIGREE, VALUES),
}

#: Native callables that do not compute in the pool, and so do not release.
NOT_CROSSING = {
    "Prep",
    "check_cip",
    "cip_aalen_johansen",
    "configure_threads",
    "core_version",
    "pg_core_rev",
    "relative_pairs_ace",
    "tetrachoric_pairs",
    "tetrachoric_table",
    "thread_budget",
    "trait_kinds",
}


def test_every_native_callable_is_classified():
    """A new native callable must be listed as crossing (and so tested below) or not."""
    public = {name for name in dir(_native) if not name.startswith("_") and callable(getattr(_native, name))}
    assert public == CROSSING.keys() | NOT_CROSSING


@pytest.mark.parametrize("name", CROSSING)
def test_a_crossing_releases_free_memory_once(name):
    count = getattr(_native, "_release_count", None)
    if count is None:
        if os.environ.get("PG_PHENOTYPE_REQUIRE_TEST_HOOKS") == "1":
            pytest.fail("the extension was built without the test-hooks feature")
        pytest.skip("installed wheel without test hooks")
    prep = pafgrs.prepare(PEDIGREE)
    before = count()
    CROSSING[name](prep)
    assert count() == before + 1
