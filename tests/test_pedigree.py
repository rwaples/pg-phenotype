"""The Pedigree contract (ADR 0006): a validated pedigree that methods share.

A Pedigree gives what its columns give, whatever its caches held before,
owns its data, and keeps the error order of the columns path.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pg_phenotype import ParameterError, Pedigree, Trait, ValidationError, pafgrs
from pg_phenotype.assortative import mate_correlation
from tests import assortative_builders as b
from tests.pedigrees import random_pedigree
from tests.scoring_fixtures import random_trait

ROOT = Path(__file__).resolve().parent.parent


def _strata(seed: int, strata: np.ndarray, unknown: float) -> np.ndarray:
    """*strata* with about *unknown* of the rows set unknown."""
    out = strata.copy()
    out[np.random.default_rng(seed).random(out.size) < unknown] = np.nan
    return out


def _run(pedigree: object, traits: list[Trait], stratum: np.ndarray | None, **kw: object) -> object:
    return b.plain(mate_correlation(pedigree, traits, stratum=stratum, **kw))


@settings(deadline=None, max_examples=20)
@given(
    seed=st.integers(0, 10_000),
    n_pairs=st.integers(20, 80),
    stratify=st.sampled_from(["none", "known", "unknown"]),
    draws=st.sampled_from([0, 19]),
)
def test_a_pedigree_gives_what_its_columns_give(seed, n_pairs, stratify, draws):
    ped, x, bv, strata = b.frame(seed, n_pairs)
    traits = [Trait(x, kind="continuous"), Trait(bv, kind="binary")]
    stratum = {"none": None, "known": strata, "unknown": _strata(seed, strata, 0.1)}[stratify]
    kw = {"permutations": draws, "bootstrap": draws, "seed": seed, "min_stratum_networks": 2}
    built = Pedigree(ped)
    want = _run(ped, traits, stratum, **kw)
    assert _run(built, traits, stratum, **kw) == want
    assert _run(built, traits, stratum, **kw) == want


@pytest.mark.parametrize("seed", range(3))
def test_pafgrs_scores_equal_through_a_pedigree(seed):
    ped = random_pedigree(seed)
    case = random_trait(ped["id"].size, seed)
    built = Pedigree(ped)
    for ndegree, probands in ((1, None), (3, ped["id"][::3])):
        preps = [pafgrs.prepare(p, ndegree=ndegree, probands=probands) for p in (ped, built)]
        uni = [pafgrs.score_univariate(p, case.trait, age=case.age, cip=case.cip, h2=0.4) for p in preps]
        assert all(np.array_equal(u, v) for u, v in zip(*(r.to_dict().values() for r in uni), strict=True))
        biv = [
            pafgrs.score_bivariate(
                p, [case.trait, case.trait], age=[case.age, case.age], cip=[case.cip, case.cip], h2=[0.4, 0.3], rg=0.5
            )
            for p in preps
        ]
        assert all(np.array_equal(u, v) for u, v in zip(*(r.to_dict().values() for r in biv), strict=True))


@pytest.mark.parametrize(
    "order",
    [
        ["unknown", "none", "known", "other"],
        ["none", "unknown", "known", "other"],
        ["none", "known", "unknown"],
    ],
)
def test_a_result_does_not_depend_on_earlier_calls(order):
    """Strata that drop pairs, strata that drop none, none, and other traits, in any order."""
    ped, x, bv, strata = b.frame(9, 120)
    calls = {
        "none": ([Trait(x, kind="continuous")], None),
        "known": ([Trait(x, kind="continuous")], strata),
        "unknown": ([Trait(x, kind="continuous"), Trait(bv, kind="binary")], _strata(1, strata, 0.15)),
        "other": ([Trait(bv, kind="binary")], None),
    }
    kw = {"permutations": 19, "bootstrap": 19, "seed": 2, "min_stratum_networks": 2}
    built = Pedigree(ped)
    for name in order:
        traits, stratum = calls[name]
        assert _run(built, traits, stratum, **kw) == _run(ped, traits, stratum, **kw), name


def pedigree_payload() -> object:
    """A stratified call on a warm Pedigree: run in a subprocess with a set thread budget."""
    ped, x, bv, strata = b.frame(21, 300)
    traits = [Trait(x, kind="continuous"), Trait(bv, kind="binary")]
    built = Pedigree(ped)
    mate_correlation(built, traits, permutations=0)
    stratum = _strata(3, strata, 0.1)
    return _run(built, traits, stratum, permutations=99, bootstrap=49, seed=4, min_stratum_networks=2)


def test_a_pedigree_result_is_the_same_on_1_and_4_threads():
    out = []
    for threads in (1, 4):
        code = "import json; from tests.test_pedigree import pedigree_payload; print(json.dumps(pedigree_payload()))"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=os.environ | {"PG_PHENOTYPE_THREADS": str(threads)},
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["settings"].pop("threads") == threads
        out.append(payload)
    assert out[0] == out[1]


def test_concurrent_first_calls_on_one_pedigree():
    ped, x, bv, _ = b.frame(4, 200)
    traits = [Trait(x, kind="continuous"), Trait(bv, kind="binary")]
    kw = {"permutations": 19, "bootstrap": 0, "seed": 1}
    want = _run(ped, traits, None, **kw)
    built = Pedigree(ped)
    barrier = threading.Barrier(4)
    got: list[object] = [None] * 4

    def call(i: int) -> None:
        barrier.wait()
        got[i] = _run(built, traits, None, **kw)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(g == want for g in got)


def test_a_pedigree_owns_its_rows():
    ped, x, _, _ = b.frame(5, 60)
    trait = Trait(x, kind="continuous")
    source = {k: v.copy() for k, v in ped.items()}
    built = Pedigree(source)
    want = _run(ped, [trait], None, permutations=0)
    source["mother"][:] = -1
    source["id"][:] = 0
    assert _run(built, [trait], None, permutations=0) == want
    del source
    gc.collect()
    assert _run(built, [trait], None, permutations=0) == want
    assert np.array_equal(built.ids, ped["id"])


def test_ids_are_a_new_read_only_array_in_row_order():
    ped = b.rows(random_pedigree(2), np.arange(random_pedigree(2)["id"].size)[::-1])
    built = Pedigree(ped)
    assert len(built) == ped["id"].size
    ids = built.ids
    assert ids.dtype == np.int64
    assert np.array_equal(ids, ped["id"])
    assert not ids.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        ids[0] = 1
    assert built.ids is not ids
    assert repr(built) == f"<Pedigree: {len(built)} rows>"


def test_a_prep_outlives_its_pedigree():
    ped = random_pedigree(1)
    case = random_trait(ped["id"].size, 1)
    built = Pedigree(ped)
    prep = pafgrs.prepare(built, ndegree=2)
    want = pafgrs.score_univariate(prep, case.trait, age=case.age, cip=case.cip, h2=0.4).to_dict()
    del built
    gc.collect()
    got = pafgrs.score_univariate(prep, case.trait, age=case.age, cip=case.cip, h2=0.4).to_dict()
    assert all(np.array_equal(got[k], want[k]) for k in want)
    assert np.array_equal(got["id"], ped["id"])


def test_a_pedigree_cannot_be_pickled():
    built = Pedigree(random_pedigree(0))
    with pytest.raises(TypeError, match="pickled"):
        pickle.dumps(built)
    with pytest.raises(TypeError):
        pickle.dumps(built._native)


def test_with_columns_errors_come_in_reading_parameter_validation_order():
    bad = {"id": [1, 1, 2], "mother": [-1, -1, 1], "father": [-1, -1, -1]}
    x = Trait([0.1, 0.2, 0.3], kind="continuous")
    with pytest.raises(ParameterError) as err:
        mate_correlation(bad, x, permutations=-1)
    assert err.value.code == "parameter_out_of_range"
    with pytest.raises(ValidationError) as err:
        pafgrs.prepare(bad, ndegree=0)
    assert err.value.code == "degree_out_of_range"
    # A column that cannot be read comes first, as in v0.2.0.
    with pytest.raises(ValidationError) as err:
        mate_correlation({"id": [1, 2]}, Trait([0.1, 0.2], kind="continuous"), permutations=-1)
    assert err.value.code == "missing_field"
    with pytest.raises(ValidationError) as err:
        pafgrs.prepare({"id": [1, 2]}, ndegree=0)
    assert err.value.code == "missing_field"
    with pytest.raises(ValidationError) as err:
        Pedigree(bad)
    assert err.value.code == "duplicate_id"
    assert dict(err.value.fields) == {"id": 1, "rows": (0, 1), "duplicate_count": 1}


def test_inputs_are_checked_against_the_pedigree_rows():
    ped, x, _, strata = b.frame(6, 40)
    built = Pedigree(ped)
    with pytest.raises(ValidationError) as err:
        mate_correlation(built, Trait(x[:-1], kind="continuous"))
    assert err.value.code == "trait_length_mismatch"
    with pytest.raises(ValidationError) as err:
        mate_correlation(built, Trait(x, kind="continuous"), stratum=strata[:-1])
    assert err.value.code == "stratum_length_mismatch"
    prep = pafgrs.prepare(built, probands=ped["id"][[5, 2]])
    assert np.array_equal(prep.proband_rows(), [2, 5])
    with pytest.raises(ValidationError) as err:
        pafgrs.prepare(built, probands=[-7])
    assert err.value.code == "unknown_proband"
    case = random_trait(ped["id"].size - 1, 0)
    with pytest.raises(ValidationError) as err:
        pafgrs.score_univariate(prep, case.trait, age=case.age, cip=case.cip, h2=0.4)
    assert err.value.code == "trait_length_mismatch"
    full = random_trait(ped["id"].size, 0)
    with pytest.raises(ValidationError) as err:
        pafgrs.score_univariate(prep, full.trait, age=full.age[:-1], cip=full.cip, h2=0.4)
    assert err.value.code == "trait_length_mismatch"


@settings(deadline=None, max_examples=30)
@given(seed=st.integers(0, 10_000), keep=st.floats(0.0, 1.0))
def test_align_moves_values_to_the_rows_their_ids_name(seed, keep):
    ped = random_pedigree(seed)
    built = Pedigree(ped)
    rng = np.random.default_rng(seed)
    rows = rng.permutation(ped["id"].size)[: round(keep * ped["id"].size)]
    ids = ped["id"][rows]
    values = rng.normal(size=rows.size)
    values[rng.random(rows.size) < 0.2] = np.nan
    want = np.full(ped["id"].size, np.nan)
    want[rows] = values
    assert np.array_equal(built.rows(ids), rows)
    assert built.rows(ids).dtype == np.int64
    got = built.align(ids, values)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, want)


def test_align_takes_frame_columns_and_matches_a_reindex():
    ped = random_pedigree(3)
    built = Pedigree(ped)
    ids = ped["id"][::-2]
    pheno = pd.DataFrame({"id": ids, "dx": pd.array([(1, None, 0)[i % 3] for i in range(ids.size)], dtype="Int64")})
    want = pheno.set_index("id").reindex(built.ids)["dx"].to_numpy(dtype=np.float64, na_value=np.nan)
    np.testing.assert_array_equal(built.align(pheno["id"], pheno["dx"]), want)
    frame = pl.DataFrame({"id": ids, "dx": [(1, None, 0)[i % 3] for i in range(ids.size)]})
    np.testing.assert_array_equal(built.align(frame["id"], frame["dx"]), want)
    assert built.align([], []).size == len(built)


def test_rows_and_align_reject_unknown_repeated_and_missing_ids():
    ped = random_pedigree(4)
    built = Pedigree(ped)
    absent = int(ped["id"].max()) + 1
    with pytest.raises(ValidationError) as err:
        built.rows([ped["id"][0], absent])
    assert err.value.code == "unknown_id"
    assert dict(err.value.fields) == {"id": absent, "position": 1}
    with pytest.raises(ValidationError) as err:
        built.align([ped["id"][2], ped["id"][1], ped["id"][2]], [1.0, 2.0, 3.0])
    assert err.value.code == "duplicate_lookup_id"
    assert dict(err.value.fields) == {"id": int(ped["id"][2]), "positions": (0, 2)}
    with pytest.raises(ValidationError) as err:
        built.align(ped["id"][:3], [1.0, 2.0])
    assert err.value.code == "pair_length_mismatch"
    assert "values must have one entry per entry of ids (3), got 2" in str(err.value)
    assert dict(err.value.fields) == {"columns": ("ids", "values"), "expected_length": 3, "actual_length": 2}
    with pytest.raises(ValidationError) as err:
        built.rows([float(ped["id"][0]), np.nan])
    assert err.value.code == "invalid_integer_value"
    assert dict(err.value.fields) == {"field": "ids", "position": 1, "value": None}
