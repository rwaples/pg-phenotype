"""The public contracts of prepare and the two scores."""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

import pg_phenotype
from pg_phenotype import pafgrs
from tests.pedigrees import random_pedigree
from tests.scoring_fixtures import Case, random_trait


@pytest.fixture(scope="module")
def case():
    ped = random_pedigree(7, generations=5)
    n = len(ped["id"])
    return ped, random_trait(n, 1), random_trait(n, 2, prevalence=0.3)


def test_output_is_one_record_per_proband_in_input_row_order(case):
    ped, t1, _ = case
    got = pafgrs.score_univariate(pafgrs.prepare(ped), t1.trait, age=t1.age, cip=t1.cip, h2=0.4)
    np.testing.assert_array_equal(got.id, ped["id"])
    assert set(got.to_dict()) == {"id", "est", "var", "n_relatives"}


def test_scores_follow_ids_not_row_positions(case):
    ped, t1, _ = case
    base = pafgrs.score_univariate(pafgrs.prepare(ped), t1.trait, age=t1.age, cip=t1.cip, h2=0.4)
    perm = np.random.default_rng(0).permutation(len(ped["id"]))
    shuffled = {k: v[perm] for k, v in ped.items()}
    trait = Case(t1.affected[perm], t1.age[perm], t1.cip)
    got = pafgrs.score_univariate(pafgrs.prepare(shuffled), trait.trait, age=trait.age, cip=trait.cip, h2=0.4)
    np.testing.assert_array_equal(got.id, base.id[perm])
    np.testing.assert_allclose(got.est, base.est[perm], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.var, base.var[perm], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(got.n_relatives, base.n_relatives[perm])


def test_probands_subset_scores_equal_the_full_run(case):
    ped, t1, t2 = case
    full_prep = pafgrs.prepare(ped, ndegree=3)
    rows = np.array([3, 50, 9, 77])
    sub_prep = pafgrs.prepare(ped, ndegree=3, probands=ped["id"][rows])
    full = pafgrs.score_univariate(full_prep, t1.trait, age=t1.age, cip=t1.cip, h2=0.4)
    sub = pafgrs.score_univariate(sub_prep, t1.trait, age=t1.age, cip=t1.cip, h2=0.4)
    order = np.sort(rows)
    np.testing.assert_array_equal(sub.id, ped["id"][order])
    np.testing.assert_array_equal(sub.est, full.est[order])
    np.testing.assert_array_equal(sub.var, full.var[order])
    bfull = pafgrs.score_bivariate(
        full_prep, (t1.trait, t2.trait), age=(t1.age, t2.age), cip=(t1.cip, t2.cip), h2=(0.4, 0.5), rg=0.3
    )
    bsub = pafgrs.score_bivariate(
        sub_prep, (t1.trait, t2.trait), age=(t1.age, t2.age), cip=(t1.cip, t2.cip), h2=(0.4, 0.5), rg=0.3
    )
    for name in ("est1", "est2", "var1", "var2", "cov12", "n_relatives", "n_obs1", "n_obs2"):
        np.testing.assert_array_equal(getattr(bsub, name), getattr(bfull, name)[order])


def test_frames_and_nullable_columns_are_accepted(case):
    ped, t1, _ = case
    base = pafgrs.score_univariate(pafgrs.prepare(ped), t1.trait, age=t1.age, cip=t1.cip, h2=0.4)
    frame = pl.DataFrame(dict(ped)).with_columns(
        pl.when(pl.col("mother") < 0).then(None).otherwise(pl.col("mother")).alias("mother")
    )
    affected = pd.array([None if np.isnan(v) else bool(v) for v in t1.affected], dtype="boolean")
    age = pl.Series(t1.age).fill_nan(None)
    trait = Case(affected, age, t1.cip)
    pandas_frame = pd.DataFrame({k: pd.array(v, dtype="Int64") for k, v in ped.items()})
    pandas_frame.loc[pandas_frame["mother"] < 0, "mother"] = pd.NA
    got = pafgrs.score_univariate(pafgrs.prepare(pandas_frame), trait.trait, age=trait.age, cip=trait.cip, h2=0.4)
    np.testing.assert_array_equal(got.est, base.est)
    got = pafgrs.score_univariate(pafgrs.prepare(frame), trait.trait, age=trait.age, cip=trait.cip, h2=0.4)
    np.testing.assert_array_equal(got.est, base.est)


def test_missingness_rules():
    ped = {"id": [1, 2, 3, 4, 5, 6], "mother": [-1, -1, 1, 1, 1, 1], "father": [-1, -1, 2, 2, 2, 2]}
    cip = pafgrs.Cip([0.0, 50.0], [0.0, 0.1])
    nan = np.nan
    trait = Case(
        affected=[nan, nan, 1, 0, 0, nan],
        age=[nan, nan, nan, 25.0, nan, 30.0],
        cip=cip,
    )
    prep = pafgrs.prepare(ped, ndegree=1)
    got = pafgrs.score_univariate(prep, trait.trait, age=trait.age, cip=trait.cip, h2=0.5)
    assert got.metadata["controls_without_age"] == 1
    # Observed: row 2, a case without an age (w = 1), and row 3, a control
    # at 25 (w = 0.5).  The control without an age (row 4) and the missing
    # statuses do not count, so row 3's only observed relative is row 2.
    assert got.n_relatives.tolist() == [2, 2, 1, 1, 2, 2]


def test_no_informative_relative_gives_the_prior(case):
    ped, t1, _ = case
    n = len(ped["id"])
    empty = Case(np.full(n, np.nan), np.full(n, np.nan), t1.cip)
    prep = pafgrs.prepare(ped)
    got = pafgrs.score_univariate(prep, empty.trait, age=empty.age, cip=empty.cip, h2=0.3)
    assert (got.est == 0).all()
    assert (got.var == 0.3).all()
    assert (got.n_relatives == 0).all()
    got = pafgrs.score_bivariate(
        prep, (empty.trait, empty.trait), age=(empty.age, empty.age), cip=(empty.cip, empty.cip), h2=(0.3, 0.6), rg=-0.5
    )
    cov_g = -0.5 * np.sqrt(0.3 * 0.6)
    assert (got.est1 == 0).all()
    assert (got.est2 == 0).all()
    assert (got.var1 == 0.3).all()
    assert (got.var2 == 0.6).all()
    assert (got.cov12 == cov_g).all()
    assert (got.n_relatives == 0).all()
    assert (got.n_obs1 == 0).all()
    assert (got.n_obs2 == 0).all()


def test_bivariate_with_rg_zero_and_one_trait_missing_is_univariate(case):
    ped, t1, _ = case
    n = len(ped["id"])
    empty = Case(np.full(n, np.nan), np.full(n, np.nan), t1.cip)
    prep = pafgrs.prepare(ped, ndegree=3)
    uni = pafgrs.score_univariate(prep, t1.trait, age=t1.age, cip=t1.cip, h2=0.4)
    biv = pafgrs.score_bivariate(
        prep, (t1.trait, empty.trait), age=(t1.age, empty.age), cip=(t1.cip, empty.cip), h2=(0.4, 0.7), rg=0.0
    )
    np.testing.assert_array_equal(biv.est1, uni.est)
    np.testing.assert_array_equal(biv.var1, uni.var)
    np.testing.assert_array_equal(biv.n_relatives, uni.n_relatives)
    assert (biv.est2 == 0).all()
    assert (biv.var2 == 0.7).all()
    assert (biv.n_obs2 == 0).all()


@pytest.mark.parametrize(
    ("kwargs", "code", "name"),
    [
        ({"h2": 0.0}, "parameter_out_of_range", "h2"),
        ({"h2": 1.2}, "parameter_out_of_range", "h2"),
        ({"h2": float("nan")}, "parameter_out_of_range", "h2"),
    ],
)
def test_univariate_parameters_are_checked_before_the_trait(kwargs, code, name):
    ped = random_pedigree(0)
    prep = pafgrs.prepare(ped)
    bad_trait = Case([0.5], [1.0], pafgrs.Cip([0.0, 1.0], [0.0, 0.1]))
    with pytest.raises(pg_phenotype.ParameterError) as err:
        pafgrs.score_univariate(prep, bad_trait.trait, age=bad_trait.age, cip=bad_trait.cip, **kwargs)
    assert err.value.code == code
    assert err.value.fields["name"] == name


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"h2": (0.3, 1.5), "rg": 0.2}, "parameter_out_of_range"),
        ({"h2": (0.3, 0.5), "rg": 1.2}, "parameter_out_of_range"),
        ({"h2": (0.3, 0.5), "rg": 0.2, "rho_within": -1.5}, "parameter_out_of_range"),
        ({"h2": (0.9, 0.9), "rg": 0.0, "rho_within": 0.5}, "inconsistent_parameters"),
    ],
)
def test_bivariate_parameters_are_checked(kwargs, code):
    ped = random_pedigree(0)
    n = len(ped["id"])
    trait = random_trait(n, 0)
    with pytest.raises(pg_phenotype.ParameterError) as err:
        pafgrs.score_bivariate(
            pafgrs.prepare(ped),
            (trait.trait, trait.trait),
            age=(trait.age, trait.age),
            cip=(trait.cip, trait.cip),
            **kwargs,
        )
    assert err.value.code == code


@pytest.mark.parametrize(
    ("affected", "age", "code", "field"),
    [
        ([2.0], [10.0], "invalid_trait_value", "trait"),
        ([1.0], [-1.0], "invalid_age", "age"),
        ([1.0], [np.inf], "invalid_age", "age"),
    ],
)
def test_trait_values_are_validated(affected, age, code, field):
    ped = {"id": [1], "mother": [-1], "father": [-1]}
    trait = Case(affected, age, pafgrs.Cip([0.0, 1.0], [0.0, 0.1]))
    with pytest.raises(pg_phenotype.ValidationError) as err:
        pafgrs.score_univariate(pafgrs.prepare(ped), trait.trait, age=trait.age, cip=trait.cip, h2=0.5)
    assert err.value.code == code
    assert dict(err.value.fields) == {
        "field": field,
        "position": 0,
        "value": pytest.approx(affected[0] if field == "trait" else age[0]),
    }


def test_trait_length_is_checked():
    ped = {"id": [1, 2], "mother": [-1, -1], "father": [-1, -1]}
    trait = Case([1.0], [10.0], pafgrs.Cip([0.0, 1.0], [0.0, 0.1]))
    with pytest.raises(pg_phenotype.ValidationError) as err:
        pafgrs.score_univariate(pafgrs.prepare(ped), trait.trait, age=trait.age, cip=trait.cip, h2=0.5)
    assert err.value.code == "trait_length_mismatch"


@pytest.mark.parametrize(
    ("ages", "cip", "position"),
    [([0.0, 1.0], [0.0, 1.0], 1), ([1.0, 0.0], [0.0, 0.1], 1), ([0.0, 1.0], [0.0, 0.0], 1), ([], [], 0)],
)
def test_cip_is_validated(ages, cip, position):
    with pytest.raises(pg_phenotype.ParameterError) as err:
        pafgrs.Cip(ages, cip)
    assert err.value.code == "invalid_cip"
    assert err.value.fields["position"] == position


def test_cip_threshold_is_the_lifetime_quantile():
    from scipy.stats import norm

    cip = pafgrs.Cip([0.0, 30.0, 80.0], [0.0, 0.01, 0.07])
    assert cip.prevalence == 0.07
    assert cip.threshold == pytest.approx(norm.isf(0.07), abs=1e-15)


def test_metadata_names_versions(case):
    ped, t1, _ = case
    got = pafgrs.score_univariate(pafgrs.prepare(ped, ndegree=2), t1.trait, age=t1.age, cip=t1.cip, h2=0.4)
    meta = got.metadata
    assert meta["pg_phenotype_version"] == pg_phenotype.__version__
    assert meta["pedigree_graph_core_rev"] == pg_phenotype.pg_core_rev()
    assert meta["n_probands"] == len(ped["id"])
    assert meta["ndegree"] == 2


_SCORES = """
import sys, numpy as np
from pg_phenotype import pafgrs
sys.path.insert(0, {root!r})
from tests.pedigrees import random_pedigree
from tests.scoring_fixtures import Case, random_trait
ped = random_pedigree(5, generations=5, size=60)
n = len(ped["id"])
prep = pafgrs.prepare(ped, ndegree=3)
t1, t2 = random_trait(n, 1), random_trait(n, 2)
u = pafgrs.score_univariate(prep, t1.trait, age=t1.age, cip=t1.cip, h2=0.35)
b = pafgrs.score_bivariate(prep, (t1.trait, t2.trait), age=(t1.age, t2.age), cip=(t1.cip, t2.cip), h2=(0.35, 0.6), rg=0.4)
parts = [u.est, u.var, b.est1, b.est2, b.var1, b.var2, b.cov12]
sys.stdout.write(np.concatenate(parts).tobytes().hex())
"""


def test_scores_are_identical_across_thread_budgets():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs = [
        subprocess.run(
            [sys.executable, "-c", _SCORES.format(root=root)],
            env={**os.environ, "PG_PHENOTYPE_THREADS": threads},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for threads in ("1", "3", "8")
    ]
    assert outputs[0] == outputs[1] == outputs[2]


def test_a_pool_built_for_another_budget_raises_runtime_error():
    code = (
        "import numpy as np\n"
        "from pg_phenotype import _native\n"
        "ped = [np.array(v, dtype=np.int64) for v in ([1, 2], [-1, -1], [-1, -1])]\n"
        "_native.prepare(*ped, None, None, ndegree=1, probands=None, threads=1)\n"
        "try:\n"
        "    _native.prepare(*ped, None, None, ndegree=1, probands=None, threads=2)\n"
        "except RuntimeError as e:\n"
        "    print('RuntimeError', e)\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert run.stdout.startswith("RuntimeError")
