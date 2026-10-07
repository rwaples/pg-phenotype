import os
import subprocess
import sys

import numpy as np
import pytest
from pedigree_graph import PedigreeGraph

import pg_phenotype
from pg_phenotype import pafgrs
from tests.pedigrees import random_pedigree


def _graph(ped):
    return PedigreeGraph.from_arrays(
        ids=ped["id"], mother_ids=ped["mother"], father_ids=ped["father"], twin_ids=ped["twin"], sex=ped["sex"]
    )


def _expected_relatives(graph, n, ndegree):
    """Per row: relatives by pg categories <= ndegree and exact kinship >= threshold."""
    pairs = graph.relationship_pairs(max_degree=ndegree)
    a = np.concatenate([b.first_rows for b in pairs.values()]).astype(np.int64)
    b = np.concatenate([b.second_rows for b in pairs.values()]).astype(np.int64)
    kin = graph.pair_kinship(a, b)
    keep = kin.astype(np.float64) >= 0.5 ** (ndegree + 1) - 1e-6
    rel: list[dict[int, np.float32]] = [{} for _ in range(n)]
    for x, y, k in zip(a[keep], b[keep], kin[keep], strict=True):
        rel[x][int(y)] = k
        rel[y][int(x)] = k
    return rel


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("ndegree", [1, 2, 3])
def test_relatives_and_kinship_match_pedigree_graph(seed, ndegree):
    ped = random_pedigree(seed)
    graph = _graph(ped)
    prep = pafgrs.prepare(ped, ndegree=ndegree)
    n = len(ped["id"])
    expected = _expected_relatives(graph, n, ndegree)
    assert prep.n_probands == n
    np.testing.assert_array_equal(prep.proband_rows(), np.arange(n))
    firsts, seconds, got = [], [], []
    for i in range(n):
        rows, kin = prep.relatives(i)
        assert rows.tolist() == sorted(expected[i])
        np.testing.assert_array_equal(kin, np.array([expected[i][r] for r in rows], dtype=np.float32))
        j, k = np.triu_indices(len(rows), 1)
        firsts.append(rows[j])
        seconds.append(rows[k])
        got.append(prep.triangle(i))
    first, second = np.concatenate(firsts).astype(np.int64), np.concatenate(seconds).astype(np.int64)
    want = graph.pair_kinship(first, second)
    np.testing.assert_array_equal(np.concatenate(got).view(np.uint32), want.view(np.uint32))


def test_probands_restrict_scored_rows_only():
    ped = random_pedigree(3)
    full = pafgrs.prepare(ped, ndegree=2)
    chosen = ped["id"][[5, 40, 2, 17]]
    sub = pafgrs.prepare(ped, ndegree=2, probands=chosen)
    rows = sorted([5, 40, 2, 17])
    np.testing.assert_array_equal(sub.proband_rows(), rows)
    for i, row in enumerate(rows):
        for got, want in zip(sub.relatives(i), full.relatives(row), strict=True):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(sub.triangle(i), full.triangle(row))


_DUMP = """
import sys, numpy as np
from pg_phenotype import pafgrs
sys.path.insert(0, {root!r})
from tests.pedigrees import random_pedigree
ped = random_pedigree(11, generations=5, size=60)
prep = pafgrs.prepare(ped, ndegree=3)
parts = [prep.proband_rows()]
for i in range(prep.n_probands):
    rows, kin = prep.relatives(i)
    parts += [rows, kin.view(np.uint32), prep.triangle(i).view(np.uint32)]
sys.stdout.write(np.concatenate([p.astype(np.uint64) for p in parts]).tobytes().hex())
"""


def test_output_is_identical_across_thread_budgets():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs = []
    for threads in ("1", "4"):
        env = {**os.environ, "PG_PHENOTYPE_THREADS": threads}
        run = subprocess.run(
            [sys.executable, "-c", _DUMP.format(root=root)], env=env, capture_output=True, text=True, check=True
        )
        outputs.append(run.stdout)
    assert outputs[0] == outputs[1]
    assert len(outputs[0]) > 1000


def test_errors_carry_codes_and_fields():
    ped = random_pedigree(0)
    with pytest.raises(pg_phenotype.ValidationError) as err:
        pafgrs.prepare(ped, ndegree=6)
    assert err.value.code == "degree_out_of_range"
    assert dict(err.value.fields) == {"value": 6, "minimum": 1, "maximum": 5}
    with pytest.raises(pg_phenotype.ValidationError) as err:
        pafgrs.prepare(ped, probands=[ped["id"][0], -5])
    assert err.value.code == "unknown_proband"
    assert dict(err.value.fields) == {"id": -5, "position": 1}
    with pytest.raises(pg_phenotype.ValidationError) as err:
        pafgrs.prepare(ped, probands=[ped["id"][3], ped["id"][3]])
    assert err.value.code == "duplicate_proband"
    cyc = {"id": [1, 2], "mother": [2, 1], "father": [-1, -1]}
    with pytest.raises(pg_phenotype.ValidationError) as err:
        pafgrs.prepare(cyc)
    assert err.value.code == "cycle"


def test_missing_parents_may_be_null():
    ped = {"id": [1, 2, 3], "mother": [None, None, 1], "father": [np.nan, np.nan, 2.0]}
    prep = pafgrs.prepare(ped, ndegree=1)
    assert prep.relatives(2)[0].tolist() == [0, 1]
