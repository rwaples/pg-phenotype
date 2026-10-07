"""pg-phenotype PA-FGRS against BioPsyk/PAFGRS's own pa_fgrs on stored fixtures (#11b).

Regenerate with tools/make_r_golden.py.  Each case stores its pedigree,
trait, and CIP, and R's score for every proband with an observed relative.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import pg_phenotype
from pg_phenotype import pafgrs

GOLDEN = sorted(Path(__file__).parent.glob("golden/r_pafgrs_*"))
CASES = [(d, case) for d in GOLDEN for case in json.loads((d / "manifest.json").read_text())["cases"]]

TOL = 1e-10


def test_golden_is_present():
    assert len(CASES) >= 4


@pytest.mark.parametrize(("directory", "case"), CASES, ids=[c["file"] for _, c in CASES])
def test_univariate_matches_r_pafgrs(directory, case):
    data = np.load(directory / case["file"])
    ped = {k: data[k] for k in ("id", "mother", "father", "twin", "sex")}
    trait = pg_phenotype.Trait(data["affected"], kind="binary")
    cip = pafgrs.Cip(data["cip_ages"], data["cip_values"])
    prep = pafgrs.prepare(ped, ndegree=case["ndegree"])
    got = pafgrs.score_univariate(prep, trait, age=data["age"], cip=cip, h2=case["h2"])
    idx = data["proband_index"]
    assert len(idx) == case["n_scored"] > 50
    np.testing.assert_allclose(got.est[idx], data["est"], rtol=0, atol=TOL)
    np.testing.assert_allclose(got.var[idx], data["var"], rtol=0, atol=TOL)
    unscored = np.setdiff1d(np.arange(prep.n_probands), idx)
    assert (got.n_relatives[unscored] == 0).all()
    assert (got.est[unscored] == 0).all()
    assert (got.var[unscored] == case["h2"]).all()
