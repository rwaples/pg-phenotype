"""The standalone assortative-mating reference: isolated from pedsum, and equal to pedsum on the goldens."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.assortative_golden import NAMES, load, mating_pairs
from tests.oracle import assortative_reference as ref

ROOT = Path(__file__).resolve().parent.parent

_ISOLATED = """
import sys
for name in ("pedsum", "numba", "polars", "pg_phenotype"):
    sys.modules[name] = None
sys.path.insert(0, {root!r})
import numpy as np
from tests.oracle import assortative_reference as ref
rng = np.random.default_rng(0)
m = rng.standard_normal(200)
f = 0.5 * m + rng.standard_normal(200)
print(ref.pearson(m, f))
"""


def test_reference_imports_without_pedsum_numba_or_polars():
    run = subprocess.run(
        [sys.executable, "-c", _ISOLATED.format(root=str(ROOT))], capture_output=True, text=True, check=True
    )
    assert 0.2 < float(run.stdout) < 0.7


def _crude(g, mother_trait, father_trait):
    """Reference crude estimates of one unstratified cell, by estimator name."""
    mothers, fathers = mating_pairs(g)
    m, f = g.values[mother_trait][mothers], g.values[father_trait][fathers]
    keep = ~np.isnan(m) & ~np.isnan(f)
    m, f = m[keep], f[keep]
    zeros = np.zeros(m.size, dtype=np.int64)
    kinds = g.kinds[mother_trait], g.kinds[father_trait]
    n_levels = g.n_levels[mother_trait], g.n_levels[father_trait]
    levels = [None if k is None else ref.shown_levels(v, zeros, k) for k, v in zip(n_levels, (m, f), strict=True)]
    pairs = ref.CellPairs(m, f, zeros, zeros, *levels)
    match kinds:
        case ("continuous", "continuous"):
            return {"pearson": ref.pearson(m, f), "spearman": ref.spearman(m, f)}
        case ("binary", "binary"):
            return {"tetrachoric": ref.polychoric(pairs), "odds_ratio": ref.odds_ratio(pairs), "phi": ref.pearson(m, f)}
        case ("continuous", _):
            name = "biserial" if kinds[1] == "binary" else "polyserial"
            out = {name: ref.polyserial(m, f, zeros, zeros, levels[1])}
        case (_, "continuous"):
            name = "biserial" if kinds[0] == "binary" else "polyserial"
            out = {name: ref.polyserial(f, m, zeros, zeros, levels[0])}
        case _:
            return {"polychoric": ref.polychoric(pairs)}
    if "binary" in kinds:
        out["point_biserial"] = ref.pearson(m, f)
    return out


def _value(estimate):
    return estimate.value if isinstance(estimate, ref.Fit) else estimate


@pytest.mark.parametrize("name", [n for n in NAMES if not load(n).stratified])
def test_reference_crude_estimates_match_pedsum(name):
    g = load(name)
    n_traits = len(g.kinds)
    cells = [(i, j) for i in range(n_traits) for j in range(n_traits)]
    for (i, j), record in zip(cells, g.payload["mate_correlation"], strict=True):
        for est_name, got in _crude(g, i, j).items():
            want = record["crude"][est_name]
            key = next(k for k in ("r", "rho", "value") if k in want)
            if isinstance(got, ref.Undefined):
                assert want[key] is None, (name, i, j, est_name)
                assert want["reason"] == got.reason, (name, i, j, est_name)
                continue
            assert want[key] is not None, (name, i, j, est_name, want)
            if math.isinf(want[key]):
                assert _value(got) == want[key]
            else:
                # Brent (reference) and Newton (pedsum) stop at different points within xatol.
                assert _value(got) == pytest.approx(want[key], abs=2e-6), (name, i, j, est_name)
            if isinstance(got, ref.Fit):
                assert got.boundary == want.get("boundary"), (name, i, j, est_name)
