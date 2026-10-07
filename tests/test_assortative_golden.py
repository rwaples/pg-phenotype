"""pg-phenotype's Mate Correlation against pedsum #13's on every golden fixture (plan v2 parity contract).

Every count, reason, flag, estimator, order and presence must match exactly,
and so must every p-value.  The floats meet D3, set from the unit 9 gate
(``docs/gates/assortative-mating``):

- closed-form estimators and the sample share: ``1e-12 * max(1, |pedsum|)``
  (largest gap over 74 cases 4.5e-15, an odds-ratio SE);
- latent estimators (tetrachoric, polychoric, biserial, polyserial): 1e-7, the
  resolution of the bounded-Brent fallback (``xatol``).  Brent's path follows
  exact NLL comparisons, and pedsum sums the NLL with NumPy's BLAS ``dot``;
  the largest gap was 3.2e-10, on a polychoric fit that fell back to Brent.
"""

import os
import subprocess
import sys

import pytest

from tests.assortative_golden import GATE_NAMES, NAMES
from tests.assortative_parity import diff

CLOSED_TOL = 1e-12
LATENT_TOL = 1e-7
LATENT = ("tetrachoric", "polychoric", "biserial", "polyserial")


def tolerance(path: str, want: float) -> float:
    if path.endswith("p_perm"):
        return 0.0
    if any(f".{name}." in path for name in LATENT) or path.endswith(".rho"):
        return LATENT_TOL
    return CLOSED_TOL * max(1.0, abs(want))


# Where pg-phenotype deliberately departs from pedsum: path -> what pedsum computes once fixed.
#
# pedsum's polychoric sandwich sums count * (d2/pi - score^2) over every cell of a populated stratum
# table, empty cells included.  In small_strat, father stratum 9 has two equal thresholds (an empty
# level), so an empty cell's phi2 corner difference is cancellation noise (6.9e-18 at pedsum's rho-hat);
# noise / 1e-300 squared overflows, 0 * inf is NaN, and pedsum withholds the SE.  pg-phenotype skips
# empty cells.  The value is pedsum's own with the same fix (its polychoric_influence and
# _threshold_cross with zero-count cells masked, run in the pinned snapshot).
DEVIATIONS = {
    ("small_strat", "mate_correlation[3].stratified.polychoric.se"): 0.10541470792145655,
}


def test_deviations_equal_pedsum_with_the_fix():
    for (name, path), fixed in DEVIATIONS.items():
        mismatches, _ = diff(name)
        hit = [m for m in mismatches if m.split(":")[0] == path]
        assert len(hit) == 1, (name, path, mismatches)
        ours = float(hit[0].split("ours ")[1].split(" pedsum")[0])
        assert ours == pytest.approx(fixed, abs=CLOSED_TOL), (name, path, ours)


@pytest.mark.parametrize("name", NAMES + GATE_NAMES)
def test_matches_pedsum(name):
    mismatches, gaps = diff(name)
    mismatches = [m for m in mismatches if (name, m.split(":")[0]) not in DEVIATIONS]
    assert not mismatches, "\n".join(mismatches)
    over = [g for g in gaps if g[3] > tolerance(g[0], g[2])]
    assert not over, "\n".join(f"{p}: ours {a!r} pedsum {b!r}" for p, a, b, _ in over)


_RUN = """
import sys
sys.path.insert(0, {root!r})
from tests.assortative_golden import GATE_NAMES, NAMES
from tests.assortative_parity import run, load
print(repr([run(load(n), threads={threads}) for n in NAMES if n in {names!r}]))
"""


def test_results_do_not_depend_on_the_thread_count():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = ["cont_bin_boot", "bin_ord_depth_boot", "ord_cont_birthyear_boot", "strong_assort"]
    out = [
        subprocess.run(
            [sys.executable, "-c", _RUN.format(root=root, threads=t, names=names)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.replace(f"'threads': {t}", "'threads': T")
        for t in (1, 4)
    ]
    assert out[0] == out[1]
