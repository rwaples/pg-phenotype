"""Write simACE's tetrachoric results over the gate's table grid (issue #2).

Runs in simACE's pixi env against a clean simACE checkout, whose
``simace`` it imports ahead of the env's own install:

    pixi run --manifest-path <simACE>/pixi.toml python -B tools/make_simace_tetrachoric_golden.py \\
        --simace <simACE checkout> --out tests/golden/simace_tetrachoric_<sha12>

The grid (``tables.json``):

- ``small``: every table with ``n <= 10`` (1001 tables, empty and constant
  ones included).
- ``expected``: the rounded expected table of ``n`` pairs from a bivariate
  normal with prevalences ``p_x``, ``p_y`` and correlation ``rho``, over
  ``n`` in 50 .. 10^6, prevalences 0.001 .. 0.5 and ``rho`` in -0.95 .. 0.999.
- ``sampled``: one multinomial draw of each ``expected`` setting, seed 2.
- ``one_empty``: one empty cell, each of the four, at ``n`` in 20 .. 10^4.

``simace.json`` holds ``tetrachoric_from_table``'s ``(r, se)`` per table
(JIT on, ``null`` for NaN) and the checkout's revision.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal, norm

NS = (50, 200, 1000, 10_000, 100_000, 1_000_000)
PREVALENCES = (0.001, 0.01, 0.05, 0.2, 0.5)
RHOS = (-0.95, -0.5, 0.0, 0.3, 0.7, 0.95, 0.99, 0.999)


def small() -> list[list[list[int]]]:
    out = []
    for n in range(11):
        for a, b, c in itertools.product(range(n + 1), repeat=3):
            if a + b + c <= n:
                out.append([[a, b], [c, n - a - b - c]])
    return out


def cell_probabilities(p_x: float, p_y: float, rho: float) -> np.ndarray:
    """P(x = i, y = j), level 1 the upper tail of prevalence ``p``."""
    h, k = norm.ppf(1 - p_x), norm.ppf(1 - p_y)
    p00 = float(multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([h, k]))
    p01 = (1 - p_x) - p00
    p10 = (1 - p_y) - p00
    return np.clip(np.array([p00, p01, p10, 1 - p00 - p01 - p10]), 0, 1)


def expected_and_sampled() -> tuple[list, list, list]:
    rng = np.random.default_rng(2)
    expected, sampled, settings = [], [], []
    for n, p_x, p_y, rho in itertools.product(NS, PREVALENCES, PREVALENCES, RHOS):
        if p_y < p_x:
            continue
        p = cell_probabilities(p_x, p_y, rho)
        counts = np.rint(p * n).astype(int)
        counts[0] += n - counts.sum()
        expected.append([[int(counts[0]), int(counts[1])], [int(counts[2]), int(counts[3])]])
        draw = rng.multinomial(n, p / p.sum())
        sampled.append([[int(draw[0]), int(draw[1])], [int(draw[2]), int(draw[3])]])
        settings.append({"n": n, "p_x": p_x, "p_y": p_y, "rho": rho})
    return expected, sampled, settings


def one_empty() -> list[list[list[int]]]:
    out = []
    for n in (20, 100, 1000, 10_000):
        base = [n // 2, n // 5, n // 10]
        for empty in range(4):
            cells = [*base[:empty], 0, *base[empty:]]
            cells[(empty + 3) % 4] += n - sum(cells)
            out.append([[cells[0], cells[1]], [cells[2], cells[3]]])
    return out


def revision(checkout: Path) -> str:
    dirty = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--", "simace"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if dirty:
        sys.exit(f"{checkout}/simace has uncommitted changes:\n{dirty}")
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def finite_or_none(v: float) -> float | None:
    return None if math.isnan(v) else float(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--simace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    checkout = args.simace.resolve()
    rev = revision(checkout)
    sys.path.insert(0, str(checkout))
    import simace
    from simace.analysis.stats.tetrachoric import tetrachoric_from_table

    if not Path(simace.__file__).resolve().is_relative_to(checkout):
        sys.exit(f"imported simace from {simace.__file__}, not {checkout}")
    logging.disable(logging.WARNING)  # the n < 50 warning

    expected, sampled, settings = expected_and_sampled()
    sets = {"small": small(), "expected": expected, "sampled": sampled, "one_empty": one_empty()}
    results = {}
    for name, tables in sets.items():
        out = []
        for (n00, n01), (n10, n11) in tables:
            r, se = tetrachoric_from_table(n11, n10, n01, n00)
            out.append([finite_or_none(r), finite_or_none(se)])
        results[name] = out
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables.json").write_text(json.dumps({"sets": sets, "expected_settings": settings}) + "\n")
    meta = {
        "simace_rev": rev,
        "function": "simace.analysis.stats.tetrachoric.tetrachoric_from_table(n11, n10, n01, n00)",
        "numpy": np.__version__,
    }
    (args.out / "simace.json").write_text(json.dumps({"meta": meta, "results": results}) + "\n")
    print(f"{sum(map(len, sets.values()))} tables at simACE {rev[:12]} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
