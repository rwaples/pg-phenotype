"""Write the SciPy and pedsum primitive values the Rust assortative-mating port is measured against.

Runs in the pinned pedsum snapshot's environment (see ``make_pedsum_am_golden.py``)::

    pixi run --manifest-path "$S/pixi.toml" python -I tools/make_pedsum_am_primitives.py \\
        --out tests/golden/pedsum_am_<sha12>/primitives.txt

One line per value: a function name, then its arguments and result as the
hex of their IEEE-754 bits, so the Rust test reads them back exactly.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
from pedsum import assortative_kernels as kernels
from pedsum.assortative_mating import bvn_cdf, bvn_pdf_and_drho
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, owens_t


def bits(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    out = p.parse_args().out
    lines = []

    def emit(name, *values):
        lines.append(" ".join([name, *(bits(v) for v in values)]))

    hs = [*np.linspace(-7, 7, 57), 0.0, 1e-9, -1e-9, 0.019, 0.61, 2.35, 4.9, 9.0, np.inf, -np.inf]
    a_s = [-np.inf, -40.0, -3.0, -1.2, -1.0, -0.999, -0.5, -0.01, 0.0, 0.02, 0.1, 0.3, 0.4, 0.7, 0.95,
           0.99999, 1.0, 1.0001, 1.5, 2.9, 7.0, 50.0, np.inf]  # fmt: skip
    for h in hs:
        for a in a_s:
            emit("owens_t", h, a, owens_t(h, a))
    for x in [*np.linspace(-40, 40, 321), -1.0, 1.0, -0.7071, 0.7072, np.inf, -np.inf]:
        emit("ndtr", x, ndtr(x))
        emit("kernel_ndtr", x, kernels.ndtr(x))
    for q in [*np.linspace(0, 1, 201), 1e-300, 1e-20, 1e-12, 0.025, 0.075, 0.425, 0.575, 0.925, 0.975, 1 - 1e-12]:
        emit("ndtri", q, kernels.ndtri(q))
    grid = [-np.inf, -3.1, -1.0, -0.2, 0.0, 0.0001, 0.3, 1.7, 4.0, np.inf]
    for rho in (-0.9999, -0.7, -0.05, 0.0, 0.3, 0.86, 0.9999):
        for h in grid:
            for k in grid:
                c = bvn_cdf(np.array([h]), np.array([k]), rho)[0]
                d, r = bvn_pdf_and_drho(np.array([h]), np.array([k]), rho)
                emit("bvn", h, k, rho, c, d[0], r[0])
    for c, amp in ((0.3, 0.1), (-0.95, 0.5), (0.9995, 0.0), (0.0, 2.0)):
        res = minimize_scalar(
            lambda x, c=c, amp=amp: (x - c) * (x - c) + amp * math.sin(5 * x),
            bounds=(-0.9999, 0.9999),
            method="bounded",
            options={"xatol": 1e-7},
        )
        emit("brent", c, amp, res.x, res.fun, res.nfev)
    out.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
