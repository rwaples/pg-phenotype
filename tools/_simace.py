"""The simACE checkout that the ``make_simace_*`` goldens and ``consumer_simace.py`` import (#23).

Every ``tests/golden/simace_*`` golden was written from simACE ``SIMACE_REV``.
Later simACE computes these statistics with pg-phenotype itself (mate
correlations from ec3d0a3, parent-offspring regressions from c655f40,
tetrachoric correlations), so a later checkout would compare pg-phenotype
with itself.  The tools refuse any checkout but a clean one at the pin:

    git -C <simACE> worktree add --detach <dir> 6355ffa6aebe

They run that checkout's source in the current simACE env, not in the env
locked at the pin (which needs pixi 0.76).  A regenerated golden must match
the committed one in every value; with the pin unchanged, a difference is env
drift.  Only recorded metadata may differ (the tetrachoric golden's NumPy
version, the relative-pair reps' ``run.yaml`` provenance), as on 2026-10-30.
``import_simace`` logs the env's versions to stderr for that check.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

SIMACE_REV = "6355ffa6aebe37238cb2724dfcde288830fdcb6c"
ENV_PACKAGES = ("numpy", "scipy", "polars", "numba", "pedigree-graph", "pg-phenotype")


def check_checkout(checkout: Path) -> None:
    """Exit unless *checkout* is a git checkout at ``SIMACE_REV`` with no changes under ``simace/``."""
    git = ["git", "-C", str(checkout)]
    try:
        rev = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            [*git, "status", "--porcelain", "--", "simace"], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError as e:
        sys.exit(f"{checkout} is not a git checkout: {e.stderr.strip()}")
    if rev != SIMACE_REV:
        sys.exit(f"{checkout} is at {rev}, not SIMACE_REV {SIMACE_REV}")
    if dirty:
        sys.exit(f"{checkout}/simace has uncommitted changes:\n{dirty}")


def import_simace(checkout: Path) -> None:
    """Check *checkout* (:func:`check_checkout`), then import ``simace`` from it ahead of the env's own."""
    checkout = checkout.resolve()
    check_checkout(checkout)
    sys.path.insert(0, str(checkout))
    import simace

    if not Path(simace.__file__).resolve().is_relative_to(checkout):
        sys.exit(f"imported simace from {simace.__file__}, not {checkout}")
    versions = []
    for name in ENV_PACKAGES:
        try:
            versions.append(f"{name} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{name} -")
    print(f"simACE {SIMACE_REV[:12]} in env: {', '.join(versions)}", file=sys.stderr)
