"""The calibration tools' shared driver: their arguments, the reps they run and the worker pool.

The workers are the parallelism, so every thread pool a worker could start is
pinned to one thread (:data:`THREAD_VARIABLES`, checked before any rep runs)
and the workers are spawned, not forked, so none inherits a parent's pools.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

THREAD_VARIABLES = (
    "PG_PHENOTYPE_THREADS",
    "RAYON_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def parser(
    doc: str, *, jobs: int | None, reps_help: str = "also write the per-rep rows here"
) -> argparse.ArgumentParser:
    """The arguments every calibration tool takes; a tool adds its own."""
    p = argparse.ArgumentParser(description=doc.splitlines()[0])
    p.add_argument("results", nargs="?", type=Path, help="simACE results folder, e.g. results/se_calibration")
    p.add_argument("--out", type=Path)
    p.add_argument("--jobs", type=int, default=jobs)
    p.add_argument("--max-reps", type=int)
    p.add_argument("--reps", type=Path, help=reps_help)
    p.add_argument("--from-reps", type=Path, help="summarise these per-rep rows instead of computing them")
    p.add_argument("--self-test", action="store_true")
    return p


def check_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Exit unless ``--out`` and exactly one of results or ``--from-reps`` are given and, with results, the
    thread variables are pinned."""
    if args.out is None or (args.results is None) == (args.from_reps is None):
        p.error("--out and one of results or --from-reps are required")
    unpinned = [v for v in THREAD_VARIABLES if os.environ.get(v) != "1"]
    if args.results is not None and unpinned:
        p.error(f"set {', '.join(unpinned)} to 1: the workers are the parallelism")


def threads() -> dict[str, str]:
    """The pinned thread variables, for a run record."""
    return {v: os.environ[v] for v in THREAD_VARIABLES}


def rep_dirs(results: Path, max_reps: int | None) -> list[tuple[Path, bool]]:
    """Each scenario's ``rep<N>`` folders holding a ``cohort.parquet``, in rep order, the first *max_reps*,
    each with whether it is its scenario's first."""
    work = []
    for scen in sorted(p for p in results.iterdir() if p.is_dir()):
        numbered = ((int(m[1]), p) for p in scen.iterdir() if (m := re.fullmatch(r"rep(\d+)", p.name)))
        reps = [p for _, p in sorted(numbered) if (p / "cohort.parquet").exists()]
        work += [(p, i == 0) for i, p in enumerate(reps[:max_reps])]
    return work


def run[T](job: Callable[..., T], work: list[tuple], jobs: int) -> Iterator[T]:
    """``job(*w)`` for each ``w`` of *work* on *jobs* spawned workers, in order, with progress every 50."""
    start = time.perf_counter()
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for done, out in enumerate(pool.map(job, *zip(*work, strict=True)), 1):
            yield out
            if done % 50 == 0:
                print(f"{done}/{len(work)} reps, {time.perf_counter() - start:.0f} s", file=sys.stderr, flush=True)
