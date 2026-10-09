"""simACE consumer experiment (simACE plan pg-phenotype-pedigree-v2, unit 7, gate 4).

Runs simACE's mate correlations three ways on one rep's ``pedigree.parquet``,
each in a fresh process in its own systemd scope:

- ``today``: simACE's NumPy code, ``compute_mate_correlation`` (stats),
  ``validate_assortative_mating`` and ``observed_mate_correlations`` for
  both traits (validation), with the ``PedigreeArrays`` validation builds.
- ``columns``: simACE #47's three calls with columns: liability for stats,
  then liability and A for validation.
- ``pedigree``: the same three calls on one ``Pedigree``.

Every estimate must match ``today`` within 1e-12 and every pair count
exactly (pass/fail); wall time and peak memory are reported, not gated.
``r_ho`` is reported but not gated: simACE sums its Float32 ``A + C + E``,
while the three calls read the Float64 ``liability`` column.
Run from the simACE root, in simACE's env, with the build under test first
on ``PYTHONPATH`` (simACE does not depend on pg-phenotype)::

    PYTHONPATH=<site> pixi run python -P external/pg-phenotype/tools/consumer_simace.py bench \\
        --rep results/test/coverage_scenario/rep1 --repeats 5 --out <jsonl>
    pixi run python -P external/pg-phenotype/tools/consumer_simace.py summary <jsonl>...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve()
PATHS = ("today", "columns", "pedigree")
TOLERANCE = 1e-12
#: Estimates the planned path takes from another input than ``today`` (see the docstring).
OTHER_INPUT = ("r_ho1", "r_ho2")


def cgroup_peak() -> int | None:
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        return int(Path(f"/sys/fs/cgroup{rel}/memory.peak").read_text())
    except (OSError, IndexError, ValueError):
        return None


def today(df: object, params: dict) -> dict:
    from simace.analysis.stats.correlations import compute_mate_correlation
    from simace.analysis.validate.am_relatedness import observed_mate_correlations
    from simace.analysis.validate.assortative_mating import validate_assortative_mating
    from simace.core.pedigree_arrays import PedigreeArrays

    stats = compute_mate_correlation(df)
    ped = PedigreeArrays.from_frame(df)
    checks = validate_assortative_mating(df, params, ped)
    observed = [observed_mate_correlations(df, ped, t) for t in (1, 2)]
    out = {
        "stats_n_pairs": stats["n_pairs"],
        **{f"stats_{i}{j}": stats["matrix"][i][j] for i in range(2) for j in range(2)},
        **{f"liability{t}": checks[f"mate_corr_liability{t}"]["observed"] for t in (1, 2)},
        **{f"A{t}": checks[f"mate_corr_A{t}"]["observed"] for t in (1, 2)},
        "validation_n_pairs": checks["mate_corr_liability1"]["n_pairs"],
        **{f"mu_A{t}": observed[t - 1][0] for t in (1, 2)},
        **{f"r_ho{t}": observed[t - 1][1] for t in (1, 2)},
        **{f"observed_n_pairs{t}": observed[t - 1][2] for t in (1, 2)},
    }
    for label in ("cross_12", "cross_21"):
        if f"mate_corr_{label}" in checks:
            out[label] = checks[f"mate_corr_{label}"]["observed"]
    return out


def planned(df: object, built: bool) -> dict:
    from pg_phenotype import Pedigree, Trait
    from pg_phenotype.assortative import mate_correlation

    pedigree = Pedigree(df) if built else df

    def call(prefix: str) -> object:
        traits = [Trait(df[f"{prefix}{t}"], kind="continuous") for t in (1, 2)]
        return mate_correlation(pedigree, traits, permutations=0)

    def r(res: object, i: int, j: int) -> float:
        return res.cell(i, j).primary.value

    stats, liability, a = call("liability"), call("liability"), call("A")
    return {
        "stats_n_pairs": stats.sample.n_total,
        **{f"stats_{i}{j}": r(stats, i, j) for i in range(2) for j in range(2)},
        **{f"liability{t}": r(liability, t - 1, t - 1) for t in (1, 2)},
        **{f"A{t}": r(a, t - 1, t - 1) for t in (1, 2)},
        "validation_n_pairs": liability.sample.n_total,
        **{f"mu_A{t}": r(a, t - 1, t - 1) for t in (1, 2)},
        **{f"r_ho{t}": r(liability, t - 1, t - 1) for t in (1, 2)},
        **{f"observed_n_pairs{t}": a.sample.n_total for t in (1, 2)},
        "cross_12": r(liability, 0, 1),
        "cross_21": r(liability, 1, 0),
    }


def run(args: argparse.Namespace) -> None:
    # Every path's modules load before the timer, so each run pays the same imports.
    import polars as pl
    import simace.analysis.stats.correlations
    import simace.analysis.validate.am_relatedness
    import simace.analysis.validate.assortative_mating  # noqa: F401
    import yaml

    import pg_phenotype.assortative

    # Read without simACE's layout check: the experiment's reps may predate it.
    df = pl.read_parquet(args.rep / "pedigree.parquet")
    params = yaml.safe_load((args.rep / "params.yaml").read_text())
    start = time.perf_counter()
    estimates = today(df, params) if args.path == "today" else planned(df, built=args.path == "pedigree")
    wall = time.perf_counter() - start
    record = {
        "native": pg_phenotype._native.__file__,
        "wall": wall,
        "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_peak": cgroup_peak(),
        "estimates": estimates,
    }
    print(json.dumps(record))


def one_run(args: argparse.Namespace, path: str) -> dict:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "POLARS_MAX_THREADS": "1", "PG_PHENOTYPE_THREADS": "1"}
    cmd = [
        "systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=12G", "-p", "MemorySwapMax=0",
        args.pixi, "run", "python", "-P", str(HERE), "run", "--rep", str(args.rep), "--path", path,
    ]  # fmt: skip
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if out.returncode:
        raise SystemExit(f"{path} failed:\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def compare(want: dict, got: dict) -> list[str]:
    """Each estimate of *want* that *got* misses or differs on, as text."""
    problems = []
    for key, value in want.items():
        if key in OTHER_INPUT:
            continue
        other = got.get(key)
        if "n_pairs" in key:
            if other != value:
                problems.append(f"{key}: {value} vs {other}")
        elif value is None or other is None or not math.isfinite(value) or abs(value - other) > TOLERANCE:
            problems.append(f"{key}: {value!r} vs {other!r}")
    return problems


def bench(args: argparse.Namespace) -> None:
    for i in range(args.repeats):
        order = PATHS if i % 2 == 0 else PATHS[::-1]
        runs = {path: one_run(args, path) for path in order}
        problems = {path: compare(runs["today"]["estimates"], runs[path]["estimates"]) for path in PATHS[1:]}
        record = {
            "rep": str(args.rep),
            "repeat": i,
            "load1": os.getloadavg()[0],
            "problems": problems,
            **{
                f"max_abs_diff_{name}": {
                    path: max(
                        abs(v - runs[path]["estimates"][k])
                        for k, v in runs["today"]["estimates"].items()
                        if "n_pairs" not in k and (k in OTHER_INPUT) == (name == "r_ho")
                    )
                    for path in PATHS[1:]
                }
                for name in ("gated", "r_ho")
            },
            "n_pairs": runs["today"]["estimates"]["validation_n_pairs"],
            **{f"{p}_{k}": runs[p][k] for p in PATHS for k in ("wall", "ru_maxrss_kib", "memory_peak")},
        }
        with args.out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if any(problems.values()):
            raise SystemExit(f"estimates differ: {problems}")


def summary(args: argparse.Namespace) -> None:
    print("| Rep | Pairs | Path | Wall (ms) | memory.peak (MiB) | ru_maxrss (MiB) | Max abs diff | r_ho diff | Equal |")
    print("|---|---|---|---|---|---|---|---|---|")
    for jsonl in args.jsonl:
        records = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        rep = records[0]["rep"].split("results/", 1)[-1]
        for path in PATHS:
            wall = statistics.median(r[f"{path}_wall"] for r in records) * 1e3
            peaks = [r[f"{path}_memory_peak"] for r in records]
            # memory.peak is None for a run outside a cgroup v2 scope.
            peak = "n/a" if None in peaks else f"{statistics.median(peaks) / 2**20:.0f}"
            rss = statistics.median(r[f"{path}_ru_maxrss_kib"] for r in records) / 1024
            diff, r_ho = (
                ("", "")
                if path == "today"
                else (f"{max(r[f'max_abs_diff_{n}'][path] for r in records):.1e}" for n in ("gated", "r_ho"))
            )
            equal = "" if path == "today" else ("yes" if not any(r["problems"][path] for r in records) else "NO")
            print(
                f"| {rep} | {records[0]['n_pairs']:,} | {path} | {wall:.0f} | {peak} | {rss:.0f} | {diff} | {r_ho} | {equal} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--rep", type=Path, required=True)
    p.add_argument("--path", choices=PATHS, required=True)
    p = sub.add_parser("bench")
    p.add_argument("--rep", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--pixi", default="pixi", help="the pixi executable (absolute inside a systemd scope)")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("summary")
    p.add_argument("jsonl", type=Path, nargs="+")
    args = parser.parse_args()
    {"run": run, "bench": bench, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
