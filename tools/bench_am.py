"""Benchmark the assortative-mating compute stage: pedsum #13 vs pg-phenotype (plan v2, unit 12).

Inputs come from pedsum's own generator (configurations ``a``: one binary
trait, unstratified; ``b``: continuous and binary, birth-year strata), written
once per size by the snapshot env::

    S=<scratch>/pedsum_am_<sha12>
    pixi run --manifest-path $S/pixi.toml python -I tools/bench_am.py gen --snapshot $S --pairs 100000 --out <dir>

``bench`` (in pg-phenotype's env) runs interleaved ABAB pairs, each run a
fresh process in its own systemd scope: wall from ``perf_counter`` around the
one compute call, memory from the scope's cgroup ``memory.peak`` and the
process's ``ru_maxrss``.  pedsum gets one untimed warm call first (numba cache
primed); cold start is not gated.  Every pair asserts equal ``draws_used`` per
cell, so both did the same work.  Records are appended as JSON lines::

    pixi run python tools/bench_am.py bench --snapshot $S --data <dir> --pairs-per-config 10 \\
        --config b --size 100000 --threads 6 --permutations 999 --bootstrap 1000 --out <jsonl>
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()


def data_path(directory: Path, config: str, size: int) -> Path:
    return directory / f"am_{config}_{size}.npz"


def gen(args: argparse.Namespace) -> None:
    import numpy as np

    sys.path.insert(0, str(args.snapshot / "benchmarks"))
    from generate_assortative_mating import generate

    df, _ = generate(
        pairs=args.pairs, seed=args.seed, r_mf=np.array([[0.30, 0.15], [0.05, 0.25]]), within_m=0.4,
        within_f=0.3, remate_fathers=0.2, remate_mothers=0.1, prevalence=0.1, year_min=1900, year_max=1980,
    )  # fmt: skip

    def column(name: str) -> np.ndarray:
        values = df[name].to_numpy()
        return np.where(values == "NA", "nan", values).astype(np.float64)

    args.out.mkdir(parents=True, exist_ok=True)
    year = df["birth_year"].to_numpy().astype(np.int64)
    for config in ("a", "b"):
        np.savez(
            data_path(args.out, config, args.pairs),
            id=df["id"].to_numpy(),
            mother=df["mother"].to_numpy(),
            father=df["father"].to_numpy(),
            liab=column("liab"),
            dx=column("dx"),
            stratum=year // 10 * 10,
        )


def cgroup_peak() -> int | None:
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        return int(Path(f"/sys/fs/cgroup{rel}/memory.peak").read_text())
    except (OSError, IndexError, ValueError):
        return None


def run_pedsum(args: argparse.Namespace) -> None:
    import numpy as np
    import polars as pl
    from pedsum.assortative_mating import Trait, compute_assortative_mating

    kinds = [("dx", "binary")] if args.config == "a" else [("liab", "continuous"), ("dx", "binary")]
    options = {"stratify_by": "birth_year", "birth_year_bin": 10} if args.config == "b" else {}

    def inputs(size: int) -> tuple:
        d = np.load(data_path(args.data, args.config, size))
        traits = [Trait(n, k, "stated", None if k == "continuous" else ("0", "1"), d[n]) for n, k in kinds]
        df = pl.DataFrame({"id": d["id"], "mother": d["mother"], "father": d["father"], "birth_year": d["stratum"]})
        return df, traits

    def call(df: pl.DataFrame, traits: list) -> dict:
        return compute_assortative_mating(
            df, traits, permutations=args.permutations, bootstrap=args.bootstrap, seed=0, **options
        )

    # One untimed warm call on the 10^4 input of the same configuration: the same
    # numba specialisations, compiled (or loaded from the cache) before timing.
    call(*inputs(10_000))
    df, traits = inputs(args.size)
    start = time.perf_counter()
    out = call(df, traits)
    wall = time.perf_counter() - start
    used = [
        e["permutations"]["draws_used"]
        for cell in out["mate_correlation"]
        for form in ("crude", "stratified")
        for e in cell.get(form, {}).values()
        if isinstance(e, dict) and "permutations" in e
    ]
    report(wall, used)


def run_pg(args: argparse.Namespace) -> None:
    import numpy as np

    from pg_phenotype import Trait
    from pg_phenotype.assortative import mate_correlation

    d = np.load(data_path(args.data, args.config, args.size))
    names = ["dx"] if args.config == "a" else ["liab", "dx"]
    traits = [Trait(d[n], kind="binary" if n == "dx" else "continuous") for n in names]
    ped = {"id": d["id"], "mother": d["mother"], "father": d["father"]}
    stratum = d["stratum"] if args.config == "b" else None
    start = time.perf_counter()
    out = mate_correlation(
        ped, traits, stratum=stratum, permutations=args.permutations, bootstrap=args.bootstrap, seed=0
    )
    wall = time.perf_counter() - start
    used = [
        r.permutation.draws_used
        for cell in out.cells
        for r in (cell.crude[0], cell.stratified.result if cell.stratified else None)
        if r is not None and r.permutation is not None
    ]
    report(wall, used)


def report(wall: float, used: list[int]) -> None:
    print(
        json.dumps(
            {
                "wall": wall,
                "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "memory_peak": cgroup_peak(),
                "draws_used": used,
            }
        )
    )


def one_run(args: argparse.Namespace, side: str) -> dict:
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": str(args.threads),
        "NUMBA_NUM_THREADS": str(args.threads),
        "PG_PHENOTYPE_THREADS": str(args.threads),
    }
    common = [
        "--data", str(args.data), "--config", args.config, "--size", str(args.size),
        "--permutations", str(args.permutations), "--bootstrap", str(args.bootstrap),
    ]  # fmt: skip
    if side == "pedsum":
        python = ["pixi", "run", "--manifest-path", str(args.snapshot / "pixi.toml"), "python", "-I"]
    else:
        python = ["pixi", "run", "--manifest-path", str(HERE.parent.parent / "pixi.toml"), "python", "-I"]
    cmd = ["systemd-run", "--user", "--scope", "-q", *python, str(HERE), f"run-{side}", *common]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def bench(args: argparse.Namespace) -> None:
    for i in range(args.pairs_per_config):
        order = ("pedsum", "pg") if i % 2 == 0 else ("pg", "pedsum")
        runs = {side: one_run(args, side) for side in order}
        if runs["pedsum"]["draws_used"] != runs["pg"]["draws_used"]:
            raise SystemExit(f"draws_used differ: {runs}")
        record = {
            "config": args.config,
            "size": args.size,
            "threads": args.threads,
            "permutations": args.permutations,
            "bootstrap": args.bootstrap,
            "pair": i,
            "first": order[0],
            "load1": os.getloadavg()[0],
            **{f"{side}_{k}": v for side, r in runs.items() for k, v in r.items() if k != "draws_used"},
            "draws_used": runs["pg"]["draws_used"],
        }
        with args.out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--snapshot", type=Path, required=True)
    g.add_argument("--pairs", type=int, required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", type=Path, required=True)
    for name in ("run-pedsum", "run-pg", "bench"):
        r = sub.add_parser(name)
        r.add_argument("--data", type=Path, required=True)
        r.add_argument("--config", choices=("a", "b"), required=True)
        r.add_argument("--size", type=int, required=True)
        r.add_argument("--permutations", type=int, default=999)
        r.add_argument("--bootstrap", type=int, default=1000)
        if name == "bench":
            r.add_argument("--snapshot", type=Path, required=True)
            r.add_argument("--threads", type=int, default=1)
            r.add_argument("--pairs-per-config", type=int, default=10)
            r.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    {"gen": gen, "run-pedsum": run_pedsum, "run-pg": run_pg, "bench": bench}[args.cmd](args)


if __name__ == "__main__":
    main()
