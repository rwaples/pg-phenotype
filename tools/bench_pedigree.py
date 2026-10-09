"""Benchmark the Pedigree against the columns path (simACE plan pg-phenotype-pedigree-v2, unit 7).

Two builds of the package are compared, each unpacked from its wheel into a
directory (``--site``): A, the base without ``Pedigree``, and B, the change.
``gen`` writes one simACE pedigree as an ``.npz`` once, in simACE's env::

    pixi run python tools/bench_pedigree.py gen --pedigree <rep>/pedigree.parquet --out <npz>

``bench`` (in pg-phenotype's env) runs interleaved ABAB pairs of every mode,
each run a fresh process in its own systemd scope: wall from
``perf_counter`` around each public call, memory from the scope's cgroup
``memory.peak`` and the process's ``ru_maxrss``.  Two continuous traits
(``liability1``, ``liability2``), ``permutations=0``.  Modes:

- ``cols``: ``T_cols``, ``mate_correlation`` on the columns.
- ``strat_unknown``: the same with ``stratum`` = generation, about 10% of
  rows unknown, so pairs are dropped.
- ``strat_known``: the same with every stratum known.
- ``prep``: ``T_prep``, ``pafgrs.prepare(columns, ndegree=2)``.
- ``ped`` (B only): ``T_ctor`` = ``Pedigree(columns)``, then ``T_first`` and
  ``T_warm``, two ``mate_correlation`` calls on it.

::

    pixi run python tools/bench_pedigree.py bench --site-a <dirA> --site-b <dirB> --data <npz> \\
        --pairs 10 --modes cols strat_unknown strat_known prep ped --out <jsonl>

``summary`` prints the medians and the gates of one or more jsonl files as Markdown::

    pixi run python tools/bench_pedigree.py summary <jsonl>...
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
PIXI_TOML = HERE.parent.parent / "pixi.toml"
MODES = ("cols", "strat_unknown", "strat_known", "prep", "ped")
#: Gate 2's floor on ``T_cols - T_warm``, ms, by row count: 80% of validation plus pairs.
WARM_SAVING_MS = {1_000_000: 170.0, 5_364_404: 1393.0}


def gen(args: argparse.Namespace) -> None:
    import numpy as np
    import polars as pl

    df = pl.read_parquet(args.pedigree)
    cols = ("id", "mother", "father", "twin", "sex", "generation")
    out = {c: df[c].fill_null(-1).to_numpy().astype(np.int64) for c in cols}
    out |= {c: df[c].to_numpy().astype(np.float64) for c in ("liability1", "liability2")}
    np.savez(args.out, **out)


def cgroup_peak() -> int | None:
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        return int(Path(f"/sys/fs/cgroup{rel}/memory.peak").read_text())
    except (OSError, IndexError, ValueError):
        return None


def run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.site))
    import numpy as np

    import pg_phenotype
    from pg_phenotype import Trait, pafgrs
    from pg_phenotype.assortative import mate_correlation

    d = np.load(args.data)
    ped = {c: d[c] for c in ("id", "mother", "father", "twin", "sex")}
    traits = [Trait(d["liability1"], kind="continuous"), Trait(d["liability2"], kind="continuous")]
    stratum = None
    if args.mode.startswith("strat"):
        stratum = d["generation"].astype(np.float64)
        if args.mode == "strat_unknown":
            stratum[np.random.default_rng(0).random(stratum.size) < 0.1] = np.nan
    walls: dict[str, float] = {}

    def timed(name: str, call: object) -> object:
        start = time.perf_counter()
        out = call()
        walls[name] = time.perf_counter() - start
        return out

    if args.mode == "prep":
        timed("T_prep", lambda: pafgrs.prepare(ped, ndegree=2))
        n_pairs = None
    elif args.mode == "ped":
        built = timed("T_ctor", lambda: pg_phenotype.Pedigree(ped))
        first = timed("T_first", lambda: mate_correlation(built, traits, permutations=0))
        warm = timed("T_warm", lambda: mate_correlation(built, traits, permutations=0))
        # repr, not ==: an undefined estimate is NaN, which equals nothing.
        assert repr(first) == repr(warm)
        n_pairs = first.sample.n_total
    else:
        result = timed("T_cols", lambda: mate_correlation(ped, traits, stratum=stratum, permutations=0))
        n_pairs = result.sample.n_total
    record = {
        "native": pg_phenotype._native.__file__,
        "n_rows": int(d["id"].size),
        "n_pairs": n_pairs,
        "walls": walls,
        "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_peak": cgroup_peak(),
    }
    print(json.dumps(record))


def one_run(args: argparse.Namespace, side: str, mode: str) -> dict:
    site = args.site_a if side == "A" else args.site_b
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PG_PHENOTYPE_THREADS": str(args.threads)}
    python = [args.pixi, "run", "--manifest-path", str(PIXI_TOML), "python", "-I"]
    cmd = [
        "systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=12G", "-p", "MemorySwapMax=0",
        *python, str(HERE), "run",
        "--site", str(site), "--data", str(args.data), "--mode", mode,
    ]  # fmt: skip
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if out.returncode:
        raise SystemExit(f"{side} {mode} failed:\n{out.stderr}")
    record = json.loads(out.stdout.strip().splitlines()[-1])
    if not record["native"].startswith(str(site)):
        raise SystemExit(f"{side} imported {record['native']}, not from {site}")
    return record


def bench(args: argparse.Namespace) -> None:
    for i in range(args.pairs):
        for mode in args.modes:
            sides = ("B",) if mode == "ped" else (("A", "B") if i % 2 == 0 else ("B", "A"))
            runs = {side: one_run(args, side, mode) for side in sides}
            pairs = {r["n_pairs"] for r in runs.values()}
            if len(pairs) != 1:
                raise SystemExit(f"pair counts differ: {runs}")
            record = {
                "mode": mode,
                "pair": i,
                "first": sides[0],
                "threads": args.threads,
                "load1": os.getloadavg()[0],
                "n_rows": runs[sides[0]]["n_rows"],
                "n_pairs": pairs.pop(),
                **{
                    f"{side}_{k}": v
                    for side, r in runs.items()
                    for k, v in (
                        *r["walls"].items(),
                        ("ru_maxrss_kib", r["ru_maxrss_kib"]),
                        ("memory_peak", r["memory_peak"]),
                    )
                },
            }
            with args.out.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


def summary(args: argparse.Namespace) -> None:
    records = [json.loads(line) for path in args.jsonl for line in path.read_text().splitlines() if line.strip()]
    by_mode: dict[str, list[dict]] = {}
    for r in records:
        by_mode.setdefault(r["mode"], []).append(r)
    n_rows = records[0]["n_rows"]

    def med(rows: list[dict], key: str) -> float:
        return statistics.median(r[key] for r in rows)

    def ratio(rows: list[dict], key: str) -> float | None:
        """The median B/A ratio, or None where a run had no reading (memory.peak outside a cgroup scope)."""
        pairs = [(r[f"B_{key}"], r[f"A_{key}"]) for r in rows]
        if any(b is None or a is None for b, a in pairs):
            return None
        return statistics.median(b / a for b, a in pairs)

    n_pairs = next((r["n_pairs"] for r in records if r["n_pairs"] is not None), None)
    print(
        f"{n_rows:,} rows, {n_pairs or 'n/a'} Mating Pairs; interleaved pairs per mode: "
        + ", ".join(f"{mode} {len(rows)}" for mode, rows in by_mode.items())
        + f"; load1 median {statistics.median(r['load1'] for r in records):.2f}\n"
    )
    print("| Mode | A wall (ms) | B wall (ms) | B/A wall | B/A memory.peak | B/A ru_maxrss | Gate 1 |")
    print("|---|---|---|---|---|---|---|")
    verdicts = []
    for mode in ("cols", "strat_unknown", "strat_known", "prep"):
        rows = by_mode.get(mode)
        if not rows:
            continue
        key = "T_prep" if mode == "prep" else "T_cols"
        ratios = [ratio(rows, key), ratio(rows, "memory_peak"), ratio(rows, "ru_maxrss_kib")]
        ok = all(x <= 1.05 for x in ratios if x is not None)
        verdicts.append(ok)
        print(
            f"| {mode} | {med(rows, f'A_{key}') * 1e3:.0f} | {med(rows, f'B_{key}') * 1e3:.0f} | "
            + " | ".join("n/a" if x is None else f"{x:.3f}" for x in ratios)
            + f" | {'pass' if ok else 'FAIL'} |"
        )
    cols, ped = by_mode.get("cols"), by_mode.get("ped")
    if cols and ped:
        t_cols = med(cols, "B_T_cols") * 1e3
        t_ctor, t_first, t_warm = (med(ped, f"B_{k}") * 1e3 for k in ("T_ctor", "T_first", "T_warm"))
        saving = t_cols - t_warm
        floor = WARM_SAVING_MS.get(n_rows)
        gate2 = floor is not None and saving >= floor
        gate3 = (t_ctor + t_first) <= 1.05 * t_cols
        print("\n| B, ms | T_cols | T_ctor | T_first | T_warm |\n|---|---|---|---|---|")
        print(f"| median | {t_cols:.0f} | {t_ctor:.0f} | {t_first:.0f} | {t_warm:.0f} |\n")
        print(
            f"- Gate 2, warm call: T_cols - T_warm = {saving:.0f} ms, floor {floor} ms: {'pass' if gate2 else 'FAIL'}"
        )
        print(
            f"- Gate 3, building first: (T_ctor + T_first) / T_cols = {(t_ctor + t_first) / t_cols:.3f}, "
            f"at most 1.05: {'pass' if gate3 else 'FAIL'}"
        )
        three_cols, three_ped = 3 * t_cols, t_ctor + t_first + 2 * t_warm
        print(
            f"- Three calls: 3 x T_cols = {three_cols:.0f} ms; T_ctor + T_first + 2 x T_warm = {three_ped:.0f} ms "
            f"({1 - three_ped / three_cols:.0%} less)"
        )
        verdicts += [gate2, gate3]
    print(f"\nAll gates: {'pass' if all(verdicts) else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("gen")
    p.add_argument("--pedigree", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--site", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p = sub.add_parser("bench")
    p.add_argument("--site-a", type=Path, required=True)
    p.add_argument("--site-b", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--pixi", default="pixi", help="the pixi executable (absolute inside a systemd scope)")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("summary")
    p.add_argument("jsonl", type=Path, nargs="+")
    args = parser.parse_args()
    {"gen": gen, "run": run, "bench": bench, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
