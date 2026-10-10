"""Write simACE's relative-pair correlations over simACE ``small_test`` reps (ADR 0009 gate).

Runs in simACE's pixi env against a clean simACE checkout, whose ``simace``
it imports ahead of the env's own install:

    pixi run --manifest-path <simACE>/pixi.toml python -B tools/make_simace_relative_pairs_golden.py \\
        --simace <simACE checkout> --reps <results>/test/small_test/rep{1,2,3} \\
        --out tests/golden/simace_relative_pairs_<sha12>

Each rep gives its full pedigree (``pedigree.parquet``) and the phenotyped
rows' ``liability1`` and ``affected1`` (``phenotype.parquet``), aligned to the
pedigree with NaN on unphenotyped rows; ``inputs.npz`` holds them.
``golden.json`` holds, per rep, its ``run.yaml`` provenance and simACE's
results on ``graph.view(ids=phenotyped)``, as ``simace stats`` computes them:

- ``liability``: ``compute_liability_correlations`` (Pearson, ``None`` below
  ten pairs);
- ``phi``: ``compute_affected_correlations``;
- ``tetrachoric``: ``compute_tetrachoric`` (``{r, se, n_pairs}``);
- ``table``: ``Stratum.affection_table``, ``[first affected][second affected]``.

Every category is single-entered there, the lower receiver row first for a
symmetric one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def plain(obj: object) -> object:
    """JSON-ready: NumPy scalars to Python, NaN to None."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def rep(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    import polars as pl
    import yaml
    from pedigree_graph import PedigreeGraph
    from simace.analysis.stats.correlations import (
        compute_affected_correlations,
        compute_liability_correlations,
        compute_tetrachoric,
    )
    from simace.analysis.stats.moments import Stratum, relationship_moments_for
    from simace.core.relationships import RELATIONSHIP_TYPES

    ped = pl.read_parquet(path / "pedigree.parquet")
    df = pl.read_parquet(path / "phenotype.parquet")
    graph = PedigreeGraph.from_frame(ped)
    moments = relationship_moments_for(df, graph.view(ids=df["id"].to_numpy()))
    stratum = Stratum(moments)
    tables = stratum.affection_table(1)
    row = {int(i): r for r, i in enumerate(ped["id"].to_numpy())}
    rows = np.array([row[int(i)] for i in df["id"].to_numpy()])
    liability = np.full(len(ped), np.nan)
    affected = np.full(len(ped), np.nan)
    liability[rows] = df["liability1"].to_numpy()
    affected[rows] = df["affected1"].to_numpy().astype(float)
    run = yaml.safe_load((path / "run.yaml").read_text())
    arrays = {
        "id": ped["id"].to_numpy().astype(np.int64),
        "mother": ped["mother"].to_numpy().astype(np.int64),
        "father": ped["father"].to_numpy().astype(np.int64),
        "twin": ped["twin"].to_numpy().astype(np.int64),
        "sex": ped["sex"].to_numpy().astype(np.int64),
        "liability1": liability,
        "affected1": affected,
    }
    result = {
        "rep": path.name,
        "run": {k: run.get(k) for k in ("simace_version", "source", "scenario", "rep", "seed")},
        "n_rows": len(ped),
        "n_phenotyped": len(df),
        "simace": plain(
            {
                "liability": compute_liability_correlations(moments=moments)["trait1"],
                "phi": compute_affected_correlations(moments=moments)["trait1"],
                "tetrachoric": compute_tetrachoric(moments=moments)["trait1"],
                "table": {code: tables[i].tolist() for code, i in stratum.report_positions()},
                "n_pairs": {code: stratum.n_pairs[i] for code, i in stratum.report_positions()},
                "categories": list(RELATIONSHIP_TYPES),
            }
        ),
    }
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simace", type=Path, required=True)
    parser.add_argument("--reps", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    status = subprocess.run(
        ["git", "-C", args.simace, "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    if status.stdout.strip():
        sys.exit(f"{args.simace} has uncommitted changes")
    rev = subprocess.run(["git", "-C", args.simace, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    sys.path.insert(0, str(args.simace.resolve()))
    import simace

    if not Path(simace.__file__).resolve().is_relative_to(args.simace.resolve()):
        sys.exit(f"imported simace from {simace.__file__}, not {args.simace}")
    args.out.mkdir(parents=True, exist_ok=True)
    reps, arrays = [], {}
    for path in args.reps:
        result, columns = rep(path)
        reps.append(result)
        arrays.update({f"{path.name}.{k}": v for k, v in columns.items()})
    golden = {"simace_rev": rev.stdout.strip(), "reps": reps}
    (args.out / "golden.json").write_text(json.dumps(golden, indent=1) + "\n")
    np.savez_compressed(args.out / "inputs.npz", **arrays)


if __name__ == "__main__":
    main()
