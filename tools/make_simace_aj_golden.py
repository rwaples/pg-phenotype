"""Write simACE's Aalen-Johansen CIFs over the gate's cases (issue #4).

Runs in simACE's pixi env against a clean simACE checkout, whose
``simace`` it imports ahead of the env's own install:

    pixi run --manifest-path <simACE>/pixi.toml python -B tools/make_simace_aj_golden.py \\
        --simace <simACE checkout> --out tests/golden/simace_aj_<sha12>

Each scenario runs simACE's pipeline (``run_simulation``, ``run_phenotype``
with the default frailty traits, ``run_censor`` with death as a competing
risk) and reads entry, exit and event per trait as
``compute_cumulative_incidence_aj`` does (``_build_entry_times``,
``_exit_event_arrays``).  A case is one trait, pooled or within one sex.
Rows simACE drops (entry after exit: death before the observation window)
are dropped here, so ``cases.npz`` holds only rows both sides read.

- ``late_entry``: the default windows, generation 3 entering at 40.
- ``no_entry``: every window ``[0, 80]``; pg-phenotype is given no entry.
- ``small``: ``late_entry`` on a 100-per-generation pedigree.
- ``integer_ages``: ``late_entry`` with every age and entry rounded up to a
  whole year, so cases, deaths and entries tie.

``simace.json`` holds, per case, ``_aalen_johansen``'s ``aj_disease`` at the
distinct case ages, and for pooled cases ``compute_cumulative_incidence_aj``'s
last value (``K`` on its default grid to the censoring age).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CENSOR_AGE = 80.0
LATE_ENTRY = {3: [40.0, 80.0], 4: [0.0, 80.0], 5: [0.0, 45.0]}
NO_ENTRY = {3: [0.0, 80.0], 4: [0.0, 80.0], 5: [0.0, 80.0]}
SCENARIOS = {
    "late_entry": {"seed": 1, "N": 2000, "windows": LATE_ENTRY},
    "no_entry": {"seed": 2, "N": 2000, "windows": NO_ENTRY},
    "small": {"seed": 3, "N": 100, "windows": LATE_ENTRY},
    "integer_ages": {"seed": 4, "N": 2000, "windows": LATE_ENTRY, "ceil": True},
}
SIMULATION = {
    "G_ped": 6,
    "mating_lambda": 0.5,
    "p_mztwin": 0.02,
    "A1": 0.5,
    "C1": 0.2,
    "E1": 0.3,
    "A2": 0.5,
    "C2": 0.2,
    "E2": 0.3,
    "rA": 0.3,
    "rC": 0.5,
}
PHENOTYPE = {
    "G_pheno": 3,
    "standardize": "global",
    "phenotype_model1": "frailty",
    "phenotype_model2": "frailty",
    "beta1": 1.0,
    "beta_sex1": 0.0,
    "phenotype_params1": {"distribution": "weibull", "scale": 2160, "rho": 0.8},
    "beta2": 1.5,
    "beta_sex2": 0.0,
    "phenotype_params2": {"distribution": "weibull", "scale": 333, "rho": 1.2},
}
DEATH = {"death_scale": 164.0, "death_rho": 2.73}


def scenario(name: str, spec: dict) -> tuple[dict[str, np.ndarray], dict]:
    from simace.analysis.stats.incidence import (
        _aalen_johansen,
        _build_entry_times,
        _exit_event_arrays,
        compute_cumulative_incidence_aj,
    )
    from simace.censoring.censor import run_censor
    from simace.phenotype.runner import run_phenotype
    from simace.simulation.simulate import run_simulation

    windows = spec["windows"]
    pedigree = run_simulation(seed=spec["seed"], N=spec["N"], **SIMULATION)
    phenotype = run_phenotype(pedigree, seed=spec["seed"], **PHENOTYPE)
    censored = run_censor(phenotype, pedigree, censor_age=CENSOR_AGE, seed=spec["seed"], gen_censoring=windows, **DEATH)
    df = censored.join(pedigree.select("id", "generation", "sex"), on="id", how="left")
    entry = _build_entry_times(df, windows)
    sex = df["sex"].to_numpy()

    arrays: dict[str, np.ndarray] = {}
    results: dict = {}
    for trait in (1, 2):
        exit_time, event = _exit_event_arrays(df, trait)
        keep = entry <= exit_time
        e, x, s, sx = entry[keep], exit_time[keep], event[keep], sex[keep]
        if spec.get("ceil"):
            # After choosing the rows simACE keeps, so no row moves from
            # before its entry to on it.  Entries are whole already.
            x = np.ceil(x)
        strata = {"pooled": np.ones(len(x), dtype=bool), "female": sx == 0, "male": sx == 1}
        for stratum, mask in strata.items():
            key = f"{name}/trait{trait}/{stratum}"
            ce, cx, cs = e[mask], x[mask], s[mask].astype(np.int64)
            grid = np.unique(cx[cs == 1])
            aj = _aalen_johansen(ce, cx, cs, grid)
            arrays[f"{key}/entry"] = ce
            arrays[f"{key}/age"] = cx
            arrays[f"{key}/status"] = cs
            record = {
                "scenario": name,
                "trait": trait,
                "stratum": stratum,
                "entry": windows is not NO_ENTRY,
                "n": int(mask.sum()),
                "n_dropped": int((~keep).sum()) if stratum == "pooled" else None,
                "n_cases": int((cs == 1).sum()),
                "n_deaths": int((cs == 2).sum()),
                "ages": grid.tolist(),
                "aj_disease": aj["aj_disease"].tolist(),
            }
            if stratum == "pooled" and not spec.get("ceil"):
                wrapper = compute_cumulative_incidence_aj(df, CENSOR_AGE, gen_censoring=windows)
                record["wrapper_k"] = wrapper[f"trait{trait}"]["aj_values"][-1]
            results[key] = record
    return arrays, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--simace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.simace))
    import simace

    if not Path(simace.__file__).resolve().is_relative_to(args.simace.resolve()):
        sys.exit(f"imported simace from {simace.__file__}, not {args.simace}")
    status = subprocess.run(
        ["git", "-C", args.simace, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout.strip():
        sys.exit(f"{args.simace} has uncommitted changes")
    rev = subprocess.run(["git", "-C", args.simace, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)

    arrays: dict[str, np.ndarray] = {}
    cases: dict = {}
    for name, spec in SCENARIOS.items():
        a, c = scenario(name, spec)
        arrays |= a
        cases |= c
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "cases.npz", **arrays)
    payload = {"simace_rev": rev.stdout.strip(), "censor_age": CENSOR_AGE, "cases": cases}
    (args.out / "simace.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
