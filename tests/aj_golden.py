"""simACE's and survfit's Aalen-Johansen CIFs over the gate's cases.

From ``tools/make_simace_aj_golden.py`` (inputs, simACE) and
``tools/make_survival_aj_golden.py`` (R survival's survfit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).parent / "golden" / "simace_aj_6355ffa6aebe"


@dataclass(frozen=True)
class Case:
    key: str
    scenario: str
    #: None when the scenario has no late entry (all entries 0).
    entry: np.ndarray | None
    age: np.ndarray
    status: np.ndarray
    n_dropped: int | None
    n_cases: int
    #: simACE's ``_aalen_johansen`` at the case ages.
    simace_ages: np.ndarray
    simace_cif: np.ndarray
    #: ``compute_cumulative_incidence_aj``'s last value, pooled continuous cases only.
    simace_wrapper_k: float | None
    survfit_ages: np.ndarray
    survfit_cif: np.ndarray


@dataclass(frozen=True)
class Golden:
    simace_rev: str
    survival_versions: str
    cases: list[Case]


def load() -> Golden:
    simace = json.loads((GOLDEN / "simace.json").read_text())
    survival = json.loads((GOLDEN / "survival.json").read_text())
    arrays = np.load(GOLDEN / "cases.npz")
    cases = []
    for key, c in simace["cases"].items():
        s = survival["cases"][key]
        cases.append(
            Case(
                key=key,
                scenario=c["scenario"],
                entry=arrays[f"{key}/entry"] if c["entry"] else None,
                age=arrays[f"{key}/age"],
                status=arrays[f"{key}/status"],
                n_dropped=c["n_dropped"],
                n_cases=c["n_cases"],
                simace_ages=np.array(c["ages"]),
                simace_cif=np.array(c["aj_disease"]),
                simace_wrapper_k=c.get("wrapper_k"),
                survfit_ages=np.array(s["ages"]),
                survfit_cif=np.array(s["cif"]),
            )
        )
    return Golden(simace["simace_rev"], survival["versions"], cases)
