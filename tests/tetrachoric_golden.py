"""simACE's tetrachoric results over the gate's table grid (``tools/make_simace_tetrachoric_golden.py``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

GOLDEN = Path(__file__).parent / "golden" / "simace_tetrachoric_6355ffa6aebe"


@dataclass(frozen=True)
class Case:
    set: str
    index: int
    table: list[list[int]]
    simace_r: float | None
    simace_se: float | None


@dataclass(frozen=True)
class Golden:
    simace_rev: str
    cases: list[Case]


def load() -> Golden:
    tables = json.loads((GOLDEN / "tables.json").read_text())["sets"]
    simace = json.loads((GOLDEN / "simace.json").read_text())
    cases = [
        Case(name, i, table, *simace["results"][name][i])
        for name, rows in tables.items()
        for i, table in enumerate(rows)
    ]
    return Golden(simace["meta"]["simace_rev"], cases)
