"""The pedsum #13 assortative-mating goldens (``tools/make_pedsum_am_golden.py``) and their Mating Pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PIN = (Path(__file__).resolve().parent.parent / "tools" / "pedsum_am.pin").read_text().strip()
GOLDEN = Path(__file__).resolve().parent / "golden" / f"pedsum_am_{PIN[:12]}"
NAMES = sorted(p.stem for p in GOLDEN.glob("*.json") if p.name != "provenance.json")
GATE = GOLDEN / "gate"
GATE_NAMES = sorted(p.stem for p in GATE.glob("*.json") if p.name != "provenance.json")


@dataclass(frozen=True)
class Golden:
    """One fixture: its inputs, settings, and pedsum's payload."""

    name: str
    id: np.ndarray
    mother: np.ndarray
    father: np.ndarray
    values: np.ndarray
    kinds: tuple[str, ...]
    n_levels: tuple[int | None, ...]
    stratum_labels: np.ndarray
    stratum_known: np.ndarray
    stratified: bool
    settings: dict
    payload: dict


def load(name: str) -> Golden:
    """Fixture ``name`` of the pinned golden directory, or of its gate set (``gate_*``)."""
    directory = GATE if name.startswith("gate_") else GOLDEN
    arrays = np.load(directory / f"{name}.npz")
    record = json.loads((directory / f"{name}.json").read_text())
    return Golden(
        name=name,
        id=arrays["id"],
        mother=arrays["mother"],
        father=arrays["father"],
        values=arrays["values"],
        kinds=tuple(str(k) for k in arrays["kinds"]),
        n_levels=tuple(None if k < 0 else int(k) for k in arrays["n_levels"]),
        stratum_labels=arrays["stratum_labels"],
        stratum_known=arrays["stratum_known"],
        stratified=record["stratified"],
        settings=record["settings"],
        payload=record["payload"],
    )


def mating_pairs(g: Golden) -> tuple[np.ndarray, np.ndarray]:
    """Mother and father rows of each Mating Pair, in (mother id, father id) order; parents must be rows."""
    row = {int(i): r for r, i in enumerate(g.id)}
    m_rows = np.array([row.get(int(m), -1) for m in g.mother])
    f_rows = np.array([row.get(int(f), -1) for f in g.father])
    both = (m_rows >= 0) & (f_rows >= 0)
    keys = sorted({(int(g.mother[c]), int(g.father[c])) for c in np.flatnonzero(both)})
    return np.array([row[m] for m, _ in keys], dtype=np.int64), np.array([row[f] for _, f in keys], dtype=np.int64)
