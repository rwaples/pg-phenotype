"""Write the R binding's parity fixtures: inputs and the Python package's results.

    pixi run python tools/r_parity.py [out-dir]   # default: r/tests/testthat/fixtures

Floats are written as C99 hex (``float.hex``), which R's ``as.numeric`` reads
exactly, so the R tests compare against the Python scores bit for bit; ``NA``
is a missing value.  Both packages run the same Rust core, so any difference
is a binding bug.  ``trait_cases.csv`` holds what ``pg_phenotype.Trait`` makes
of a few inputs, which R's ``trait()`` must reproduce.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pg_phenotype import Trait, ValidationError, pafgrs  # noqa: E402
from tests.pedigrees import random_pedigree  # noqa: E402
from tests.scoring_fixtures import random_trait  # noqa: E402

PEDIGREES = {
    "small": random_pedigree(3),
    "deep": random_pedigree(11, generations=6, size=18),
}


def cell(value: object) -> str:
    if isinstance(value, float | np.floating):
        return "NA" if np.isnan(value) else float(value).hex()
    return str(value)


def write(path: Path, columns: Mapping[str, object]) -> None:
    names = list(columns)
    with path.open("w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(names)
        out.writerows(zip(*(map(cell, np.asarray(columns[n]).tolist()) for n in names), strict=True))


def subset(ped: dict[str, np.ndarray]) -> list[int]:
    """Every third id, listed in reverse, so input order is not output order."""
    return [int(i) for i in ped["id"][::3][::-1]]


@dataclass(frozen=True)
class Case:
    """One score call; two ``h2`` values make it bivariate."""

    pedigree: str
    ndegree: int
    h2: tuple[float, ...]
    rg: float = np.nan
    rho_within: float | None = None
    probands: bool = False


CASES = [
    *(case for nd in (1, 2, 3) for case in (Case("small", nd, (0.5,)), Case("small", nd, (0.5, 0.3), rg=0.4))),
    Case("deep", 2, (0.35,)),
    Case("deep", 3, (0.8,), probands=True),
    Case("deep", 2, (0.6, 0.45), rg=-0.5, rho_within=0.1),
    Case("deep", 3, (0.3, 0.7), rg=0.8, rho_within=0.55, probands=True),
]

# (name, R type, space-separated values with NA for missing, kind).  Values
# never contain spaces.
TRAIT_CASES = [
    ("logical", "logical", "TRUE FALSE NA TRUE", None),
    ("binary_double", "double", "1 0 NA 0", None),
    ("binary_integer", "integer", "0 NA 1", None),
    ("binary_explicit", "double", "0 2", "binary"),
    ("continuous", "double", "0.5 1.25 NA 3", None),
    ("continuous_explicit", "integer", "1 2 3", "continuous"),
    ("ambiguous", "integer", "0 1 2", None),
    ("ordinal", "integer", "2 0 1 NA", "ordinal"),
    ("categorical_fraction", "double", "1 2.5", "categorical"),
    ("strings", "character", "b a NA c a", None),
    ("strings_codepoint_order", "character", "b B a _", None),
    ("strings_ordinal", "character", "lo hi lo", "ordinal"),
    ("strings_all_missing", "character", "NA NA", None),
    ("empty", "double", "", None),
    ("unknown_kind", "double", "0 1", "nominal"),
]


def trait_input(r_type: str, tokens: list[str]) -> list[object]:
    """The Python input R's vector of *r_type* corresponds to."""
    parse = {
        "logical": lambda t: t == "TRUE",
        "integer": int,
        "double": float,
        "character": str,
    }[r_type]
    return [None if t == "NA" else parse(t) for t in tokens]


def trait_row(name: str, r_type: str, values: str, kind: str | None) -> dict[str, object]:
    row: dict[str, object] = {"case": name, "type": r_type, "values": values, "kind": kind or "NA"}
    try:
        # The unknown_kind case passes a kind outside TraitKind on purpose.
        t = Trait(trait_input(r_type, values.split()), kind)  # ty: ignore[invalid-argument-type]
    except ValidationError as err:
        return row | {
            "out_kind": "NA",
            "out_values": "",
            "levels": "NA",
            "code": err.code,
            "field_names": " ".join(sorted(err.fields)),
            "position": err.fields.get("position", "NA"),
            "error_value": cell(err.fields["value"]) if "value" in err.fields else "NA",
        }
    return row | {
        "out_kind": t.kind,
        "out_values": " ".join(cell(v) for v in t.values.tolist()),
        "levels": "NA" if t.levels is None else " ".join(t.levels),
        "code": "NA",
        "field_names": "",
        "position": "NA",
        "error_value": "NA",
    }


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    traits = {}
    for name, ped in PEDIGREES.items():
        n = len(ped["id"])
        t1, t2 = random_trait(n, 101), random_trait(n, 202, prevalence=0.08)
        traits[name] = (t1, t2)
        write(out_dir / f"pedigree_{name}.csv", ped)
        write(
            out_dir / f"traits_{name}.csv",
            {"trait1": t1.trait.values, "age1": t1.age, "trait2": t2.trait.values, "age2": t2.age},
        )
    # random_trait's table depends only on the prevalence, so one per trait suffices.
    for k, t in enumerate(traits["small"], 1):
        write(out_dir / f"cip{k}.csv", {"ages": t.cip.ages, "cip": t.cip.cip})

    rows: list[dict[str, object]] = []
    for i, case in enumerate(CASES, 1):
        ped = PEDIGREES[case.pedigree]
        probands = subset(ped) if case.probands else None
        prep = pafgrs.prepare(ped, ndegree=case.ndegree, probands=probands)
        t1, t2 = traits[case.pedigree]
        got: pafgrs.UnivariateScores | pafgrs.BivariateScores
        if len(case.h2) == 1:
            got = pafgrs.score_univariate(prep, t1.trait, age=t1.age, cip=t1.cip, h2=case.h2[0])
        else:
            got = pafgrs.score_bivariate(
                prep,
                [t1.trait, t2.trait],
                age=[t1.age, t2.age],
                cip=[t1.cip, t2.cip],
                h2=case.h2,
                rg=case.rg,
                rho_within=case.rho_within,
            )
        name = f"case{i:02d}"
        write(out_dir / f"expected_{name}.csv", got.to_dict())
        meta = got.metadata
        rows.append(
            {
                "case": name,
                "kind": "uni" if len(case.h2) == 1 else "biv",
                "pedigree": case.pedigree,
                "ndegree": case.ndegree,
                "probands": " ".join(map(str, probands)) if probands else "",
                "h2_1": case.h2[0],
                "h2_2": case.h2[1] if len(case.h2) == 2 else np.nan,
                "rg": case.rg,
                "rho_within": np.nan if case.rho_within is None else case.rho_within,
                "rho_within_used": meta.get("rho_within", np.nan),
                "controls_without_age": " ".join(map(str, np.ravel(np.asarray(meta["controls_without_age"])))),
                "threshold": " ".join(map(cell, np.ravel(np.asarray(meta["threshold"])).tolist())),
            }
        )
    write(out_dir / "cases.csv", {k: [r[k] for r in rows] for k in rows[0]})

    trait_rows = [trait_row(*case) for case in TRAIT_CASES]
    write(out_dir / "trait_cases.csv", {k: [r[k] for r in trait_rows] for k in trait_rows[0]})


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "r" / "tests" / "testthat" / "fixtures")
