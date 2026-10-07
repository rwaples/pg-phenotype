"""Write pedsum #13's assortative-mating results on fixed fixtures as goldens.

Runs inside the pinned pedsum snapshot's own environment, never the live
worktree (plan v2, D6)::

    S=<scratch>/pedsum_am_<sha12>
    git -C ../pedsum-issue13 archive <sha> | tar -x -C "$S"
    (cd "$S" && pixi install --locked)
    pixi run --manifest-path "$S/pixi.toml" python -I tools/make_pedsum_am_golden.py \\
        --snapshot "$S" --out tests/golden/pedsum_am_<sha12>

Each fixture writes ``<name>.npz`` (the inputs pg-phenotype reads: pedigree,
trait values, kinds and level counts, stratum labels) and ``<name>.json``
(the settings and pedsum's ``compute_assortative_mating`` payload).
``provenance.json`` records the SHA, the generator, package versions, the
snapshot's lock hash and the numba thread count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numba
import numpy as np
import polars as pl
import scipy
from pedsum.assortative_mating import Trait, classify_trait, compute_assortative_mating, strata
from pedsum.base import PedigreeError
from pedsum.pedigree_ops import IdIndex, _parent_rows, _structural_depth

KINDS = ("continuous", "binary", "ordinal")
ORDINAL_CUTS = (-0.8, 0.0, 0.8)


@dataclass
class Pedigree:
    """Rows of a fixture pedigree, with a latent 2-vector and a birth year per row."""

    ids: np.ndarray
    mother: np.ndarray
    father: np.ndarray
    latent: np.ndarray
    year: np.ndarray


@dataclass
class Fixture:
    """One golden case: a pedigree recipe, the traits taken from it, strata and settings."""

    name: str
    seed: int
    kinds: tuple[str, ...]
    founders: int = 300
    generations: int = 3
    assort: float = 0.5
    remate: float = 0.15
    missing: float = 0.08
    stratify_by: str | None = None
    unknown_stratum: float = 0.0
    birth_year_bin: int = 10
    min_stratum_networks: int = 10
    permutations: int = 199
    bootstrap: int = 0
    draw_seed: int = 0
    edits: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)


def _mate(rng: np.random.Generator, n: int, assort: float, mothers: np.ndarray, fathers: np.ndarray, latent):
    """Pair ``n`` mothers and fathers rank-wise on a noisy first latent trait, so mates correlate."""
    noise = math.sqrt(max(1 - assort * assort, 1e-9)) / max(assort, 1e-9)
    m = rng.choice(mothers, n, replace=False)
    f = rng.choice(fathers, n, replace=False)
    m = m[np.argsort(latent[m, 0] + noise * rng.standard_normal(n))]
    f = f[np.argsort(latent[f, 0] + noise * rng.standard_normal(n))]
    return m, f


def population(fx: Fixture) -> Pedigree:
    """A multi-generation pedigree: rank-wise assortative mating, remating, married-in founders."""
    rng = np.random.default_rng(fx.seed)
    within = np.array([[1.0, 0.4], [0.4, 1.0]])
    latent = [rng.multivariate_normal(np.zeros(2), within, size=fx.founders)]
    sex = [rng.integers(0, 2, fx.founders)]
    year = [rng.integers(1900, 1930, fx.founders)]
    mother = [np.full(fx.founders, -1)]
    father = [np.full(fx.founders, -1)]
    n = fx.founders
    current = np.arange(fx.founders)
    for _ in range(fx.generations):
        n_in = len(current) // 3
        lat_in = rng.multivariate_normal(np.zeros(2), within, size=n_in)
        latent.append(lat_in)
        sex.append(rng.integers(0, 2, n_in))
        base = int(np.median(np.concatenate(year)[current])) if len(current) else 1930
        year.append(rng.integers(base - 5, base + 5, n_in))
        mother.append(np.full(n_in, -1))
        father.append(np.full(n_in, -1))
        pool = np.concatenate([current, n + np.arange(n_in)])
        n += n_in
        all_latent, all_sex, all_year = np.concatenate(latent), np.concatenate(sex), np.concatenate(year)
        females, males = pool[all_sex[pool] == 0], pool[all_sex[pool] == 1]
        n_pairs = min(len(females), len(males)) * 4 // 5
        if n_pairs == 0:
            break
        m, f = _mate(rng, n_pairs, fx.assort, females, males, all_latent)
        n_re = int(fx.remate * n_pairs)
        if n_re:
            again = rng.choice(n_pairs, n_re, replace=False)
            m = np.concatenate([m, rng.permutation(m[again])])
            f = np.concatenate([f, f[again]])
        kids = rng.integers(1, 4, len(m))
        cm, cf = np.repeat(m, kids), np.repeat(f, kids)
        n_kids = len(cm)
        child = 0.5 * (all_latent[cm] + all_latent[cf]) + math.sqrt(0.5) * rng.multivariate_normal(
            np.zeros(2), within, size=n_kids
        )
        latent.append(child)
        sex.append(rng.integers(0, 2, n_kids))
        year.append(np.maximum(all_year[cm], all_year[cf]) + 20 + rng.integers(0, 15, n_kids))
        mother.append(cm)
        father.append(cf)
        current = n + np.arange(n_kids)
        n += n_kids
    rows = np.concatenate(mother), np.concatenate(father)
    ids = 1000 + 3 * np.arange(n)
    ids = ids[rng.permutation(n)]
    to_id = lambda r: np.where(r >= 0, ids[np.maximum(r, 0)], -1)  # noqa: E731
    order = rng.permutation(n)
    return Pedigree(
        ids[order],
        to_id(rows[0])[order],
        to_id(rows[1])[order],
        np.concatenate(latent)[order],
        np.concatenate(year)[order],
    )


def trait_values(kind: str, latent: np.ndarray) -> np.ndarray:
    """A trait of ``kind`` from one latent column: as is, thresholded at 0.5 (binary), or cut into 4 levels."""
    if kind == "continuous":
        return np.round(latent, 6)
    if kind == "binary":
        return (latent > 0.5).astype(np.float64)
    return np.digitize(latent, ORDINAL_CUTS).astype(np.float64)


def apply_edits(fx: Fixture, ped: Pedigree, values: list[np.ndarray], rng: np.random.Generator) -> None:
    """The fixture's structural edits, in place."""
    n = len(ped.ids)
    for edit in fx.edits:
        if edit == "dangling_parents":
            hit = np.flatnonzero(ped.father >= 0)
            hit = rng.choice(hit, len(hit) // 20, replace=False)
            ped.father[hit[: len(hit) // 2]] = 10**9 + np.arange(len(hit) // 2)
            ped.father[hit[len(hit) // 2 :]] = -1
        elif edit == "phantoms":
            hit = np.flatnonzero(ped.father >= 0)
            hit = rng.choice(hit, len(hit) // 10, replace=False)
            phantom = 2 * 10**9 + np.arange(len(hit))
            ped.father[hit] = phantom
            ped.ids = np.concatenate([ped.ids, phantom])
            ped.mother = np.concatenate([ped.mother, np.full(len(hit), -1)])
            ped.father = np.concatenate([ped.father, np.full(len(hit), -1)])
            ped.latent = np.concatenate([ped.latent, np.zeros((len(hit), 2))])
            ped.year = np.concatenate([ped.year, np.full(len(hit), 1950)])
            for t in range(len(values)):
                values[t] = np.concatenate([values[t], np.full(len(hit), np.nan)])
        elif edit == "constant_father_bin":
            # Fathers of complete pairs all 0 on trait 0, the rest of the rows still take 1.
            fathers = np.isin(ped.ids, ped.father)
            values[0][fathers] = 0.0
            values[0][~fathers & ~np.isnan(values[0])] = 1.0
        elif edit == "degenerate_bin_stratum":
            # Every row of one birth-year bin takes one value of trait 0.
            year_bin = ped.year // fx.birth_year_bin * fx.birth_year_bin
            target = np.unique(year_bin)[len(np.unique(year_bin)) // 2]
            values[0][year_bin == target] = 1.0
        else:
            raise ValueError(edit)
    assert all(len(v) == len(ped.ids) for v in values), n


def two_by_two(table: list[list[int]], seed: int) -> tuple[Pedigree, list[np.ndarray]]:
    """Founder pairs, each with one child, whose binary traits give exactly ``table`` (mother level x father level)."""
    rng = np.random.default_rng(seed)
    m_vals, f_vals = [], []
    for i in range(2):
        for j in range(2):
            m_vals += [i] * table[i][j]
            f_vals += [j] * table[i][j]
    p = len(m_vals)
    order = rng.permutation(p)
    m_vals, f_vals = np.array(m_vals)[order], np.array(f_vals)[order]
    ids = np.arange(3 * p) + 1
    mother = np.concatenate([np.full(2 * p, -1), ids[:p]])
    father = np.concatenate([np.full(2 * p, -1), ids[p : 2 * p]])
    trait = np.concatenate([m_vals, f_vals, np.full(p, np.nan)]).astype(np.float64)
    ped = Pedigree(ids, mother, father, np.zeros((3 * p, 2)), np.full(3 * p, 1950))
    return ped, [trait]


FIXTURES: list[Fixture] = [
    Fixture("cont_bin", 1, ("continuous", "binary")),
    Fixture("cont_bin_boot", 1, ("continuous", "binary"), permutations=99, bootstrap=200, draw_seed=11),
    Fixture(
        "cont_ord_birthyear",
        2,
        ("continuous", "ordinal"),
        stratify_by="birth_year",
        unknown_stratum=0.05,
        min_stratum_networks=5,
    ),
    Fixture(
        "bin_ord_depth_boot",
        3,
        ("binary", "ordinal"),
        stratify_by="depth",
        min_stratum_networks=5,
        bootstrap=150,
        draw_seed=5,
    ),
    Fixture(
        "ord_cont_birthyear_boot",
        4,
        ("ordinal", "continuous"),
        stratify_by="birth_year",
        birth_year_bin=20,
        min_stratum_networks=8,
        bootstrap=200,
        draw_seed=-7,
    ),
    Fixture("cont_only_boot", 5, ("continuous",), bootstrap=100, draw_seed=3),
    Fixture("bin_only", 6, ("binary",), permutations=299),
    Fixture("ord_only_depth", 7, ("ordinal",), stratify_by="depth", min_stratum_networks=3),
    Fixture("thin_birthyear", 8, ("continuous", "binary"), founders=150, stratify_by="birth_year"),
    Fixture("all_thin_birthyear", 8, ("continuous", "binary"), founders=60, stratify_by="birth_year", birth_year_bin=5),
    Fixture("dangling_and_phantoms", 9, ("continuous", "ordinal"), edits=("dangling_parents", "phantoms")),
    Fixture("constant_cell_margin", 10, ("binary", "continuous"), edits=("constant_father_bin",)),
    Fixture(
        "degenerate_stratum",
        11,
        ("binary", "continuous"),
        stratify_by="birth_year",
        min_stratum_networks=2,
        edits=("degenerate_bin_stratum",),
    ),
    Fixture("no_permutations", 12, ("continuous", "ordinal"), permutations=0, bootstrap=50, draw_seed=1),
    Fixture("strong_assort", 13, ("binary", "continuous"), assort=0.95, permutations=999, bootstrap=100),
    Fixture("null_assort", 14, ("ordinal", "binary"), assort=0.0, permutations=999),
    Fixture("founders_only", 15, ("continuous", "binary"), generations=0),
    Fixture("boundary_2x2", 0, ("binary",), bootstrap=200, extra={"table": [[40, 0], [12, 30]]}),
    Fixture("boundary_2x2_sandwich", 0, ("binary",), extra={"table": [[25, 0], [0, 25]]}),
    Fixture("ties_2x2", 16, ("binary",), permutations=499, extra={"table": [[3, 2], [2, 3]]}),
    Fixture("single_network", 0, ("continuous",), bootstrap=20, extra={"chain": 30}),
    Fixture("perfect_2x2_boot", 0, ("binary",), bootstrap=200, draw_seed=3, extra={"table": [[25, 0], [0, 25]]}),
    Fixture(
        "separated_biserial_boot", 24, ("continuous", "binary"), bootstrap=200, draw_seed=8, extra={"separated": 60}
    ),
    Fixture(
        "near_perfect_ord_cont_depth",
        22,
        ("ordinal", "continuous"),
        assort=0.999,
        missing=0.0,
        bootstrap=200,
        draw_seed=3,
        stratify_by="depth",
        min_stratum_networks=3,
    ),
    Fixture("single_network_sandwich", 1, ("continuous",), extra={"chain": 25}),
    Fixture("small_boot_2x2", 17, ("binary",), bootstrap=300, draw_seed=4, extra={"table": [[4, 2], [1, 3]]}),
    Fixture(
        "small_strat",
        18,
        ("binary", "ordinal"),
        founders=70,
        stratify_by="birth_year",
        min_stratum_networks=2,
        permutations=299,
        bootstrap=200,
        draw_seed=9,
    ),
    Fixture(
        "small_strat_cont",
        19,
        ("continuous", "ordinal"),
        founders=60,
        stratify_by="depth",
        min_stratum_networks=2,
        permutations=299,
        bootstrap=200,
        draw_seed=2,
    ),
]


def gate_fixtures() -> list[Fixture]:
    """The unit 9 gate: pedsum's benchmark configurations at 10^4 pairs, and random breadth cases."""
    out = [
        Fixture(f"gate_{config}_1e4_s{seed}", seed, kinds, permutations=999, bootstrap=1000, draw_seed=seed,
                stratify_by="birth_year" if config == "b" else None, extra={"pedsum_generator": 10_000, "config": config})
        for config, kinds in (("a", ("binary",)), ("b", ("continuous", "binary")))
        for seed in range(3)
    ]  # fmt: skip
    rng = np.random.default_rng(20261007)
    for i in range(40):
        n_traits = int(rng.integers(1, 3))
        stratify = [None, "depth", "birth_year"][int(rng.integers(0, 3))]
        out.append(
            Fixture(
                f"gate_random_{i:02d}",
                100 + i,
                tuple(str(k) for k in rng.choice(KINDS, n_traits)),
                founders=int(rng.integers(60, 500)),
                generations=int(rng.integers(1, 5)),
                assort=float(rng.uniform(0, 0.9)),
                remate=float(rng.uniform(0, 0.4)),
                missing=float(rng.uniform(0, 0.3)),
                stratify_by=stratify,
                unknown_stratum=float(rng.choice([0.0, 0.05])),
                birth_year_bin=int(rng.choice([5, 10, 20])),
                min_stratum_networks=int(rng.integers(1, 11)),
                permutations=int(rng.choice([0, 199, 499])),
                bootstrap=int(rng.choice([0, 100, 200])),
                draw_seed=int(rng.integers(-(2**40), 2**40)),
            )
        )
    return out


def pedsum_generator(fx: Fixture, snapshot: Path) -> tuple[Pedigree, list[np.ndarray]]:
    """pedsum's benchmark pedigree (``benchmarks/generate_assortative_mating.py``), config ``a`` or ``b``."""
    sys.path.insert(0, str(snapshot / "benchmarks"))
    from generate_assortative_mating import generate

    df, _ = generate(
        pairs=fx.extra["pedsum_generator"], seed=fx.seed, r_mf=np.array([[0.30, 0.15], [0.05, 0.25]]),
        within_m=0.4, within_f=0.3, remate_fathers=0.2, remate_mothers=0.1, prevalence=0.1,
        year_min=1900, year_max=1980,
    )  # fmt: skip
    liab = np.array([np.nan if v == "NA" else float(v) for v in df["liab"]])
    dx = np.array([np.nan if v == "NA" else float(v) for v in df["dx"]])
    n = len(df)
    ped = Pedigree(
        df["id"].to_numpy(),
        df["mother"].to_numpy(),
        df["father"].to_numpy(),
        np.zeros((n, 2)),
        df["birth_year"].to_numpy(),
    )
    return ped, [dx] if fx.extra["config"] == "a" else [liab, dx]


def build(fx: Fixture, snapshot: Path | None = None) -> tuple[Pedigree, list[np.ndarray]]:
    """The fixture's pedigree and trait values; pedsum's benchmark generator needs the ``snapshot``."""
    if "pedsum_generator" in fx.extra:
        assert snapshot is not None
        return pedsum_generator(fx, snapshot)
    if "table" in fx.extra:
        return two_by_two(fx.extra["table"], fx.seed)
    if "separated" in fx.extra:
        # Founder pairs whose father's binary trait 1 is 1 exactly when the mother's trait 0 is positive.
        p = fx.extra["separated"]
        rng = np.random.default_rng(fx.seed)
        ids = np.arange(3 * p) + 1
        mother = np.concatenate([np.full(2 * p, -1), ids[:p]])
        father = np.concatenate([np.full(2 * p, -1), ids[p : 2 * p]])
        m0 = rng.standard_normal(p)
        t0 = np.concatenate([m0, rng.standard_normal(p), np.full(p, np.nan)])
        t1 = np.concatenate([rng.integers(0, 2, p), (m0 > 0).astype(float), np.full(p, np.nan)]).astype(float)
        return Pedigree(ids, mother, father, np.zeros((3 * p, 2)), np.full(3 * p, 1950)), [t0, t1]
    if "chain" in fx.extra:
        # Mother i has children by fathers i and i + 1: one Mate Network of 2p - 1 pairs.
        p = fx.extra["chain"]
        rng = np.random.default_rng(fx.seed)
        ids = np.arange(2 * p + 2 * p - 1) + 1
        mothers, fathers = ids[:p], ids[p : 2 * p]
        pair_m = np.concatenate([mothers, mothers[:-1]])
        pair_f = np.concatenate([fathers, fathers[1:]])
        n = len(ids)
        mother = np.concatenate([np.full(2 * p, -1), pair_m])
        father = np.concatenate([np.full(2 * p, -1), pair_f])
        trait = np.concatenate([rng.standard_normal(2 * p), np.full(n - 2 * p, np.nan)])
        return Pedigree(ids, mother, father, np.zeros((n, 2)), np.full(n, 1950)), [trait]
    ped = population(fx)
    rng = np.random.default_rng(fx.seed + 1000)
    values = []
    for t, kind in enumerate(fx.kinds):
        v = trait_values(kind, ped.latent[:, t])
        v[rng.random(len(v)) < fx.missing] = np.nan
        values.append(v)
    apply_edits(fx, ped, values, rng)
    return ped, values


def pedsum_trait(t: int, kind: str, values: np.ndarray) -> Trait:
    """A pedsum ``Trait`` over already-coded values, with levels as ``classify_trait`` would record."""
    levels = None if kind == "continuous" else tuple(str(int(v)) for v in np.unique(values[~np.isnan(values)]))
    return Trait(f"t{t}", kind, "stated", levels, values)


def run(fx: Fixture, out: Path, snapshot: Path) -> None:
    """Compute the fixture with pedsum and write its inputs and payload."""
    ped, values = build(fx, snapshot)
    traits = [pedsum_trait(t, k, v) for t, (k, v) in enumerate(zip(fx.kinds, values, strict=True))]
    columns = {
        "id": ped.ids.astype(np.int64),
        "mother": ped.mother.astype(np.int64),
        "father": ped.father.astype(np.int64),
    }
    rng = np.random.default_rng(fx.seed + 2000)
    if fx.stratify_by == "depth":
        index = IdIndex(columns["id"])
        depth = _structural_depth(_parent_rows(columns["mother"], index)[0], _parent_rows(columns["father"], index)[0])
        columns["ped_depth"] = depth.astype(np.int64)
    year = ped.year.astype(np.int64)
    year[rng.random(len(year)) < fx.unknown_stratum] = -1
    columns["birth_year"] = year
    df = pl.DataFrame(columns)
    payload = compute_assortative_mating(
        df,
        traits,
        permutations=fx.permutations,
        bootstrap=fx.bootstrap,
        seed=fx.draw_seed,
        stratify_by=fx.stratify_by,
        birth_year_bin=fx.birth_year_bin,
        min_stratum_networks=fx.min_stratum_networks,
    )
    labels = strata(df, fx.stratify_by, fx.birth_year_bin) if fx.stratify_by else np.zeros(len(df), dtype=np.int64)
    np.savez_compressed(
        out / f"{fx.name}.npz",
        id=columns["id"],
        mother=columns["mother"],
        father=columns["father"],
        values=np.stack(values),
        kinds=np.array(fx.kinds),
        n_levels=np.array([-1 if t.levels is None else len(t.levels) for t in traits]),
        stratum_labels=labels.astype(np.int64),
        stratum_known=labels != -1,
    )
    record = {
        "name": fx.name,
        "fixture": {k: v for k, v in fx.__dict__.items() if k != "name"},
        "stratified": fx.stratify_by is not None,
        "settings": {
            "permutations": fx.permutations,
            "bootstrap": fx.bootstrap,
            "seed": fx.draw_seed,
            "min_stratum_networks": fx.min_stratum_networks,
        },
        "payload": payload,
    }
    (out / f"{fx.name}.json").write_text(json.dumps(record, indent=1, default=_plain) + "\n")


def _plain(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (tuple, np.ndarray)):
        return list(value)
    raise TypeError(type(value))


TRAIT_ERRORS = {"all_missing_trait": [None, None, None], "constant_trait": ["5", "5.0", None]}


def trait_errors() -> dict[str, str]:
    """pedsum's message for each trait it refuses at parse time (D9)."""
    out = {}
    for code, tokens in TRAIT_ERRORS.items():
        try:
            classify_trait("t", np.array(tokens, dtype=object))
        except PedigreeError as err:
            out[code] = str(err)
        else:
            raise AssertionError(code)
    return out


def main(argv: list[str] | None = None) -> int:
    """Write every fixture and the provenance record."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--set", choices=("fixtures", "gate"), default="fixtures")
    args = p.parse_args(argv)
    snapshot = args.snapshot.resolve()
    import pedsum

    if Path(pedsum.__file__).resolve().parent.parent != snapshot:
        raise SystemExit(f"pedsum imports from {pedsum.__file__}, not the snapshot {snapshot}")
    sha = (Path(__file__).resolve().parent / "pedsum_am.pin").read_text().strip()
    if f"pedsum_am_{sha[:12]}" not in args.out.parts:
        raise SystemExit(f"--out must be under pedsum_am_{sha[:12]}")
    args.out.mkdir(parents=True, exist_ok=True)
    fixtures = FIXTURES if args.set == "fixtures" else gate_fixtures()
    for fx in fixtures:
        run(fx, args.out, snapshot)
        print(fx.name, file=sys.stderr)
    generator = Path(__file__).resolve()
    provenance = {
        "pedsum_sha": sha,
        "snapshot": "git archive of the pin, extracted outside the worktree, its own locked pixi env",
        "generator": generator.name,
        "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "pg_phenotype_commit": subprocess.run(
            ["git", "-C", str(generator.parent), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "fixtures": [fx.name for fx in fixtures],
        "pixi_lock_sha256": hashlib.sha256((snapshot / "pixi.lock").read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
            "polars": pl.__version__,
        },
        "numba_threads": numba.get_num_threads(),
        "trait_errors": trait_errors(),
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
