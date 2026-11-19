"""A Pedigree's caches are filled in the steps the formal-validate model assumes.

TLC checks the model ``PoolCaches.tla`` (``models/pool_caches/`` in the
private companion repo formal-validate), not the code. The model checks that
two host threads sharing one Pedigree never deadlock the pool: a worker
waiting on a cache must never be the worker whose stack holds that cache's
fill (``pedigree.rs``: "The caches are filled serially"). These tests pin each
model step to the code it stands for. They read the source, because a fill
that turns parallel deadlocks only when a stolen job and a second call meet
in one window, which a timing test rarely sees. When one fails, the code no
longer matches the model: update the model (or the code), rerun
``pixi run pool-caches`` in formal-validate with ``PG_PHENOTYPE_ROOT``
pointing at this checkout (``bash models/pool_caches/check.sh --mutants`` for
the mutants), and move the sync commit in its
``.agents/skills/pool-caches/SKILL.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "crates" / "core" / "src"
PEDIGREE_RS = CORE / "pedigree.rs"
MATE_NETWORK_RS = CORE / "mate_network.rs"
ASSORTATIVE_RS = CORE / "assortative" / "mod.rs"
REGRESSION_RS = CORE / "regression.rs"
CROSSINGS = (
    ROOT / "crates" / "python" / "src" / "crossing.rs",
    ROOT / "r" / "src" / "rust" / "src" / "crossing.rs",
)

# Anything that can hand work to rayon or wait on it.
PARALLEL = re.compile(r"\brayon\b|\bpar_\w+|\binto_par_\w+|\bjoin\s*\(|\bscope\s*\(|\binstall\s*\(|\bspawn\w*\s*\(")
CACHE_READS = re.compile(r"\.(mating_pairs|networks|parent_networks)\(\)")


# One Rust token the scan skips or keeps whole: a comment, a string, a char.
_TOKEN = re.compile(r"""//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|'(?:\\u\{[0-9a-fA-F]+\}|\\.|[^\\'])'""", re.S)


def code(path: Path) -> str:
    """``path`` without its test module, comments, and string contents, whitespace squashed."""
    text = path.read_text(encoding="utf-8")

    def keep(token: re.Match[str]) -> str:
        t = token.group()
        return " " if t.startswith("/") else '""' if t.startswith('"') else "''"

    text = _TOKEN.sub(keep, text)
    text = re.split(r"#\[cfg\(test\)\]\s*mod tests\b", text)[0]
    return re.sub(r"\s+", " ", text)


def body(src: str, signature: str) -> str:
    """The braced body after the first match of the regex ``signature``."""
    match = re.search(signature, src)
    assert match, f"no {signature!r}"
    start = src.index("{", match.end())
    depth = 0
    for i in range(start, len(src)):
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        if depth == 0:
            return src[start + 1 : i]
    raise AssertionError(f"unbalanced body after {signature!r}")


def in_order(text: str, *parts: str) -> bool:
    at = 0
    for part in parts:
        at = text.find(part, at)
        if at < 0:
            return False
        at += len(part)
    return True


PEDIGREE = code(PEDIGREE_RS)


@pytest.mark.parametrize(
    ("field", "accessor"),
    [("pairs", "mating_pairs"), ("networks", "networks"), ("parent_networks", "parent_networks")],
)
def test_each_cache_is_a_oncelock_filled_by_get_or_init(field: str, accessor: str) -> None:
    """Model ``Run`` "get" and "fill": OnceLock::get_or_init reads, claims, or blocks."""
    assert re.search(rf"\b{field}: OnceLock<", PEDIGREE)
    accessor_body = body(PEDIGREE, rf"fn {accessor}\(&self\)")
    assert f"self.{field} .get_or_init(" in accessor_body or f"self.{field}.get_or_init(" in accessor_body


def test_the_networks_fill_reads_the_pairs_first() -> None:
    """Model ``Nets``: get networks, then Pairs, then fill networks."""
    fill = body(PEDIGREE, r"fn networks\(&self\)")
    assert in_order(fill, "get_or_init(", "self.mating_pairs()", "Networks::of_pairs(")


def test_the_parents_fill_reads_pairs_then_networks() -> None:
    """Model ``Parents``: get parents, Pairs, Nets, then fill parents (left-to-right arguments)."""
    fill = body(PEDIGREE, r"fn parent_networks\(&self\)")
    assert in_order(fill, "get_or_init(", "ParentNetworks::of(", "self.mating_pairs()", "self.networks()")


def test_every_fill_is_serial() -> None:
    """Model ``Fill(c)`` under Bug = "none": no fill runs a rayon step.

    The fills are ``mating_pairs``, ``Networks::of_pairs`` and
    ``ParentNetworks::of``, all in mate_network.rs, which imports nothing, so
    they cannot reach parallel code elsewhere in the crate.
    """
    mate_network = code(MATE_NETWORK_RS)
    for fill in (r"fn mating_pairs\(", r"fn of_pairs\(", r"fn of\(pairs: &MatingPairs"):
        assert body(mate_network, fill)
    assert not PARALLEL.search(mate_network), PARALLEL.search(mate_network)
    assert not re.search(r"\buse\b|\bcrate::|\bsuper::|\bpedigree_graph_core\b", mate_network)
    # The closures call only those fills and the other cache accessors.
    for accessor in ("mating_pairs", "networks", "parent_networks"):
        assert not PARALLEL.search(body(PEDIGREE, rf"fn {accessor}\(&self\)"))


def test_every_cache_reader_is_a_modelled_method() -> None:
    """Model ``Prog``: only mate_correlation and parent_offspring read the caches."""
    readers = set()
    for path in CORE.rglob("*.rs"):
        if path != PEDIGREE_RS and CACHE_READS.search(code(path)):
            readers.add(path.relative_to(CORE).as_posix())
    assert readers == {"assortative/mod.rs", "regression.rs"}


def test_mate_correlation_reads_pairs_then_networks() -> None:
    """Model ``Prog(call, mate)``: Pairs, then Nets, then the parallel inference."""
    build = body(code(ASSORTATIVE_RS), r"fn build\(pedigree: &'p Pedigree")
    assert in_order(build, "pedigree.mating_pairs()", "pedigree.networks()")


def test_parent_offspring_reads_parents_and_fits_serially() -> None:
    """Model ``Prog(call, po)``: Parents, then serial fits."""
    regression = code(REGRESSION_RS)
    assert "pedigree.parent_networks()" in regression
    assert not PARALLEL.search(regression), PARALLEL.search(regression)


@pytest.mark.parametrize("path", CROSSINGS, ids=["python", "r"])
def test_every_call_runs_as_one_pool_job(path: Path) -> None:
    """Model ``Call``: the host installs the call in the pool and waits for it."""
    run = body(code(path), r"fn run<T, F>\(")
    assert "pool.install(compute)" in run
    if "python" in path.parts:
        # The GIL is released, so two Python threads run two jobs at once.
        assert in_order(run, "py.detach(", "pool.install(compute)")
