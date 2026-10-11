"""Each ADR has its own number, its title carries it, and every cited ADR file exists."""

import collections
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
ADR_DIR = ROOT / "docs" / "adr"
NAME = re.compile(r"\d{4}-[a-z0-9-]+\.md")
CITING = ["*.md", "docs/**/*.md", "crates/**/*.rs", "pg_phenotype/**/*.py", "tests/**/*.py", "tools/**/*.py", "r/R/*.R"]


def _adrs():
    adrs = sorted(ADR_DIR.iterdir())
    assert adrs, ADR_DIR
    return adrs


def test_every_adr_file_is_named_nnnn_slug():
    assert [p.name for p in _adrs() if not NAME.fullmatch(p.name)] == []


def test_adr_numbers_are_unique():
    counts = collections.Counter(p.name[:4] for p in _adrs())
    assert {n: c for n, c in counts.items() if c > 1} == {}


def test_adr_title_matches_its_file_number():
    for p in _adrs():
        title = p.read_text(encoding="utf-8").splitlines()[0]
        assert re.match(rf"# ADR {p.name[:4]}: ", title), (p.name, title)


def test_every_cited_adr_file_exists():
    names = {p.name for p in _adrs()}
    missing = {
        (str(path.relative_to(ROOT)), cited)
        for pattern in CITING
        for path in ROOT.glob(pattern)
        for cited in NAME.findall(path.read_text(encoding="utf-8"))
        if cited not in names
    }
    assert missing == set()
