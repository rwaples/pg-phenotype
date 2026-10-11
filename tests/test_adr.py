"""Each ADR has its own number, and its title carries that number."""

import collections
import pathlib
import re

ADR_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs" / "adr"


def _adrs():
    adrs = sorted(ADR_DIR.glob("[0-9][0-9][0-9][0-9]-*.md"))
    assert adrs, ADR_DIR
    return adrs


def test_adr_numbers_are_unique():
    counts = collections.Counter(p.name[:4] for p in _adrs())
    assert {n: c for n, c in counts.items() if c > 1} == {}


def test_adr_title_matches_its_file_number():
    for p in _adrs():
        title = p.read_text().splitlines()[0]
        assert re.match(rf"# ADR {p.name[:4]}: ", title), (p.name, title)
