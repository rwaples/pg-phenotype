"""``release_free_memory``: free heap pages go back to the system before native work."""

from __future__ import annotations

import sys

import numpy as np

from pg_phenotype import Trait, _memory
from pg_phenotype import assortative as am


def test_release_free_memory_runs_and_finds_glibc_on_linux():
    """It is safe to call anywhere; on glibc Linux it binds ``malloc_trim``."""
    _memory.release_free_memory()
    if sys.platform.startswith("linux") and _memory._MALLOC_TRIM is None:
        import platform

        assert platform.libc_ver()[0] != "glibc"


def test_mate_correlation_releases_free_memory_first(monkeypatch):
    """``mate_correlation`` hands free memory back once, before the pool allocates."""
    calls = []
    monkeypatch.setattr(am, "release_free_memory", lambda: calls.append(1))
    ids = np.arange(1, 7)
    pedigree = {"id": ids, "mother": [-1, -1, -1, -1, 1, 2], "father": [-1, -1, -1, -1, 3, 4]}
    am.mate_correlation(pedigree, Trait([0.1, 0.9, 0.4, 0.2, np.nan, np.nan]), permutations=0)
    assert calls == [1]
