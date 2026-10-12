from collections.abc import Sequence
from typing import Any

import numpy as np

from pg_phenotype._trait import Trait

def core_version() -> str: ...
def pg_core_rev() -> str: ...
def trait_kinds() -> list[str]: ...
def configure_threads(n: int) -> None: ...
def thread_budget() -> int: ...
def _reset_thread_budget() -> None: ...
def _panic_for_test() -> None: ...
def _release_count() -> int: ...

_Columns = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
_Strata = tuple[np.ndarray, np.ndarray]

class Pedigree:
    def __new__(
        cls,
        ids: np.ndarray,
        mother: np.ndarray,
        father: np.ndarray,
        twin: np.ndarray | None,
        sex: np.ndarray | None,
        /,
    ) -> Pedigree: ...
    def __len__(self) -> int: ...
    @property
    def ids(self) -> np.ndarray: ...
    def rows(self, ids: np.ndarray, /) -> np.ndarray: ...
    def align(self, ids: np.ndarray, values: np.ndarray, /) -> np.ndarray: ...

class Prep:
    @property
    def n_rows(self) -> int: ...
    @property
    def ndegree(self) -> int: ...
    @property
    def n_probands(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def proband_rows(self) -> np.ndarray: ...
    def relatives(self, i: int) -> tuple[np.ndarray, np.ndarray]: ...
    def triangle(self, i: int) -> np.ndarray: ...

def prepare(
    pedigree: Pedigree | _Columns,
    /,
    *,
    ndegree: int,
    probands: np.ndarray | None,
) -> Prep: ...

_CipTable = tuple[list[float], list[float], str]

def check_cip(cip: _CipTable, /) -> tuple[float, float]: ...
def cip_aalen_johansen(
    age: np.ndarray, status: np.ndarray, entry: np.ndarray | None, /
) -> tuple[list[float], list[float]]: ...
def score_univariate(
    prep: Prep,
    trait_: Trait,
    age: np.ndarray,
    cip: _CipTable,
    /,
    *,
    h2: float,
) -> dict[str, Any]: ...
def score_bivariate(
    prep: Prep,
    traits: tuple[Trait, Trait],
    ages: tuple[np.ndarray, np.ndarray],
    cips: tuple[_CipTable, _CipTable],
    /,
    *,
    h2: tuple[float, float],
    rg: float,
    rho_within: float | None,
) -> dict[str, Any]: ...
def mate_correlation(
    pedigree: Pedigree | _Columns,
    traits: Sequence[Trait],
    strata: _Strata | None,
    /,
    *,
    permutations: int | None,
    bootstrap: int,
    seed: int,
    min_stratum_networks: int,
    spearman: bool,
    inference: bool,
) -> dict[str, Any]: ...
def tetrachoric_table(table: tuple[tuple[int, int], tuple[int, int]], /) -> dict[str, Any]: ...
def tetrachoric_pairs(x: np.ndarray, y: np.ndarray, /) -> dict[str, Any]: ...
def parent_offspring(
    pedigree: Pedigree | _Columns,
    trait_: Trait,
    strata: _Strata | None,
    /,
) -> dict[str, Any]: ...
def relative_pairs_correlation(
    pedigree: Pedigree | _Columns,
    trait_: Trait,
    categories: list[tuple[int, str, list[str] | None]] | None,
    memory_budget_bytes: int,
    /,
) -> dict[str, Any]: ...
