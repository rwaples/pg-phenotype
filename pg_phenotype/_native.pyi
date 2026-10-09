from collections.abc import Sequence
from typing import Any

import numpy as np

def core_version() -> str: ...
def pg_core_rev() -> str: ...
def trait_kinds() -> list[str]: ...
def configure_threads(n: int) -> None: ...
def thread_budget() -> int: ...
def _reset_thread_budget() -> None: ...
def _panic_for_test() -> None: ...

_Columns = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]

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
    threads: int,
) -> Prep: ...
def check_cip(ages: list[float], cip: list[float]) -> tuple[float, float]: ...
def score_univariate(
    prep: Prep,
    values: np.ndarray,
    kind: str,
    age: np.ndarray,
    cip_ages: list[float],
    cip_values: list[float],
    /,
    *,
    h2: float,
    threads: int,
) -> dict[str, Any]: ...
def score_bivariate(
    prep: Prep,
    values1: np.ndarray,
    kind1: str,
    age1: np.ndarray,
    values2: np.ndarray,
    kind2: str,
    age2: np.ndarray,
    cip1: tuple[list[float], list[float]],
    cip2: tuple[list[float], list[float]],
    /,
    *,
    h2: tuple[float, float],
    rg: float,
    rho_within: float | None,
    threads: int,
) -> dict[str, Any]: ...
def mate_correlation(
    pedigree: Pedigree | _Columns,
    traits: Sequence[tuple[np.ndarray, str, int | None]],
    stratum_labels: np.ndarray | None,
    stratum_known: np.ndarray | None,
    /,
    *,
    permutations: int,
    bootstrap: int,
    seed: int,
    min_stratum_networks: int,
    spearman: bool,
    threads: int,
) -> dict[str, Any]: ...
