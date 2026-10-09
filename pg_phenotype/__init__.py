"""Phenotypes in the context of a pedigree, on pedigree-graph's core.

Each method is a submodule: :mod:`pg_phenotype.pafgrs` for PA-FGRS,
:mod:`pg_phenotype.assortative` for assortative mating, and
:mod:`pg_phenotype.correlation` for correlation estimators on their own.  The
:class:`Pedigree` and :class:`Trait` inputs, the errors, and the thread
budget are shared.
"""

from __future__ import annotations

from importlib.metadata import version as _dist_version

from pg_phenotype import _native, assortative, correlation, pafgrs
from pg_phenotype._errors import ParameterError, PgPhenotypeError, ResourceError, ValidationError
from pg_phenotype._pedigree import Pedigree
from pg_phenotype._threads import configure_threads, thread_budget
from pg_phenotype._trait import Trait

__all__ = [
    "ParameterError",
    "Pedigree",
    "PgPhenotypeError",
    "ResourceError",
    "Trait",
    "ValidationError",
    "__version__",
    "assortative",
    "configure_threads",
    "correlation",
    "pafgrs",
    "pg_core_rev",
    "thread_budget",
]

__version__ = _dist_version("pg-phenotype")


def pg_core_rev() -> str:
    """Return the pedigree-graph-core git revision this build links."""
    return _native.pg_core_rev()
