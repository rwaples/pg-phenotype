//! `parent_offspring`: the regressions as nested dicts of plain values (the
//! core's `to_value` tree), which `pg_phenotype.regression` turns into
//! frozen dataclasses.

use crate::crossing::{self, StrataArg, TraitArg};
use crate::PedigreeSource;
use pg_phenotype_core::regression;
use pyo3::prelude::*;

/// The parent-offspring regressions of one trait, in the pool.
#[pyfunction]
#[pyo3(signature = (pedigree, trait_, strata, /))]
pub(crate) fn parent_offspring<'py>(
    py: Python<'py>,
    pedigree: PedigreeSource<'py>,
    trait_: TraitArg<'py>,
    strata: StrataArg<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pedigree = pedigree.arg()?;
    let trait_ = trait_.get()?;
    let strata = crossing::strata(&strata)?;
    let result = crossing::run(py, || {
        regression::parent_offspring(pedigree, trait_, strata).map(|r| r.to_value())
    })?;
    crossing::result(py, result)
}
