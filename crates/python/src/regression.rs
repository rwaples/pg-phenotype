//! `parent_offspring`: the regressions as nested dicts of plain values (the
//! core's `to_value` tree), which `pg_phenotype.regression` turns into
//! frozen dataclasses.

use crate::crossing::{self, StrataArg, TraitArg};
use crate::PedigreeSource;
use pg_phenotype_core::regression::{self, Lines, Parent};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The parent-offspring regressions of one trait, in the pool, of the
/// lines named in `lines`.
#[pyfunction]
#[pyo3(signature = (pedigree, trait_, strata, /, *, lines, inference))]
pub(crate) fn parent_offspring<'py>(
    py: Python<'py>,
    pedigree: PedigreeSource<'py>,
    trait_: TraitArg<'py>,
    strata: StrataArg<'py>,
    lines: Vec<String>,
    inference: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let lines = lines
        .iter()
        .map(|name| {
            Parent::parse(name)
                .ok_or_else(|| PyValueError::new_err(format!("{name:?} is not a line")))
        })
        .collect::<PyResult<Lines>>()?;
    let pedigree = pedigree.arg()?;
    let trait_ = trait_.get()?;
    let strata = crossing::strata(&strata)?;
    let result = crossing::run(py, || {
        let settings = regression::Settings { lines, inference };
        regression::parent_offspring(pedigree, trait_, strata, settings).map(|r| r.to_value())
    })?;
    crossing::result(py, result)
}
