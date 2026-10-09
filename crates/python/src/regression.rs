//! `parent_offspring`: the regressions as nested dicts of plain values (the
//! core's `to_value` tree), which `pg_phenotype.regression` turns into
//! frozen dataclasses.

use crate::assortative::to_object;
use crate::{checked_pool, to_pyerr, trait_kind, PedigreeSource};
use numpy::PyReadonlyArray1;
use pg_phenotype_core::assortative::Strata;
use pg_phenotype_core::regression;
use pg_phenotype_core::Trait;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The parent-offspring regressions of one trait, in the pool.
#[pyfunction]
#[pyo3(signature = (pedigree, values, kind, n_levels, stratum_labels, stratum_known, /, *, threads))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn parent_offspring<'py>(
    py: Python<'py>,
    pedigree: PedigreeSource<'py>,
    values: PyReadonlyArray1<'py, f64>,
    kind: String,
    n_levels: Option<usize>,
    stratum_labels: Option<PyReadonlyArray1<'py, i64>>,
    stratum_known: Option<PyReadonlyArray1<'py, bool>>,
    threads: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let pedigree = pedigree.arg()?;
    let trait_ = Trait {
        values: values.as_slice()?,
        kind: trait_kind(&kind)?,
        n_levels,
    };
    let strata = match (&stratum_labels, &stratum_known) {
        (Some(labels), Some(known)) => Some(Strata {
            labels: labels.as_slice()?,
            known: known.as_slice()?,
        }),
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "stratum_labels and stratum_known are passed together",
            ))
        }
    };
    let pool = checked_pool(py, threads)?;
    let result = py
        .detach(|| pool.install(|| regression::parent_offspring(pedigree, trait_, strata)))
        .map_err(|e| to_pyerr(py, e))?;
    to_object(py, &result.to_value())
}
