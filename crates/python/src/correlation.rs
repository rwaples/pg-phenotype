//! `tetrachoric`: a 2 x 2 table's tetrachoric correlation as a dict of plain
//! values (the core's `to_value` tree), which `pg_phenotype.correlation`
//! turns into a frozen dataclass.

use crate::assortative::to_object;
use crate::to_pyerr;
use numpy::PyReadonlyArray1;
use pg_phenotype_core::correlation;
use pyo3::prelude::*;

/// The tetrachoric correlation of a table, rows by the `x` level.
#[pyfunction]
#[pyo3(signature = (table, /))]
pub(crate) fn tetrachoric_table(
    py: Python<'_>,
    table: [[u64; 2]; 2],
) -> PyResult<Bound<'_, PyAny>> {
    let result = correlation::tetrachoric(table).map_err(|e| to_pyerr(py, e))?;
    to_object(py, &result.to_value())
}

/// The tetrachoric correlation of paired binary values, `NaN` missing.
#[pyfunction]
#[pyo3(signature = (x, y, /))]
pub(crate) fn tetrachoric_pairs<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let result = correlation::tetrachoric_pairs(x.as_slice()?, y.as_slice()?)
        .map_err(|e| to_pyerr(py, e))?;
    to_object(py, &result.to_value())
}
