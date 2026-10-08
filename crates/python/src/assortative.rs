//! `mate_correlation`: the Mate Correlation result as nested dicts of plain
//! values (the core's `to_value` tree), which `pg_phenotype.assortative`
//! turns into frozen dataclasses.

use crate::{checked_pool, to_pyerr, trait_kind, PedigreeArgs};
use numpy::PyReadonlyArray1;
use pg_phenotype_core::assortative::{self, Settings, Strata};
use pg_phenotype_core::value::Value;
use pg_phenotype_core::Trait;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

/// A core result tree as nested Python dicts, lists and tuples.
pub(crate) fn to_object<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(v) => v.into_pyobject(py)?.to_owned().into_any(),
        Value::Count(v) => v.into_pyobject(py)?.into_any(),
        Value::Int(v) => v.into_pyobject(py)?.into_any(),
        Value::Float(v) => v.into_pyobject(py)?.into_any(),
        Value::Str(v) => v.into_pyobject(py)?.into_any(),
        Value::Floats(v) => PyTuple::new(py, v)?.into_any(),
        Value::Counts(v) => PyList::new(py, v)?.into_any(),
        Value::List(items) => {
            let out = PyList::empty(py);
            for item in items {
                out.append(to_object(py, item)?)?;
            }
            out.into_any()
        }
        Value::Map(entries) => {
            let out = PyDict::new(py);
            for (key, item) in entries {
                out.set_item(key, to_object(py, item)?)?;
            }
            out.into_any()
        }
    })
}

/// One trait as the host passes it: values, kind, declared level count.
type TraitArg<'py> = (PyReadonlyArray1<'py, f64>, String, Option<usize>);

/// The Mate Correlation of one or two traits, in the pool.
#[pyfunction]
#[pyo3(signature = (ids, mother, father, twin, sex, traits, stratum_labels, stratum_known, /, *, permutations, bootstrap, seed, min_stratum_networks, threads))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mate_correlation<'py>(
    py: Python<'py>,
    ids: PyReadonlyArray1<'py, i64>,
    mother: PyReadonlyArray1<'py, i64>,
    father: PyReadonlyArray1<'py, i64>,
    twin: Option<PyReadonlyArray1<'py, i64>>,
    sex: Option<PyReadonlyArray1<'py, i64>>,
    traits: Vec<TraitArg<'py>>,
    stratum_labels: Option<PyReadonlyArray1<'py, i64>>,
    stratum_known: Option<PyReadonlyArray1<'py, bool>>,
    permutations: i64,
    bootstrap: i64,
    seed: i64,
    min_stratum_networks: i64,
    threads: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let pedigree = PedigreeArgs::new(ids, mother, father, twin, sex);
    let input = pedigree.input()?;
    let traits = traits
        .iter()
        .map(|(values, kind, n_levels)| {
            Ok(Trait {
                values: values.as_slice()?,
                kind: trait_kind(kind)?,
                n_levels: *n_levels,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let strata = match (&stratum_labels, &stratum_known) {
        (Some(labels), Some(known)) => Some(Strata {
            labels: labels.as_slice()?,
            known: known.as_slice()?,
        }),
        (None, None) => None,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "stratum_labels and stratum_known are passed together",
            ))
        }
    };
    let settings = Settings {
        permutations,
        bootstrap,
        seed,
        min_stratum_networks,
    };
    let pool = checked_pool(py, threads)?;
    let result = py
        .detach(|| pool.install(|| assortative::mate_correlation(input, &traits, strata, settings)))
        .map_err(|e| to_pyerr(py, e))?;
    to_object(py, &result.to_value())
}
