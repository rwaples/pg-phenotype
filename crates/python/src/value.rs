//! The core's result tree (`pg_phenotype_core::value::Value`) as nested
//! Python dicts, lists and tuples, for every method's binding.

use pg_phenotype_core::value::Value;
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
        Value::Text(v) => v.into_pyobject(py)?.into_any(),
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
