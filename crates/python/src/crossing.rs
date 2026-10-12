//! How a method crosses from Python into the core, held once for every
//! method: the Trait whole, the free-memory release, the pool the thread
//! budget sizes with the GIL released, errors, and the result tree stamped
//! with the versions.  A method's binding states only its own parameters.
//!
//! Every call that runs in the pool first hands the process's free heap
//! pages back to the system (`malloc_trim`, glibc only).  The pool threads
//! allocate in arenas of their own, so memory the caller freed on its
//! thread (parsing the input, say) would otherwise stay resident beside
//! them.  In v0.1.1 this lowered `mate_correlation`'s peak in pedsum's CLI;
//! a trim costs microseconds on a quiet heap and about a millisecond after
//! millions of frees.

use crate::value::to_object;
use crate::{budget_error, to_pyerr, trait_kind};
use numpy::PyReadonlyArray1;
use pg_phenotype_core::assortative::Strata;
use pg_phenotype_core::value::Value;
use pg_phenotype_core::{threads, Error, Trait};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::num::NonZeroUsize;

/// A `pg_phenotype.Trait`, read by its attributes; the wrapper checked its type.
#[derive(FromPyObject)]
pub(crate) struct TraitArg<'py> {
    #[pyo3(attribute)]
    values: PyReadonlyArray1<'py, f64>,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    levels: Option<Bound<'py, PyTuple>>,
}

impl TraitArg<'_> {
    pub(crate) fn get(&self) -> PyResult<Trait<'_>> {
        Ok(Trait {
            values: self.values.as_slice()?,
            kind: trait_kind(&self.kind)?,
            n_levels: self.levels.as_ref().map(|levels| levels.len()),
        })
    }
}

/// Stratum labels and known flags, as `pg_phenotype._input.strata` returns them.
pub(crate) type StrataArg<'py> = Option<(PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, bool>)>;

pub(crate) fn strata<'a>(arg: &'a StrataArg<'_>) -> PyResult<Option<Strata<'a>>> {
    arg.as_ref()
        .map(|(labels, known)| {
            Ok(Strata {
                labels: labels.as_slice()?,
                known: known.as_slice()?,
            })
        })
        .transpose()
}

/// Run `compute` in the pool the committed thread budget sizes, with the
/// GIL released, after handing free memory back.
pub(crate) fn run<T, F>(py: Python<'_>, compute: F) -> PyResult<T>
where
    F: FnOnce() -> Result<T, Error> + Send,
    T: Send,
{
    let threads = threads::budget().map_err(budget_error)?;
    let threads = NonZeroUsize::new(threads).expect("a committed budget is at least 1");
    let pool = pg_phenotype_core::configure_pool(threads).map_err(|e| to_pyerr(py, e))?;
    py.detach(|| {
        release_free_memory();
        pool.install(compute)
    })
    .map_err(|e| to_pyerr(py, e))
}

/// A method's result tree as nested Python dicts, with its `metadata`.
pub(crate) fn result(py: Python<'_>, value: Value) -> PyResult<Bound<'_, PyAny>> {
    let Value::Map(mut entries) = value else {
        return Err(PyRuntimeError::new_err(
            "a method's result tree is not a map",
        ));
    };
    entries.push(("metadata", metadata(Vec::new())));
    to_object(py, &Value::Map(entries))
}

/// A result's metadata: the method's `entries`, then the versions.
pub(crate) fn metadata(mut entries: Vec<(&'static str, Value)>) -> Value {
    entries.push(("pg_phenotype_version", env!("CARGO_PKG_VERSION").into()));
    entries.push((
        "pedigree_graph_core_rev",
        pg_phenotype_core::PG_CORE_REV.into(),
    ));
    Value::Map(entries)
}

/// Releases so far, which the tests read to see that every crossing releases.
#[cfg(feature = "test-hooks")]
pub(crate) static RELEASES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Return the process's free heap pages to the system.
pub(crate) fn release_free_memory() {
    #[cfg(feature = "test-hooks")]
    RELEASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    malloc_trim();
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn malloc_trim() {
    // SAFETY: glibc declares `int malloc_trim(size_t pad)`.  It takes and
    // returns an integer, and no pointer crosses.
    unsafe extern "C" {
        #[link_name = "malloc_trim"]
        safe fn glibc_malloc_trim(pad: usize) -> std::ffi::c_int;
    }
    glibc_malloc_trim(0);
}

/// musl, macOS and Windows have no `malloc_trim`.
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn malloc_trim() {}
