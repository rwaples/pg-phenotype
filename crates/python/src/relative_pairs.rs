//! `relative_pairs_correlation`: the Relative-Pair Correlation as nested
//! dicts of plain values (the core's `to_value` tree), which
//! `pg_phenotype.relative_pairs` turns into frozen dataclasses.

use crate::assortative::to_object;
use crate::{checked_pool, to_pyerr, trait_kind, PedigreeSource};
use numpy::PyReadonlyArray1;
use pg_phenotype_core::relative_pairs::{self, Entry};
use pg_phenotype_core::Trait;
use pyo3::prelude::*;

/// One entry as the wrapper passes it: its position, its code or group
/// name, and a group's member codes.
type HostEntry = (usize, String, Option<Vec<String>>);

/// The Relative-Pair Correlation of one trait, in the pool.  Each entry of
/// `categories` is `(position, code, None)` or `(position, group name, member
/// codes)`, `position` its index in the caller's request.
#[pyfunction]
#[pyo3(signature = (pedigree, values, kind, n_levels, categories, memory_budget_bytes, /, *, threads))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn relative_pairs_correlation<'py>(
    py: Python<'py>,
    pedigree: PedigreeSource<'py>,
    values: PyReadonlyArray1<'py, f64>,
    kind: String,
    n_levels: Option<usize>,
    categories: Option<Vec<HostEntry>>,
    memory_budget_bytes: i64,
    threads: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let pedigree = pedigree.arg()?;
    let trait_ = Trait {
        values: values.as_slice()?,
        kind: trait_kind(&kind)?,
        n_levels,
    };
    let entries: Option<Vec<Entry>> = categories.map(|entries| {
        entries
            .into_iter()
            .map(|(position, name, members)| match members {
                None => Entry::Category {
                    code: name,
                    position,
                },
                Some(members) => Entry::Group {
                    name,
                    members,
                    position,
                },
            })
            .collect()
    });
    let pool = checked_pool(py, threads)?;
    let result = py
        .detach(|| {
            pool.install(|| {
                relative_pairs::correlation(
                    pedigree,
                    trait_,
                    entries.as_deref(),
                    memory_budget_bytes,
                )
            })
        })
        .map_err(|e| to_pyerr(py, e))?;
    to_object(py, &result.to_value())
}
