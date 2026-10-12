//! `relative_pairs_correlation`: the Relative-Pair Correlation as nested
//! dicts of plain values (the core's `to_value` tree), which
//! `pg_phenotype.relative_pairs` turns into frozen dataclasses.

use crate::crossing::{self, TraitArg};
use crate::PedigreeSource;
use pg_phenotype_core::relative_pairs::{self, Entry};
use pyo3::prelude::*;

/// One entry as the wrapper passes it: its position, its code or group
/// name, and a group's member codes.
type HostEntry = (usize, String, Option<Vec<String>>);

/// The Relative-Pair Correlation of one trait, in the pool.  Each entry of
/// `categories` is `(position, code, None)` or `(position, group name, member
/// codes)`, `position` its index in the caller's request.
#[pyfunction]
#[pyo3(signature = (pedigree, trait_, categories, memory_budget_bytes, /))]
pub(crate) fn relative_pairs_correlation<'py>(
    py: Python<'py>,
    pedigree: PedigreeSource<'py>,
    trait_: TraitArg<'py>,
    categories: Option<Vec<HostEntry>>,
    memory_budget_bytes: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let pedigree = pedigree.arg()?;
    let trait_ = trait_.get()?;
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
    let result = crossing::run(py, || {
        relative_pairs::correlation(pedigree, trait_, entries.as_deref(), memory_budget_bytes)
            .map(|r| r.to_value())
    })?;
    crossing::result(py, result)
}
