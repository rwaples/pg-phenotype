//! `mate_correlation`: the Mate Correlation result as nested dicts of plain
//! values (the core's `to_value` tree), which `pg_phenotype.assortative`
//! turns into frozen dataclasses.

use crate::value::to_object;
use crate::{checked_pool, to_pyerr, trait_kind, PedigreeSource};
use numpy::PyReadonlyArray1;
use pg_phenotype_core::assortative::{self, Settings, Strata};
use pg_phenotype_core::Trait;
use pyo3::prelude::*;

/// One trait as the host passes it: values, kind, declared level count.
type TraitArg<'py> = (PyReadonlyArray1<'py, f64>, String, Option<usize>);

/// The Mate Correlation of one or two traits, in the pool.
#[pyfunction]
#[pyo3(signature = (pedigree, traits, stratum_labels, stratum_known, /, *, permutations, bootstrap, seed, min_stratum_networks, spearman, inference, threads))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mate_correlation<'py>(
    py: Python<'py>,
    pedigree: PedigreeSource<'py>,
    traits: Vec<TraitArg<'py>>,
    stratum_labels: Option<PyReadonlyArray1<'py, i64>>,
    stratum_known: Option<PyReadonlyArray1<'py, bool>>,
    permutations: Option<i64>,
    bootstrap: i64,
    seed: i64,
    min_stratum_networks: i64,
    spearman: bool,
    inference: bool,
    threads: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let pedigree = pedigree.arg()?;
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
        spearman,
        inference,
    };
    let pool = checked_pool(py, threads)?;
    let result = py
        .detach(|| {
            pool.install(|| assortative::mate_correlation(pedigree, &traits, strata, settings))
        })
        .map_err(|e| to_pyerr(py, e))?;
    to_object(py, &result.to_value())
}
