//! `mate_correlation`: the Mate Correlation result as nested dicts of plain
//! values (the core's `to_value` tree), which `pg_phenotype.assortative`
//! turns into frozen dataclasses.

use crate::crossing::{self, StrataArg, TraitArg};
use crate::PedigreeSource;
use pg_phenotype_core::assortative::{self, Settings};
use pyo3::prelude::*;

/// The Mate Correlation of one or two traits, in the pool.
#[pyfunction]
#[pyo3(signature = (pedigree, traits, strata, /, *, permutations, bootstrap, seed, min_stratum_networks, spearman, inference))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mate_correlation<'py>(
    py: Python<'py>,
    pedigree: PedigreeSource<'py>,
    traits: Vec<TraitArg<'py>>,
    strata: StrataArg<'py>,
    permutations: Option<i64>,
    bootstrap: i64,
    seed: i64,
    min_stratum_networks: i64,
    spearman: bool,
    inference: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let pedigree = pedigree.arg()?;
    let traits = traits
        .iter()
        .map(TraitArg::get)
        .collect::<PyResult<Vec<_>>>()?;
    let strata = crossing::strata(&strata)?;
    let settings = Settings {
        permutations,
        bootstrap,
        seed,
        min_stratum_networks,
        spearman,
        inference,
    };
    let result = crossing::run(py, || {
        assortative::mate_correlation(pedigree, &traits, strata, settings).map(|r| r.to_value())
    })?;
    crossing::result(py, result)
}
