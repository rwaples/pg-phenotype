//! `relative_pairs_correlation` and `relative_pairs_ace`: the
//! Relative-Pair Correlation and its ACE Fit as nested dicts of plain values
//! (the core's `to_value` trees), which `pg_phenotype.relative_pairs` turns
//! into frozen dataclasses.

use crate::crossing::{self, StrataArg, TraitArg};
use crate::{to_pyerr, PedigreeSource};
use pg_phenotype_core::ace::{
    self, AceCell, AceRequest, Components, CrudeEstimate, Estimate, Estimator, Pairs,
    StratifiedEstimate,
};
use pg_phenotype_core::relative_pairs::{self, Entry};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// One entry as the wrapper passes it: its position, its code or group
/// name, and a group's member codes.
type HostEntry = (usize, String, Option<Vec<String>>);

/// The Relative-Pair Correlation of one or two traits, in the pool.  Each entry of
/// `categories` is `(position, code, None)` or `(position, group name, member
/// codes)`, `position` its index in the caller's request.
#[pyfunction]
#[pyo3(signature = (pedigree, traits, strata, categories, memory_budget_bytes, /))]
pub(crate) fn relative_pairs_correlation<'py>(
    py: Python<'py>,
    pedigree: PedigreeSource<'py>,
    traits: Vec<TraitArg<'py>>,
    strata: StrataArg<'py>,
    categories: Option<Vec<HostEntry>>,
    memory_budget_bytes: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let pedigree = pedigree.arg()?;
    let traits = traits
        .iter()
        .map(TraitArg::get)
        .collect::<PyResult<Vec<_>>>()?;
    let strata = crossing::strata(&strata)?;
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
        relative_pairs::correlation(
            pedigree,
            &traits,
            strata,
            entries.as_deref(),
            memory_budget_bytes,
        )
        .map(|r| r.to_value())
    })?;
    crossing::result(py, result)
}

/// One cell as the wrapper passes it: name, member codes (`None` within
/// person), first and second trait, `(estimator, value, reason)` per crude
/// estimate, `n`, `n_entries`, and the stratified `(value, reason, n,
/// n_entries)`.
type HostCell = (
    String,
    Option<Vec<String>>,
    usize,
    usize,
    Vec<(String, Option<f64>, Option<String>)>,
    u64,
    Option<u64>,
    Option<(Option<f64>, Option<String>, u64, Option<u64>)>,
);

/// A result's estimate value, or the reason it is undefined.
fn value(value: Option<f64>, reason: Option<String>) -> Result<f64, String> {
    value.ok_or_else(|| reason.unwrap_or_default())
}

fn cell(
    (name, members, first_trait, second_trait, crude, n, n_entries, stratified): HostCell,
) -> Result<AceCell, pg_phenotype_core::Error> {
    Ok(AceCell {
        pairs: match members {
            None => Pairs::WithinPerson,
            Some(codes) => Pairs::relatives(&codes)?,
        },
        name,
        first_trait,
        second_trait,
        crude: crude
            .into_iter()
            .map(|(estimator, v, reason)| CrudeEstimate {
                estimator,
                value: value(v, reason),
            })
            .collect(),
        n,
        n_entries,
        stratified: stratified.map(|(v, reason, n, n_entries)| StratifiedEstimate {
            value: value(v, reason),
            n,
            n_entries,
        }),
    })
}

fn named<T>(what: &str, name: &str, parsed: Option<T>) -> PyResult<T> {
    parsed
        .ok_or_else(|| PyValueError::new_err(format!("{what} {name:?} is not one this fit takes")))
}

/// The ACE Fit of a result's cells, which the wrapper checked and resolved.
#[pyfunction]
#[pyo3(signature = (cells, categories, components, shared_environment, estimator, estimate, /))]
pub(crate) fn relative_pairs_ace<'py>(
    py: Python<'py>,
    cells: Vec<HostCell>,
    categories: Option<Vec<String>>,
    components: Vec<String>,
    shared_environment: Option<Vec<String>>,
    estimator: String,
    estimate: String,
) -> PyResult<Bound<'py, PyAny>> {
    let names: Vec<&str> = components.iter().map(String::as_str).collect();
    let request = AceRequest {
        categories,
        components: named(
            "components",
            &components.join(""),
            Components::parse(&names),
        )?,
        shared_environment,
        estimator: named("estimator", &estimator, Estimator::parse(&estimator))?,
        estimate: named("estimate", &estimate, Estimate::parse(&estimate))?,
    };
    let fit = cells
        .into_iter()
        .map(cell)
        .collect::<Result<Vec<_>, _>>()
        .and_then(|cells| ace::fit(&cells, &request))
        .map_err(|e| to_pyerr(py, e))?;
    crossing::result(py, fit.to_value())
}
