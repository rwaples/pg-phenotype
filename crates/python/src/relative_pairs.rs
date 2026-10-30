//! `relative_pairs_correlation` and `relative_pairs_ace`: the
//! Relative-Pair Correlation and its ACE Fit as nested dicts of plain values
//! (the core's `to_value` trees), which `pg_phenotype.relative_pairs` turns
//! into frozen dataclasses.

use crate::crossing::{self, StrataArg, TraitArg};
use crate::{to_pyerr, PedigreeSource};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pg_phenotype_core::ace::{
    self, AceCell, AceRequest, Components, CrudeEstimate, Estimate, Estimator, Pairs,
    StratifiedEstimate, Weights,
};
use pg_phenotype_core::relative_pairs::{self, CellEntries, Covariance, Entry, Se};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// One entry as the wrapper passes it: its position, its code or group
/// name, and a group's member codes.
type HostEntry = (usize, String, Option<Vec<String>>);

/// The Relative-Pair Correlation of one or two traits, in the pool.  Each entry of
/// `categories` is `(position, code, None)` or `(position, group name, member
/// codes)`, `position` its index in the caller's request.  `se` asks for
/// each estimate's cluster sandwich SE, `covariance`, with it, for the
/// covariance of the estimates the ACE Fit reads, and `influence`, with it,
/// for each cell's entries and their influences, under the result's
/// `entries` key.
#[pyfunction]
#[pyo3(signature = (pedigree, traits, strata, categories, memory_budget_bytes, se, covariance, influence, /))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn relative_pairs_correlation<'py>(
    py: Python<'py>,
    pedigree: PedigreeSource<'py>,
    traits: Vec<TraitArg<'py>>,
    strata: StrataArg<'py>,
    categories: Option<Vec<HostEntry>>,
    memory_budget_bytes: i64,
    se: bool,
    covariance: bool,
    influence: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let se = Se::from_flags(se, covariance)
        .ok_or_else(|| PyValueError::new_err("covariance needs se"))?;
    if influence && se == Se::Off {
        return Err(PyValueError::new_err("influence needs se"));
    }
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
    let (result, cells) = crossing::run(py, || {
        relative_pairs::correlation_with(
            pedigree,
            &traits,
            strata,
            entries.as_deref(),
            memory_budget_bytes,
            se,
            influence,
        )
        .map(|mut r| (r.to_value(), r.entries.take()))
    })?;
    let out = crossing::result(py, result)?;
    let cells = cells
        .map(|cells| -> PyResult<_> {
            let list = PyList::empty(py);
            for cell in cells {
                list.append(cell_entries(py, cell)?)?;
            }
            Ok(list.into_any())
        })
        .transpose()?;
    out.set_item("entries", cells)?;
    Ok(out)
}

/// One cell's entries as a dict of numpy arrays, each influence moved
/// without a copy.
fn cell_entries(py: Python<'_>, cell: CellEntries) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("first_id", cell.first_id.into_pyarray(py))?;
    out.set_item("second_id", cell.second_id.into_pyarray(py))?;
    out.set_item("mrca_id", cell.mrca_id.into_pyarray(py))?;
    let influence = PyList::empty(py);
    for column in cell.influence {
        influence.append(column.map(|c| c.into_pyarray(py)))?;
    }
    out.set_item("influence", influence)?;
    Ok(out)
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
    py: Python<'_>,
    (name, members, first_trait, second_trait, crude, n, n_entries, stratified): HostCell,
) -> PyResult<AceCell> {
    Ok(AceCell {
        pairs: match members {
            None => Pairs::WithinPerson,
            Some(codes) => Pairs::relatives(&codes).map_err(|e| to_pyerr(py, e))?,
        },
        first_trait,
        second_trait,
        crude: crude
            .into_iter()
            .map(|(estimator, v, reason)| {
                Ok(CrudeEstimate {
                    estimator: relative_pairs::Estimator::parse(&estimator).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "cell {name:?}: crude estimate {estimator:?} is not an estimator"
                        ))
                    })?,
                    value: value(v, reason),
                })
            })
            .collect::<PyResult<_>>()?,
        name,
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

/// A result's covariance as the wrapper passes it: the pooled, phi and
/// stratified matrices, each flat and row-major.
type HostCovariance<'py> = (
    PyReadonlyArray1<'py, f64>,
    Option<PyReadonlyArray1<'py, f64>>,
    Option<PyReadonlyArray1<'py, f64>>,
);

fn flat(matrix: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    Ok(matrix.as_slice()?.to_vec())
}

/// The ACE Fit of a result's cells, which the wrapper checked, with the
/// result's covariance when it has one; `estimate` and `weights` `None`
/// are the core's defaults.
#[pyfunction]
#[pyo3(signature = (cells, categories, components, shared_environment, estimator, estimate, weights, covariance, /))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn relative_pairs_ace<'py>(
    py: Python<'py>,
    cells: Vec<HostCell>,
    categories: Option<Vec<String>>,
    components: Vec<String>,
    shared_environment: Option<Vec<String>>,
    estimator: String,
    estimate: Option<String>,
    weights: Option<String>,
    covariance: Option<HostCovariance<'py>>,
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
        estimate: estimate
            .map(|e| named("estimate", &e, Estimate::parse(&e)))
            .transpose()?,
        weights: weights
            .map(|w| named("weights", &w, Weights::parse(&w)))
            .transpose()?,
    };
    let matrices = covariance
        .map(|(pooled, phi, stratified)| -> PyResult<_> {
            Ok((
                flat(&pooled)?,
                phi.as_ref().map(flat).transpose()?,
                stratified.as_ref().map(flat).transpose()?,
            ))
        })
        .transpose()?;
    let cells = cells
        .into_iter()
        .map(|c| cell(py, c))
        .collect::<PyResult<Vec<_>>>()?;
    let fit = matrices
        .map(|(pooled, phi, stratified)| Covariance::new(cells.len(), pooled, phi, stratified))
        .transpose()
        .and_then(|covariance| ace::fit(&cells, &request, covariance.as_ref()))
        .map_err(|e| to_pyerr(py, e))?;
    crossing::result(py, fit.to_value())
}
