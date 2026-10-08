//! `mate_correlation`: the Mate Correlation result as nested dicts of plain
//! values, which `pg_phenotype.assortative` turns into frozen dataclasses.

use crate::{checked_pool, to_pyerr, trait_kind, PedigreeArgs};
use numpy::PyReadonlyArray1;
use pg_phenotype_core::assortative::{
    self, Cell, Draws, Estimate, EstimatorResult, MateCorrelation, Permutation, Point, Reason,
    Settings, Strata, WithinPerson,
};
use pg_phenotype_core::Trait;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

type Dict<'py> = Bound<'py, PyDict>;

fn reason_or<'py, T: IntoPyObject<'py>>(
    py: Python<'py>,
    out: &Dict<'py>,
    key: &str,
    value: Result<T, Reason>,
) -> PyResult<()> {
    match value {
        Ok(v) => {
            out.set_item(key, v)?;
            out.set_item(format!("{key}_unavailable_reason"), py.None())
        }
        Err(r) => {
            out.set_item(key, py.None())?;
            out.set_item(format!("{key}_unavailable_reason"), r.name())
        }
    }
}

fn draws<'py>(py: Python<'py>, d: &Draws) -> PyResult<Dict<'py>> {
    let out = PyDict::new(py);
    out.set_item("requested", d.requested)?;
    out.set_item("valid", d.valid)?;
    out.set_item("failed", d.failed)?;
    let reasons = PyDict::new(py);
    for (r, n) in &d.failure_reasons {
        reasons.set_item(r.name(), n)?;
    }
    out.set_item("failure_reasons", reasons)?;
    Ok(out)
}

fn permutation<'py>(py: Python<'py>, p: &Permutation) -> PyResult<Dict<'py>> {
    let out = PyDict::new(py);
    out.set_item("statistic", p.statistic.name())?;
    reason_or(py, &out, "p", p.p)?;
    out.set_item("draws", draws(py, &p.draws)?)?;
    out.set_item("seed", p.seed)?;
    out.set_item("n_fixed_fathers", p.n_fixed_fathers)?;
    out.set_item("stopped_early", p.stopped_early)?;
    out.set_item("draws_used", p.draws_used)?;
    out.set_item("sequential_h", p.sequential_h)?;
    Ok(out)
}

fn point<'py>(out: &Dict<'py>, p: &Point) -> PyResult<()> {
    out.set_item("value", p.value)?;
    out.set_item("boundary", p.boundary)
}

fn estimate<'py>(py: Python<'py>, out: &Dict<'py>, e: &Estimate) -> PyResult<()> {
    point(out, &e.point)?;
    reason_or(py, out, "se", e.se)?;
    reason_or(py, out, "ci", e.ci.map(|c| (c.bounds[0], c.bounds[1])))?;
    out.set_item("ci_method", e.ci.ok().map(|c| c.method.name()))?;
    out.set_item(
        "bootstrap",
        e.bootstrap.as_ref().map(|d| draws(py, d)).transpose()?,
    )?;
    out.set_item(
        "permutation",
        e.permutation
            .as_ref()
            .map(|p| permutation(py, p))
            .transpose()?,
    )
}

fn estimator_result<'py>(py: Python<'py>, r: &EstimatorResult) -> PyResult<Dict<'py>> {
    let out = PyDict::new(py);
    out.set_item("estimator", r.estimator.name())?;
    out.set_item("primary", r.primary)?;
    match &r.outcome {
        Ok(e) => {
            out.set_item("reason", py.None())?;
            estimate(py, &out, e)?;
        }
        Err(reason) => out.set_item("reason", reason.name())?,
    }
    Ok(out)
}

fn cell<'py>(py: Python<'py>, c: &Cell) -> PyResult<Dict<'py>> {
    let out = PyDict::new(py);
    out.set_item("mother_trait", c.mother_trait)?;
    out.set_item("father_trait", c.father_trait)?;
    out.set_item("n", c.n)?;
    let d = &c.n_dropped;
    let dropped = PyDict::new(py);
    dropped.set_item("mother_missing", d.mother_missing)?;
    dropped.set_item("father_missing", d.father_missing)?;
    dropped.set_item("both_missing", d.both_missing)?;
    dropped.set_item("small_stratum", d.small_stratum)?;
    dropped.set_item("degenerate_stratum", d.degenerate_stratum)?;
    out.set_item("n_dropped", dropped)?;
    out.set_item("n_mate_networks", c.n_mate_networks)?;
    out.set_item("largest_mate_network_share", c.largest_mate_network_share)?;
    out.set_item("table", c.table.map(|t| t.map(|r| r.to_vec()).to_vec()))?;
    let crude = PyList::empty(py);
    for r in &c.crude {
        crude.append(estimator_result(py, r)?)?;
    }
    out.set_item("crude", crude)?;
    let stratified = match &c.stratified {
        None => None,
        Some(s) => {
            let d = estimator_result(py, &s.result)?;
            d.set_item("n_strata_mothers", s.n_strata_mothers)?;
            d.set_item("n_strata_fathers", s.n_strata_fathers)?;
            Some(d)
        }
    };
    out.set_item("stratified", stratified)?;
    Ok(out)
}

fn within<'py>(py: Python<'py>, w: &WithinPerson) -> PyResult<Dict<'py>> {
    let out = PyDict::new(py);
    out.set_item("estimator", w.estimator.name())?;
    out.set_item("n", w.n)?;
    match &w.outcome {
        Ok(p) => {
            point(&out, p)?;
            out.set_item("reason", py.None())?;
        }
        Err(r) => {
            out.set_item("value", py.None())?;
            out.set_item("boundary", py.None())?;
            out.set_item("reason", r.name())?;
        }
    }
    Ok(out)
}

fn to_dict<'py>(py: Python<'py>, r: &MateCorrelation) -> PyResult<Dict<'py>> {
    let out = PyDict::new(py);
    let s = &r.sample;
    let sample = PyDict::new(py);
    sample.set_item("n_total", s.n_total)?;
    sample.set_item("n_dropped_unknown_stratum", s.n_dropped_unknown_stratum)?;
    sample.set_item("n_mate_networks", s.n_mate_networks)?;
    sample.set_item("largest_mate_network_share", s.largest_mate_network_share)?;
    sample.set_item("n_mothers_multiple_mates", s.n_mothers_multiple_mates)?;
    sample.set_item("n_fathers_multiple_mates", s.n_fathers_multiple_mates)?;
    out.set_item("sample", sample)?;
    let cells = PyList::empty(py);
    for c in &r.cells {
        cells.append(cell(py, c)?)?;
    }
    out.set_item("cells", cells)?;
    let wp = match &r.within_person {
        None => None,
        Some(w) => {
            let d = PyDict::new(py);
            d.set_item("mothers", within(py, &w.mothers)?)?;
            d.set_item("fathers", within(py, &w.fathers)?)?;
            Some(d)
        }
    };
    out.set_item("within_person", wp)?;
    let e = &r.settings;
    let settings = PyDict::new(py);
    settings.set_item("permutations", e.permutations)?;
    settings.set_item("bootstrap", e.bootstrap)?;
    settings.set_item("seed", e.seed)?;
    settings.set_item("threads", e.threads)?;
    settings.set_item("ci_level", e.ci_level)?;
    settings.set_item("min_stratum_networks", e.min_stratum_networks)?;
    out.set_item("settings", settings)?;
    let m = &r.method;
    let method = PyDict::new(py);
    method.set_item("se_method", m.se_method)?;
    method.set_item("ci_scale", m.ci_scale)?;
    method.set_item("bootstrap_unit", m.bootstrap_unit)?;
    method.set_item("bootstrap_assumption", m.bootstrap_assumption)?;
    method.set_item("bootstrap_method", m.bootstrap_method)?;
    method.set_item("permutation_null", m.permutation_null)?;
    method.set_item("permutation_blocks", m.permutation_blocks)?;
    method.set_item("permutation_statistic", m.permutation_statistic)?;
    method.set_item("permutation_stopping", m.permutation_stopping)?;
    method.set_item("permutation_stop_h", m.permutation_stop_h)?;
    out.set_item("method", method)?;
    Ok(out)
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
) -> PyResult<Dict<'py>> {
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
    to_dict(py, &result)
}
