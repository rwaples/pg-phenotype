//! `pg_phenotype._native`: the PyO3 host binding of pg-phenotype-core.
//!
//! Core errors cross as the structured classes of `pg_phenotype._errors`, keyed
//! by `.code` with keyword fields from `Error::fields`; a thread-pool conflict
//! crosses as `RuntimeError`, as `configure_threads` raises it.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pg_phenotype_core::error::{Class, FieldValue};
use pg_phenotype_core::pafgrs::{self, BivParams, Cip};
use pg_phenotype_core::{threads, Error, PedigreeArg, PedigreeInput, Trait, TraitKind};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::num::NonZeroUsize;

mod assortative;
#[cfg(feature = "test-hooks")]
mod test_hooks;

/// A proband's relative rows and their kinship to it.
type Relatives<'py> = (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>);

/// The Cargo workspace version, which is also the Python distribution version.
#[pyfunction]
fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The pedigree-graph-core git revision this build links.
#[pyfunction]
fn pg_core_rev() -> &'static str {
    pg_phenotype_core::PG_CORE_REV
}

fn field_object(py: Python<'_>, value: FieldValue) -> PyResult<Py<PyAny>> {
    Ok(match value {
        FieldValue::Int(v) => v.into_pyobject(py)?.into_any().unbind(),
        FieldValue::Float(v) => v.into_pyobject(py)?.into_any().unbind(),
        FieldValue::Str(v) => v.into_pyobject(py)?.into_any().unbind(),
        FieldValue::Ints(v) => PyTuple::new(py, v)?.into_any().unbind(),
        FieldValue::Strs(v) => PyTuple::new(py, v)?.into_any().unbind(),
    })
}

pub(crate) fn to_pyerr(py: Python<'_>, err: Error) -> PyErr {
    if err.code() == "thread_pool_conflict" {
        return PyRuntimeError::new_err(err.to_string());
    }
    let class_name = match err.class() {
        Class::Validation => "ValidationError",
        Class::Parameter => "ParameterError",
        Class::Resource => "ResourceError",
        Class::Usage => return PyValueError::new_err(err.to_string()),
    };
    let raise = || -> PyResult<PyErr> {
        let fields = PyDict::new(py);
        for (name, value) in err.fields() {
            fields.set_item(name, field_object(py, value)?)?;
        }
        let class = py.import("pg_phenotype._errors")?.getattr(class_name)?;
        let instance = class.call((err.code(), err.to_string()), Some(&fields))?;
        Ok(PyErr::from_value(instance))
    };
    raise().unwrap_or_else(|e| e)
}

fn budget_error(err: threads::BudgetError) -> PyErr {
    use threads::{BudgetError, ENV_VAR, MAX_THREADS};
    match err {
        BudgetError::OutOfRange { requested } => PyValueError::new_err(format!(
            "configure_threads(n) requires an int from 1 to {MAX_THREADS}, got {requested}"
        )),
        BudgetError::Conflict {
            committed,
            requested,
        } => PyRuntimeError::new_err(format!(
            "the committed thread budget is {committed} and cannot be changed to {requested}; \
             call configure_threads() before the first thread_budget() call"
        )),
        BudgetError::InvalidEnv { raw } => PyValueError::new_err(format!(
            "{ENV_VAR} must be a decimal integer from 1 to {MAX_THREADS}, got {raw:?}"
        )),
    }
}

/// Record the package thread budget (`pg_phenotype.configure_threads`).
#[pyfunction]
fn configure_threads(n: usize) -> PyResult<()> {
    threads::configure(n).map_err(budget_error)
}

/// The committed package thread budget, committing it on first call.
#[pyfunction]
fn thread_budget() -> PyResult<usize> {
    threads::budget().map_err(budget_error)
}

/// Clear the budget; for tests only (the pool keeps its size).
#[pyfunction]
fn _reset_thread_budget() {
    threads::reset();
}

pub(crate) fn checked_pool(
    py: Python<'_>,
    threads: usize,
) -> PyResult<&'static pg_phenotype_core::rayon::ThreadPool> {
    let threads = NonZeroUsize::new(threads)
        .ok_or_else(|| PyValueError::new_err("threads must be at least 1"))?;
    pg_phenotype_core::configure_pool(threads).map_err(|e| to_pyerr(py, e))
}

/// The relative structure of one pedigree at one degree, held in memory.
#[pyclass(module = "pg_phenotype._native", frozen)]
struct Prep {
    inner: pafgrs::Prep,
}

#[pymethods]
impl Prep {
    #[getter]
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    #[getter]
    fn ndegree(&self) -> u8 {
        self.inner.ndegree()
    }

    #[getter]
    fn n_probands(&self) -> usize {
        self.inner.n_probands()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.bytes()
    }

    /// Proband rows, ascending.
    fn proband_rows<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.probands().into_pyarray(py)
    }

    /// Proband `i`'s relative rows and their kinship to it.
    fn relatives<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Relatives<'py>> {
        self.check_proband(i)?;
        let (rows, kin) = self.inner.relatives(i);
        Ok((
            rows.to_vec().into_pyarray(py),
            kin.to_vec().into_pyarray(py),
        ))
    }

    /// Proband `i`'s relative-relative kinship, row-major upper triangle.
    fn triangle<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.check_proband(i)?;
        let d = self.inner.relatives(i).0.len();
        let values: Vec<f32> = (0..d)
            .flat_map(|j| (j + 1..d).map(move |k| (j, k)))
            .map(|(j, k)| self.inner.pair_kinship(i, j, k))
            .collect();
        Ok(values.into_pyarray(py))
    }
}

impl Prep {
    fn check_proband(&self, i: usize) -> PyResult<()> {
        let n = self.inner.n_probands();
        if i >= n {
            return Err(PyValueError::new_err(format!(
                "proband index {i} is not below {n}"
            )));
        }
        Ok(())
    }
}

/// One pedigree column as `pg_phenotype._input.pedigree_arrays` passes it.
type Column<'py> = PyReadonlyArray1<'py, i64>;

/// The pedigree columns `(id, mother, father, twin, sex)`.
#[derive(FromPyObject)]
pub(crate) struct PedigreeArgs<'py>(
    Column<'py>,
    Column<'py>,
    Column<'py>,
    Option<Column<'py>>,
    Option<Column<'py>>,
);

impl PedigreeArgs<'_> {
    fn input(&self) -> PyResult<PedigreeInput<'_>> {
        Ok(PedigreeInput {
            ids: self.0.as_slice()?,
            mother: self.1.as_slice()?,
            father: self.2.as_slice()?,
            twin: self.3.as_ref().map(|a| a.as_slice()).transpose()?,
            sex: self.4.as_ref().map(|a| a.as_slice()).transpose()?,
        })
    }
}

/// A validated pedigree that methods share (`pg_phenotype.Pedigree`).
#[pyclass(module = "pg_phenotype._native", name = "Pedigree", frozen)]
struct Pedigree {
    inner: pg_phenotype_core::Pedigree,
}

#[pymethods]
impl Pedigree {
    /// Validate the columns, with the GIL released.
    #[new]
    #[pyo3(signature = (ids, mother, father, twin, sex, /))]
    fn new<'py>(
        py: Python<'py>,
        ids: Column<'py>,
        mother: Column<'py>,
        father: Column<'py>,
        twin: Option<Column<'py>>,
        sex: Option<Column<'py>>,
    ) -> PyResult<Pedigree> {
        let columns = PedigreeArgs(ids, mother, father, twin, sex);
        let input = columns.input()?;
        let inner = py
            .detach(|| pg_phenotype_core::Pedigree::new(input))
            .map_err(|e| to_pyerr(py, e))?;
        Ok(Pedigree { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// A new array of the row ids, in input order.
    #[getter]
    fn ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, self.inner.ids())
    }
}

/// What a method takes: a [`Pedigree`] or its columns.
#[derive(FromPyObject)]
pub(crate) enum PedigreeSource<'py> {
    Built(Bound<'py, Pedigree>),
    Columns(PedigreeArgs<'py>),
}

impl PedigreeSource<'_> {
    pub(crate) fn arg(&self) -> PyResult<PedigreeArg<'_>> {
        Ok(match self {
            PedigreeSource::Built(pedigree) => PedigreeArg::Built(&pedigree.get().inner),
            PedigreeSource::Columns(columns) => PedigreeArg::Columns(columns.input()?),
        })
    }
}

/// An int as i64, saturated at the int64 bounds, so a range the core checks
/// reports its own error for any int rather than an `OverflowError`.
fn saturating_i64(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    match value.extract::<i64>() {
        Ok(v) => Ok(v),
        Err(err) if err.is_instance_of::<PyOverflowError>(value.py()) => {
            Ok(if value.lt(0)? { i64::MIN } else { i64::MAX })
        }
        Err(err) => Err(err),
    }
}

/// Build a pedigree's relative structure in the pool, validating columns.
#[pyfunction]
#[pyo3(signature = (pedigree, /, *, ndegree, probands, threads))]
fn prepare<'py>(
    py: Python<'py>,
    pedigree: PedigreeSource<'py>,
    ndegree: &Bound<'py, PyAny>,
    probands: Option<PyReadonlyArray1<'py, i64>>,
    threads: usize,
) -> PyResult<Prep> {
    let ndegree = saturating_i64(ndegree)?;
    let pedigree = pedigree.arg()?;
    let probands = probands.as_ref().map(|a| a.as_slice()).transpose()?;
    let pool = checked_pool(py, threads)?;
    let inner = py
        .detach(|| pool.install(|| pafgrs::prepare(pedigree, ndegree, probands)))
        .map_err(|e| to_pyerr(py, e))?;
    Ok(Prep { inner })
}

pub(crate) fn trait_kind(kind: &str) -> PyResult<TraitKind> {
    TraitKind::from_name(kind).ok_or_else(|| {
        PyValueError::new_err(format!(
            "trait kind must be one of {}, got {kind:?}",
            trait_kinds().join(", ")
        ))
    })
}

/// The trait kind names, in the order hosts list them.
#[pyfunction]
fn trait_kinds() -> Vec<&'static str> {
    TraitKind::ALL.map(TraitKind::name).to_vec()
}

fn checked_cip(py: Python<'_>, ages: Vec<f64>, cip: Vec<f64>) -> PyResult<Cip> {
    Cip::new(ages, cip).map_err(|e| to_pyerr(py, e))
}

fn f64_array<'py>(py: Python<'py>, values: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    values.into_pyarray(py)
}

/// Validate a CIP table and return its prevalence and threshold.
#[pyfunction]
fn check_cip(py: Python<'_>, ages: Vec<f64>, cip: Vec<f64>) -> PyResult<(f64, f64)> {
    let cip = checked_cip(py, ages, cip)?;
    Ok((cip.prevalence(), cip.threshold()))
}

/// Univariate scores of every proband, in the pool.
#[pyfunction]
#[pyo3(signature = (prep, values, kind, age, cip_ages, cip_values, /, *, h2, threads))]
#[allow(clippy::too_many_arguments)]
fn score_univariate<'py>(
    py: Python<'py>,
    prep: &Prep,
    values: PyReadonlyArray1<'py, f64>,
    kind: &str,
    age: PyReadonlyArray1<'py, f64>,
    cip_ages: Vec<f64>,
    cip_values: Vec<f64>,
    h2: f64,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let cip = checked_cip(py, cip_ages, cip_values)?;
    let values = Trait {
        values: values.as_slice()?,
        kind: trait_kind(kind)?,
        n_levels: None,
    };
    let age = age.as_slice()?;
    let pool = checked_pool(py, threads)?;
    let scores = py
        .detach(|| pool.install(|| pafgrs::score_univariate(&prep.inner, values, age, &cip, h2)))
        .map_err(|e| to_pyerr(py, e))?;
    let out = PyDict::new(py);
    out.set_item("id", scores.ids.into_pyarray(py))?;
    out.set_item("est", f64_array(py, scores.est))?;
    out.set_item("var", f64_array(py, scores.var))?;
    out.set_item("n_relatives", scores.n_relatives.into_pyarray(py))?;
    out.set_item("controls_without_age", scores.controls_without_age)?;
    out.set_item("threshold", scores.threshold)?;
    Ok(out)
}

/// Bivariate scores of every proband, in the pool.
#[pyfunction]
#[pyo3(signature = (prep, values1, kind1, age1, values2, kind2, age2, cip1, cip2, /, *, h2, rg, rho_within, threads))]
#[allow(clippy::too_many_arguments)]
fn score_bivariate<'py>(
    py: Python<'py>,
    prep: &Prep,
    values1: PyReadonlyArray1<'py, f64>,
    kind1: &str,
    age1: PyReadonlyArray1<'py, f64>,
    values2: PyReadonlyArray1<'py, f64>,
    kind2: &str,
    age2: PyReadonlyArray1<'py, f64>,
    cip1: (Vec<f64>, Vec<f64>),
    cip2: (Vec<f64>, Vec<f64>),
    h2: (f64, f64),
    rg: f64,
    rho_within: Option<f64>,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let params = BivParams::new([h2.0, h2.1], rg, rho_within).map_err(|e| to_pyerr(py, e))?;
    let cip1 = checked_cip(py, cip1.0, cip1.1)?;
    let cip2 = checked_cip(py, cip2.0, cip2.1)?;
    let values = [
        Trait {
            values: values1.as_slice()?,
            kind: trait_kind(kind1)?,
            n_levels: None,
        },
        Trait {
            values: values2.as_slice()?,
            kind: trait_kind(kind2)?,
            n_levels: None,
        },
    ];
    let ages = [age1.as_slice()?, age2.as_slice()?];
    let pool = checked_pool(py, threads)?;
    let scores = py
        .detach(|| {
            pool.install(|| {
                pafgrs::score_bivariate(&prep.inner, values, ages, [&cip1, &cip2], params)
            })
        })
        .map_err(|e| to_pyerr(py, e))?;
    let out = PyDict::new(py);
    let [est1, est2] = scores.est;
    let [var1, var2] = scores.var;
    let [n_obs1, n_obs2] = scores.n_obs;
    out.set_item("id", scores.ids.into_pyarray(py))?;
    out.set_item("est1", f64_array(py, est1))?;
    out.set_item("est2", f64_array(py, est2))?;
    out.set_item("var1", f64_array(py, var1))?;
    out.set_item("var2", f64_array(py, var2))?;
    out.set_item("cov12", f64_array(py, scores.cov12))?;
    out.set_item("n_relatives", scores.n_relatives.into_pyarray(py))?;
    out.set_item("n_obs1", n_obs1.into_pyarray(py))?;
    out.set_item("n_obs2", n_obs2.into_pyarray(py))?;
    out.set_item("controls_without_age", scores.controls_without_age.to_vec())?;
    out.set_item("threshold", scores.threshold.to_vec())?;
    out.set_item("rho_within", params.rho_within())?;
    Ok(out)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(pg_core_rev, m)?)?;
    m.add_function(wrap_pyfunction!(trait_kinds, m)?)?;
    m.add_function(wrap_pyfunction!(configure_threads, m)?)?;
    m.add_function(wrap_pyfunction!(thread_budget, m)?)?;
    m.add_function(wrap_pyfunction!(_reset_thread_budget, m)?)?;
    m.add_function(wrap_pyfunction!(prepare, m)?)?;
    m.add_function(wrap_pyfunction!(check_cip, m)?)?;
    m.add_function(wrap_pyfunction!(score_univariate, m)?)?;
    m.add_function(wrap_pyfunction!(score_bivariate, m)?)?;
    m.add_function(wrap_pyfunction!(assortative::mate_correlation, m)?)?;
    m.add_class::<Prep>()?;
    m.add_class::<Pedigree>()?;
    #[cfg(feature = "test-hooks")]
    m.add_function(wrap_pyfunction!(test_hooks::_panic_for_test, m)?)?;
    Ok(())
}
