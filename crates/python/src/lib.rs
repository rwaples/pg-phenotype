//! `pg_phenotype._native`: the PyO3 host binding of pg-phenotype-core.
//!
//! Core errors cross as the structured classes of `pg_phenotype._errors`, keyed
//! by `.code` with keyword fields from `Error::fields`; a thread-pool conflict
//! crosses as `RuntimeError`, as `configure_threads` raises it.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pg_phenotype_core::error::{Class, FieldValue, IndexBase};
use pg_phenotype_core::pafgrs::{self, BivParams, Cip, Interpolation};
use pg_phenotype_core::{threads, Error, PedigreeArg, PedigreeInput, TraitKind};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

mod assortative;
mod correlation;
mod crossing;
mod regression;
mod relative_pairs;
#[cfg(feature = "test-hooks")]
mod test_hooks;
mod value;

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
        FieldValue::Text(v) => v.into_pyobject(py)?.into_any().unbind(),
        FieldValue::Texts(v) => PyTuple::new(py, v)?.into_any().unbind(),
        FieldValue::Ints(v) => PyTuple::new(py, v)?.into_any().unbind(),
        FieldValue::Strs(v) => PyTuple::new(py, v)?.into_any().unbind(),
    })
}

pub(crate) fn to_pyerr(py: Python<'_>, err: Error) -> PyErr {
    if err.code() == "thread_pool_conflict" {
        return PyRuntimeError::new_err(err.message(IndexBase::Zero));
    }
    let class_name = match err.class() {
        Class::Validation => "ValidationError",
        Class::Parameter => "ParameterError",
        Class::Resource => "ResourceError",
        Class::Usage => return PyValueError::new_err(err.message(IndexBase::Zero)),
    };
    let raise = || -> PyResult<PyErr> {
        let fields = PyDict::new(py);
        for (name, value) in err.fields(IndexBase::Zero) {
            fields.set_item(name, field_object(py, value)?)?;
        }
        let class = py.import("pg_phenotype._errors")?.getattr(class_name)?;
        let instance = class.call((err.code(), err.message(IndexBase::Zero)), Some(&fields))?;
        Ok(PyErr::from_value(instance))
    };
    raise().unwrap_or_else(|e| e)
}

pub(crate) fn budget_error(err: threads::BudgetError) -> PyErr {
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
    /// Validate the columns, with the GIL released, after handing free
    /// memory back.
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
            .detach(|| {
                crossing::release_free_memory();
                pg_phenotype_core::Pedigree::new(input)
            })
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

    /// The row of each of `ids`, with the GIL released.
    fn rows<'py>(&self, py: Python<'py>, ids: Column<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let ids = ids.as_slice()?;
        let rows = py
            .detach(|| self.inner.rows(ids))
            .map_err(|e| to_pyerr(py, e))?;
        Ok(rows
            .into_iter()
            .map(|row| row as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py))
    }

    /// `values` per `ids` on the rows, NaN elsewhere, with the GIL released.
    fn align<'py>(
        &self,
        py: Python<'py>,
        ids: Column<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (ids, values) = (ids.as_slice()?, values.as_slice()?);
        let aligned = py
            .detach(|| self.inner.align(ids, values, f64::NAN))
            .map_err(|e| to_pyerr(py, e))?;
        Ok(f64_array(py, aligned))
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
#[pyo3(signature = (pedigree, /, *, ndegree, probands))]
fn prepare<'py>(
    py: Python<'py>,
    pedigree: PedigreeSource<'py>,
    ndegree: &Bound<'py, PyAny>,
    probands: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Prep> {
    let ndegree = saturating_i64(ndegree)?;
    let pedigree = pedigree.arg()?;
    let probands = probands.as_ref().map(|a| a.as_slice()).transpose()?;
    let inner = crossing::run(py, || pafgrs::prepare(pedigree, ndegree, probands))?;
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

/// A CIP table as Python passes it: ages, values and the interpolation name.
type CipArg = (Vec<f64>, Vec<f64>, String);

fn checked_cip(py: Python<'_>, (ages, cip, interpolation): CipArg) -> PyResult<Cip> {
    let interpolation = Interpolation::from_name(&interpolation).ok_or_else(|| {
        PyValueError::new_err(format!(
            "interpolation must be 'linear' or 'step', got {interpolation:?}"
        ))
    })?;
    Cip::new(ages, cip, interpolation).map_err(|e| to_pyerr(py, e))
}

fn f64_array<'py>(py: Python<'py>, values: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    values.into_pyarray(py)
}

/// Validate a CIP table and return its prevalence and threshold.
#[pyfunction]
fn check_cip(py: Python<'_>, cip: CipArg) -> PyResult<(f64, f64)> {
    let cip = checked_cip(py, cip)?;
    Ok((cip.prevalence(), cip.threshold()))
}

/// The Aalen-Johansen CIP table of ages and survival statuses, as
/// `(ages, cip)`.
#[pyfunction]
#[pyo3(signature = (age, status, entry, /))]
fn cip_aalen_johansen<'py>(
    py: Python<'py>,
    age: PyReadonlyArray1<'py, f64>,
    status: PyReadonlyArray1<'py, f64>,
    entry: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (age, status) = (age.as_slice()?, status.as_slice()?);
    let entry = entry.as_ref().map(|e| e.as_slice()).transpose()?;
    py.detach(|| Cip::aalen_johansen(age, status, entry))
        .map(Cip::into_parts)
        .map_err(|e| to_pyerr(py, e))
}

/// Univariate scores of every proband, in the pool.
#[pyfunction]
#[pyo3(signature = (prep, trait_, age, cip, /, *, h2))]
fn score_univariate<'py>(
    py: Python<'py>,
    prep: &Prep,
    trait_: crossing::TraitArg<'py>,
    age: PyReadonlyArray1<'py, f64>,
    cip: CipArg,
    h2: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let cip = checked_cip(py, cip)?;
    let trait_ = trait_.get()?;
    let age = age.as_slice()?;
    let scores = crossing::run(py, || {
        pafgrs::score_univariate(&prep.inner, trait_, age, &cip, h2)
    })?;
    let out = PyDict::new(py);
    let metadata = crossing::metadata(scores.metadata());
    out.set_item("id", scores.ids.into_pyarray(py))?;
    out.set_item("est", f64_array(py, scores.est))?;
    out.set_item("var", f64_array(py, scores.var))?;
    out.set_item("n_relatives", scores.n_relatives.into_pyarray(py))?;
    out.set_item("metadata", value::to_object(py, &metadata)?)?;
    Ok(out)
}

/// Bivariate scores of every proband, in the pool.
#[pyfunction]
#[pyo3(signature = (prep, traits, ages, cips, /, *, h2, rg, rho_within))]
#[allow(clippy::too_many_arguments)]
fn score_bivariate<'py>(
    py: Python<'py>,
    prep: &Prep,
    traits: (crossing::TraitArg<'py>, crossing::TraitArg<'py>),
    ages: (PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>),
    cips: (CipArg, CipArg),
    h2: (f64, f64),
    rg: f64,
    rho_within: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let params = BivParams::new([h2.0, h2.1], rg, rho_within).map_err(|e| to_pyerr(py, e))?;
    let cip1 = checked_cip(py, cips.0)?;
    let cip2 = checked_cip(py, cips.1)?;
    let values = [traits.0.get()?, traits.1.get()?];
    let ages = [ages.0.as_slice()?, ages.1.as_slice()?];
    let scores = crossing::run(py, || {
        pafgrs::score_bivariate(&prep.inner, values, ages, [&cip1, &cip2], params)
    })?;
    let out = PyDict::new(py);
    let metadata = crossing::metadata(scores.metadata());
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
    out.set_item("metadata", value::to_object(py, &metadata)?)?;
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
    m.add_function(wrap_pyfunction!(cip_aalen_johansen, m)?)?;
    m.add_function(wrap_pyfunction!(score_univariate, m)?)?;
    m.add_function(wrap_pyfunction!(score_bivariate, m)?)?;
    m.add_function(wrap_pyfunction!(assortative::mate_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::tetrachoric_table, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::tetrachoric_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(regression::parent_offspring, m)?)?;
    m.add_function(wrap_pyfunction!(
        relative_pairs::relative_pairs_correlation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(relative_pairs::relative_pairs_ace, m)?)?;
    m.add_class::<Prep>()?;
    m.add_class::<Pedigree>()?;
    #[cfg(feature = "test-hooks")]
    m.add_function(wrap_pyfunction!(test_hooks::_panic_for_test, m)?)?;
    #[cfg(feature = "test-hooks")]
    m.add_function(wrap_pyfunction!(test_hooks::_release_count, m)?)?;
    Ok(())
}
