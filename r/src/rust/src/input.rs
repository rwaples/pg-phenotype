//! Host coercion of R columns, with Python's rules (`pg_phenotype/_input.py`).
//!
//! An id column passes as integer, whole finite double, or
//! `bit64::integer64` (a double vector whose bits are an `i64`, read without
//! depending on bit64); `NA` is `-1` where a column allows it.  An all-`NA`
//! logical is all missing.  Positions in errors are 1-based.

use crate::errors::{HostError, HostResult};
use extendr_api::prelude::*;

/// The R storage an id column arrived in, so ids go back out in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdType {
    Integer,
    Double,
    Integer64,
}

impl IdType {
    /// `ids`, which came in as this storage, back as an R vector of it.
    pub fn to_robj(self, ids: &[i64]) -> Robj {
        match self {
            IdType::Integer => Integers::from_values(ids.iter().map(|&v| v as i32)).into_robj(),
            IdType::Double => Doubles::from_values(ids.iter().map(|&v| v as f64)).into_robj(),
            IdType::Integer64 => crate::errors::integer64(ids),
        }
    }
}

/// One coerced column: its values, `-1` where missing, and its storage.
pub struct Coerced {
    pub values: Vec<i64>,
    pub first_missing: Option<usize>,
    pub storage: IdType,
}

const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;

fn invalid(field: &'static str, position: usize, value: Robj) -> HostError {
    HostError::validation(
        "invalid_integer_value",
        format!(
            "'{field}' value at position {} is not a lossless integer",
            position + 1
        ),
        vec![
            ("field", field.into()),
            ("position", ((position + 1) as f64).into()),
            ("value", value),
        ],
    )
}

fn describe(column: &Robj) -> Robj {
    if let Some(class) = column.class().and_then(|mut c| c.next()) {
        return class.into();
    }
    match column.rtype() {
        Rtype::Logicals => "logical",
        Rtype::Strings => "character",
        Rtype::Complexes => "complex",
        Rtype::List => "list",
        Rtype::Raw => "raw",
        _ => "unsupported",
    }
    .into()
}

/// Coerce one column to int64, `NA` to `-1`.
pub fn coerce(field: &'static str, column: &Robj) -> HostResult<Coerced> {
    let n = column.len();
    if column.inherits("factor") {
        return Err(invalid(field, 0, "factor".into()));
    }
    let mut first_missing = None;
    let mut missing = |position: usize| {
        first_missing.get_or_insert(position);
        -1
    };
    let (values, storage) = match column.rtype() {
        Rtype::Integers => {
            let slice = column.as_integer_slice().unwrap_or_default();
            let values = slice
                .iter()
                .enumerate()
                .map(|(p, &v)| {
                    if v == i32::MIN {
                        missing(p)
                    } else {
                        i64::from(v)
                    }
                })
                .collect();
            (values, IdType::Integer)
        }
        Rtype::Doubles if column.inherits("integer64") => {
            let slice = column.as_real_slice().unwrap_or_default();
            let values = slice
                .iter()
                .enumerate()
                .map(|(p, v)| match v.to_bits() as i64 {
                    i64::MIN => missing(p),
                    v => v,
                })
                .collect();
            (values, IdType::Integer64)
        }
        Rtype::Doubles => {
            let slice = column.as_real_slice().unwrap_or_default();
            let mut values = Vec::with_capacity(n);
            for (position, &v) in slice.iter().enumerate() {
                if v.is_nan() {
                    values.push(missing(position));
                } else if v.is_finite() && v == v.trunc() && (-TWO_POW_63..TWO_POW_63).contains(&v)
                {
                    values.push(v as i64);
                } else {
                    return Err(invalid(field, position, v.into()));
                }
            }
            (values, IdType::Double)
        }
        // `NA` alone is a logical in R, so `mother = NA` is a column of nulls.
        Rtype::Logicals => {
            let slice = column.as_logical_slice().unwrap_or_default();
            if let Some(position) = slice.iter().position(|v| !v.is_na()) {
                return Err(invalid(field, position, slice[position].to_bool().into()));
            }
            (vec![-1; n], IdType::Integer)
        }
        _ if n == 0 => (Vec::new(), IdType::Integer),
        _ => return Err(invalid(field, 0, describe(column))),
    };
    Ok(Coerced {
        values,
        first_missing,
        storage,
    })
}

/// Coerce a column that has no missing form (`id`, `probands`).
pub fn coerce_required(field: &'static str, column: &Robj) -> HostResult<Coerced> {
    let coerced = coerce(field, column)?;
    if let Some(position) = coerced.first_missing {
        let na = Strings::from_values([Rstr::na()]).into_robj();
        return Err(invalid(field, position, na));
    }
    Ok(coerced)
}

/// A pedigree column the caller must pass (`NULL` when the data had none).
pub fn present<'a>(field: &'static str, column: &'a Robj) -> HostResult<&'a Robj> {
    if column.is_null() {
        return Err(HostError::validation(
            "missing_field",
            format!("pedigree has no '{field}' column"),
            vec![("field", field.into())],
        ));
    }
    Ok(column)
}

/// A double vector the R wrapper already coerced.
pub fn doubles<'a>(name: &str, column: &'a Robj) -> HostResult<&'a [f64]> {
    column
        .as_real_slice()
        .ok_or_else(|| HostError::usage(format!("`{name}` must be a double vector")))
}

/// One double; `NA` comes through as a NaN for the core to judge.
pub fn number(name: &str, x: &Robj) -> HostResult<f64> {
    match doubles(name, x)? {
        &[x] => Ok(x),
        _ => Err(HostError::usage(format!(
            "`{name}` must be a single number"
        ))),
    }
}

/// A whole number from R, which passes every number as a double.
pub fn whole(name: &str, x: f64) -> HostResult<i64> {
    if x.is_finite() && x == x.trunc() && x.abs() < 2f64.powi(53) {
        return Ok(x as i64);
    }
    Err(HostError::usage(format!(
        "`{name}` must be a whole number, got {x}"
    )))
}
