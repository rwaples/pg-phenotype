//! The core's result tree (`pg_phenotype_core::value::Value`) as nested R
//! lists and vectors, for every method's binding.

use crate::errors::int_values;
use extendr_api::prelude::*;
use pg_phenotype_core::value::Value;

/// A core result tree as nested R lists and vectors, `NULL` where Python
/// has `None`.
pub(crate) fn to_robj(value: &Value) -> Robj {
    match value {
        Value::Null => ().into_robj(),
        Value::Bool(v) => (*v).into(),
        Value::Count(n) => counts(&[*n]),
        Value::Int(v) => int_values(vec![*v]),
        Value::Float(v) => (*v).into(),
        Value::Str(v) => (*v).into(),
        Value::Text(v) => v.as_str().into(),
        Value::Floats(v) => Doubles::from_values(v.iter().copied()).into_robj(),
        Value::Counts(v) => counts(v),
        Value::List(items) => List::from_values(items.iter().map(to_robj)).into_robj(),
        Value::Map(entries) => {
            let (names, values): (Vec<&str>, Vec<Robj>) =
                entries.iter().map(|(k, v)| (*k, to_robj(v))).unzip();
            crate::named_list(names, values)
        }
    }
}

/// Counts as an R integer vector, or through [`int_values`] (a double, or
/// `integer64` past 2^53) when one passes R's integer range.
fn counts(values: &[u64]) -> Robj {
    let small: Option<Vec<i32>> = values.iter().map(|&n| i32::try_from(n).ok()).collect();
    match small {
        Some(small) => Integers::from_values(small).into_robj(),
        None => int_values(
            values
                .iter()
                .map(|&n| i64::try_from(n).unwrap_or(i64::MAX))
                .collect(),
        ),
    }
}
