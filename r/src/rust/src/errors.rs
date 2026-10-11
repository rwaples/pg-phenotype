//! Failures handed to R as data, never raised from Rust.
//!
//! A binding that fails returns a list classed `pgphenotype_native_error`
//! carrying the condition class, code, message and fields; the R wrapper
//! `.pgp_call()` turns it into a classed condition.  Raising from Rust would
//! give an untyped `simpleError` and unwind through R's longjmp.

use extendr_api::prelude::*;
use pg_phenotype_core::error::{Class, FieldValue, IndexBase};
use pg_phenotype_core::Error;

/// One failure on its way to R.
pub struct HostError {
    class: &'static str,
    code: Option<&'static str>,
    message: String,
    fields: Vec<(&'static str, Robj)>,
}

pub type HostResult<T> = std::result::Result<T, HostError>;

impl HostError {
    /// An input-contract failure the binding detects itself (host coercion).
    pub fn validation(
        code: &'static str,
        message: String,
        fields: Vec<(&'static str, Robj)>,
    ) -> HostError {
        HostError {
            class: "validation",
            code: Some(code),
            message,
            fields,
        }
    }

    /// A capacity the binding itself enforces (an R representation limit).
    pub fn resource(
        code: &'static str,
        message: String,
        fields: Vec<(&'static str, Robj)>,
    ) -> HostError {
        HostError {
            class: "resource",
            code: Some(code),
            message,
            fields,
        }
    }

    /// API misuse: a bad argument, with no code and no fields.
    pub fn usage(message: String) -> HostError {
        HostError {
            class: "usage",
            code: None,
            message,
            fields: Vec::new(),
        }
    }

    /// A thread budget changed after it was committed.
    pub fn thread_conflict(committed: usize, requested: usize) -> HostError {
        HostError {
            class: "thread_conflict",
            code: Some("thread_pool_conflict"),
            message: format!(
                "the committed thread budget is {committed} and cannot be changed to {requested}; \
                 call configure_threads() before the first thread_budget() call"
            ),
            fields: vec![
                ("configured", (committed as f64).into()),
                ("requested", (requested as f64).into()),
            ],
        }
    }

    /// The R list `.pgp_call()` signals.
    pub fn into_robj(self) -> Robj {
        let (names, values): (Vec<&str>, Vec<Robj>) = self.fields.into_iter().unzip();
        let fields = List::from_names_and_values(names, values)
            .expect("names and values have one entry each")
            .into_robj();
        let code: Robj = match self.code {
            Some(code) => code.into(),
            None => Strings::from_values([Rstr::na()]).into_robj(),
        };
        let mut out = List::from_names_and_values(
            ["class", "code", "message", "fields"],
            [self.class.into(), code, self.message.into(), fields],
        )
        .expect("four names and four values")
        .into_robj();
        out.set_class(["pgphenotype_native_error"])
            .expect("a character class attribute");
        out
    }
}

impl From<Error> for HostError {
    fn from(err: Error) -> HostError {
        let class = match err.class() {
            Class::Validation => "validation",
            Class::Parameter => "parameter",
            Class::Resource => "resource",
            Class::Usage => "usage",
        };
        let code = match err.code() {
            "" => None,
            code => Some(code),
        };
        let fields = err
            .fields(IndexBase::One)
            .into_iter()
            .map(|(name, value)| (name, field_robj(value)))
            .collect();
        HostError {
            class,
            code,
            message: err.message(IndexBase::One),
            fields,
        }
    }
}

/// The largest magnitude a double holds exactly.
const EXACT_IN_DOUBLE: i64 = 1 << 53;

/// Integer field values: doubles when every value is exact in one, else
/// bit64 `integer64`, so an id above 2^53 is never reported as its neighbour.
pub fn int_values(values: Vec<i64>) -> Robj {
    if values
        .iter()
        .all(|n| n.unsigned_abs() <= EXACT_IN_DOUBLE as u64)
    {
        return Doubles::from_values(values.iter().map(|&n| n as f64)).into_robj();
    }
    integer64(&values)
}

/// `values` as a bit64 `integer64` vector, read without depending on bit64.
pub fn integer64(values: &[i64]) -> Robj {
    let mut out =
        Doubles::from_values(values.iter().map(|&n| f64::from_bits(n as u64))).into_robj();
    out.set_class(["integer64"])
        .expect("a character class attribute");
    out
}

fn field_robj(value: FieldValue) -> Robj {
    match value {
        FieldValue::Int(n) => int_values(vec![n]),
        FieldValue::Ints(v) => int_values(v),
        FieldValue::Float(x) => x.into(),
        FieldValue::Str(s) => s.into(),
        FieldValue::Text(s) => s.into(),
        FieldValue::Texts(v) => Strings::from_values(v).into_robj(),
        FieldValue::Strs(v) => Strings::from_values(v).into_robj(),
    }
}

/// A binding's value, or its failure as the list `.pgp_call()` signals.
pub fn finish(result: HostResult<Robj>) -> Robj {
    result.unwrap_or_else(HostError::into_robj)
}
