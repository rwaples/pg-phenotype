//! A host-neutral tree of a result, so each binding converts leaves only.
//!
//! The keys and their order are the result's contract (Python dict keys,
//! R list names); a host maps each variant onto its own types once.

/// One node of a result.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Python `None`, R `NULL`.
    Null,
    Bool(bool),
    /// A count: a Python int, an R integer (a double, or `integer64` past
    /// 2^53, beyond R's integer range).
    Count(u64),
    /// A full-range integer (a seed): a Python int, an R double, or
    /// `integer64` beyond 2^53.
    Int(i64),
    Float(f64),
    Str(&'static str),
    /// A fixed-length tuple of floats (a CI): a Python tuple, an R double vector.
    Floats(Vec<f64>),
    /// A row of counts (a table row): a Python list, an R integer vector
    /// (as for [`Value::Count`] when one passes R's integer range).
    Counts(Vec<u64>),
    /// A Python list, an unnamed R list.
    List(Vec<Value>),
    /// A Python dict, a named R list, in key order.
    Map(Vec<(&'static str, Value)>),
}

impl From<bool> for Value {
    fn from(v: bool) -> Value {
        Value::Bool(v)
    }
}

impl From<u64> for Value {
    fn from(v: u64) -> Value {
        Value::Count(v)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Value {
        Value::Float(v)
    }
}

impl From<&'static str> for Value {
    fn from(v: &'static str) -> Value {
        Value::Str(v)
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(v: Option<T>) -> Value {
        v.map_or(Value::Null, Into::into)
    }
}
