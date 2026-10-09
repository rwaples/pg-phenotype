//! `assortative_mate_correlation()`: the Mate Correlation as nested lists,
//! the core's `to_value` tree that the Python binding converts too.

use crate::errors::{finish, int_values, HostError, HostResult};
use crate::input::{self, Pedigree};
use crate::threads;
use extendr_api::prelude::*;
use pg_phenotype_core::assortative::{self, Settings, Strata};
use pg_phenotype_core::value::Value;
use pg_phenotype_core::Trait;

/// A core result tree as nested R lists and vectors, `NULL` where Python
/// has `None`.
fn to_robj(value: &Value) -> Robj {
    match value {
        Value::Null => ().into_robj(),
        Value::Bool(v) => (*v).into(),
        Value::Count(n) => counts(&[*n]),
        Value::Int(v) => int_values(vec![*v]),
        Value::Float(v) => (*v).into(),
        Value::Str(v) => (*v).into(),
        Value::Floats(v) => Doubles::from_values(v.iter().copied()).into_robj(),
        Value::Counts(v) => counts(v),
        Value::List(items) => List::from_values(items.iter().map(to_robj)).into_robj(),
        Value::Map(entries) => {
            let (names, values): (Vec<&str>, Vec<Robj>) =
                entries.iter().map(|(k, v)| (*k, to_robj(v))).unzip();
            List::from_names_and_values(names, values)
                .expect("names and values of one length")
                .into_robj()
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

#[allow(clippy::too_many_arguments)]
fn mate_correlation_impl(
    columns: [Robj; 5],
    values: &List,
    kinds: &Robj,
    n_levels: &Robj,
    stratum: &Robj,
    counts: [&Robj; 4],
    spearman: bool,
) -> HostResult<Robj> {
    let pedigree = Pedigree::coerce(columns)?;
    let [permutations, bootstrap, seed, min_stratum_networks] = counts;
    let whole =
        |name: &'static str, x: &Robj| input::number(name, x).and_then(|v| input::whole(name, v));
    let n_levels = input::doubles("n_levels", n_levels)?;
    let kinds = kinds
        .as_str_vector()
        .ok_or_else(|| HostError::usage("a trait must come from trait()".to_string()))?;
    let value_robjs: Vec<Robj> = values.values().collect();
    let traits = value_robjs
        .iter()
        .zip(&kinds)
        .zip(n_levels)
        .map(|((v, k), &n)| {
            Ok(Trait {
                values: input::doubles("trait", v)?,
                kind: input::trait_kind(k)?,
                n_levels: (!n.is_nan()).then_some(n as usize),
            })
        })
        .collect::<HostResult<Vec<_>>>()?;
    let strata = (!stratum.is_null())
        .then(|| input::strata(stratum))
        .transpose()?;
    let settings = Settings {
        permutations: whole("permutations", permutations)?,
        bootstrap: whole("bootstrap", bootstrap)?,
        seed: whole("seed", seed)?,
        min_stratum_networks: whole("min_stratum_networks", min_stratum_networks)?,
        spearman,
    };
    let pool = threads::pool()?;
    let result = pool.install(|| {
        assortative::mate_correlation(
            pedigree.input(),
            &traits,
            strata
                .as_ref()
                .map(|(labels, known)| Strata { labels, known }),
            settings,
        )
    })?;
    Ok(to_robj(&result.to_value()))
}

/// The Mate Correlation of one or two traits, in the pool.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn assortative_mate_correlation(
    id: Robj,
    mother: Robj,
    father: Robj,
    twin: Robj,
    sex: Robj,
    values: List,
    kinds: Robj,
    n_levels: Robj,
    stratum: Robj,
    permutations: Robj,
    bootstrap: Robj,
    seed: Robj,
    min_stratum_networks: Robj,
    spearman: bool,
) -> Robj {
    finish(mate_correlation_impl(
        [id, mother, father, twin, sex],
        &values,
        &kinds,
        &n_levels,
        &stratum,
        [&permutations, &bootstrap, &seed, &min_stratum_networks],
        spearman,
    ))
}

extendr_module! {
    mod assortative;
    fn assortative_mate_correlation;
}
