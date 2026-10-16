//! `relative_pairs_correlation()` and `relative_pairs_ace()`: the
//! Relative-Pair Correlation and its ACE Fit as nested lists, the core's
//! `to_value` trees that the Python binding converts too.

use crate::crossing::{self, TraitArg};
use crate::errors::{finish, HostError, HostResult};
use crate::input;
use crate::pedigree::Source;
use extendr_api::prelude::*;
use pg_phenotype_core::ace::{
    self, AceCell, AceRequest, Components, CrudeEstimate, Estimate, Estimator, Pairs,
    StratifiedEstimate,
};
use pg_phenotype_core::assortative::Strata;
use pg_phenotype_core::relative_pairs::{self, Entry};

fn strings(what: &str, x: &Robj) -> HostResult<Vec<String>> {
    x.as_string_vector()
        .ok_or_else(|| HostError::usage(format!("{what} must be a character vector")))
}

/// The entries the R wrapper normalized: `names[i]` with `members[[i]]`
/// `NULL` for a code, a character vector for a group.
fn entries(names: &Robj, members: &Robj) -> HostResult<Option<Vec<Entry>>> {
    if names.is_null() {
        return Ok(None);
    }
    let names = strings("category names", names)?;
    let members = members
        .as_list()
        .filter(|l| l.len() == names.len())
        .ok_or_else(|| HostError::usage("category members must be a list, one per name".into()))?;
    names
        .into_iter()
        .zip(members.values())
        .enumerate()
        .map(|(position, (name, codes))| {
            Ok(if codes.is_null() {
                Entry::Category {
                    code: name,
                    position,
                }
            } else {
                let members = strings(&format!("category group {name:?}"), &codes)?;
                Entry::Group {
                    name,
                    members,
                    position,
                }
            })
        })
        .collect::<HostResult<Vec<Entry>>>()
        .map(Some)
}

fn correlation_impl(
    pedigree: &Robj,
    traits: &List,
    stratum: &Robj,
    names: &Robj,
    members: &Robj,
    memory_budget_bytes: &Robj,
    se: bool,
) -> HostResult<Robj> {
    let pedigree = Source::read(pedigree)?;
    let traits = traits
        .values()
        .map(|t| TraitArg::read(&t))
        .collect::<HostResult<Vec<_>>>()?;
    let traits = traits
        .iter()
        .map(TraitArg::get)
        .collect::<HostResult<Vec<_>>>()?;
    let strata = (!stratum.is_null())
        .then(|| input::strata(stratum))
        .transpose()?;
    let strata = strata
        .as_ref()
        .map(|(labels, known)| Strata { labels, known });
    let entries = entries(names, members)?;
    let budget = input::saturating_whole(
        "memory_budget_bytes",
        input::number("memory_budget_bytes", memory_budget_bytes)?,
    )?;
    let result = crossing::run(|| {
        relative_pairs::correlation(
            pedigree.arg(),
            &traits,
            strata,
            entries.as_deref(),
            budget,
            se,
        )
        .map(|r| r.to_value())
    })?;
    crossing::result(result)
}

/// The Relative-Pair Correlation of one or two traits, in the pool.
#[extendr]
fn relative_pairs_correlation(
    pedigree: Robj,
    traits: List,
    stratum: Robj,
    names: Robj,
    members: Robj,
    memory_budget_bytes: Robj,
    se: bool,
) -> Robj {
    finish(correlation_impl(
        &pedigree,
        &traits,
        &stratum,
        &names,
        &members,
        &memory_budget_bytes,
        se,
    ))
}

const CELL_FIELDS: [&str; 10] = [
    "name",
    "members",
    "first_trait",
    "second_trait",
    "estimators",
    "values",
    "reasons",
    "n",
    "n_entries",
    "stratified",
];

/// `value`, or `reason` when it is `NA`.
fn value(value: f64, reason: &str) -> Result<f64, String> {
    if value.is_nan() {
        Err(reason.to_string())
    } else {
        Ok(value)
    }
}

fn whole(what: &str, x: &Robj) -> HostResult<f64> {
    let x = input::number(what, x)?;
    if !(x.is_finite() && x >= 0.0 && x == x.trunc()) {
        return Err(HostError::usage(format!("{what} must be a count, got {x}")));
    }
    Ok(x)
}

/// A count that is `NULL` where the result has none.
fn optional_count(what: &str, x: &Robj) -> HostResult<Option<u64>> {
    if x.is_null() {
        Ok(None)
    } else {
        Ok(Some(whole(what, x)? as u64))
    }
}

/// One cell as the R wrapper normalized it: `members` `NULL` within person,
/// traits from 0, and the crude estimates as parallel vectors, `NA` where
/// undefined.
fn cell(robj: &Robj) -> HostResult<AceCell> {
    let [name, members, first, second, estimators, values, reasons, n, n_entries, stratified] =
        input::elements(robj, "cell", "relative_pairs_correlation()", CELL_FIELDS)?;
    let estimators = strings("estimators", &estimators)?;
    let values = input::doubles("values", &values)?;
    let reasons = strings("reasons", &reasons)?;
    if values.len() != estimators.len() || reasons.len() != estimators.len() {
        return Err(HostError::usage(
            "a cell's estimators, values and reasons must have one length".into(),
        ));
    }
    let stratified = if stratified.is_null() {
        None
    } else {
        let [v, reason, n, n_entries] = input::list_of(&stratified, "a cell's stratified form")?;
        let reason = strings("stratified reason", &reason)?;
        Some(StratifiedEstimate {
            value: value(input::number("stratified value", &v)?, &reason.concat()),
            n: whole("stratified n", &n)? as u64,
            n_entries: optional_count("stratified n_entries", &n_entries)?,
        })
    };
    Ok(AceCell {
        name: strings("name", &name)?.concat(),
        pairs: if members.is_null() {
            Pairs::WithinPerson
        } else {
            Pairs::relatives(&strings("members", &members)?)?
        },
        first_trait: whole("first_trait", &first)? as usize,
        second_trait: whole("second_trait", &second)? as usize,
        crude: estimators
            .into_iter()
            .zip(values.iter().zip(&reasons))
            .map(|(estimator, (&v, reason))| CrudeEstimate {
                estimator,
                value: value(v, reason),
            })
            .collect(),
        n: whole("n", &n)? as u64,
        n_entries: optional_count("n_entries", &n_entries)?,
        stratified,
    })
}

fn one(what: &str, x: &Robj) -> HostResult<String> {
    match strings(what, x)?.as_slice() {
        [x] => Ok(x.clone()),
        _ => Err(HostError::usage(format!("{what} must be a single string"))),
    }
}

fn named<T>(what: &str, name: &str, parsed: Option<T>) -> HostResult<T> {
    parsed.ok_or_else(|| HostError::usage(format!("{what} {name:?} is not one this fit takes")))
}

fn ace_impl(
    cells: &List,
    categories: &Robj,
    components: &Robj,
    shared_environment: &Robj,
    estimator: &Robj,
    estimate: &Robj,
) -> HostResult<Robj> {
    let optional = |what: &str, x: &Robj| (!x.is_null()).then(|| strings(what, x)).transpose();
    let components = strings("components", components)?;
    let names: Vec<&str> = components.iter().map(String::as_str).collect();
    let (estimator, estimate) = (one("estimator", estimator)?, one("estimate", estimate)?);
    let request = AceRequest {
        categories: optional("categories", categories)?,
        components: named(
            "components",
            &components.concat(),
            Components::parse(&names),
        )?,
        shared_environment: optional("shared_environment", shared_environment)?,
        estimator: named("estimator", &estimator, Estimator::parse(&estimator))?,
        estimate: named("estimate", &estimate, Estimate::parse(&estimate))?,
    };
    let cells = cells
        .values()
        .map(|c| cell(&c))
        .collect::<HostResult<Vec<_>>>()?;
    crossing::result(ace::fit(&cells, &request)?.to_value())
}

/// The ACE Fit of a result's cells, which the R wrapper checked and resolved.
#[extendr]
fn relative_pairs_ace(
    cells: List,
    categories: Robj,
    components: Robj,
    shared_environment: Robj,
    estimator: Robj,
    estimate: Robj,
) -> Robj {
    finish(ace_impl(
        &cells,
        &categories,
        &components,
        &shared_environment,
        &estimator,
        &estimate,
    ))
}

extendr_module! {
    mod relative_pairs;
    fn relative_pairs_correlation;
    fn relative_pairs_ace;
}
