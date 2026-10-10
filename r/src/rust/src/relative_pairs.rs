//! `relative_pairs_correlation()`: the Relative-Pair Correlation as nested
//! lists, the core's `to_value` tree that the Python binding converts too.

use crate::assortative::to_robj;
use crate::errors::{finish, HostError, HostResult};
use crate::input;
use crate::pedigree::Source;
use crate::threads;
use extendr_api::prelude::*;
use pg_phenotype_core::relative_pairs::{self, Entry};
use pg_phenotype_core::Trait;

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

#[allow(clippy::too_many_arguments)]
fn correlation_impl(
    ptr: &Robj,
    columns: [Robj; 5],
    values: &Robj,
    kind: &str,
    n_levels: &Robj,
    names: &Robj,
    members: &Robj,
    memory_budget_bytes: &Robj,
) -> HostResult<Robj> {
    let pedigree = Source::read(ptr, columns)?;
    let n_levels = input::number("n_levels", n_levels)?;
    let trait_ = Trait {
        values: input::doubles("trait", values)?,
        kind: input::trait_kind(kind)?,
        n_levels: (!n_levels.is_nan()).then_some(n_levels as usize),
    };
    let entries = entries(names, members)?;
    let budget = input::saturating_whole(
        "memory_budget_bytes",
        input::number("memory_budget_bytes", memory_budget_bytes)?,
    )?;
    let pool = threads::pool()?;
    let result = pool.install(|| {
        relative_pairs::correlation(pedigree.arg(), trait_, entries.as_deref(), budget)
    })?;
    Ok(to_robj(&result.to_value()))
}

/// The Relative-Pair Correlation of one trait, in the pool, over a
/// `pedigree()` pointer or, when it is `NULL`, the columns.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn relative_pairs_correlation(
    pedigree: Robj,
    id: Robj,
    mother: Robj,
    father: Robj,
    twin: Robj,
    sex: Robj,
    values: Robj,
    kind: &str,
    n_levels: Robj,
    names: Robj,
    members: Robj,
    memory_budget_bytes: Robj,
) -> Robj {
    finish(correlation_impl(
        &pedigree,
        [id, mother, father, twin, sex],
        &values,
        kind,
        &n_levels,
        &names,
        &members,
        &memory_budget_bytes,
    ))
}

extendr_module! {
    mod relative_pairs;
    fn relative_pairs_correlation;
}
