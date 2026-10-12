//! `relative_pairs_correlation()`: the Relative-Pair Correlation as nested
//! lists, the core's `to_value` tree that the Python binding converts too.

use crate::crossing::{self, TraitArg};
use crate::errors::{finish, HostError, HostResult};
use crate::input;
use crate::pedigree::Source;
use extendr_api::prelude::*;
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
    trait_: &Robj,
    names: &Robj,
    members: &Robj,
    memory_budget_bytes: &Robj,
) -> HostResult<Robj> {
    let pedigree = Source::read(pedigree)?;
    let trait_ = TraitArg::read(trait_)?;
    let trait_ = trait_.get()?;
    let entries = entries(names, members)?;
    let budget = input::saturating_whole(
        "memory_budget_bytes",
        input::number("memory_budget_bytes", memory_budget_bytes)?,
    )?;
    let result = crossing::run(|| {
        relative_pairs::correlation(pedigree.arg(), trait_, entries.as_deref(), budget)
            .map(|r| r.to_value())
    })?;
    crossing::result(result)
}

/// The Relative-Pair Correlation of one trait, in the pool.
#[extendr]
fn relative_pairs_correlation(
    pedigree: Robj,
    trait_: Robj,
    names: Robj,
    members: Robj,
    memory_budget_bytes: Robj,
) -> Robj {
    finish(correlation_impl(
        &pedigree,
        &trait_,
        &names,
        &members,
        &memory_budget_bytes,
    ))
}

extendr_module! {
    mod relative_pairs;
    fn relative_pairs_correlation;
}
