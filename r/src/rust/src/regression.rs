//! `regression_parent_offspring()`: the regressions as nested lists, the
//! core's `to_value` tree that the Python binding converts too.

use crate::crossing::{self, TraitArg};
use crate::errors::{finish, HostError, HostResult};
use crate::input;
use crate::pedigree::Source;
use extendr_api::prelude::*;
use pg_phenotype_core::assortative::Strata;
use pg_phenotype_core::regression::{self, Lines, Parent};

fn parent_offspring_impl(
    pedigree: &Robj,
    trait_: &Robj,
    stratum: &Robj,
    lines: &Robj,
    inference: bool,
) -> HostResult<Robj> {
    let lines = lines
        .as_str_iter()
        .ok_or_else(|| HostError::usage("lines must be a character vector".into()))?
        .map(|name| {
            Parent::parse(name).ok_or_else(|| HostError::usage(format!("{name:?} is not a line")))
        })
        .collect::<HostResult<Lines>>()?;
    let settings = regression::Settings { lines, inference };
    let pedigree = Source::read(pedigree)?;
    let trait_ = TraitArg::read(trait_)?;
    let trait_ = trait_.get()?;
    let strata = (!stratum.is_null())
        .then(|| input::strata(stratum))
        .transpose()?;
    let strata = strata
        .as_ref()
        .map(|(labels, known)| Strata { labels, known });
    let result = crossing::run(|| {
        regression::parent_offspring(pedigree.arg(), trait_, strata, settings).map(|r| r.to_value())
    })?;
    crossing::result(result)
}

/// The parent-offspring regressions of one trait, in the pool, of the
/// lines named in `lines`.
#[extendr]
fn regression_parent_offspring(
    pedigree: Robj,
    trait_: Robj,
    stratum: Robj,
    lines: Robj,
    inference: bool,
) -> Robj {
    finish(parent_offspring_impl(
        &pedigree, &trait_, &stratum, &lines, inference,
    ))
}

extendr_module! {
    mod regression;
    fn regression_parent_offspring;
}
