//! `regression_parent_offspring()`: the regressions as nested lists, the
//! core's `to_value` tree that the Python binding converts too.

use crate::crossing::{self, TraitArg};
use crate::errors::{finish, HostResult};
use crate::input;
use crate::pedigree::Source;
use extendr_api::prelude::*;
use pg_phenotype_core::assortative::Strata;
use pg_phenotype_core::regression;

fn parent_offspring_impl(
    pedigree: &Robj,
    trait_: &Robj,
    stratum: &Robj,
    settings: regression::Settings,
) -> HostResult<Robj> {
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

/// The parent-offspring regressions of one trait, in the pool.  `lines`
/// is one flag per line (midparent, mother, father), as the R wrapper
/// builds it.
#[extendr]
fn regression_parent_offspring(
    pedigree: Robj,
    trait_: Robj,
    stratum: Robj,
    lines: Robj,
    inference: bool,
) -> Robj {
    let flags = lines.as_logical_slice().unwrap_or(&[]);
    let lines = [0, 1, 2].map(|i| flags.get(i).is_some_and(|b| b.is_true()));
    let settings = regression::Settings { lines, inference };
    finish(parent_offspring_impl(
        &pedigree, &trait_, &stratum, settings,
    ))
}

extendr_module! {
    mod regression;
    fn regression_parent_offspring;
}
