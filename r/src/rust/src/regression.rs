//! `regression_parent_offspring()`: the regressions as nested lists, the
//! core's `to_value` tree that the Python binding converts too.

use crate::assortative::to_robj;
use crate::errors::{finish, HostResult};
use crate::input;
use crate::pedigree::Source;
use crate::threads;
use extendr_api::prelude::*;
use pg_phenotype_core::assortative::Strata;
use pg_phenotype_core::regression;
use pg_phenotype_core::Trait;

fn parent_offspring_impl(
    ptr: &Robj,
    columns: [Robj; 5],
    values: &Robj,
    kind: &str,
    n_levels: &Robj,
    stratum: &Robj,
) -> HostResult<Robj> {
    let pedigree = Source::read(ptr, columns)?;
    let n_levels = input::number("n_levels", n_levels)?;
    let trait_ = Trait {
        values: input::doubles("trait", values)?,
        kind: input::trait_kind(kind)?,
        n_levels: (!n_levels.is_nan()).then_some(n_levels as usize),
    };
    let strata = (!stratum.is_null())
        .then(|| input::strata(stratum))
        .transpose()?;
    let pool = threads::pool()?;
    let result = pool.install(|| {
        regression::parent_offspring(
            pedigree.arg(),
            trait_,
            strata
                .as_ref()
                .map(|(labels, known)| Strata { labels, known }),
        )
    })?;
    Ok(to_robj(&result.to_value()))
}

/// The parent-offspring regressions of one trait, in the pool, over a
/// `pedigree()` pointer or, when it is `NULL`, the columns.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn regression_parent_offspring(
    pedigree: Robj,
    id: Robj,
    mother: Robj,
    father: Robj,
    twin: Robj,
    sex: Robj,
    values: Robj,
    kind: &str,
    n_levels: Robj,
    stratum: Robj,
) -> Robj {
    finish(parent_offspring_impl(
        &pedigree,
        [id, mother, father, twin, sex],
        &values,
        kind,
        &n_levels,
        &stratum,
    ))
}

extendr_module! {
    mod regression;
    fn regression_parent_offspring;
}
