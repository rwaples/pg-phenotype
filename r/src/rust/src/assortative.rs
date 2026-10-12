//! `assortative_mate_correlation()`: the Mate Correlation as nested lists,
//! the core's `to_value` tree that the Python binding converts too.

use crate::crossing::{self, TraitArg};
use crate::errors::{finish, HostResult};
use crate::input;
use crate::pedigree::Source;
use extendr_api::prelude::*;
use pg_phenotype_core::assortative::{self, Settings, Strata};

fn mate_correlation_impl(
    pedigree: &Robj,
    traits: &List,
    stratum: &Robj,
    counts: [&Robj; 4],
    spearman: bool,
    inference: bool,
) -> HostResult<Robj> {
    let pedigree = Source::read(pedigree)?;
    let [permutations, bootstrap, seed, min_stratum_networks] = counts;
    let whole =
        |name: &'static str, x: &Robj| input::number(name, x).and_then(|v| input::whole(name, v));
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
    let settings = Settings {
        permutations: (!permutations.is_null())
            .then(|| whole("permutations", permutations))
            .transpose()?,
        bootstrap: whole("bootstrap", bootstrap)?,
        seed: whole("seed", seed)?,
        min_stratum_networks: whole("min_stratum_networks", min_stratum_networks)?,
        spearman,
        inference,
    };
    let result = crossing::run(|| {
        assortative::mate_correlation(pedigree.arg(), &traits, strata, settings)
            .map(|r| r.to_value())
    })?;
    crossing::result(result)
}

/// The Mate Correlation of one or two traits, in the pool.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn assortative_mate_correlation(
    pedigree: Robj,
    traits: List,
    stratum: Robj,
    permutations: Robj,
    bootstrap: Robj,
    seed: Robj,
    min_stratum_networks: Robj,
    spearman: bool,
    inference: bool,
) -> Robj {
    finish(mate_correlation_impl(
        &pedigree,
        &traits,
        &stratum,
        [&permutations, &bootstrap, &seed, &min_stratum_networks],
        spearman,
        inference,
    ))
}

extendr_module! {
    mod assortative;
    fn assortative_mate_correlation;
}
