//! PA-FGRS: `pafgrs_prepare()`, `pafgrs_cip()` and the score functions.

use crate::crossing::{self, TraitArg};
use crate::errors::{finish, HostError, HostResult};
use crate::input::{self, IdType};
use crate::pedigree::Source;
use crate::value::to_robj;
use extendr_api::prelude::*;
use pg_phenotype_core::pafgrs::{self, BivParams, Cip, Interpolation, Prep};

/// What a `pgphenotype_pafgrs_prep` external pointer owns.
struct Handle {
    prep: Prep,
    id_type: IdType,
}

fn handle(prep: &Robj) -> HostResult<&Handle> {
    input::handle(prep, "prep", "pafgrs_prepare()")
}

fn prepare_impl(pedigree: &Robj, ndegree: &Robj, probands: Robj) -> HostResult<Robj> {
    let pedigree = Source::read(pedigree)?;
    let ndegree = input::saturating_whole("ndegree", input::number("ndegree", ndegree)?)?;
    let probands = (!probands.is_null())
        .then(|| input::coerce_required("probands", &probands))
        .transpose()?;
    let prep = crossing::run(|| {
        pafgrs::prepare(
            pedigree.arg(),
            ndegree,
            probands.as_ref().map(|c| c.values.as_slice()),
        )
    })?;
    let out: Robj = ExternalPtr::new(Handle {
        prep,
        id_type: pedigree.id_type(),
    })
    .into();
    Ok(crate::with_class(out, "pgphenotype_pafgrs_prep"))
}

/// Build a pedigree's relative structure in the pool.
#[extendr]
fn pafgrs_prepare(pedigree: Robj, ndegree: Robj, probands: Robj) -> Robj {
    finish(prepare_impl(&pedigree, &ndegree, probands))
}

/// `list(n_rows, n_probands, ndegree)` of a prep.
#[extendr]
fn pafgrs_prep_info(prep: Robj) -> Robj {
    finish(handle(&prep).map(|h| {
        list!(
            n_rows = h.prep.n_rows() as i32,
            n_probands = h.prep.n_probands() as i32,
            ndegree = i32::from(h.prep.ndegree())
        )
        .into_robj()
    }))
}

/// A CIP table's ages, values and interpolation name.
type CipArg<'a> = [&'a Robj; 3];

fn cip_of([ages, cip, interpolation]: CipArg<'_>) -> HostResult<Cip> {
    let ages = input::doubles("ages", ages)?.to_vec();
    let cip = input::doubles("cip", cip)?.to_vec();
    let interpolation = Interpolation::from_name(interpolation.as_str().unwrap_or_default())
        .ok_or_else(|| HostError::usage("a CIP must come from pafgrs_cip()".to_string()))?;
    Ok(Cip::new(ages, cip, interpolation)?)
}

/// `list(prevalence, threshold)` of a valid CIP table.
#[extendr]
fn pafgrs_check_cip(ages: Robj, cip: Robj, interpolation: Robj) -> Robj {
    finish(
        cip_of([&ages, &cip, &interpolation])
            .map(|c| list!(prevalence = c.prevalence(), threshold = c.threshold()).into_robj()),
    )
}

/// `list(ages, cip)` of the Aalen-Johansen CIP table; `entry` may be NULL.
#[extendr]
fn pafgrs_cip_aalen_johansen(age: Robj, status: Robj, entry: Robj) -> Robj {
    finish(aalen_johansen(&age, &status, &entry))
}

fn aalen_johansen(age: &Robj, status: &Robj, entry: &Robj) -> HostResult<Robj> {
    let entry = (!entry.is_null())
        .then(|| input::doubles("entry", entry))
        .transpose()?;
    let cip = Cip::aalen_johansen(
        input::doubles("age", age)?,
        input::doubles("status", status)?,
        entry,
    )?;
    let (ages, cip) = cip.into_parts();
    Ok(list!(ages = ages, cip = cip).into_robj())
}

/// The table of a `pafgrs_cip()`.
fn cip_table(cip: &Robj) -> HostResult<Cip> {
    let [ages, values, interpolation] =
        input::elements(cip, "cip", "pafgrs_cip()", ["ages", "cip", "interpolation"])?;
    cip_of([&ages, &values, &interpolation])
}

fn counts(values: &[u32]) -> Robj {
    Integers::from_values(values.iter().map(|&v| v as i32)).into_robj()
}

fn score_univariate_impl(
    prep: &Robj,
    trait_: &Robj,
    age: &Robj,
    cip: &Robj,
    h2: &Robj,
) -> HostResult<Robj> {
    let h = handle(prep)?;
    let h2 = input::number("h2", h2)?;
    let cip = cip_table(cip)?;
    let trait_ = TraitArg::read(trait_)?;
    let trait_ = trait_.get()?;
    let age = input::doubles("age", age)?;
    let s = crossing::run(|| pafgrs::score_univariate(&h.prep, trait_, age, &cip, h2))?;
    let metadata = to_robj(&crossing::metadata(s.metadata()));
    Ok(list!(
        id = h.id_type.to_robj(&s.ids),
        est = s.est,
        var = s.var,
        n_relatives = counts(&s.n_relatives),
        metadata = metadata
    )
    .into_robj())
}

/// Univariate scores of every proband, in the pool.
#[extendr]
fn pafgrs_score_univariate(prep: Robj, trait_: Robj, age: Robj, cip: Robj, h2: Robj) -> Robj {
    finish(score_univariate_impl(&prep, &trait_, &age, &cip, &h2))
}

fn score_bivariate_impl(
    prep: &Robj,
    [traits, ages, cips]: [&Robj; 3],
    h2: &Robj,
    rg: &Robj,
    rho_within: &Robj,
) -> HostResult<Robj> {
    let h = handle(prep)?;
    let rg = input::number("rg", rg)?;
    let h2 = match input::doubles("h2", h2)? {
        &[a, b] => [a, b],
        _ => return Err(HostError::usage("`h2` must hold two values".to_string())),
    };
    let rho_within = match rho_within.is_null() {
        true => None,
        false => Some(input::number("rho_within", rho_within)?),
    };
    let params = BivParams::new(h2, rg, rho_within)?;
    let [cip1, cip2] = input::list_of::<2>(cips, "cips")?;
    let cip = [cip_table(&cip1)?, cip_table(&cip2)?];
    let [trait1, trait2] = input::list_of::<2>(traits, "traits")?;
    let traits = [TraitArg::read(&trait1)?, TraitArg::read(&trait2)?];
    let values = [traits[0].get()?, traits[1].get()?];
    let [age1, age2] = input::list_of::<2>(ages, "ages")?;
    let ages = [
        input::doubles("age1", &age1)?,
        input::doubles("age2", &age2)?,
    ];
    let s = crossing::run(|| {
        pafgrs::score_bivariate(&h.prep, values, ages, [&cip[0], &cip[1]], params)
    })?;
    let metadata = to_robj(&crossing::metadata(s.metadata()));
    let [est1, est2] = s.est;
    let [var1, var2] = s.var;
    Ok(list!(
        id = h.id_type.to_robj(&s.ids),
        est1 = est1,
        est2 = est2,
        var1 = var1,
        var2 = var2,
        cov12 = s.cov12,
        n_relatives = counts(&s.n_relatives),
        n_obs1 = counts(&s.n_obs[0]),
        n_obs2 = counts(&s.n_obs[1]),
        metadata = metadata
    )
    .into_robj())
}

/// Bivariate scores of every proband, in the pool.  `traits`, `ages` and
/// `cips` are lists of two.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn pafgrs_score_bivariate(
    prep: Robj,
    traits: Robj,
    ages: Robj,
    cips: Robj,
    h2: Robj,
    rg: Robj,
    rho_within: Robj,
) -> Robj {
    finish(score_bivariate_impl(
        &prep,
        [&traits, &ages, &cips],
        &h2,
        &rg,
        &rho_within,
    ))
}

extendr_module! {
    mod pafgrs;
    fn pafgrs_prepare;
    fn pafgrs_prep_info;
    fn pafgrs_check_cip;
    fn pafgrs_cip_aalen_johansen;
    fn pafgrs_score_univariate;
    fn pafgrs_score_bivariate;
}
