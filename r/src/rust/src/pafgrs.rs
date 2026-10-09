//! PA-FGRS: `pafgrs_prepare()`, `pafgrs_cip()` and the score functions.

use crate::errors::{finish, HostError, HostResult};
use crate::input::{self, IdType, Pedigree};
use crate::threads;
use extendr_api::prelude::*;
use pg_phenotype_core::pafgrs::{self, BivParams, Cip, Prep};
use pg_phenotype_core::Trait;

/// What a `pgphenotype_pafgrs_prep` external pointer owns.
struct Handle {
    prep: Prep,
    id_type: IdType,
}

fn handle(prep: &Robj) -> HostResult<&Handle> {
    let ptr = <&ExternalPtr<Handle>>::try_from(prep).map_err(|err| match err {
        extendr_api::Error::ExpectedExternalNonNullPtr(_) => HostError::usage(
            "this prep is empty: a prep does not survive saveRDS() or a new R session; \
             rebuild it with pafgrs_prepare()"
                .to_string(),
        ),
        _ => HostError::usage("`prep` must come from pafgrs_prepare()".to_string()),
    })?;
    ptr.try_addr()
        .map_err(|_| HostError::usage("`prep` must come from pafgrs_prepare()".to_string()))
}

fn prepare_impl(columns: [Robj; 5], ndegree: &Robj, probands: Robj) -> HostResult<Robj> {
    let pedigree = Pedigree::coerce(columns)?;
    let ndegree = input::saturating_whole("ndegree", input::number("ndegree", ndegree)?)?;
    let probands = (!probands.is_null())
        .then(|| input::coerce_required("probands", &probands))
        .transpose()?;
    let pool = threads::pool()?;
    let prep = pool.install(|| {
        pafgrs::prepare(
            pedigree.input(),
            ndegree,
            probands.as_ref().map(|c| c.values.as_slice()),
        )
    })?;
    let mut out: Robj = ExternalPtr::new(Handle {
        prep,
        id_type: pedigree.ids.storage,
    })
    .into();
    out.set_class(["pgphenotype_pafgrs_prep"])
        .expect("a character class attribute");
    Ok(out)
}

/// Validate a pedigree and build its relative structure in the pool.
#[extendr]
fn pafgrs_prepare(
    id: Robj,
    mother: Robj,
    father: Robj,
    twin: Robj,
    sex: Robj,
    ndegree: Robj,
    probands: Robj,
) -> Robj {
    finish(prepare_impl(
        [id, mother, father, twin, sex],
        &ndegree,
        probands,
    ))
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

fn cip_of(ages: &Robj, cip: &Robj) -> HostResult<Cip> {
    let ages = input::doubles("ages", ages)?.to_vec();
    let cip = input::doubles("cip", cip)?.to_vec();
    Ok(Cip::new(ages, cip)?)
}

/// `list(prevalence, threshold)` of a valid CIP table.
#[extendr]
fn pafgrs_check_cip(ages: Robj, cip: Robj) -> Robj {
    finish(
        cip_of(&ages, &cip)
            .map(|c| list!(prevalence = c.prevalence(), threshold = c.threshold()).into_robj()),
    )
}

/// The `values` and `kind` of an R `trait()`; the R side checked both.
fn trait_of<'a>(values: &'a Robj, kind: &Robj) -> HostResult<Trait<'a>> {
    Ok(Trait {
        values: input::doubles("trait", values)?,
        kind: input::trait_kind(kind.as_str().unwrap_or_default())?,
        n_levels: None,
    })
}

fn counts(values: &[u32]) -> Robj {
    Integers::from_values(values.iter().map(|&v| v as i32)).into_robj()
}

fn score_univariate_impl(
    prep: &Robj,
    values: [&Robj; 3],
    cip: [&Robj; 2],
    h2: &Robj,
) -> HostResult<Robj> {
    let h = handle(prep)?;
    let h2 = input::number("h2", h2)?;
    let cip = cip_of(cip[0], cip[1])?;
    let [values, kind, age] = values;
    let values = trait_of(values, kind)?;
    let age = input::doubles("age", age)?;
    let pool = threads::pool()?;
    let s = pool.install(|| pafgrs::score_univariate(&h.prep, values, age, &cip, h2))?;
    Ok(list!(
        id = h.id_type.to_robj(&s.ids),
        est = s.est,
        var = s.var,
        n_relatives = counts(&s.n_relatives),
        controls_without_age = s.controls_without_age as i32,
        threshold = s.threshold,
        prevalence = cip.prevalence()
    )
    .into_robj())
}

/// Univariate scores of every proband, in the pool.
#[extendr]
fn pafgrs_score_univariate(
    prep: Robj,
    values: Robj,
    kind: Robj,
    age: Robj,
    cip_ages: Robj,
    cip_values: Robj,
    h2: Robj,
) -> Robj {
    finish(score_univariate_impl(
        &prep,
        [&values, &kind, &age],
        [&cip_ages, &cip_values],
        &h2,
    ))
}

fn score_bivariate_impl(
    prep: &Robj,
    traits: [[&Robj; 3]; 2],
    cips: [[&Robj; 2]; 2],
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
    let cip = [
        cip_of(cips[0][0], cips[0][1])?,
        cip_of(cips[1][0], cips[1][1])?,
    ];
    let [[values1, kind1, age1], [values2, kind2, age2]] = traits;
    let values = [trait_of(values1, kind1)?, trait_of(values2, kind2)?];
    let ages = [input::doubles("age1", age1)?, input::doubles("age2", age2)?];
    let pool = threads::pool()?;
    let s = pool
        .install(|| pafgrs::score_bivariate(&h.prep, values, ages, [&cip[0], &cip[1]], params))?;
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
        controls_without_age = s.controls_without_age.map(|c| c as i32).to_vec(),
        threshold = s.threshold.to_vec(),
        prevalence = vec![cip[0].prevalence(), cip[1].prevalence()],
        rho_within = params.rho_within()
    )
    .into_robj())
}

/// Bivariate scores of every proband, in the pool.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn pafgrs_score_bivariate(
    prep: Robj,
    values1: Robj,
    kind1: Robj,
    age1: Robj,
    values2: Robj,
    kind2: Robj,
    age2: Robj,
    cip1_ages: Robj,
    cip1_values: Robj,
    cip2_ages: Robj,
    cip2_values: Robj,
    h2: Robj,
    rg: Robj,
    rho_within: Robj,
) -> Robj {
    finish(score_bivariate_impl(
        &prep,
        [[&values1, &kind1, &age1], [&values2, &kind2, &age2]],
        [[&cip1_ages, &cip1_values], [&cip2_ages, &cip2_values]],
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
    fn pafgrs_score_univariate;
    fn pafgrs_score_bivariate;
}
