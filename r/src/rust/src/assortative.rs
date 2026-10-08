//! `assortative_mate_correlation()`: the Mate Correlation as nested lists
//! with the keys of the Python binding's dicts, `NULL` where Python has
//! `None`.

use crate::errors::{finish, int_values, HostError, HostResult};
use crate::input::{self, Pedigree};
use crate::threads;
use extendr_api::prelude::*;
use pg_phenotype_core::assortative::{
    self, Cell, Draws, Estimate, EstimatorResult, MateCorrelation, Permutation, Point, Reason,
    Settings, Strata, WithinPerson,
};
use pg_phenotype_core::Trait;

fn named(pairs: Vec<(&str, Robj)>) -> Robj {
    let (names, values): (Vec<&str>, Vec<Robj>) = pairs.into_iter().unzip();
    List::from_names_and_values(names, values)
        .expect("names and values of one length")
        .into_robj()
}

fn null() -> Robj {
    ().into_robj()
}

fn opt<T: Into<Robj>>(value: Option<T>) -> Robj {
    value.map_or_else(null, Into::into)
}

fn count(n: u64) -> Robj {
    (n as i32).into_robj()
}

fn reason_or(key: &'static str, value: Result<Robj, Reason>) -> [(String, Robj); 2] {
    let reason_key = format!("{key}_unavailable_reason");
    match value {
        Ok(v) => [(key.to_string(), v), (reason_key, null())],
        Err(r) => [(key.to_string(), null()), (reason_key, r.name().into())],
    }
}

fn draws(d: &Draws) -> Robj {
    let reasons: Vec<(&str, Robj)> = d
        .failure_reasons
        .iter()
        .map(|(r, n)| (r.name(), count(*n)))
        .collect();
    named(vec![
        ("requested", count(d.requested)),
        ("valid", count(d.valid)),
        ("failed", count(d.failed)),
        ("failure_reasons", named(reasons)),
    ])
}

fn owned(pairs: Vec<(String, Robj)>) -> Robj {
    named(pairs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect())
}

fn permutation(p: &Permutation) -> Robj {
    let mut out = vec![("statistic".to_string(), p.statistic.name().into())];
    out.extend(reason_or("p", p.p.map(Into::into)));
    out.extend([
        ("draws".to_string(), draws(&p.draws)),
        ("seed".to_string(), int_values(vec![p.seed])),
        ("n_fixed_fathers".to_string(), count(p.n_fixed_fathers)),
        ("stopped_early".to_string(), p.stopped_early.into()),
        ("draws_used".to_string(), count(p.draws_used)),
        ("sequential_h".to_string(), count(p.sequential_h)),
    ]);
    owned(out)
}

fn point(p: &Point) -> [(String, Robj); 2] {
    [
        ("value".to_string(), p.value.into()),
        ("boundary".to_string(), opt(p.boundary)),
    ]
}

fn estimate(e: &Estimate) -> Vec<(String, Robj)> {
    let mut out = point(&e.point).to_vec();
    out.extend(reason_or("se", e.se.map(Into::into)));
    out.extend(reason_or("ci", e.ci.map(|c| c.bounds.to_vec().into())));
    out.extend([
        (
            "ci_method".to_string(),
            opt(e.ci.ok().map(|c| c.method.name())),
        ),
        (
            "bootstrap".to_string(),
            e.bootstrap.as_ref().map_or_else(null, draws),
        ),
        (
            "permutation".to_string(),
            e.permutation.as_ref().map_or_else(null, permutation),
        ),
    ]);
    out
}

fn estimator_result(r: &EstimatorResult, extra: Vec<(String, Robj)>) -> Robj {
    let mut out = vec![
        ("estimator".to_string(), r.estimator.name().into()),
        ("primary".to_string(), r.primary.into()),
    ];
    match &r.outcome {
        Ok(e) => {
            out.push(("reason".to_string(), null()));
            out.extend(estimate(e));
        }
        Err(reason) => out.push(("reason".to_string(), reason.name().into())),
    }
    out.extend(extra);
    owned(out)
}

fn cell(c: &Cell) -> Robj {
    let d = &c.n_dropped;
    let crude: Vec<Robj> = c
        .crude
        .iter()
        .map(|r| estimator_result(r, Vec::new()))
        .collect();
    let stratified = c.stratified.as_ref().map_or_else(null, |s| {
        estimator_result(
            &s.result,
            vec![
                ("n_strata_mothers".to_string(), count(s.n_strata_mothers)),
                ("n_strata_fathers".to_string(), count(s.n_strata_fathers)),
            ],
        )
    });
    let table = c.table.map_or_else(null, |t| {
        List::from_values(t.map(|row| Integers::from_values(row.map(|v| v as i32)).into_robj()))
            .into_robj()
    });
    named(vec![
        ("mother_trait", count(c.mother_trait as u64)),
        ("father_trait", count(c.father_trait as u64)),
        ("n", count(c.n)),
        (
            "n_dropped",
            named(vec![
                ("mother_missing", count(d.mother_missing)),
                ("father_missing", count(d.father_missing)),
                ("both_missing", count(d.both_missing)),
                ("small_stratum", count(d.small_stratum)),
                ("degenerate_stratum", count(d.degenerate_stratum)),
            ]),
        ),
        ("n_mate_networks", count(c.n_mate_networks)),
        (
            "largest_mate_network_share",
            opt(c.largest_mate_network_share),
        ),
        ("table", table),
        ("crude", List::from_values(crude).into_robj()),
        ("stratified", stratified),
    ])
}

fn within(w: &WithinPerson) -> Robj {
    let mut out = vec![
        ("estimator".to_string(), w.estimator.name().into()),
        ("n".to_string(), count(w.n)),
    ];
    match &w.outcome {
        Ok(p) => {
            out.extend(point(p));
            out.push(("reason".to_string(), null()));
        }
        Err(r) => out.extend([
            ("value".to_string(), null()),
            ("boundary".to_string(), null()),
            ("reason".to_string(), r.name().into()),
        ]),
    }
    owned(out)
}

fn to_list(r: &MateCorrelation) -> Robj {
    let s = &r.sample;
    let e = &r.settings;
    let m = &r.method;
    named(vec![
        (
            "sample",
            named(vec![
                ("n_total", count(s.n_total)),
                (
                    "n_dropped_unknown_stratum",
                    count(s.n_dropped_unknown_stratum),
                ),
                ("n_mate_networks", count(s.n_mate_networks)),
                (
                    "largest_mate_network_share",
                    opt(s.largest_mate_network_share),
                ),
                (
                    "n_mothers_multiple_mates",
                    count(s.n_mothers_multiple_mates),
                ),
                (
                    "n_fathers_multiple_mates",
                    count(s.n_fathers_multiple_mates),
                ),
            ]),
        ),
        (
            "cells",
            List::from_values(r.cells.iter().map(cell)).into_robj(),
        ),
        (
            "within_person",
            r.within_person.as_ref().map_or_else(null, |w| {
                named(vec![
                    ("mothers", within(&w.mothers)),
                    ("fathers", within(&w.fathers)),
                ])
            }),
        ),
        (
            "settings",
            named(vec![
                ("permutations", count(e.permutations)),
                ("bootstrap", count(e.bootstrap)),
                ("seed", int_values(vec![e.seed])),
                ("threads", count(e.threads as u64)),
                ("ci_level", e.ci_level.into()),
                (
                    "min_stratum_networks",
                    opt(e.min_stratum_networks.map(|n| n as i32)),
                ),
            ]),
        ),
        (
            "method",
            named(vec![
                ("se_method", m.se_method.into()),
                ("ci_scale", m.ci_scale.into()),
                ("bootstrap_unit", m.bootstrap_unit.into()),
                ("bootstrap_assumption", m.bootstrap_assumption.into()),
                ("bootstrap_method", m.bootstrap_method.into()),
                ("permutation_null", m.permutation_null.into()),
                ("permutation_blocks", m.permutation_blocks.into()),
                ("permutation_statistic", m.permutation_statistic.into()),
                ("permutation_stopping", m.permutation_stopping.into()),
                ("permutation_stop_h", count(m.permutation_stop_h)),
            ]),
        ),
    ])
}

#[allow(clippy::too_many_arguments)]
fn mate_correlation_impl(
    columns: [Robj; 5],
    values: &List,
    kinds: &Robj,
    n_levels: &Robj,
    stratum: &Robj,
    counts: [&Robj; 4],
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
    Ok(to_list(&result))
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
) -> Robj {
    finish(mate_correlation_impl(
        [id, mother, father, twin, sex],
        &values,
        &kinds,
        &n_levels,
        &stratum,
        [&permutations, &bootstrap, &seed, &min_stratum_networks],
    ))
}

extendr_module! {
    mod assortative;
    fn assortative_mate_correlation;
}
