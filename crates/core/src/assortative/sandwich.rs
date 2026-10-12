//! The Mate Network cluster-robust two-step sandwich: per-pair influence on
//! each estimator, clustered by network (pedsum `assortative_mating.py:575-659`,
//! `:726-816`, `:1311-1352`).

use super::estimators::{cell_tables, two_by_two, Serial, Sides};
use super::kernels::{
    inverse_sd, pearson_influence, polyserial_influence, stratum_moments, Side, Status,
};
use super::result::{Estimator, Reason};
use super::sample::CellPairs;
use crate::mate_network::{Networks, Sandwich};
use crate::polychoric::table::Grid;
use crate::polychoric::{cell_influence, Point};
use crate::wald::CiScale;

/// Mate Network sums of the two-step polychoric influences at `rho`.
fn polychoric_sums(
    pairs: &CellPairs,
    m_levels: &Grid<bool>,
    f_levels: &Grid<bool>,
    w: &[f64],
    rho: f64,
    sandwich: Sandwich<'_>,
) -> Result<Vec<f64>, Reason> {
    let tables = cell_tables(pairs, m_levels, f_levels, w).map_err(Status::reason)?;
    let influence = cell_influence(&tables, rho)?;
    Ok(sandwich.sums((0..pairs.len()).map(|p| {
        if w[p] == 0.0 {
            0.0
        } else {
            influence.at(
                pairs.m_stratum[p],
                pairs.f_stratum[p],
                pairs.m[p] as usize,
                pairs.f[p] as usize,
            )
        }
    })))
}

fn polyserial_influence_of(serial: Serial<'_>, w: &[f64], rho: f64) -> Result<Vec<f64>, Reason> {
    let first = serial.first_step(w).map_err(Status::reason)?;
    let inv_sd = inverse_sd(&first.var);
    let (out, a_rr) = polyserial_influence(&serial.kernel(&first, &inv_sd, w), &first.var, rho);
    if a_rr < 0.0 {
        Ok(out)
    } else {
        Err(Reason::SandwichUndefined)
    }
}

fn pearson_influence_of(pairs: &CellPairs, w: &[f64]) -> Vec<f64> {
    let (n_m, n_f) = pairs.n_strata();
    let mm = stratum_moments(&pairs.m, &pairs.m_stratum, w, n_m, true);
    let mf = stratum_moments(&pairs.f, &pairs.f_stratum, w, n_f, true);
    pearson_influence(
        Side {
            x: &pairs.m,
            stratum: &pairs.m_stratum,
            mean: &mm.mean,
            var: &mm.var,
        },
        Side {
            x: &pairs.f,
            stratum: &pairs.f_stratum,
            mean: &mf.mean,
            var: &mf.var,
        },
        w,
    )
}

fn odds_ratio_sums(
    pairs: &CellPairs,
    w: &[f64],
    value: f64,
    sandwich: Sandwich<'_>,
) -> Result<Vec<f64>, Reason> {
    if !(value.is_finite() && value > 0.0) {
        return Err(Reason::InfiniteOddsRatio);
    }
    let table = two_by_two(pairs, w);
    Ok(sandwich.sums((0..pairs.len()).map(|i| {
        if w[i] == 0.0 {
            return 0.0;
        }
        let (m, f) = (pairs.m[i] as usize, pairs.f[i] as usize);
        let sign = if m == f { 1.0 } else { -1.0 };
        sign / table[m][f]
    })))
}

/// Mate Network influence sums of a defined estimate, on its CI scale; `None` for an
/// estimator without one (Spearman).
fn influence_sums(
    est: Estimator,
    stratified: bool,
    pairs: &CellPairs,
    value: f64,
    sandwich: Sandwich<'_>,
) -> Option<Result<Vec<f64>, Reason>> {
    let w = vec![1.0; pairs.len()];
    let pooled;
    let pairs = if stratified {
        pairs
    } else {
        pooled = pairs.pooled();
        &*pooled
    };
    Some(match est {
        Estimator::Spearman => return None,
        Estimator::Phi | Estimator::PointBiserial => {
            Ok(sandwich.sums(pearson_influence_of(pairs, &w)))
        }
        Estimator::OddsRatio => odds_ratio_sums(pairs, &w, value, sandwich),
        Estimator::Pearson
        | Estimator::Tetrachoric
        | Estimator::Polychoric
        | Estimator::Biserial
        | Estimator::Polyserial => match Sides::of(pairs) {
            Sides::Continuous => Ok(sandwich.sums(pearson_influence_of(pairs, &w))),
            Sides::Tables { m, f } => polychoric_sums(pairs, m, f, &w, value, sandwich),
            Sides::Serial(serial) => {
                polyserial_influence_of(serial, &w, value).map(|out| sandwich.sums(out))
            }
        },
    })
}

/// The sandwich SE of a defined estimate, or why there is none.
pub(crate) fn sandwich_se(
    est: Estimator,
    stratified: bool,
    pairs: &CellPairs,
    point: Point,
    networks: &Networks,
) -> Result<f64, Reason> {
    let Some(sandwich) = networks.sandwich() else {
        return Err(Reason::SingleMateNetwork);
    };
    if est == Estimator::Spearman {
        return Err(Reason::BootstrapNotRequested);
    }
    if point.boundary == Some(true)
        || (est.ci_scale() == CiScale::FisherZ && point.value.abs() >= 1.0)
    {
        return Err(Reason::Boundary);
    }
    let sums = influence_sums(est, stratified, pairs, point.value, sandwich)
        .unwrap_or(Err(Reason::BootstrapNotRequested))?;
    let se = sandwich.se(&sums);
    if se.is_finite() {
        Ok(se)
    } else {
        Err(Reason::SandwichUndefined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mate_network::Relabel;

    #[test]
    fn streamed_table_sums_match_materialized_influences_bitwise() {
        for (k_m, k_f) in [(2, 2), (3, 4)] {
            let mut pairs = CellPairs {
                m: vec![],
                f: vec![],
                m_stratum: vec![],
                f_stratum: vec![],
                m_levels: None,
                f_levels: None,
            };
            for repeat in 0..7 {
                for s in 0..2 {
                    for t in 0..2 {
                        for m in 0..k_m {
                            for f in 0..k_f {
                                pairs.m.push(m as f64);
                                pairs.f.push(f as f64);
                                pairs.m_stratum.push(s);
                                pairs.f_stratum.push(t);
                                if m == f && repeat % 2 == 0 {
                                    pairs.m.push(m as f64);
                                    pairs.f.push(f as f64);
                                    pairs.m_stratum.push(s);
                                    pairs.f_stratum.push(t);
                                }
                            }
                        }
                    }
                }
            }
            let pairs = pairs.with_levels(Some(k_m), Some(k_f));
            let labels: Vec<usize> = (0..pairs.len()).map(|i| i % 3).collect();
            let networks = Relabel::new(3).networks(&labels);
            assert_eq!(networks.labels(), labels);
            let sandwich = networks.sandwich().unwrap();
            let w = vec![1.0; pairs.len()];
            let bits = |sums: Vec<f64>| sums.into_iter().map(f64::to_bits).collect::<Vec<_>>();
            for stratified in [false, true] {
                let pooled = pairs.pooled();
                let sample = if stratified { &pairs } else { &*pooled };
                let m_levels = sample.m_levels.as_ref().unwrap();
                let f_levels = sample.f_levels.as_ref().unwrap();
                let rho = 0.23;
                let tables = cell_tables(sample, m_levels, f_levels, &w).unwrap();
                let influence = cell_influence(&tables, rho).unwrap();
                let materialized: Vec<f64> = (0..sample.len())
                    .map(|p| {
                        if w[p] == 0.0 {
                            0.0
                        } else {
                            influence.at(
                                sample.m_stratum[p],
                                sample.f_stratum[p],
                                sample.m[p] as usize,
                                sample.f[p] as usize,
                            )
                        }
                    })
                    .collect();
                let expected = sandwich.sums(materialized);
                assert!(expected.iter().all(|v| v.is_finite()));
                let streamed =
                    influence_sums(Estimator::Polychoric, stratified, &pairs, rho, sandwich)
                        .unwrap()
                        .unwrap();
                assert_eq!(bits(streamed), bits(expected));

                if k_m == 2 && k_f == 2 {
                    let table = two_by_two(sample, &w);
                    let materialized: Vec<f64> = (0..sample.len())
                        .map(|i| {
                            if w[i] == 0.0 {
                                return 0.0;
                            }
                            let (m, f) = (sample.m[i] as usize, sample.f[i] as usize);
                            let sign = if m == f { 1.0 } else { -1.0 };
                            sign / table[m][f]
                        })
                        .collect();
                    let expected = sandwich.sums(materialized);
                    assert!(expected.iter().all(|v| v.is_finite()));
                    let streamed = influence_sums(
                        Estimator::OddsRatio,
                        stratified,
                        &pairs,
                        table[0][0] * table[1][1] / (table[0][1] * table[1][0]),
                        sandwich,
                    )
                    .unwrap()
                    .unwrap();
                    assert_eq!(bits(streamed), bits(expected));
                }
            }
        }
    }
}
