//! The Mate Network cluster-robust two-step sandwich: per-pair influence on
//! each estimator, clustered by network (pedsum `assortative_mating.py:575-659`,
//! `:726-816`, `:1311-1352`).

use super::estimators::{cell_tables, two_by_two, Serial, Sides};
use super::kernels::{
    inverse_sd, pearson_influence, polyserial_influence, stratum_moments, Side, Status,
};
use super::result::{Estimator, Reason};
use super::sample::CellPairs;
use crate::polychoric::table::Grid;
use crate::polychoric::{cell_influence, Point};
use crate::wald::CiScale;

/// Per-pair influence on the two-step polychoric ρ̂ at `rho`.
fn polychoric_influence(
    pairs: &CellPairs,
    m_levels: &Grid<bool>,
    f_levels: &Grid<bool>,
    w: &[f64],
    rho: f64,
) -> Result<Vec<f64>, Reason> {
    let tables = cell_tables(pairs, m_levels, f_levels, w).map_err(Status::reason)?;
    let influence = cell_influence(&tables, rho)?;
    Ok((0..pairs.len())
        .map(|p| {
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
        })
        .collect())
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

fn odds_ratio_influence(pairs: &CellPairs, w: &[f64], value: f64) -> Result<Vec<f64>, Reason> {
    if !(value.is_finite() && value > 0.0) {
        return Err(Reason::InfiniteOddsRatio);
    }
    let table = two_by_two(pairs, w);
    Ok((0..pairs.len())
        .map(|i| {
            if w[i] == 0.0 {
                return 0.0;
            }
            let (m, f) = (pairs.m[i] as usize, pairs.f[i] as usize);
            let sign = if m == f { 1.0 } else { -1.0 };
            sign / table[m][f]
        })
        .collect())
}

/// Per-pair influence of a defined estimate, on its CI scale; `None` for an
/// estimator without one (Spearman).
fn influence(
    est: Estimator,
    stratified: bool,
    pairs: &CellPairs,
    value: f64,
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
        Estimator::Phi | Estimator::PointBiserial => Ok(pearson_influence_of(pairs, &w)),
        Estimator::OddsRatio => odds_ratio_influence(pairs, &w, value),
        Estimator::Pearson
        | Estimator::Tetrachoric
        | Estimator::Polychoric
        | Estimator::Biserial
        | Estimator::Polyserial => match Sides::of(pairs) {
            Sides::Continuous => Ok(pearson_influence_of(pairs, &w)),
            Sides::Tables { m, f } => polychoric_influence(pairs, m, f, &w, value),
            Sides::Serial(serial) => polyserial_influence_of(serial, &w, value),
        },
    })
}

/// `√(G/(G−1) · Σ_g (Σ_{i∈g} IF_i)²)` over the networks of `labels`.
pub(crate) fn cluster_se(influence: &[f64], labels: &[usize]) -> f64 {
    let g = labels.iter().max().map_or(0, |&l| l + 1);
    let mut sums = vec![0.0; g];
    for (&l, &v) in labels.iter().zip(influence) {
        sums[l] += v;
    }
    let ss = sums.iter().fold(0.0, |a, &s| a + s * s);
    (g as f64 / (g as f64 - 1.0) * ss).sqrt()
}

/// The sandwich SE of a defined estimate, or why there is none.
pub(crate) fn sandwich_se(
    est: Estimator,
    stratified: bool,
    pairs: &CellPairs,
    point: Point,
    labels: &[usize],
) -> Result<f64, Reason> {
    if labels.iter().max().map_or(0, |&l| l + 1) < 2 {
        return Err(Reason::SingleMateNetwork);
    }
    if est == Estimator::Spearman {
        return Err(Reason::BootstrapNotRequested);
    }
    if point.boundary == Some(true)
        || (est.ci_scale() == CiScale::FisherZ && point.value.abs() >= 1.0)
    {
        return Err(Reason::Boundary);
    }
    let infl = influence(est, stratified, pairs, point.value)
        .unwrap_or(Err(Reason::BootstrapNotRequested))?;
    let se = cluster_se(&infl, labels);
    if se.is_finite() {
        Ok(se)
    } else {
        Err(Reason::SandwichUndefined)
    }
}
