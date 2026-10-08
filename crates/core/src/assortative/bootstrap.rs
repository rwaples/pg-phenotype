//! The one-step Mate Network bootstrap (pedsum `assortative_kernels.py:1337-1717`,
//! `assortative_mating.py:919-1130`, `:1273-1308`, `:1756-1780`).
//!
//! A draw resamples the cell's G Mate Networks with replacement; a pair's
//! weight is its network's multiplicity.  Closed-form estimators are exact
//! on the weighted draw; a latent correlation refits its first step and
//! takes one Newton step from the observed ρ̂ with the draw's score and the
//! full-sample Hessian.  A draw whose Hessian is not positive is refit in full.

use super::bvn;
use super::estimators::{argsort, point, table_shape, Serial, Sides};
use super::kernels::{
    count_table, inverse_sd, margin, margin_status, pearson, sorted_ranks_into, stratum_moments,
    thresholds, Grid, Moments, Polyserial, Status, Table, LATENT_BOUND, TINY,
};
use super::result::{Draws, Estimator, Reason};
use super::rng::{network_multiplicities, network_weights};
use super::sample::{n_codes, pooled_levels, CellPairs};
use rayon::prelude::*;
use std::collections::BTreeMap;

/// A CI is published only when at least this share of draws is valid.
const MIN_VALID_DRAW_SHARE: f64 = 0.95;

/// One estimator's value on one draw.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Draw {
    Value(f64),
    Failed(Status),
    /// The one-step Hessian at ρ̂ is not positive: refit in full.
    Refit,
}

/// The per-draw scratch of one worker.
struct Scratch {
    mult: Vec<u32>,
    w: Vec<f64>,
    spare: Vec<f64>,
}

/// The cell's draws: one column per fitted estimator, crude first, the
/// stratified one last, from the observed values `starts` (`None` where
/// undefined).
pub(crate) fn draws(
    estimators: &[(Estimator, bool)],
    starts: &[Option<f64>],
    pairs: &CellPairs,
    labels: &[usize],
    n_draws: u64,
    seed: i64,
) -> Vec<Vec<Draw>> {
    let g = labels.iter().max().map_or(0, |&l| l + 1);
    let stratified = estimators.last().is_some_and(|e| e.1);
    let columns: Vec<Vec<Draw>> = match Sides::of(pairs) {
        Sides::Continuous => continuous(pairs, labels, g, n_draws, seed, stratified),
        Sides::Tables { m, f } => {
            let two = estimators[0].0 == Estimator::Tetrachoric;
            tables(
                pairs,
                [m, f],
                labels,
                g,
                n_draws,
                seed,
                starts,
                two,
                stratified,
            )
        }
        Sides::Serial(s) => serial(
            s,
            labels,
            g,
            n_draws,
            seed,
            starts,
            estimators.len(),
            stratified,
        ),
    };
    columns
        .into_iter()
        .zip(estimators.iter().zip(starts))
        .map(|(column, (&(est, strat), start))| {
            column
                .into_par_iter()
                .enumerate()
                .map(|(d, draw)| match draw {
                    Draw::Refit => {
                        let mut w = vec![0.0; labels.len()];
                        let mut mult = vec![0; g];
                        network_weights(&mut w, &mut mult, labels, seed, d as u64);
                        match point(est, strat, pairs, &w, *start) {
                            Ok(p) => Draw::Value(p.value),
                            Err(status) => Draw::Failed(status),
                        }
                    }
                    other => other,
                })
                .collect()
        })
        .collect()
}

/// Run `per_draw` over draws `0..n_draws` on the pool, each with a worker's
/// scratch filled with that draw's network multiplicities and, when
/// `weights`, its pair weights, plus a spare buffer the draw may reuse;
/// transpose to columns.
fn run(
    labels: &[usize],
    g: usize,
    n_draws: u64,
    seed: i64,
    n_columns: usize,
    weights: bool,
    per_draw: impl Fn(&[f64], &[u32], &mut Vec<f64>) -> Vec<Draw> + Sync,
) -> Vec<Vec<Draw>> {
    let rows: Vec<Vec<Draw>> = (0..n_draws)
        .into_par_iter()
        .map_init(
            || Scratch {
                mult: vec![0; g],
                w: vec![0.0; if weights { labels.len() } else { 0 }],
                spare: Vec::new(),
            },
            |s, d| {
                if weights {
                    network_weights(&mut s.w, &mut s.mult, labels, seed, d);
                } else {
                    network_multiplicities(&mut s.mult, seed, d);
                }
                per_draw(&s.w, &s.mult, &mut s.spare)
            },
        )
        .collect();
    (0..n_columns)
        .map(|c| rows.iter().map(|r| r[c]).collect())
        .collect()
}

fn kernel_draw(k: Result<f64, Status>) -> Draw {
    match k {
        Ok(v) => Draw::Value(v),
        Err(s) => Draw::Failed(s),
    }
}

/// Continuous x continuous: exact Pearson, Spearman and stratified Pearson.
fn continuous(
    pairs: &CellPairs,
    labels: &[usize],
    g: usize,
    n_draws: u64,
    seed: i64,
    stratified: bool,
) -> Vec<Vec<Draw>> {
    let order_m = argsort(&pairs.m);
    let order_f = argsort(&pairs.f);
    let mut inv_f = vec![0; order_f.len()];
    for (t, &i) in order_f.iter().enumerate() {
        inv_f[i] = t;
    }
    let m_sorted: Vec<f64> = order_m.iter().map(|&i| pairs.m[i]).collect();
    let f_sorted: Vec<f64> = order_f.iter().map(|&i| pairs.f[i]).collect();
    let labels_m: Vec<usize> = order_m.iter().map(|&i| labels[i]).collect();
    let labels_f: Vec<usize> = order_f.iter().map(|&i| labels[i]).collect();
    let f_pos: Vec<usize> = order_m.iter().map(|&i| inv_f[i]).collect();
    let (n_m, n_f) = pairs.n_strata();
    let n_columns = 2 + usize::from(stratified);
    run(
        labels,
        g,
        n_draws,
        seed,
        n_columns,
        true,
        |w, mult, rank_f| {
            let mut out = vec![kernel_draw(pearson(&pairs.m, &pairs.f, w, false))];
            rank_f.resize(f_sorted.len(), 0.0);
            sorted_ranks_into(rank_f, &f_sorted, |t| mult[labels_f[t]] as f64);
            out.push(kernel_draw(spearman_sorted(
                &m_sorted,
                |q| mult[labels_m[q]] as f64,
                rank_f,
                &f_pos,
            )));
            if stratified {
                out.push(kernel_draw(stratified_pearson(pairs, n_m, n_f, w)));
            }
            out
        },
    )
}

/// Weighted Pearson of the ranks, walked in `m`'s sorted order and centred
/// at `(T + 1) / 2`.
fn spearman_sorted(
    m_sorted: &[f64],
    w_sorted: impl Fn(usize) -> f64,
    rank_f: &[f64],
    f_pos: &[usize],
) -> Result<f64, Status> {
    let n = m_sorted.len();
    let t = (0..n).fold(0.0, |a, q| a + w_sorted(q));
    if t == 0.0 {
        return Err(Status::NoPairs);
    }
    let centre = (t + 1.0) / 2.0;
    let (mut sxy, mut sxx, mut syy, mut cum) = (0.0, 0.0, 0.0, 0.0);
    let mut i = 0;
    while i < n {
        let value = m_sorted[i];
        let mut j = i;
        let mut group = 0.0;
        while j < n && m_sorted[j] == value {
            group += w_sorted(j);
            j += 1;
        }
        if group > 0.0 {
            let dx = cum + (group + 1.0) / 2.0 - centre;
            for q in i..j {
                let wq = w_sorted(q);
                if wq > 0.0 {
                    let dy = rank_f[f_pos[q]] - centre;
                    sxy += wq * dx * dy;
                    sxx += wq * dx * dx;
                    syy += wq * dy * dy;
                }
            }
            cum += group;
        }
        i = j;
    }
    if sxx == 0.0 || syy == 0.0 {
        return Err(Status::ConstantMargin);
    }
    Ok((sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0))
}

/// The draw kernels' stratified Pearson: one pass after both sides' checks.
fn stratified_pearson(pairs: &CellPairs, n_m: usize, n_f: usize, w: &[f64]) -> Result<f64, Status> {
    let mm = stratum_moments(&pairs.m, &pairs.m_stratum, w, n_m, false);
    if mm.total.iter().fold(0.0, |a, &v| a + v) == 0.0 {
        return Err(Status::NoPairs);
    }
    if mm.any_degenerate() {
        return Err(Status::DegenerateStratum);
    }
    let mf = stratum_moments(&pairs.f, &pairs.f_stratum, w, n_f, false);
    if mf.any_degenerate() {
        return Err(Status::DegenerateStratum);
    }
    let (inv_m, inv_f) = (inverse_sd(&mm.var), inverse_sd(&mf.var));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (i, &wi) in w.iter().enumerate() {
        if wi > 0.0 {
            let (s, t) = (pairs.m_stratum[i], pairs.f_stratum[i]);
            let zm = (pairs.m[i] - mm.mean[s]) * inv_m[s];
            let zf = (pairs.f[i] - mf.mean[t]) * inv_f[t];
            sxy += wi * zm * zf;
            sxx += wi * zm * zm;
            syy += wi * zf * zf;
        }
    }
    Ok((sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0))
}

/// The polyserial first step's checks of the continuous side.
fn continuous_status(m: &Moments) -> Option<Status> {
    let mut present = false;
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for s in 0..m.total.len() {
        if m.total[s] > 0.0 {
            present = true;
            lo = lo.min(m.lo[s]);
            hi = hi.max(m.hi[s]);
        }
    }
    if !present {
        Some(Status::NoPairs)
    } else if lo == hi {
        Some(Status::ConstantMargin)
    } else if m.any_degenerate() {
        Some(Status::DegenerateStratum)
    } else {
        None
    }
}

/// One form's one-step polyserial draw from the draw's first step and the
/// full-sample Hessian `hess`.
#[allow(clippy::too_many_arguments)]
fn polyserial_form(
    rho: f64,
    hess: f64,
    serial: &Serial<'_>,
    moments: &Moments,
    y_margin: &Grid<f64>,
    levels: &Grid<bool>,
    w: &[f64],
) -> Draw {
    if let Some(status) = continuous_status(moments).or_else(|| margin_status(y_margin, levels)) {
        return Draw::Failed(status);
    }
    if !(hess > 0.0) {
        return Draw::Refit;
    }
    let inv_sd = inverse_sd(&moments.var);
    let tau = thresholds(y_margin);
    let kernel = Polyserial {
        x: serial.x,
        x_stratum: serial.x_stratum,
        mean: &moments.mean,
        inv_sd: &inv_sd,
        y: serial.y,
        y_stratum: serial.y_stratum,
        tau: &tau,
        w,
    };
    let grad = kernel.grad(rho, false);
    Draw::Value((rho - grad / hess).clamp(-LATENT_BOUND, LATENT_BOUND))
}

/// Every stratum's moments taken together, repeated in each of `n` rows.
fn pooled_moments(m: &Moments, n: usize) -> Moments {
    let mut t = 0.0;
    let mut mean = 0.0;
    for s in 0..m.total.len() {
        t += m.total[s];
        mean += m.total[s] * m.mean[s];
    }
    if t > 0.0 {
        mean /= t;
    }
    let mut v = 0.0;
    for s in 0..m.total.len() {
        if m.total[s] > 0.0 {
            let d = m.mean[s] - mean;
            v += m.total[s] * (m.var[s] + d * d);
        }
    }
    if t > 0.0 {
        v /= t;
    }
    let lo = m.lo.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = m.hi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Moments {
        total: vec![t; n],
        mean: vec![mean; n],
        var: vec![v; n],
        lo: vec![lo; n],
        hi: vec![hi; n],
    }
}

/// The full-sample `d²NLL/dρ²` of a polyserial form at `rho` (NaN when `rho` is).
fn polyserial_hessian(rho: f64, serial: &Serial<'_>) -> f64 {
    if rho.is_nan() {
        return f64::NAN;
    }
    let w = vec![1.0; serial.x.len()];
    match serial.first_step(&w) {
        Ok(first) => {
            let inv_sd = inverse_sd(&first.var);
            serial.kernel(&first, &inv_sd, &w).terms(rho, true).hess
        }
        Err(_) => f64::NAN,
    }
}

/// Continuous x discrete: one-step polyserial draws (crude, stratified) and
/// the exact point-biserial of a binary side.
#[allow(clippy::too_many_arguments)]
fn serial(
    s: Serial<'_>,
    labels: &[usize],
    g: usize,
    n_draws: u64,
    seed: i64,
    starts: &[Option<f64>],
    n_fitted: usize,
    stratified: bool,
) -> Vec<Vec<Draw>> {
    let point_biserial = n_fitted == 2 + usize::from(stratified);
    let zeros = vec![0; s.x.len()];
    let crude = Serial {
        x_stratum: &zeros,
        y_stratum: &zeros,
        y_levels: &pooled_levels(s.y_levels),
        ..s
    };
    let rho_crude = starts[0].unwrap_or(f64::NAN);
    let rho_strat = if stratified {
        starts[n_fitted - 1].unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };
    let hess_crude = polyserial_hessian(rho_crude, &crude);
    let hess_strat = polyserial_hessian(rho_strat, &s);
    let (n_ys, k) = (s.y_levels.rows, s.y_levels.cols);
    let n_xs = n_codes(s.x_stratum);
    let mut repeated_levels = Grid::filled(n_ys, k, false);
    for c in 0..k {
        let shown = (0..n_ys).any(|r| s.y_levels.at(r, c));
        for r in 0..n_ys {
            *repeated_levels.at_mut(r, c) = shown;
        }
    }
    run(labels, g, n_draws, seed, n_fitted, true, |w, _, _| {
        let moments = stratum_moments(s.x, s.x_stratum, w, n_xs, false);
        let y_margin = margin(s.y, s.y_stratum, w, n_ys, k, false);
        let mut out = vec![Draw::Value(f64::NAN)];
        if !rho_crude.is_nan() {
            let mut pooled_margin = Grid::filled(n_ys, k, 0.0);
            for c in 0..k {
                let count = (0..n_ys).fold(0.0, |a, r| a + y_margin.at(r, c));
                for r in 0..n_ys {
                    *pooled_margin.at_mut(r, c) = count;
                }
            }
            out[0] = polyserial_form(
                rho_crude,
                hess_crude,
                &s,
                &pooled_moments(&moments, n_xs),
                &pooled_margin,
                &repeated_levels,
                w,
            );
        }
        if point_biserial {
            out.push(kernel_draw(pearson(s.x, s.y, w, false)));
        }
        if stratified {
            out.push(if rho_strat.is_nan() {
                Draw::Value(f64::NAN)
            } else {
                polyserial_form(
                    rho_strat, hess_strat, &s, &moments, &y_margin, s.y_levels, w,
                )
            });
        }
        out
    })
}

/// Per draw table: `(grad, hess)` of the NLL in ρ at `rho`, thresholds refit
/// from the table's margins, or the margins' failure.
fn polychoric_terms(
    n: &Table,
    m_levels: &Grid<bool>,
    f_levels: &Grid<bool>,
    rho: f64,
) -> Result<(f64, f64), Status> {
    if !n.data.iter().any(|&v| v != 0.0) {
        return Err(Status::NoPairs);
    }
    let (m_margin, f_margin) = (n.mother_margin(), n.father_margin());
    for (margin, levels) in [(&m_margin, m_levels), (&f_margin, f_levels)] {
        if let Some(status) = margin_status(margin, levels) {
            return Err(status);
        }
    }
    let (a, b) = (thresholds(&m_margin), thresholds(&f_margin));
    let (mut grad, mut hess) = (0.0, 0.0);
    for s in 0..n.n_ms {
        for u in 0..n.n_fs {
            let grid = |f: &dyn Fn(f64, f64) -> f64| {
                let mut g = Grid::filled(n.k_m + 1, n.k_f + 1, 0.0);
                for i in 0..=n.k_m {
                    for j in 0..=n.k_f {
                        *g.at_mut(i, j) = f(a.at(s, i), b.at(u, j));
                    }
                }
                g
            };
            let cdf = grid(&|h, k| bvn::cdf(h, k, rho));
            let pdf = grid(&|h, k| bvn::pdf_and_drho(h, k, rho).0);
            let drho = grid(&|h, k| bvn::pdf_and_drho(h, k, rho).1);
            for i in 0..n.k_m {
                for j in 0..n.k_f {
                    let count = n.at(s, u, i, j);
                    if count > 0.0 {
                        let c = |g: &Grid<f64>| {
                            g.at(i + 1, j + 1) - g.at(i, j + 1) - g.at(i + 1, j) + g.at(i, j)
                        };
                        let pi = c(&cdf).max(TINY);
                        let score = c(&pdf) / pi;
                        grad += count * score;
                        hess += count * (c(&drho) / pi - score * score);
                    }
                }
            }
        }
    }
    Ok((-grad, -hess))
}

/// One-step polychoric draws from per-draw tables and the observed table.
fn polychoric_step(
    table: &Table,
    observed_hess: f64,
    m_levels: &Grid<bool>,
    f_levels: &Grid<bool>,
    rho: f64,
) -> Draw {
    match polychoric_terms(table, m_levels, f_levels, rho) {
        Err(status) => Draw::Failed(status),
        Ok(_) if !(observed_hess > 0.0) => Draw::Refit,
        Ok((grad, _)) => {
            Draw::Value((rho - grad / observed_hess).clamp(-LATENT_BOUND, LATENT_BOUND))
        }
    }
}

/// Odds ratio and phi of a pooled 2 x 2 table, undefined on the same draws.
fn two_by_two(t: &Table) -> (Draw, Draw) {
    let (a, b, c, d) = (t.data[0], t.data[1], t.data[2], t.data[3]);
    let rows = [a + b, c + d];
    let cols = [a + c, b + d];
    if a + b + c + d == 0.0 {
        return (Draw::Failed(Status::NoPairs), Draw::Failed(Status::NoPairs));
    }
    if rows.contains(&0.0) || cols.contains(&0.0) {
        return (
            Draw::Failed(Status::ConstantMargin),
            Draw::Failed(Status::ConstantMargin),
        );
    }
    let odds = a * d / (b * c);
    let phi = ((a * d - b * c) / (rows[0] * rows[1] * (cols[0] * cols[1])).sqrt()).clamp(-1.0, 1.0);
    (Draw::Value(odds), Draw::Value(phi))
}

/// Discrete x discrete: one-step polychoric draws (crude, stratified) and the
/// exact odds ratio and phi of a 2 x 2 cell.
#[allow(clippy::too_many_arguments)]
fn tables(
    pairs: &CellPairs,
    [m_levels, f_levels]: [&Grid<bool>; 2],
    labels: &[usize],
    g: usize,
    n_draws: u64,
    seed: i64,
    starts: &[Option<f64>],
    two: bool,
    stratified: bool,
) -> Vec<Vec<Draw>> {
    let shape = table_shape(pairs, m_levels, f_levels);
    let ones = vec![1.0; pairs.len()];
    let observed = count_table(
        &pairs.m_stratum,
        &pairs.f_stratum,
        &pairs.m,
        &pairs.f,
        &ones,
        shape,
        true,
    );
    let (pooled_m, pooled_f) = (pooled_levels(m_levels), pooled_levels(f_levels));
    let rho_crude = starts[0].unwrap_or(f64::NAN);
    let rho_strat = starts.last().copied().flatten().unwrap_or(f64::NAN);
    let hess = |t: &Table, ml: &Grid<bool>, fl: &Grid<bool>, rho: f64| {
        polychoric_terms(t, ml, fl, rho).map_or(f64::NAN, |(_, h)| h)
    };
    let hess_crude = hess(&observed.pooled(), &pooled_m, &pooled_f, rho_crude);
    let hess_strat = hess(&observed, m_levels, f_levels, rho_strat);
    let n_columns = 1 + 2 * usize::from(two) + usize::from(stratified);
    let cell_of: Vec<usize> = (0..pairs.len())
        .map(|i| {
            observed.index(
                pairs.m_stratum[i],
                pairs.f_stratum[i],
                pairs.m[i] as usize,
                pairs.f[i] as usize,
            )
        })
        .collect();
    run(labels, g, n_draws, seed, n_columns, false, |_, mult, _| {
        let table = draw_table(&cell_of, labels, mult, shape);
        let pooled = table.pooled();
        let mut out = vec![polychoric_step(
            &pooled, hess_crude, &pooled_m, &pooled_f, rho_crude,
        )];
        if two {
            let (odds, phi) = two_by_two(&pooled);
            out.extend([odds, phi]);
        }
        if stratified {
            out.push(polychoric_step(
                &table, hess_strat, m_levels, f_levels, rho_strat,
            ));
        }
        out
    })
}

/// A draw's count table, summed pair by pair (pedsum `table_draws`): pair
/// `i` adds its network's multiplicity to its table entry `cell_of[i]`.
fn draw_table(
    cell_of: &[usize],
    labels: &[usize],
    mult: &[u32],
    [n_ms, n_fs, k_m, k_f]: [usize; 4],
) -> Table {
    let mut t = Table::zeros(n_ms, n_fs, k_m, k_f);
    for (&at, &l) in cell_of.iter().zip(labels) {
        t.data[at] += f64::from(mult[l]);
    }
    t
}

/// Percentile CI over the valid draws, with draw accounting and the reason
/// a CI is withheld.
pub(crate) fn record(
    draws: &[Draw],
    requested: u64,
    n_networks: u64,
) -> (Draws, Result<[f64; 2], Reason>) {
    let mut valid: Vec<f64> = draws
        .iter()
        .filter_map(|d| match d {
            Draw::Value(v) => Some(*v),
            _ => None,
        })
        .collect();
    let counts = Draws {
        requested,
        valid: valid.len() as u64,
        failed: (draws.len() - valid.len()) as u64,
        failure_reasons: failure_reasons(draws),
    };
    let ci = if n_networks < 2 {
        Err(Reason::SingleMateNetwork)
    } else if (valid.len() as f64) < MIN_VALID_DRAW_SHARE * requested as f64 {
        Err(Reason::TooManyFailedDraws)
    } else {
        valid.sort_by(f64::total_cmp);
        let tail = (1.0 - super::CI_LEVEL) / 2.0;
        Ok([inverted_cdf(&valid, tail), inverted_cdf(&valid, 1.0 - tail)])
    };
    (counts, ci)
}

/// NumPy's `quantile(..., method="inverted_cdf")` of sorted values.
fn inverted_cdf(sorted: &[f64], q: f64) -> f64 {
    let index = sorted.len() as f64 * q - 1.0;
    let previous = index.floor();
    let at = if index - previous == 0.0 {
        previous
    } else {
        previous + 1.0
    };
    sorted[at.max(0.0) as usize]
}

/// Failure counts by reason, in name order.
pub(crate) fn failure_reasons(draws: &[Draw]) -> Vec<(Reason, u64)> {
    let mut counts = BTreeMap::new();
    for d in draws {
        if let Draw::Failed(status) = d {
            *counts.entry(status.reason()).or_insert(0) += 1;
        }
    }
    counts.into_iter().collect()
}
