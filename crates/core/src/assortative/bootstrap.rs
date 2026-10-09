//! The one-step Mate Network bootstrap (pedsum `assortative_kernels.py:1337-1717`,
//! `assortative_mating.py:919-1130`, `:1273-1308`, `:1756-1780`).
//!
//! A draw resamples the cell's G Mate Networks with replacement; a pair's
//! weight is its network's multiplicity.  Closed-form estimators are exact
//! on the weighted draw; a latent correlation refits its first step and
//! takes one Newton step from the observed ρ̂ with the draw's score and the
//! full-sample Hessian.  A draw whose Hessian is not positive is refit in full.

use super::estimators::{argsort, point, table_shape, Serial, Sides, Tables};
use super::kernels::{
    count_table, inverse_sd, margin, margin_status, pearson, sorted_ranks_into, stratum_moments,
    thresholds, Grid, Moments, Polyserial, Status, Table, LATENT_BOUND,
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

/// One estimator of a cell: the estimator and whether it is the stratified form.
type Key = (Estimator, bool);

/// A cell's fitted estimators and their observed values, looked up by
/// estimator rather than by position.
#[derive(Clone, Copy)]
struct Fitted<'a> {
    estimators: &'a [Key],
    starts: &'a [Option<f64>],
}

impl Fitted<'_> {
    fn has(&self, key: Key) -> bool {
        self.estimators.contains(&key)
    }

    /// The observed value of `key`, NaN when it was not fitted or is undefined.
    fn start(&self, key: Key) -> f64 {
        self.estimators
            .iter()
            .position(|&k| k == key)
            .and_then(|i| self.starts[i])
            .unwrap_or(f64::NAN)
    }
}

/// The resampling of one cell's G Mate Networks.
#[derive(Clone, Copy)]
struct Resample<'a> {
    /// Each pair's Mate Network.
    labels: &'a [usize],
    /// The number of networks.
    g: usize,
    n_draws: u64,
    seed: i64,
}

/// The cell's draws: one column per entry of `estimators`, in that order,
/// from the observed values `starts` (`None` where undefined).  An estimator
/// no draw kernel computes gets no draws, so its CI is withheld.
pub(crate) fn draws(
    estimators: &[Key],
    starts: &[Option<f64>],
    pairs: &CellPairs,
    labels: &[usize],
    n_draws: u64,
    seed: i64,
) -> Vec<Vec<Draw>> {
    let fitted = Fitted { estimators, starts };
    let plan = Resample {
        labels,
        g: n_codes(labels),
        n_draws,
        seed,
    };
    let mut columns: Vec<(Key, Vec<Draw>)> = match Sides::of(pairs) {
        Sides::Continuous => continuous(pairs, plan, fitted),
        Sides::Tables { m, f } => tables(pairs, [m, f], plan, fitted),
        Sides::Serial(s) => serial(s, plan, fitted),
    };
    // An estimator no kernel produced a column for refits every draw, so a
    // key the kernels miss costs time, never the interval.
    estimators
        .iter()
        .zip(starts)
        .map(|(&(est, strat), start)| {
            let column = columns
                .iter_mut()
                .find(|(key, _)| *key == (est, strat))
                .map_or_else(
                    || vec![Draw::Refit; n_draws as usize],
                    |(_, column)| std::mem::take(column),
                );
            column
                .into_par_iter()
                .enumerate()
                .map(|(d, draw)| match draw {
                    Draw::Refit => {
                        let mut w = vec![0.0; labels.len()];
                        let mut mult = vec![0; plan.g];
                        network_weights(&mut w, &mut mult, labels, seed, d as u64);
                        kernel_draw(point(est, strat, pairs, &w, *start).map(|p| p.value))
                    }
                    other => other,
                })
                .collect()
        })
        .collect()
}

/// Run `per_draw` over the draws on the pool, each with a worker's scratch
/// filled with that draw's network multiplicities and, when `weights`, its
/// pair weights, plus a spare buffer the draw may reuse.  `per_draw` returns
/// one value per entry of `keys`, in that order; the result is one column
/// per key.
fn run(
    plan: Resample<'_>,
    keys: Vec<Key>,
    weights: bool,
    per_draw: impl Fn(&[f64], &[u32], &mut Vec<f64>) -> Vec<Draw> + Sync,
) -> Vec<(Key, Vec<Draw>)> {
    let Resample {
        labels,
        g,
        n_draws,
        seed,
    } = plan;
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
                let row = per_draw(&s.w, &s.mult, &mut s.spare);
                debug_assert_eq!(row.len(), keys.len());
                row
            },
        )
        .collect();
    keys.into_iter()
        .enumerate()
        .map(|(c, key)| (key, rows.iter().map(|r| r[c]).collect()))
        .collect()
}

fn kernel_draw(k: Result<f64, Status>) -> Draw {
    match k {
        Ok(v) => Draw::Value(v),
        Err(s) => Draw::Failed(s),
    }
}

/// Continuous x continuous: exact Pearson, Spearman and stratified Pearson.
fn continuous(pairs: &CellPairs, plan: Resample<'_>, fitted: Fitted<'_>) -> Vec<(Key, Vec<Draw>)> {
    let labels = plan.labels;
    let stratified = fitted.has((Estimator::Pearson, true));
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
    let mut keys = vec![(Estimator::Pearson, false), (Estimator::Spearman, false)];
    if stratified {
        keys.push((Estimator::Pearson, true));
    }
    run(plan, keys, true, |w, mult, rank_f| {
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
    })
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

/// Continuous x discrete: one-step polyserial (biserial) draws, crude and
/// stratified, and the exact point-biserial of a binary side.
fn serial(s: Serial<'_>, plan: Resample<'_>, fitted: Fitted<'_>) -> Vec<(Key, Vec<Draw>)> {
    let primary = if fitted.has((Estimator::Biserial, false)) {
        Estimator::Biserial
    } else {
        Estimator::Polyserial
    };
    let point_biserial = fitted.has((Estimator::PointBiserial, false));
    let stratified = fitted.has((primary, true));
    let zeros = vec![0; s.x.len()];
    let crude = Serial {
        x_stratum: &zeros,
        y_stratum: &zeros,
        y_levels: &pooled_levels(s.y_levels),
        ..s
    };
    let rho_crude = fitted.start((primary, false));
    let rho_strat = fitted.start((primary, true));
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
    let mut keys = vec![(primary, false)];
    if point_biserial {
        keys.push((Estimator::PointBiserial, false));
    }
    if stratified {
        keys.push((primary, true));
    }
    run(plan, keys, true, |w, _, _| {
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

/// One-step polychoric draw from a draw's table: thresholds refit from its
/// margins, one Newton step from the observed `rho` with the observed Hessian.
fn polychoric_step(
    table: Table,
    observed_hess: f64,
    m_levels: &Grid<bool>,
    f_levels: &Grid<bool>,
    rho: f64,
) -> Draw {
    match Tables::from_counts(table, m_levels, f_levels) {
        Err(status) => Draw::Failed(status),
        Ok(_) if !(observed_hess > 0.0) => Draw::Refit,
        Ok(t) => {
            let (grad, _) = t.grad_hess(rho);
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

/// Discrete x discrete: one-step polychoric (tetrachoric) draws, crude and
/// stratified, and the exact odds ratio and phi of a 2 x 2 cell.
fn tables(
    pairs: &CellPairs,
    [m_levels, f_levels]: [&Grid<bool>; 2],
    plan: Resample<'_>,
    fitted: Fitted<'_>,
) -> Vec<(Key, Vec<Draw>)> {
    let primary = if fitted.has((Estimator::Tetrachoric, false)) {
        Estimator::Tetrachoric
    } else {
        Estimator::Polychoric
    };
    let two = fitted.has((Estimator::OddsRatio, false)) || fitted.has((Estimator::Phi, false));
    let stratified = fitted.has((primary, true));
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
    let rho_crude = fitted.start((primary, false));
    let rho_strat = fitted.start((primary, true));
    let hess = |t: Table, ml: &Grid<bool>, fl: &Grid<bool>, rho: f64| {
        Tables::from_counts(t, ml, fl).map_or(f64::NAN, |t| t.grad_hess(rho).1)
    };
    let hess_crude = hess(observed.pooled(), &pooled_m, &pooled_f, rho_crude);
    let hess_strat = hess(observed.clone(), m_levels, f_levels, rho_strat);
    let mut keys = vec![(primary, false)];
    if two {
        keys.extend([(Estimator::OddsRatio, false), (Estimator::Phi, false)]);
    }
    if stratified {
        keys.push((primary, true));
    }
    let labels = plan.labels;
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
    run(plan, keys, false, |_, mult, _| {
        let table = draw_table(&cell_of, labels, mult, shape);
        let pooled = table.pooled();
        let mut out = Vec::with_capacity(4);
        let two_by_two = two.then(|| two_by_two(&pooled));
        out.push(polychoric_step(
            pooled, hess_crude, &pooled_m, &pooled_f, rho_crude,
        ));
        if let Some((odds, phi)) = two_by_two {
            out.extend([odds, phi]);
        }
        if stratified {
            out.push(polychoric_step(
                table, hess_strat, m_levels, f_levels, rho_strat,
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

#[cfg(test)]
mod tests {
    use super::super::cell_estimators;
    use super::super::input::Kind;
    use super::*;

    /// A side's values on 40 pairs in two strata: deterministic, every level used.
    fn side(kind: Kind, salt: usize) -> Vec<f64> {
        (0..40)
            .map(|i| {
                let x = ((i * 7 + salt * 3) % 11) as f64 / 11.0 + (i % 3) as f64 * 0.1;
                match kind {
                    Kind::Continuous => x,
                    Kind::Binary => f64::from(x > 0.5),
                    Kind::Ordinal { k } => ((x * k as f64) as usize).min(k - 1) as f64,
                }
            })
            .collect()
    }

    #[test]
    fn every_fitted_estimator_gets_its_draws() {
        let kinds = [Kind::Continuous, Kind::Binary, Kind::Ordinal { k: 3 }];
        let labels: Vec<usize> = (0..40).map(|i| i / 2).collect();
        for m in kinds {
            for f in kinds {
                let pairs = CellPairs {
                    m: side(m, 1),
                    f: side(f, 2),
                    m_stratum: (0..40).map(|i| i % 2).collect(),
                    f_stratum: (0..40).map(|i| (i / 3) % 2).collect(),
                    m_levels: None,
                    f_levels: None,
                }
                .with_levels(m.levels(), f.levels());
                let (crude, strat) = cell_estimators(m, f);
                let estimators: Vec<Key> = crude
                    .iter()
                    .map(|&e| (e, false))
                    .chain([(strat, true)])
                    .collect();
                let starts: Vec<Option<f64>> = estimators
                    .iter()
                    .map(|&(e, s)| {
                        let ones = vec![1.0; pairs.len()];
                        point(e, s, &pairs, &ones, None).ok().map(|p| p.value)
                    })
                    .collect();
                let columns = draws(&estimators, &starts, &pairs, &labels, 7, 3);
                assert_eq!(columns.len(), estimators.len());
                for (key, column) in estimators.iter().zip(&columns) {
                    assert_eq!(column.len(), 7, "{m:?} x {f:?}: {key:?} has no draws");
                }
            }
        }
    }
}
