//! The father-permutation test of each primary estimate (pedsum
//! `assortative_mating.py:1185-1214`, `:1381-1438`, `:1457-1753`;
//! `assortative_kernels.py:877-1335`).
//!
//! At ρ = 0 every latent likelihood's score is `Σ e_m e_f`, with `e` a side's
//! conditional latent mean; normalised by its information it is the Pearson r
//! of the two sides' scores.  Fathers' trait vectors are exchanged within
//! blocks of stratum x missingness pattern; mothers' scores stay fixed and
//! the father side is refit per draw.  Draws run in batches and each tested
//! form stops at its `h`-th exceedance (Besag & Clifford 1991, closed scheme).

use super::kernels::{latent_means, margin, margin_status, Grid, Status};
use super::rng::shuffle_index;
use super::sample::CellPairs;
use rayon::prelude::*;

/// Draws in the first batch; a later batch at least doubles the draws so far.
const FIRST_BATCH: u64 = 64;
/// Exceedances that stop a form (`INFERENCE["permutation_stop_h"]`).
pub(crate) const SEQUENTIAL_H: u64 = super::METHOD.permutation_stop_h;

/// Permutation block of each distinct father: stratum x which traits he has,
/// numbered in (stratum, pattern) order.
pub(crate) fn father_blocks(stratum: &[usize], present: &[Vec<bool>]) -> Vec<usize> {
    let n_traits = present.len();
    let keys: Vec<usize> = (0..stratum.len())
        .map(|i| {
            let pattern = (0..n_traits).fold(0, |p, t| p | (usize::from(present[t][i]) << t));
            (stratum[i] << n_traits) | pattern
        })
        .collect();
    let mut distinct = keys.clone();
    distinct.sort_unstable();
    distinct.dedup();
    keys.iter()
        .map(|k| distinct.binary_search(k).unwrap_or(0))
        .collect()
}

/// One cell's permutation input.
pub(crate) struct Cell<'a> {
    pub pairs: &'a CellPairs,
    /// The father trait's index.
    pub trait_index: usize,
    /// Each pair's distinct father (father-id order).
    pub pair_father: &'a [usize],
    /// Crude and stratified per-pair mother scores.
    pub e_m: [Vec<f64>; 2],
    /// Which forms (crude, stratified) are tested.
    pub tested: [bool; 2],
}

/// Per pair, one side's conditional latent mean given its value, margins per
/// stratum; a continuous side standardised within its stratum.
pub(crate) fn latent_scores(
    values: &[f64],
    stratum: &[usize],
    k: Option<usize>,
) -> Option<Vec<f64>> {
    let Some(k) = k else {
        return super::estimators::standardise(values, stratum, &vec![1.0; values.len()]).ok();
    };
    let n_strata = super::sample::n_codes(stratum);
    let m = margin(values, stratum, &vec![1.0; values.len()], n_strata, k, true);
    let e = latent_means(&m);
    Some(
        values
            .iter()
            .zip(stratum)
            .map(|(&v, &s)| e.at(s, v as usize))
            .collect(),
    )
}

/// One form's statistic before standardising: `Σ e_m e_f` and the father
/// side's `Σ e_f²`.
#[derive(Clone, Copy, Debug)]
struct Score {
    cross: f64,
    ss_f: f64,
}

/// One father stratum of a cell: the shift its values are summed about (a
/// continuous trait's observed mean, else 0), its pairs, and the summed
/// crude and stratified mother scores.
#[derive(Clone, Copy, Debug)]
struct StratumSums {
    shift: f64,
    pairs: f64,
    e1: f64,
    e2: f64,
}

/// One cell collapsed onto its distinct fathers, in row order.
struct Fathers {
    row: Vec<u32>,
    count: Vec<i32>,
    e1: Vec<f32>,
    e2: Vec<f32>,
    /// Offsets of each run of fathers in one stratum, then the end.
    runs: Vec<usize>,
    run_stratum: Vec<usize>,
    sums: Vec<StratumSums>,
    n_pairs: f64,
    /// `None` for a continuous father trait.
    levels: Option<Grid<bool>>,
    trait_index: usize,
}

fn bincount(index: &[usize], weights: impl Iterator<Item = f64>, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (&i, w) in index.iter().zip(weights) {
        out[i] += w;
    }
    out
}

/// The cells of a run packed for the draws: fathers renumbered block by block.
pub(crate) struct Packed {
    block_start: Vec<usize>,
    /// Trait `t` of the father at row `r`, at `rows[t][r]`.
    rows: Vec<Vec<f32>>,
    cells: Vec<Fathers>,
    ss_m: Vec<[f64; 2]>,
}

/// NumPy's pairwise summation of `x` (`np.add.reduce` of a float64 array).
fn numpy_sum(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 8 {
        return x.iter().fold(0.0, |a, &v| a + v);
    }
    if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&x[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += x[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += x[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    numpy_sum(&x[..n2]) + numpy_sum(&x[n2..])
}

/// `nanmean` of `column`: the pairwise sum of the present values over their count.
fn nanmean(column: &[f64]) -> f64 {
    let present: Vec<f64> = column
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v })
        .collect();
    numpy_sum(&present) / column.iter().filter(|v| !v.is_nan()).count() as f64
}

/// `column -= nanmean(column)`, then `/= nanstd(column)` when positive, in
/// float64, so its variation survives float32 storage.
fn standardise_column(column: &mut [f64]) {
    let mean = nanmean(column);
    for v in column.iter_mut() {
        *v -= mean;
    }
    let centre = nanmean(column);
    let squares: Vec<f64> = column
        .iter()
        .map(|&v| {
            if v.is_nan() {
                0.0
            } else {
                (v - centre) * (v - centre)
            }
        })
        .collect();
    let count = column.iter().filter(|v| !v.is_nan()).count() as f64;
    let sd = (numpy_sum(&squares) / count).sqrt();
    if sd > 0.0 {
        for v in column.iter_mut() {
            *v /= sd;
        }
    }
}

impl Packed {
    /// `trait_values[t][i]`: trait `t` of distinct father `i`; `blocks[i]` his block.
    pub fn new(cells: &[Cell<'_>], trait_values: &[Vec<f64>], blocks: &[usize]) -> Packed {
        let n_fathers = blocks.len();
        let mut order: Vec<usize> = (0..n_fathers).collect();
        order.sort_by_key(|&i| blocks[i]);
        let n_blocks = blocks.iter().max().map_or(0, |&b| b + 1);
        let mut block_start = vec![0; n_blocks + 1];
        for &b in blocks {
            block_start[b + 1] += 1;
        }
        for b in 0..n_blocks {
            block_start[b + 1] += block_start[b];
        }
        let mut rank = vec![0; n_fathers];
        for (r, &i) in order.iter().enumerate() {
            rank[i] = r;
        }
        let rows: Vec<Vec<f32>> = trait_values
            .iter()
            .enumerate()
            .map(|(t, values)| {
                let mut column: Vec<f64> = order.iter().map(|&i| values[i]).collect();
                if cells
                    .iter()
                    .any(|c| c.trait_index == t && c.pairs.f_levels.is_none())
                {
                    standardise_column(&mut column);
                }
                column.iter().map(|&v| v as f32).collect()
            })
            .collect();
        let packed_cells = cells
            .iter()
            .map(|c| Packed::fathers(c, &rank, &rows[c.trait_index]))
            .collect();
        let ss_m = cells
            .iter()
            .map(|c| c.e_m.clone().map(|e| e.iter().fold(0.0, |a, &v| a + v * v)))
            .collect();
        Packed {
            block_start,
            rows,
            cells: packed_cells,
            ss_m,
        }
    }

    fn fathers(c: &Cell<'_>, rank: &[usize], values: &[f32]) -> Fathers {
        let n_rows = rank.len();
        let row: Vec<usize> = c.pair_father.iter().map(|&f| rank[f]).collect();
        let mut pairs_per_row = vec![0i32; n_rows];
        for &r in &row {
            pairs_per_row[r] += 1;
        }
        let father_row: Vec<usize> = (0..n_rows).filter(|&r| pairs_per_row[r] > 0).collect();
        let count: Vec<i32> = father_row.iter().map(|&r| pairs_per_row[r]).collect();
        let e = |k: usize| {
            let sums = bincount(&row, c.e_m[k].iter().copied(), n_rows);
            father_row
                .iter()
                .map(|&r| sums[r] as f32)
                .collect::<Vec<f32>>()
        };
        let (e1, e2) = (e(0), e(1));
        let mut row_stratum = vec![0; n_rows];
        for (&r, &s) in row.iter().zip(&c.pairs.f_stratum) {
            row_stratum[r] = s;
        }
        let stratum: Vec<usize> = father_row.iter().map(|&r| row_stratum[r]).collect();
        let mut runs: Vec<usize> = (0..stratum.len())
            .filter(|&j| j == 0 || stratum[j] != stratum[j - 1])
            .collect();
        let run_stratum = runs.iter().map(|&j| stratum[j]).collect();
        runs.push(stratum.len());
        let n_strata = c.pairs.n_strata().1;
        let n_s = bincount(&stratum, count.iter().map(|&k| f64::from(k)), n_strata);
        let mut shift = vec![0.0; n_strata];
        if c.pairs.f_levels.is_none() {
            let total = bincount(
                &stratum,
                father_row
                    .iter()
                    .zip(&count)
                    .map(|(&r, &k)| f64::from(k) * f64::from(values[r])),
                n_strata,
            );
            for s in 0..n_strata {
                if n_s[s] > 0.0 {
                    shift[s] = total[s] / n_s[s];
                }
            }
        }
        let s1 = bincount(&stratum, e1.iter().map(|&v| f64::from(v)), n_strata);
        let s2 = bincount(&stratum, e2.iter().map(|&v| f64::from(v)), n_strata);
        Fathers {
            row: father_row.iter().map(|&r| r as u32).collect(),
            count,
            e1,
            e2,
            runs,
            run_stratum,
            sums: (0..n_strata)
                .map(|s| StratumSums {
                    shift: shift[s],
                    pairs: n_s[s],
                    e1: s1[s],
                    e2: s2[s],
                })
                .collect(),
            n_pairs: c.pair_father.len() as f64,
            levels: c.pairs.f_levels.clone(),
            trait_index: c.trait_index,
        }
    }

    /// Each form's score of cell `c` for one arrangement (`values[r]`: the
    /// father trait at row `r`), or why the arrangement has none.
    fn statistic(&self, c: usize, values: &[f32]) -> [Result<Score, Status>; 2] {
        let cell = &self.cells[c];
        match &cell.levels {
            None => continuous_cell(cell, values),
            Some(levels) => discrete_cell(cell, values, levels),
        }
    }

    /// The standardised statistic of the unpermuted fathers per cell and form.
    pub fn observed(&self) -> Vec<[f64; 2]> {
        (0..self.cells.len())
            .map(|c| self.standardised(c, self.statistic(c, &self.rows[self.cells[c].trait_index])))
            .collect()
    }

    fn standardised(&self, c: usize, stats: [Result<Score, Status>; 2]) -> [f64; 2] {
        [0, 1].map(|form| match stats[form] {
            Ok(score) => score.cross / (self.ss_m[c][form] * score.ss_f).sqrt(),
            Err(_) => f64::NAN,
        })
    }

    /// The null draws of every cell and form, run in batches until every
    /// tested form has stopped: per cell and form, the draws read.
    pub fn permuted(
        &self,
        observed: &[[f64; 2]],
        tested: &[[bool; 2]],
        permutations: u64,
        seed: i64,
    ) -> Vec<[Vec<NullDraw>; 2]> {
        let n_cells = self.cells.len();
        let mut stopping = Stopping::start(observed, tested);
        let mut out: Vec<[Vec<NullDraw>; 2]> =
            (0..n_cells).map(|_| [Vec::new(), Vec::new()]).collect();
        let mut active: Vec<usize> = stopping.active();
        let mut first = 0u64;
        while first < permutations && !active.is_empty() {
            let end = (2 * first)
                .max(FIRST_BATCH)
                .max(stopping.horizon())
                .min(permutations);
            let needed: Vec<usize> = {
                let mut t: Vec<usize> = active.iter().map(|&c| self.cells[c].trait_index).collect();
                t.sort_unstable();
                t.dedup();
                t
            };
            let n_rows = self.rows.first().map_or(0, Vec::len);
            let batch: Vec<Vec<[NullDraw; 2]>> = (first..end)
                .into_par_iter()
                .map_init(
                    || {
                        (
                            vec![0u32; n_rows],
                            vec![vec![0f32; n_rows]; self.rows.len()],
                        )
                    },
                    |(order, arranged), d| {
                        shuffle_index(order, &self.block_start, seed, d);
                        for &t in &needed {
                            for (r, &o) in order.iter().enumerate() {
                                arranged[t][r] = self.rows[t][o as usize];
                            }
                        }
                        active
                            .iter()
                            .map(|&c| {
                                let stats = self.statistic(c, &arranged[self.cells[c].trait_index]);
                                let values = self.standardised(c, stats);
                                [0, 1].map(|form| match stats[form] {
                                    Ok(_) => NullDraw::Value(values[form]),
                                    Err(status) => NullDraw::Failed(status),
                                })
                            })
                            .collect()
                    },
                )
                .collect();
            for (i, &c) in active.iter().enumerate() {
                for form in 0..2 {
                    out[c][form].extend(batch.iter().map(|draw| draw[i][form]));
                }
            }
            stopping.scan(&active, &batch, first);
            active = stopping.active();
            first = end;
        }
        for (forms, used) in out.iter_mut().zip(&stopping.used) {
            for (draws, &n) in forms.iter_mut().zip(used) {
                draws.truncate(n as usize);
            }
        }
        out
    }
}

/// One permutation draw's statistic, or why the permuted sample has none.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NullDraw {
    Value(f64),
    Failed(Status),
}

/// Whether `|t| >= |t_obs|` to `1e-12` relative.
pub(crate) fn extreme(t: f64, observed: f64) -> bool {
    t.abs() >= observed.abs() - 1e-12 * observed.abs().max(1.0)
}

/// The closed-scheme state of every cell and form.
struct Stopping {
    observed: Vec<[f64; 2]>,
    exceedances: Vec<[u64; 2]>,
    used: Vec<[u64; 2]>,
    done: Vec<[bool; 2]>,
}

impl Stopping {
    fn start(observed: &[[f64; 2]], tested: &[[bool; 2]]) -> Stopping {
        Stopping {
            observed: observed.to_vec(),
            exceedances: vec![[0; 2]; observed.len()],
            used: vec![[0; 2]; observed.len()],
            done: tested.iter().map(|t| [!t[0], !t[1]]).collect(),
        }
    }

    fn active(&self) -> Vec<usize> {
        (0..self.done.len())
            .filter(|&c| !(self.done[c][0] && self.done[c][1]))
            .collect()
    }

    /// The draw by which the nearest open form reaches `h` at its rate so far.
    fn horizon(&self) -> u64 {
        let mut best = f64::INFINITY;
        for c in 0..self.done.len() {
            for form in 0..2 {
                if !self.done[c][form] {
                    let rate = (SEQUENTIAL_H * self.used[c][form]) as f64
                        / self.exceedances[c][form].max(1) as f64;
                    best = best.min(rate.ceil());
                }
            }
        }
        if best.is_finite() {
            best as u64
        } else {
            0
        }
    }

    fn scan(&mut self, active: &[usize], batch: &[Vec<[NullDraw; 2]>], first: u64) {
        for (i, &c) in active.iter().enumerate() {
            for form in 0..2 {
                if self.done[c][form] {
                    continue;
                }
                let mut total = self.exceedances[c][form];
                let mut reached = None;
                for (d, draw) in batch.iter().enumerate() {
                    if let NullDraw::Value(v) = draw[i][form] {
                        if extreme(v, self.observed[c][form]) {
                            total += 1;
                            if total >= SEQUENTIAL_H && reached.is_none() {
                                reached = Some(d);
                            }
                        }
                    }
                }
                match reached {
                    Some(d) => {
                        self.used[c][form] = first + d as u64 + 1;
                        self.done[c][form] = true;
                    }
                    None => self.used[c][form] = first + batch.len() as u64,
                }
                self.exceedances[c][form] = total.min(SEQUENTIAL_H);
            }
        }
    }
}

/// Crude and stratified `Σ e_m e_f` with the fathers' values standardised on
/// the permuted pairs, in one pass over the cell's fathers.
fn continuous_cell(cell: &Fathers, values: &[f32]) -> [Result<Score, Status>; 2] {
    let n_strata = cell.sums.len();
    let mut dev = vec![0.0; n_strata];
    let mut sq = vec![0.0; n_strata];
    let mut cross1 = vec![0.0; n_strata];
    let mut cross2 = vec![0.0; n_strata];
    let mut lo = vec![f64::INFINITY; n_strata];
    let mut hi = vec![f64::NEG_INFINITY; n_strata];
    for r in 0..cell.run_stratum.len() {
        let s = cell.run_stratum[r];
        let shift = cell.sums[s].shift;
        let (mut d_sum, mut sq_sum, mut c1, mut c2) = (0.0, 0.0, 0.0, 0.0);
        let (mut v_lo, mut v_hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for j in cell.runs[r]..cell.runs[r + 1] {
            let v = f64::from(values[cell.row[j] as usize]);
            let d = v - shift;
            let count = f64::from(cell.count[j]);
            d_sum += count * d;
            sq_sum += count * d * d;
            c1 += f64::from(cell.e1[j]) * d;
            c2 += f64::from(cell.e2[j]) * d;
            v_lo = v_lo.min(v);
            v_hi = v_hi.max(v);
        }
        dev[s] += d_sum;
        sq[s] += sq_sum;
        cross1[s] += c1;
        cross2[s] += c2;
        lo[s] = lo[s].min(v_lo);
        hi[s] = hi[s].max(v_hi);
    }
    let lo_min = lo.iter().copied().fold(f64::INFINITY, f64::min);
    let hi_max = hi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lo_min == hi_max {
        return [Err(Status::ConstantMargin); 2];
    }
    let n = cell.n_pairs;
    let mut pooled_mean = 0.0;
    let mut degenerate = false;
    for s in 0..n_strata {
        let sums = cell.sums[s];
        if sums.pairs > 0.0 {
            pooled_mean += sums.shift * sums.pairs + dev[s];
            degenerate = degenerate || lo[s] == hi[s];
        }
    }
    pooled_mean /= n;
    let (mut ss_pooled, mut cross_pooled, mut cross_strat) = (0.0, 0.0, 0.0);
    for s in 0..n_strata {
        let StratumSums {
            shift,
            pairs,
            e1,
            e2,
        } = cell.sums[s];
        if pairs > 0.0 {
            let delta = shift - pooled_mean;
            ss_pooled += sq[s] + delta * (2.0 * dev[s] + delta * pairs);
            cross_pooled += cross1[s] + delta * e1;
            if !degenerate {
                let mean_dev = dev[s] / pairs;
                let ss_s = sq[s] - mean_dev * dev[s];
                cross_strat += (cross2[s] - mean_dev * e2) / (ss_s / pairs).sqrt();
            }
        }
    }
    let cross_crude = cross_pooled / (ss_pooled / n).sqrt();
    let crude = Ok(Score {
        cross: cross_crude,
        ss_f: n,
    });
    if degenerate {
        return [crude, Err(Status::DegenerateStratum)];
    }
    let strat = Score {
        cross: cross_strat,
        ss_f: n,
    };
    [crude, Ok(strat)]
}

/// Crude and stratified `Σ e_m e_f` with the fathers' thresholds refit on the
/// permuted pairs' margins.
fn discrete_cell(
    cell: &Fathers,
    values: &[f32],
    levels: &Grid<bool>,
) -> [Result<Score, Status>; 2] {
    let (n_strata, k) = (levels.rows, levels.cols);
    // Whole pair counts: integer sums equal pedsum's float sums exactly.
    let mut pair_counts = vec![0i64; n_strata * k];
    let mut score1 = Grid::filled(n_strata, k, 0.0);
    let mut score2 = Grid::filled(n_strata, k, 0.0);
    for r in 0..cell.run_stratum.len() {
        let s = cell.run_stratum[r];
        let (fathers, row) = (cell.runs[r]..cell.runs[r + 1], &cell.row);
        for j in fathers {
            let at = s * k + values[row[j] as usize] as usize;
            pair_counts[at] += i64::from(cell.count[j]);
            score1.data[at] += f64::from(cell.e1[j]);
            score2.data[at] += f64::from(cell.e2[j]);
        }
    }
    let counts = Grid {
        rows: n_strata,
        cols: k,
        data: pair_counts.iter().map(|&n| n as f64).collect(),
    };
    let mut pooled = Grid::filled(1, k, 0.0);
    let mut pooled_levels = Grid::filled(1, k, false);
    for s in 0..n_strata {
        for c in 0..k {
            *pooled.at_mut(0, c) += counts.at(s, c);
            let shown = pooled_levels.at(0, c) || levels.at(s, c);
            *pooled_levels.at_mut(0, c) = shown;
        }
    }
    let status_crude = margin_status(&pooled, &pooled_levels);
    let status_strat = margin_status(&counts, levels);
    let e_crude = latent_means(&pooled);
    let e_strat = latent_means(&counts);
    let (mut ss_crude, mut ss_strat, mut cross_crude, mut cross_strat) = (0.0, 0.0, 0.0, 0.0);
    for c in 0..k {
        ss_crude += pooled.at(0, c) * e_crude.at(0, c) * e_crude.at(0, c);
        let mut level_score = 0.0;
        for s in 0..n_strata {
            ss_strat += counts.at(s, c) * e_strat.at(s, c) * e_strat.at(s, c);
            level_score += score1.at(s, c);
            cross_strat += score2.at(s, c) * e_strat.at(s, c);
        }
        cross_crude += level_score * e_crude.at(0, c);
    }
    let form = |status: Option<Status>, cross, ss_f| match status {
        None => Ok(Score { cross, ss_f }),
        Some(status) => Err(status),
    };
    [
        form(status_crude, cross_crude, ss_crude),
        form(status_strat, cross_strat, ss_strat),
    ]
}
