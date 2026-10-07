//! Assortative mating: the Mate Correlation of one or two traits over a
//! pedigree's Mating Pairs, ported from pedsum #13 at
//! 142adf300d5b0802f59af03f3f5211af910fc852 (plan `pg-phenotype-assortative-v2`).
//!
//! A Mating Pair is a mother and father with a child in the pedigree; a Mate
//! Network is a connected set of pairs through shared mates.  Each cell
//! (mother trait x father trait) gets crude estimators on its pooled pairs,
//! and with strata the primary estimator standardised within each sex x
//! stratum; inference is a Mate Network cluster-robust sandwich, an optional
//! one-step Mate Network bootstrap, and a father-permutation test of each
//! primary estimate.

// `!(x > 0.0)` is meant: NaN fails each of these checks, as `not x > 0` does in pedsum.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod bootstrap;
mod bvn;
mod cephes;
mod estimators;
mod fit;
mod input;
mod kernels;
mod permutation;
mod result;
mod rng;
mod sample;
mod sandwich;

#[cfg(test)]
mod primitives_parity;

use crate::error::Error;
use crate::input::{PedigreeInput, Trait};
use bootstrap::Draw;
use input::{check_strata, check_traits, Kind};
use permutation::{NullDraw, Packed, SEQUENTIAL_H};
use sample::{
    drop_thin_strata, mate_networks, mating_pairs, network_summary, stratum_codes, CellPairs, Kept,
};

pub use input::{Settings, Strata};
pub use result::{
    Cell, Ci, CiMethod, CiScale, Draws, Dropped, Estimate, Estimator, EstimatorResult,
    MateCorrelation, Method, Permutation, PermutationStatistic, Point, Reason, Sample,
    SettingsEcho, Stratified, WithinPerson, WithinPersonPair, METHOD,
};

/// The confidence level of every interval.
pub(crate) const CI_LEVEL: f64 = 0.95;

/// A cell kind's estimators: the crude ones, primary first, and the
/// stratified form of the primary (pedsum `CELL_ESTIMATORS`).
fn cell_estimators(m: Kind, f: Kind) -> (&'static [Estimator], Estimator) {
    use Estimator::*;
    match (m, f) {
        (Kind::Continuous, Kind::Continuous) => (&[Pearson, Spearman], Pearson),
        (Kind::Binary, Kind::Binary) => (&[Tetrachoric, OddsRatio, Phi], Tetrachoric),
        (Kind::Continuous, Kind::Binary) | (Kind::Binary, Kind::Continuous) => {
            (&[Biserial, PointBiserial], Biserial)
        }
        (Kind::Continuous, Kind::Ordinal { .. }) | (Kind::Ordinal { .. }, Kind::Continuous) => {
            (&[Polyserial], Polyserial)
        }
        _ => (&[Polychoric], Polychoric),
    }
}

/// The Mate Correlation of `traits` (one or two, on the pedigree's rows) over
/// the pedigree's Mating Pairs, optionally stratified, on the current pool.
///
/// # Errors
///
/// A settings [`Error::ParameterOutOfRange`], then any pedigree-graph-core
/// validation error, then the trait and stratum errors of the input boundary
/// (`trait_count`, `trait_length_mismatch`, `unsupported_trait_kind`,
/// `invalid_trait_value`, `all_missing_trait`, `constant_trait`,
/// `sparse_ordinal_codes`, `unused_level`, `stratum_length_mismatch`).
pub fn mate_correlation(
    pedigree: PedigreeInput<'_>,
    traits: &[Trait<'_>],
    strata: Option<Strata<'_>>,
    settings: Settings,
) -> Result<MateCorrelation, Error> {
    let settings = settings.check()?;
    let graph = pedigree.validate()?;
    let n_rows = graph.len();
    let columns = check_traits(traits, n_rows)?;
    if let Some(s) = strata {
        check_strata(s, n_rows)?;
    }
    let stratified = strata.is_some();
    let ids = &graph.ids;
    let (all_mothers, all_fathers) = mating_pairs(ids, &graph.mother_rows, &graph.father_rows);
    let codes = stratum_codes(strata, n_rows);
    let known: Vec<usize> = (0..all_mothers.len())
        .filter(|&p| codes[all_mothers[p]].is_some() && codes[all_fathers[p]].is_some())
        .collect();
    let mothers: Vec<usize> = known.iter().map(|&p| all_mothers[p]).collect();
    let fathers: Vec<usize> = known.iter().map(|&p| all_fathers[p]).collect();
    let code = |row: usize| codes[row].unwrap_or(0);

    let mut fathers_by_id: Vec<usize> = fathers.clone();
    fathers_by_id.sort_unstable_by_key(|&r| ids[r]);
    fathers_by_id.dedup();
    let pair_father: Vec<usize> = fathers
        .iter()
        .map(|&r| fathers_by_id.partition_point(|&o| ids[o] < ids[r]))
        .collect();
    let present: Vec<Vec<bool>> = columns
        .iter()
        .map(|c| {
            fathers_by_id
                .iter()
                .map(|&r| !c.values[r].is_nan())
                .collect()
        })
        .collect();
    let blocks = permutation::father_blocks(
        &fathers_by_id.iter().map(|&r| code(r)).collect::<Vec<_>>(),
        &present,
    );
    let mut block_size = vec![0usize; blocks.iter().max().map_or(0, |&b| b + 1)];
    for &b in &blocks {
        block_size[b] += 1;
    }
    let fixed: Vec<bool> = blocks.iter().map(|&b| block_size[b] == 1).collect();

    let labels = mate_networks(&mothers, &fathers);
    let (n_mate_networks, largest_mate_network_share) = network_summary(&labels);
    let multiple = |rows: &[usize], n: usize| {
        let mut count = vec![0u32; n];
        for &r in rows {
            count[r] += 1;
        }
        count.iter().filter(|&&c| c > 1).count() as u64
    };
    let sample = Sample {
        n_total: all_mothers.len() as u64,
        n_dropped_unknown_stratum: (all_mothers.len() - known.len()) as u64,
        n_mate_networks,
        largest_mate_network_share,
        n_mothers_multiple_mates: multiple(&mothers, n_rows),
        n_fathers_multiple_mates: multiple(&pair_father, fathers_by_id.len()),
    };

    let mut cells = Vec::with_capacity(columns.len() * columns.len());
    let mut fitted = Vec::with_capacity(cells.capacity());
    for (mt, mother_trait) in columns.iter().enumerate() {
        for (ft, father_trait) in columns.iter().enumerate() {
            let m_all: Vec<f64> = mothers.iter().map(|&r| mother_trait.values[r]).collect();
            let f_all: Vec<f64> = fathers.iter().map(|&r| father_trait.values[r]).collect();
            let mut dropped = Dropped::default();
            let mut complete = Vec::new();
            for p in 0..mothers.len() {
                match (m_all[p].is_nan(), f_all[p].is_nan()) {
                    (false, false) => complete.push(p),
                    (true, false) => dropped.mother_missing += 1,
                    (false, true) => dropped.father_missing += 1,
                    (true, true) => dropped.both_missing += 1,
                }
            }
            let mut pairs = CellPairs {
                m: complete.iter().map(|&p| m_all[p]).collect(),
                f: complete.iter().map(|&p| f_all[p]).collect(),
                m_stratum: complete.iter().map(|&p| code(mothers[p])).collect(),
                f_stratum: complete.iter().map(|&p| code(fathers[p])).collect(),
                m_levels: None,
                f_levels: None,
            };
            let cell_mothers: Vec<usize> = complete.iter().map(|&p| mothers[p]).collect();
            let cell_fathers: Vec<usize> = complete.iter().map(|&p| fathers[p]).collect();
            let n_complete = complete.len() as u64;
            let cell_labels = if stratified {
                let Kept {
                    keep,
                    n_small,
                    labels: cell_labels,
                } = drop_thin_strata(
                    &pairs,
                    &cell_mothers,
                    &cell_fathers,
                    settings.min_stratum_networks,
                );
                complete = keep.iter().map(|&i| complete[i]).collect();
                pairs = pairs.take(&keep);
                dropped.small_stratum = n_small;
                dropped.degenerate_stratum = n_complete - complete.len() as u64 - n_small;
                cell_labels
            } else {
                mate_networks(&cell_mothers, &cell_fathers)
            };
            let pairs = pairs.with_levels(mother_trait.kind.levels(), father_trait.kind.levels());
            let (crude, strat) = cell_estimators(mother_trait.kind, father_trait.kind);
            let estimators: Vec<(Estimator, bool)> = crude
                .iter()
                .map(|&e| (e, false))
                .chain(stratified.then_some((strat, true)))
                .collect();
            let results = fit_cell(
                &estimators,
                &pairs,
                &cell_labels,
                settings.bootstrap,
                settings.seed,
            );
            let (n_mate_networks, largest_mate_network_share) = network_summary(&cell_labels);
            let n_strata = |code: &[usize]| {
                let mut distinct = code.to_vec();
                distinct.sort_unstable();
                distinct.dedup();
                distinct.len() as u64
            };
            let mut results = results.into_iter();
            let crude_results: Vec<EstimatorResult> = results.by_ref().take(crude.len()).collect();
            let stratified_result = results.next().map(|result| Stratified {
                result,
                n_strata_mothers: n_strata(&pairs.m_stratum),
                n_strata_fathers: n_strata(&pairs.f_stratum),
            });
            let table = (crude[0] == Estimator::Tetrachoric).then(|| {
                let t = estimators::two_by_two(&pairs.pooled(), &vec![1.0; pairs.len()]);
                t.map(|row| row.map(|v| v as u64))
            });
            let cell_pair_father: Vec<usize> = complete.iter().map(|&p| pair_father[p]).collect();
            cells.push(Cell {
                mother_trait: mt,
                father_trait: ft,
                n: pairs.len() as u64,
                n_dropped: dropped,
                n_mate_networks,
                largest_mate_network_share,
                table,
                crude: crude_results,
                stratified: stratified_result,
            });
            fitted.push((pairs, cell_pair_father, ft));
        }
    }

    let statistic_of = |cell: &Cell| {
        if cell.crude[0].estimator == Estimator::Pearson {
            PermutationStatistic::Pearson
        } else {
            PermutationStatistic::ScoreAtZero
        }
    };
    let tested: Vec<[bool; 2]> = cells
        .iter()
        .map(|cell| {
            [
                cell.crude[0].outcome.is_ok(),
                cell.stratified
                    .as_ref()
                    .is_some_and(|s| s.result.outcome.is_ok()),
            ]
        })
        .collect();
    let informative: Vec<usize> = (0..cells.len())
        .filter(|&c| (tested[c][0] || tested[c][1]) && !fitted[c].1.iter().all(|&f| fixed[f]))
        .collect();
    let mut null: Vec<[Vec<NullDraw>; 2]> =
        (0..cells.len()).map(|_| [Vec::new(), Vec::new()]).collect();
    let mut observed_stat = vec![[f64::NAN; 2]; cells.len()];
    if !informative.is_empty() && settings.permutations > 0 {
        let perm_cells: Vec<permutation::Cell<'_>> = informative
            .iter()
            .map(|&c| {
                let (pairs, pair_father, ft) = &fitted[c];
                let k_m = columns[cells[c].mother_trait].kind.levels();
                let scores = |stratum: &[usize]| {
                    permutation::latent_scores(&pairs.m, stratum, k_m)
                        .unwrap_or_else(|| vec![0.0; pairs.len()])
                };
                permutation::Cell {
                    pairs,
                    trait_index: *ft,
                    pair_father,
                    e_m: [scores(&vec![0; pairs.len()]), scores(&pairs.m_stratum)],
                    tested: tested[c],
                }
            })
            .collect();
        let trait_values: Vec<Vec<f64>> = columns
            .iter()
            .map(|c| fathers_by_id.iter().map(|&r| c.values[r]).collect())
            .collect();
        let packed = Packed::new(&perm_cells, &trait_values, &blocks);
        let observed = packed.observed();
        let tests: Vec<[bool; 2]> = perm_cells.iter().map(|c| c.tested).collect();
        let draws = packed.permuted(&observed, &tests, settings.permutations, settings.seed);
        for ((&c, stat), cell_draws) in informative.iter().zip(observed).zip(draws) {
            observed_stat[c] = stat;
            null[c] = cell_draws;
        }
    }
    for (c, cell) in cells.iter_mut().enumerate() {
        let mut fathers: Vec<usize> = fitted[c].1.clone();
        fathers.sort_unstable();
        fathers.dedup();
        let n_fixed = fathers.iter().filter(|&&f| fixed[f]).count() as u64;
        let statistic = statistic_of(cell);
        let forms = [
            Some(&mut cell.crude[0]),
            cell.stratified.as_mut().map(|s| &mut s.result),
        ];
        for (form, result) in forms.into_iter().enumerate() {
            if let Some(EstimatorResult {
                outcome: Ok(estimate),
                ..
            }) = result
            {
                estimate.permutation = Some(permutation_record(
                    observed_stat[c][form],
                    &null[c][form],
                    settings.permutations,
                    settings.seed,
                    n_fixed,
                    statistic,
                ));
            }
        }
    }

    let within_person = (columns.len() == 2).then(|| {
        let one = |rows: &[usize]| within_person(&columns, ids, rows);
        WithinPersonPair {
            mothers: one(&mothers),
            fathers: one(&fathers),
        }
    });
    Ok(MateCorrelation {
        sample,
        cells,
        within_person,
        settings: SettingsEcho {
            permutations: settings.permutations,
            bootstrap: settings.bootstrap,
            seed: settings.seed,
            threads: rayon::current_num_threads(),
            ci_level: CI_LEVEL,
            min_stratum_networks: stratified.then_some(settings.min_stratum_networks),
        },
        method: METHOD,
    })
}

/// Observed estimates of one cell, and their records with SE, CI and draws.
fn fit_cell(
    estimators: &[(Estimator, bool)],
    pairs: &CellPairs,
    labels: &[usize],
    bootstrap: u64,
    seed: i64,
) -> Vec<EstimatorResult> {
    let ones = vec![1.0; pairs.len()];
    let observed: Vec<_> = estimators
        .iter()
        .map(|&(e, strat)| estimators::point(e, strat, pairs, &ones, None))
        .collect();
    let starts: Vec<Option<f64>> = observed
        .iter()
        .map(|o| o.as_ref().ok().map(|p| p.value))
        .collect();
    let n_networks = labels.iter().max().map_or(0, |&l| l + 1) as u64;
    let draws: Vec<Vec<Draw>> = if n_networks >= 2 && bootstrap > 0 {
        bootstrap::draws(estimators, &starts, pairs, labels, bootstrap, seed)
    } else {
        vec![Vec::new(); estimators.len()]
    };
    estimators
        .iter()
        .zip(&observed)
        .zip(&draws)
        .enumerate()
        .map(|(k, ((&(est, strat), obs), draws))| EstimatorResult {
            estimator: est,
            primary: k == 0 || strat,
            outcome: match obs {
                Err(status) => Err(status.reason()),
                Ok(point) => {
                    let se = sandwich::sandwich_se(est, strat, pairs, *point, labels);
                    let (counts, ci) = if bootstrap > 0 {
                        let (counts, ci) = bootstrap::record(draws, bootstrap, n_networks);
                        (
                            Some(counts),
                            ci.map(|bounds| Ci {
                                bounds,
                                method: CiMethod::Bootstrap,
                            }),
                        )
                    } else {
                        let ci = se.map(|se| Ci {
                            bounds: sandwich::wald_ci(point.value, se, est.ci_scale()),
                            method: CiMethod::Sandwich,
                        });
                        (None, ci)
                    };
                    Ok(Estimate {
                        point: *point,
                        se,
                        ci,
                        bootstrap: counts,
                        permutation: None,
                    })
                }
            },
        })
        .collect()
}

/// Two-sided sequential Monte Carlo p-value by magnitude (Besag & Clifford
/// 1991, closed scheme): stop at the `h`-th valid draw at least as extreme.
fn permutation_record(
    observed: f64,
    draws: &[NullDraw],
    requested: u64,
    seed: i64,
    n_fixed_fathers: u64,
    statistic: PermutationStatistic,
) -> Permutation {
    let h = SEQUENTIAL_H;
    let mut hits = 0;
    let mut used = draws.len();
    for (d, draw) in draws.iter().enumerate() {
        if let NullDraw::Value(v) = draw {
            if permutation::extreme(*v, observed) {
                hits += 1;
                if hits == h {
                    used = d + 1;
                    break;
                }
            }
        }
    }
    let read = &draws[..used];
    let valid: Vec<f64> = read
        .iter()
        .filter_map(|d| match d {
            NullDraw::Value(v) => Some(*v),
            NullDraw::Failed(_) => None,
        })
        .collect();
    let failures: Vec<Draw> = read
        .iter()
        .map(|d| match d {
            NullDraw::Value(v) => Draw::Value(*v),
            NullDraw::Failed(s) => Draw::Failed(*s),
        })
        .collect();
    let p = if requested == 0 {
        Err(Reason::NotRequested)
    } else if read.is_empty() {
        Err(Reason::NoInformativePermutations)
    } else if valid.is_empty() {
        Err(Reason::NoValidPermutations)
    } else {
        let g = valid
            .iter()
            .filter(|&&v| permutation::extreme(v, observed))
            .count() as u64;
        let b = valid.len() as f64;
        Ok(if g >= h {
            h as f64 / b
        } else {
            (g as f64 + 1.0) / (b + 1.0)
        })
    };
    Permutation {
        statistic,
        p,
        draws: Draws {
            requested,
            valid: valid.len() as u64,
            failed: (read.len() - valid.len()) as u64,
            failure_reasons: bootstrap::failure_reasons(&failures),
        },
        seed,
        n_fixed_fathers,
        stopped_early: (used as u64) < requested,
        draws_used: used as u64,
        sequential_h: h,
    }
}

/// The Within-Person Cross-Trait Correlation of the distinct people in
/// `pair_rows` with both traits.
fn within_person(columns: &[input::Column<'_>], ids: &[i64], pair_rows: &[usize]) -> WithinPerson {
    let mut rows = pair_rows.to_vec();
    rows.sort_unstable();
    rows.dedup();
    rows.sort_by_key(|&r| ids[r]);
    let (first, second) = (&columns[0], &columns[1]);
    let both: Vec<usize> = rows
        .into_iter()
        .filter(|&r| !first.values[r].is_nan() && !second.values[r].is_nan())
        .collect();
    let n = both.len();
    let people = CellPairs {
        m: both.iter().map(|&r| first.values[r]).collect(),
        f: both.iter().map(|&r| second.values[r]).collect(),
        m_stratum: vec![0; n],
        f_stratum: vec![0; n],
        m_levels: None,
        f_levels: None,
    }
    .with_levels(first.kind.levels(), second.kind.levels());
    let estimator = cell_estimators(first.kind, second.kind).0[0];
    WithinPerson {
        estimator,
        n: n as u64,
        outcome: estimators::point(estimator, false, &people, &vec![1.0; n], None)
            .map_err(kernels::Status::reason),
    }
}
