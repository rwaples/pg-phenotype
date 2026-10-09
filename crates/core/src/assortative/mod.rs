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
mod value;

#[cfg(test)]
mod primitives_parity;

use crate::error::Error;
use crate::input::Trait;
use crate::pedigree::{Pedigree, PedigreeArg};
use bootstrap::Draw;
use input::{check_strata, check_traits, Kind};
use permutation::{NullDraw, Packed, SEQUENTIAL_H};
use sample::{distinct_rows, drop_thin_strata, network_summary, stratum_codes, CellPairs, Kept};
use std::borrow::Cow;

pub use input::{Settings, Strata};
pub use result::{
    Cell, Ci, CiMethod, CiScale, Draws, Dropped, Estimate, Estimator, EstimatorResult,
    MateCorrelation, Method, Permutation, PermutationStatistic, Point, Reason, Sample,
    SettingsEcho, Stratified, WithinPerson, WithinPersonPair, METHOD,
};
pub(crate) use sample::{mate_networks, mating_pairs, MatingPairs};

/// The confidence level of every interval.
pub(crate) const CI_LEVEL: f64 = 0.95;

/// The two-sided normal quantile of [`CI_LEVEL`] for Wald intervals: SciPy's
/// cephes `ndtri((1 + 0.95) / 2)`, as pedsum computes it, one ulp above the
/// kernels' AS 241.  A literal for parity; `ci_level_and_wald_z_agree` ties
/// it to `CI_LEVEL`.
pub(crate) const WALD_Z: f64 = 1.959_963_984_540_054;

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
/// A built `pedigree` keeps its Mating Pairs and their Mate Networks for
/// later calls; columns are validated for this call alone.
///
/// # Errors
///
/// A settings [`Error::ParameterOutOfRange`], then any pedigree-graph-core
/// validation error of columns, then the trait and stratum errors of the input boundary
/// (`trait_count`, `trait_length_mismatch`, `unsupported_trait_kind`,
/// `invalid_trait_value`, `all_missing_trait`, `constant_trait`,
/// `sparse_ordinal_codes`, `unused_level`, `stratum_length_mismatch`).
pub fn mate_correlation(
    pedigree: PedigreeArg<'_>,
    traits: &[Trait<'_>],
    strata: Option<Strata<'_>>,
    settings: Settings,
) -> Result<MateCorrelation, Error> {
    let settings = settings.check()?;
    let pedigree = pedigree.resolve()?;
    let n_rows = pedigree.len();
    let columns = check_traits(traits, n_rows)?;
    if let Some(s) = strata {
        check_strata(s, n_rows)?;
    }
    let stratified = strata.is_some();
    let ids = pedigree.ids();
    let pairs = Pairs::build(&pedigree, strata);
    let fathers = DistinctFathers::build(&pairs, ids, &columns);
    let sample = pairs.sample(n_rows);

    let mut cells = Vec::with_capacity(columns.len() * columns.len());
    let mut fitted = Vec::with_capacity(cells.capacity());
    for mt in 0..columns.len() {
        for ft in 0..columns.len() {
            let (cell, fit) = analyse_cell(&pairs, &fathers, &columns, [mt, ft], settings);
            cells.push(cell);
            fitted.push(fit);
        }
    }
    let nulls = permutation_nulls(&cells, &fitted, &columns, &fathers, settings);
    attach_permutations(&mut cells, &fitted, &nulls, settings);

    let within_person = (columns.len() == 2).then(|| {
        let one = |rows: &[usize]| within_person(&columns, ids, rows);
        WithinPersonPair {
            mothers: one(&pairs.mothers),
            fathers: one(&pairs.fathers),
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
            spearman: settings.spearman,
        },
        method: METHOD,
    })
}

/// The Mating Pairs whose parents both have a known stratum: the
/// Pedigree's own pairs and networks when no pair is dropped.
struct Pairs<'p> {
    /// Every Mating Pair, before the unknown-stratum drop.
    n_total: usize,
    mothers: Cow<'p, [usize]>,
    fathers: Cow<'p, [usize]>,
    /// Each row's stratum code, `None` where unknown; without strata every
    /// row is code 0.
    codes: Option<Vec<Option<usize>>>,
    /// Each pair's Mate Network.
    labels: Cow<'p, [usize]>,
}

impl<'p> Pairs<'p> {
    fn build(pedigree: &'p Pedigree, strata: Option<Strata<'_>>) -> Pairs<'p> {
        let all = pedigree.mating_pairs();
        let codes = strata.map(stratum_codes);
        let all_pairs = || all.mothers.iter().zip(&all.fathers);
        let known =
            |c: &[Option<usize>], (&m, &f): (&usize, &usize)| c[m].is_some() && c[f].is_some();
        let dropping = codes
            .as_deref()
            .filter(|c| !all_pairs().all(|pair| known(c, pair)));
        let (mothers, fathers, labels) = match dropping {
            Some(c) => {
                let (mothers, fathers): (Vec<usize>, Vec<usize>) =
                    all_pairs().filter(|&pair| known(c, pair)).unzip();
                let labels = mate_networks(&mothers, &fathers);
                (Cow::Owned(mothers), Cow::Owned(fathers), Cow::Owned(labels))
            }
            None => (
                Cow::Borrowed(all.mothers.as_slice()),
                Cow::Borrowed(all.fathers.as_slice()),
                Cow::Borrowed(pedigree.networks()),
            ),
        };
        Pairs {
            n_total: all.mothers.len(),
            mothers,
            fathers,
            codes,
            labels,
        }
    }

    fn stratified(&self) -> bool {
        self.codes.is_some()
    }

    /// Row `row`'s stratum code, 0 where unknown (no kept pair has one).
    fn code(&self, row: usize) -> usize {
        self.codes.as_ref().map_or(0, |c| c[row].unwrap_or(0))
    }

    fn sample(&self, n_rows: usize) -> Sample {
        let (n_mate_networks, largest_mate_network_share) = network_summary(&self.labels);
        let multiple = |rows: &[usize], n: usize| {
            let mut count = vec![0u32; n];
            for &r in rows {
                count[r] += 1;
            }
            count.iter().filter(|&&c| c > 1).count() as u64
        };
        Sample {
            n_total: self.n_total as u64,
            n_dropped_unknown_stratum: (self.n_total - self.mothers.len()) as u64,
            n_mate_networks,
            largest_mate_network_share,
            n_mothers_multiple_mates: multiple(&self.mothers, n_rows),
            n_fathers_multiple_mates: multiple(&self.fathers, n_rows),
        }
    }
}

/// The distinct fathers of the kept pairs, in father-id order, with their
/// permutation blocks.
struct DistinctFathers {
    rows: Vec<usize>,
    /// Each kept pair's father, as an index into `rows`.
    of_pair: Vec<usize>,
    /// Each father's permutation block: stratum x which traits he has.
    blocks: Vec<usize>,
    /// Whether a father is alone in his block, so never exchanged.
    fixed: Vec<bool>,
}

impl DistinctFathers {
    fn build(pairs: &Pairs<'_>, ids: &[i64], columns: &[input::Column<'_>]) -> DistinctFathers {
        let rows = distinct_rows(&pairs.fathers, ids);
        let mut index = vec![0; ids.len()];
        for (i, &r) in rows.iter().enumerate() {
            index[r] = i;
        }
        let of_pair = pairs.fathers.iter().map(|&r| index[r]).collect();
        let present: Vec<Vec<bool>> = columns
            .iter()
            .map(|c| rows.iter().map(|&r| !c.values[r].is_nan()).collect())
            .collect();
        let blocks = permutation::father_blocks(
            &rows.iter().map(|&r| pairs.code(r)).collect::<Vec<_>>(),
            &present,
        );
        let mut block_size = vec![0usize; sample::n_codes(&blocks)];
        for &b in &blocks {
            block_size[b] += 1;
        }
        let fixed = blocks.iter().map(|&b| block_size[b] == 1).collect();
        DistinctFathers {
            rows,
            of_pair,
            blocks,
            fixed,
        }
    }
}

/// What the permutation test needs of a fitted cell.
struct CellFit {
    /// The cell's distinct fathers alone in their permutation block.
    n_fixed_fathers: u64,
    /// Whether some pair's father can be exchanged.
    exchangeable: bool,
    /// The test's input, kept only when permutations are requested.
    test: Option<CellTest>,
}

/// A cell's pairs as the permutation test reads them.
struct CellTest {
    pairs: CellPairs,
    /// Each of the cell's pairs' father, as an index into the distinct fathers.
    pair_father: Vec<usize>,
    father_trait: usize,
}

/// One cell (mother trait `mt` x father trait `ft`): its complete pairs, the
/// thin-stratum drop, and every estimator with its SE and CI.
fn analyse_cell(
    pairs: &Pairs<'_>,
    fathers: &DistinctFathers,
    columns: &[input::Column<'_>],
    [mt, ft]: [usize; 2],
    settings: input::Checked,
) -> (Cell, CellFit) {
    let stratified = pairs.stratified();
    let (mother_trait, father_trait) = (&columns[mt], &columns[ft]);
    let mut dropped = Dropped::default();
    let n_pairs = pairs.mothers.len();
    let mut complete = Vec::with_capacity(n_pairs);
    let (mut m, mut f) = (Vec::with_capacity(n_pairs), Vec::with_capacity(n_pairs));
    for (p, (&mr, &fr)) in pairs.mothers.iter().zip(pairs.fathers.iter()).enumerate() {
        let (x, y) = (mother_trait.values[mr], father_trait.values[fr]);
        match (x.is_nan(), y.is_nan()) {
            (false, false) => {
                complete.push(p);
                m.push(x);
                f.push(y);
            }
            (true, false) => dropped.mother_missing += 1,
            (false, true) => dropped.father_missing += 1,
            (true, true) => dropped.both_missing += 1,
        }
    }
    let stratum_of = |rows: &[usize]| -> Vec<usize> {
        if stratified {
            complete.iter().map(|&p| pairs.code(rows[p])).collect()
        } else {
            vec![0; complete.len()]
        }
    };
    let mut cell_pairs = CellPairs {
        m,
        f,
        m_stratum: stratum_of(&pairs.mothers),
        f_stratum: stratum_of(&pairs.fathers),
        m_levels: None,
        f_levels: None,
    };
    let cell_mothers = || -> Vec<usize> { complete.iter().map(|&p| pairs.mothers[p]).collect() };
    let cell_fathers = || -> Vec<usize> { complete.iter().map(|&p| pairs.fathers[p]).collect() };
    let n_complete = complete.len() as u64;
    let labels: Cow<'_, [usize]> = if stratified {
        let Kept {
            keep,
            n_small,
            labels,
        } = drop_thin_strata(
            &cell_pairs,
            &cell_mothers(),
            &cell_fathers(),
            settings.min_stratum_networks,
        );
        complete = keep.iter().map(|&i| complete[i]).collect();
        cell_pairs = cell_pairs.take(&keep);
        dropped.small_stratum = n_small;
        dropped.degenerate_stratum = n_complete - complete.len() as u64 - n_small;
        Cow::Owned(labels)
    } else if complete.len() == pairs.mothers.len() {
        Cow::Borrowed(&pairs.labels)
    } else {
        Cow::Owned(mate_networks(&cell_mothers(), &cell_fathers()))
    };
    let cell_pairs = cell_pairs.with_levels(mother_trait.kind.levels(), father_trait.kind.levels());
    let (all_crude, strat) = cell_estimators(mother_trait.kind, father_trait.kind);
    let crude: Vec<Estimator> = all_crude
        .iter()
        .copied()
        .filter(|&e| e != Estimator::Spearman || settings.spearman)
        .collect();
    let estimators: Vec<(Estimator, bool)> = crude
        .iter()
        .map(|&e| (e, false))
        .chain(stratified.then_some((strat, true)))
        .collect();
    let results = fit_cell(
        &estimators,
        &cell_pairs,
        &labels,
        settings.bootstrap,
        settings.seed,
    );
    let (n_mate_networks, largest_mate_network_share) = network_summary(&labels);
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
        n_strata_mothers: n_strata(&cell_pairs.m_stratum),
        n_strata_fathers: n_strata(&cell_pairs.f_stratum),
    });
    let table = (crude[0] == Estimator::Tetrachoric).then(|| {
        let t = estimators::two_by_two(&cell_pairs.pooled(), &vec![1.0; cell_pairs.len()]);
        t.map(|row| row.map(|v| v as u64))
    });
    let cell = Cell {
        mother_trait: mt,
        father_trait: ft,
        n: cell_pairs.len() as u64,
        n_dropped: dropped,
        n_mate_networks,
        largest_mate_network_share,
        table,
        crude: crude_results,
        stratified: stratified_result,
    };
    let pair_father: Vec<usize> = complete.iter().map(|&p| fathers.of_pair[p]).collect();
    let mut seen = vec![false; fathers.rows.len()];
    let fit = CellFit {
        n_fixed_fathers: pair_father
            .iter()
            .filter(|&&f| fathers.fixed[f] && !std::mem::replace(&mut seen[f], true))
            .count() as u64,
        exchangeable: pair_father.iter().any(|&f| !fathers.fixed[f]),
        test: (settings.permutations > 0).then_some(CellTest {
            pairs: cell_pairs,
            pair_father,
            father_trait: ft,
        }),
    };
    (cell, fit)
}

/// Per cell, the observed permutation statistic and the null draws read of
/// each form (crude, stratified); empty where a cell is not tested.
struct Nulls {
    observed: Vec<[f64; 2]>,
    draws: Vec<[Vec<NullDraw>; 2]>,
}

/// Run the father-permutation test of every cell with a defined primary
/// estimate and an exchangeable father.
fn permutation_nulls(
    cells: &[Cell],
    fitted: &[CellFit],
    columns: &[input::Column<'_>],
    fathers: &DistinctFathers,
    settings: input::Checked,
) -> Nulls {
    let mut nulls = Nulls {
        observed: vec![[f64::NAN; 2]; cells.len()],
        draws: (0..cells.len()).map(|_| [Vec::new(), Vec::new()]).collect(),
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
    let informative: Vec<(usize, &CellTest)> = fitted
        .iter()
        .enumerate()
        .filter(|&(c, fit)| (tested[c][0] || tested[c][1]) && fit.exchangeable)
        .filter_map(|(c, fit)| Some((c, fit.test.as_ref()?)))
        .collect();
    if informative.is_empty() {
        return nulls;
    }
    let perm_cells: Vec<permutation::Cell<'_>> = informative
        .iter()
        .map(|&(c, fit)| {
            let pairs = &fit.pairs;
            let k_m = columns[cells[c].mother_trait].kind.levels();
            let scores = |stratum: &[usize]| {
                permutation::latent_scores(&pairs.m, stratum, k_m)
                    .unwrap_or_else(|| vec![0.0; pairs.len()])
            };
            permutation::Cell {
                pairs,
                trait_index: fit.father_trait,
                pair_father: &fit.pair_father,
                e_m: [scores(&vec![0; pairs.len()]), scores(&pairs.m_stratum)],
                tested: tested[c],
            }
        })
        .collect();
    let trait_values: Vec<Vec<f64>> = columns
        .iter()
        .map(|c| fathers.rows.iter().map(|&r| c.values[r]).collect())
        .collect();
    let packed = Packed::new(&perm_cells, &trait_values, &fathers.blocks);
    let observed = packed.observed();
    let tests: Vec<[bool; 2]> = perm_cells.iter().map(|c| c.tested).collect();
    let draws = packed.permuted(&observed, &tests, settings.permutations, settings.seed);
    for ((&(c, _), stat), cell_draws) in informative.iter().zip(observed).zip(draws) {
        nulls.observed[c] = stat;
        nulls.draws[c] = cell_draws;
    }
    nulls
}

/// Give every defined primary estimate its permutation record.
fn attach_permutations(
    cells: &mut [Cell],
    fitted: &[CellFit],
    nulls: &Nulls,
    settings: input::Checked,
) {
    for (c, cell) in cells.iter_mut().enumerate() {
        let statistic = if cell.crude[0].estimator == Estimator::Pearson {
            PermutationStatistic::Pearson
        } else {
            PermutationStatistic::ScoreAtZero
        };
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
                    nulls.observed[c][form],
                    &nulls.draws[c][form],
                    settings.permutations,
                    settings.seed,
                    fitted[c].n_fixed_fathers,
                    statistic,
                ));
            }
        }
    }
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
/// 1991, closed scheme) from the draws [`Packed::permuted`] read: they end at
/// the `h`-th valid draw at least as extreme, or at the last one requested.
fn permutation_record(
    observed: f64,
    read: &[NullDraw],
    requested: u64,
    seed: i64,
    n_fixed_fathers: u64,
    statistic: PermutationStatistic,
) -> Permutation {
    let h = SEQUENTIAL_H;
    let used = read.len();
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
    let (first, second) = (&columns[0], &columns[1]);
    let both: Vec<usize> = distinct_rows(pair_rows, ids)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_level_and_wald_z_agree() {
        let z = kernels::ndtri(0.5 + CI_LEVEL / 2.0);
        assert!(
            (z - WALD_Z).abs() <= 2.0 * f64::EPSILON * WALD_Z,
            "{z} vs {WALD_Z}"
        );
    }
}
