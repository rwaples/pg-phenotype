//! Each cell's entries and every estimate's influence on them (issue #25):
//! what the cluster sandwich SE (ADR 0013) sums, one entry at a time, so a
//! caller can form the meat itself.
//!
//! The SE's pass folds each entry into per-key sums and keeps no pair, so
//! this is a second engine pass after the fits.  Each entry the requests'
//! cells read becomes one [`Oriented`]: its request, its members' rows and
//! its MRCA cluster.  The lanes' records are sorted after the pass, so the
//! result does not depend on lanes or threads.  Each influence is the
//! estimate's [`super::influence::Weight`] on the entry's engine cell,
//! applied to the entry's own stride, so the influences a cluster holds sum
//! to what the SE's sweep sums for it, and the SE is the square root of the
//! inclusion-exclusion meat over them.

use super::influence::{Influence, Weights};
use super::mrca::{self, Ancestry, Cluster};
use super::{Cell, CellEntries, Fit, Members, Request, SeReason, View};
use crate::error::Error;
use crate::pedigree::Pedigree;
use pedigree_graph_core::error::Error as PgError;
use pedigree_graph_core::relationships::{
    self, reduce_pairs_in, Category, CategorySet, MaxDegree, Product, Progress, Reducer, RowOrder,
    Side, Symmetric, N_CATEGORIES,
};
use rayon::prelude::*;
use std::num::NonZeroUsize;

/// The operation the entries' budget error names.
pub(super) const OPERATION: &str = "relative_pairs_influence";

/// One entry a cell reads: request, first and second member's rows, MRCA
/// cluster, and the category, which orders two entries of one pair in a
/// group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Oriented {
    request: u32,
    first: u32,
    second: u32,
    category: u8,
    cluster: Cluster,
}

/// The pass's reducer: each entry some cell of its request reads, as an
/// [`Oriented`].
struct Emit<'a> {
    labels: &'a [i32],
    ancestry: &'a Ancestry<'a>,
    /// Per category, its request, or `u32::MAX` for one no request reads.
    request: [u32; N_CATEGORIES],
    /// Per request, per engine label pair `first * n_labels + second`,
    /// whether one of its cells reads the entry.
    reads: Vec<Vec<bool>>,
    n_labels: usize,
}

impl Reducer for Emit<'_> {
    type Lane = Vec<Vec<Oriented>>;

    fn lane(&self) -> Result<Self::Lane, PgError> {
        Ok(vec![Vec::new()])
    }

    #[inline]
    fn reduce(&self, lane: &mut Self::Lane, cat: Category, first: u32, second: u32) {
        let request = self.request[cat.index()];
        let Some(reads) = self.reads.get(request as usize) else {
            return;
        };
        let (la, lb) = (
            self.labels[first as usize] as usize,
            self.labels[second as usize] as usize,
        );
        if !reads[la * self.n_labels + lb] {
            return;
        }
        lane[0].push(Oriented {
            request,
            first,
            second,
            category: cat.index() as u8,
            cluster: self.ancestry.mrca(cat, first, second),
        });
    }

    fn merge(&self, into: &mut Self::Lane, from: Self::Lane) {
        into.extend(from);
    }
}

/// The traits each of `views` reads shown by the members of an engine
/// label pair, per pair `first * n_labels + second`.
fn reads(members: &Members, views: &[View], n_labels: usize) -> Vec<bool> {
    let layout = members.layout;
    let n_joint = layout.n_joint();
    let shows = |label: usize, t: usize| layout.shows(label % n_joint, t);
    (0..n_labels * n_labels)
        .map(|pair| {
            let (a, b) = (pair / n_labels, pair % n_labels);
            views
                .iter()
                .any(|v| shows(a, v.first) && shows(b, v.second))
        })
        .collect()
}

/// A cell's entries: a directed cell's pairs, a symmetric same-trait
/// cell's in both orientations, a symmetric cross-trait cell's complete
/// orientations.
fn n_entries(cell: &Cell<Fit>) -> u64 {
    match (cell.n_entries, cell.symmetric) {
        (Some(n), _) => n,
        (None, true) => 2 * cell.n,
        (None, false) => cell.n,
    }
}

/// A cell's fits in the order hosts present them: the crude estimates,
/// each stratum's, then the stratified fit.
pub(super) fn fits<E>(cell: &Cell<E>) -> impl Iterator<Item = &E> {
    let strata = cell.strata.iter().flatten().flat_map(|s| &s.estimates);
    cell.estimates
        .iter()
        .chain(strata)
        .chain(cell.stratified.as_ref().map(|s| &s.estimate))
}

/// The bytes the entries take: the records, as the lanes may hold them at
/// twice their length plus their concatenation; the cells' ids and
/// influences; the largest request's [`Weights`] and its estimates' columns;
/// the requests' label-pair tables and one cell's; and the `n_external`
/// external ids.
fn bytes(
    cells: &[(usize, Cell<Fit>)],
    influences: &[Vec<Result<Influence, SeReason>>],
    members: &Members,
    n_requests: usize,
    n_external: usize,
) -> u64 {
    let (mut records, mut out) = (0u64, 0u64);
    for (_, cell) in cells {
        let n = n_entries(cell);
        let columns = 3 + fits(cell).filter(|f| f.influence.is_some()).count() as u64;
        records = records.saturating_add(n);
        out = out.saturating_add(n.saturating_mul(columns).saturating_mul(8));
    }
    let weights = influences
        .iter()
        .map(|list| {
            Weights::bytes(members.layout, list)
                .saturating_add((list.len() * size_of::<Option<usize>>()) as u64)
        })
        .max()
        .unwrap_or(0);
    let n_labels = (members.layout.n_codes * members.layout.n_joint()) as u64;
    let tables = (n_requests as u64 + 1).saturating_mul(n_labels * n_labels);
    records
        .saturating_mul(3 * size_of::<Oriented>() as u64)
        .saturating_add(out)
        .saturating_add(weights)
        .saturating_add(tables)
        .saturating_add((n_external * size_of::<i64>()) as u64)
}

/// [`entries`]' inputs but the budget.
pub(super) struct Args<'a, 'p> {
    pub(super) pedigree: &'p Pedigree,
    pub(super) ped: &'p relationships::Pedigree<'p>,
    pub(super) max_degree: MaxDegree,
    pub(super) union: CategorySet,
    pub(super) requests: &'p [Request],
    pub(super) members: &'a Members,
    pub(super) ancestry: &'a Ancestry<'a>,
    pub(super) threads: NonZeroUsize,
    /// Each request's cells, with the request's index, in result order.
    pub(super) cells: &'a [(usize, Cell<Fit>)],
    pub(super) influences: &'a [Vec<Result<Influence, SeReason>>],
}

/// Each of `cells`' [`CellEntries`], in their order.  `held` is what the
/// call holds through the pass, and `max_degree` covers `union`.
///
/// # Errors
///
/// `memory_budget_exceeded` (operation [`OPERATION`]) before the pass when
/// `held` and the entries' bytes do not fit `budget`; the engine's errors.
pub(super) fn entries(
    args: Args<'_, '_>,
    held: u64,
    budget: u64,
) -> Result<Vec<CellEntries>, Error> {
    let Args {
        pedigree,
        ped,
        max_degree,
        union,
        requests,
        members,
        ancestry,
        threads,
        cells,
        influences,
    } = args;
    let external = mrca::external(pedigree);
    let need = held.saturating_add(bytes(
        cells,
        influences,
        members,
        requests.len(),
        external.len(),
    ));
    if need > budget {
        return Err(PgError::MemoryBudgetExceeded {
            operation: OPERATION,
            estimated_bytes: need,
            budget_bytes: budget,
        }
        .into());
    }
    let n_labels = members.layout.n_codes * members.layout.n_joint();
    let mut request = [u32::MAX; N_CATEGORIES];
    for (r, req) in requests.iter().enumerate() {
        for cat in &req.members {
            request[cat.index()] = r as u32;
        }
    }
    let mut views = vec![Vec::new(); requests.len()];
    for (r, cell) in cells {
        views[*r].push(View {
            first: cell.first_trait,
            second: cell.second_trait,
            symmetric: cell.symmetric,
        });
    }
    let emit = Emit {
        labels: &members.labels,
        ancestry,
        request,
        reads: views.iter().map(|v| reads(members, v, n_labels)).collect(),
        n_labels,
    };
    let reduced = reduce_pairs_in(
        ped,
        max_degree,
        union,
        None,
        Symmetric::Both,
        threads,
        &emit,
        &Progress::default(),
        RowOrder::Families,
    )?;
    drop(emit);
    let runs = reduced.lane;
    let mut all = Vec::with_capacity(runs.iter().map(Vec::len).sum());
    for run in runs {
        all.extend(run);
    }
    all.par_sort_unstable();
    let ids = pedigree.ids();
    let k = members.exponents.len();
    let products = members.products();
    let label = |row: u32| members.labels[row as usize] as usize;
    let mut m = vec![0.0; 1 + 4 * k + products.len()];
    let mut out = Vec::with_capacity(cells.len());
    // A request's cells are consecutive, and one Weights serves them all.
    for group in cells.chunk_by(|a, b| a.0 == b.0) {
        let r = group[0].0;
        let weights = Weights::build(members.layout, &members.exponents, &influences[r]);
        let start = all.partition_point(|e| (e.request as usize) < r);
        let end = start + all[start..].partition_point(|e| e.request as usize == r);
        for (_, cell) in group {
            let view = View {
                first: cell.first_trait,
                second: cell.second_trait,
                symmetric: cell.symmetric,
            };
            let read = reads(members, &[view], n_labels);
            let n = n_entries(cell) as usize;
            // Per estimate with an influence, its column of `influence`.
            let mut column = vec![None; influences[r].len()];
            let mut influence: Vec<Option<Vec<f64>>> = Vec::new();
            for (c, fit) in fits(cell).enumerate() {
                let estimate = fit.influence.filter(|&i| influences[r][i].is_ok());
                if let Some(i) = estimate {
                    column[i] = Some(c);
                }
                influence.push(estimate.map(|_| vec![0.0; n]));
            }
            let (mut first_id, mut second_id, mut mrca_id) = (
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            );
            let mine = all[start..end]
                .iter()
                .filter(|e| read[label(e.first) * n_labels + label(e.second)]);
            for (at, e) in mine.enumerate() {
                first_id.push(ids[e.first as usize]);
                second_id.push(ids[e.second as usize]);
                mrca_id.push(mrca::id(ids, &external, e.cluster).unwrap_or(-1));
                single(&members.values, k, products, e.first, e.second, &mut m);
                for w in weights.of(label(e.first) * n_labels + label(e.second)) {
                    if let Some(list) =
                        column[w.estimate as usize].and_then(|c| influence[c].as_mut())
                    {
                        list[at] = w.apply(&m);
                    }
                }
            }
            debug_assert_eq!(first_id.len(), n, "{}", cell.category);
            out.push(CellEntries {
                first_id,
                second_id,
                mrca_id,
                influence,
            });
        }
    }
    Ok(out)
}

/// One entry's stride, CellReducer's layout over a count of 1: the count,
/// the first member's values, the second's, their squares, then each
/// product, first x second.
fn single(
    values: &[i64],
    k: usize,
    products: &[Product],
    first: u32,
    second: u32,
    out: &mut [f64],
) {
    let (a, b) = (first as usize * k, second as usize * k);
    let (x, y) = (&values[a..a + k], &values[b..b + k]);
    out[0] = 1.0;
    for c in 0..k {
        let (x, y) = (i128::from(x[c]), i128::from(y[c]));
        out[1 + c] = x as f64;
        out[1 + k + c] = y as f64;
        out[1 + 2 * k + c] = (x * x) as f64;
        out[1 + 3 * k + c] = (y * y) as f64;
    }
    for (i, p) in products.iter().enumerate() {
        debug_assert!(p.a.side == Side::First && p.b.side == Side::Second);
        out[1 + 4 * k + i] = (i128::from(x[p.a.column]) * i128::from(y[p.b.column])) as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::super::mrca::tests::generated;
    use super::super::{
        correlation_with, Estimate, RelativePairCorrelation, Se, DEFAULT_MEMORY_BUDGET_BYTES,
    };
    use super::*;
    use crate::assortative::Strata;
    use crate::input::PedigreeInput;
    use crate::input::{Trait, TraitKind};
    use crate::pedigree::PedigreeArg;
    use std::collections::HashMap;

    fn noise(i: usize) -> f64 {
        ((i * 7919 + 13) % 101) as f64 / 101.0
    }

    /// `n_traits` traits of `kind` on a generated pedigree (external
    /// fathers, parentless MZ twins, double first cousins), a few values
    /// missing, with generation strata (every 13th unknown) when `strata`.
    fn run(
        kind: TraitKind,
        n_traits: usize,
        strata: bool,
        se: Se,
        influence: bool,
        budget: i64,
    ) -> Result<RelativePairCorrelation, Error> {
        let cols = generated(300, 4, 3);
        let values: Vec<Vec<f64>> = (0..n_traits)
            .map(|t| {
                (0..cols.ids.len())
                    .map(|i| match ((i + 4 * t) % 9, kind) {
                        (0, _) => f64::NAN,
                        (_, TraitKind::Binary) => f64::from(u8::from(
                            noise(i + 31 * t) + 0.4 * noise(cols.mother[i].max(0) as usize) < 0.5,
                        )),
                        _ => noise(i + 31 * t) + 0.5 * noise(cols.mother[i].max(0) as usize),
                    })
                    .collect()
            })
            .collect();
        let traits: Vec<Trait<'_>> = values
            .iter()
            .map(|values| Trait {
                values,
                kind,
                n_levels: None,
            })
            .collect();
        let labels: Vec<i64> = cols.generation.clone();
        let known: Vec<bool> = (0..cols.ids.len()).map(|i| i % 13 != 0).collect();
        let input = PedigreeInput {
            ids: &cols.ids,
            mother: &cols.mother,
            father: &cols.father,
            twin: Some(&cols.twin),
            sex: None,
        };
        correlation_with(
            PedigreeArg::Columns(input),
            &traits,
            strata.then_some(Strata {
                labels: &labels,
                known: &known,
            }),
            None,
            budget,
            se,
            influence,
        )
    }

    /// A cluster key of the MRCA design: the MRCA's id, or the unordered
    /// pair for one with none.
    type Key = (i64, i64, i64);

    /// The meat `Σ_m Φ_m² + Σ_i S_i² − Σ_(i,m) S_(i,m)²` of one estimate's
    /// influences `phi` on `entries`.
    fn meat(entries: &CellEntries, phi: &[f64]) -> f64 {
        let mut by_mrca: HashMap<Key, f64> = HashMap::new();
        let mut by_person: HashMap<i64, f64> = HashMap::new();
        let mut by_both: HashMap<(i64, Key), f64> = HashMap::new();
        for (e, &p) in phi.iter().enumerate() {
            let (a, b) = (entries.first_id[e], entries.second_id[e]);
            let m = match entries.mrca_id[e] {
                -1 => (-1, a.min(b), a.max(b)),
                m => (m, -1, -1),
            };
            *by_mrca.entry(m).or_default() += p;
            for i in [a, b] {
                *by_person.entry(i).or_default() += p;
                *by_both.entry((i, m)).or_default() += p;
            }
        }
        let squares = |sums: Vec<f64>| sums.iter().map(|s| s * s).sum::<f64>();
        squares(by_mrca.into_values().collect()) + squares(by_person.into_values().collect())
            - squares(by_both.into_values().collect())
    }

    /// Every SE is the square root of the meat over its entries'
    /// influences, each cell holds its entries, and the estimates do not
    /// change.
    #[test]
    fn the_se_is_the_meat_over_the_entries() {
        for kind in [TraitKind::Binary, TraitKind::Continuous] {
            for n_traits in [1, 2] {
                for strata in [false, true] {
                    let what = format!("{kind:?} traits {n_traits} strata {strata}");
                    let budget = DEFAULT_MEMORY_BUDGET_BYTES;
                    let res = run(kind, n_traits, strata, Se::Estimates, true, budget).unwrap();
                    let off = run(kind, n_traits, strata, Se::Estimates, false, budget).unwrap();
                    assert_eq!(res.cells, off.cells, "{what}");
                    assert!(off.entries.is_none(), "{what}");
                    let covariance = run(kind, n_traits, strata, Se::Covariance, true, budget);
                    assert_eq!(covariance.unwrap().entries, res.entries, "{what}");
                    let entries = res.entries.as_ref().unwrap();
                    assert_eq!(entries.len(), res.cells.len(), "{what}");
                    let (mut checked, mut lone) = (0, 0);
                    for (cell, entries) in res.cells.iter().zip(entries) {
                        let what = format!(
                            "{what} {} ({}, {})",
                            cell.category, cell.first_trait, cell.second_trait
                        );
                        let n = match (cell.n_entries, cell.symmetric) {
                            (Some(n), _) => n,
                            (None, true) => 2 * cell.n,
                            (None, false) => cell.n,
                        };
                        assert_eq!(entries.first_id.len() as u64, n, "{what}");
                        assert_eq!(entries.second_id.len() as u64, n, "{what}");
                        assert_eq!(entries.mrca_id.len() as u64, n, "{what}");
                        lone += entries.mrca_id.iter().filter(|&&m| m == -1).count();
                        let estimates: Vec<&Estimate> = fits(cell).collect();
                        assert_eq!(estimates.len(), entries.influence.len(), "{what}");
                        for (estimate, phi) in estimates.iter().zip(&entries.influence) {
                            let se = estimate.se.unwrap();
                            let Some(phi) = phi else {
                                assert!(se.is_err(), "{what}");
                                continue;
                            };
                            assert_eq!(phi.len() as u64, n, "{what}");
                            let Ok(se) = se else { continue };
                            let rebuilt = meat(entries, phi).sqrt();
                            assert!(
                                (rebuilt - se).abs() <= 1e-9 * se,
                                "{what} {:?}: {rebuilt} {se}",
                                estimate.estimator
                            );
                            checked += 1;
                        }
                    }
                    assert!(checked >= 10, "{what}: {checked}");
                    assert!(lone > 0, "{what}");
                }
            }
        }
    }

    /// The entries are in row order and their ids are the pedigree's, an
    /// external MRCA its own id; a symmetric cell holds both orientations.
    #[test]
    fn entries_are_ordered_and_oriented() {
        let res = run(
            TraitKind::Continuous,
            1,
            false,
            Se::Estimates,
            true,
            1 << 30,
        )
        .unwrap();
        let entries = res.entries.unwrap();
        let cols = generated(300, 4, 3);
        let row: HashMap<i64, usize> = cols
            .ids
            .iter()
            .enumerate()
            .map(|(r, &id)| (id, r))
            .collect();
        let mut external = false;
        for (cell, entries) in res.cells.iter().zip(&entries) {
            let rows: Vec<(usize, usize)> = entries
                .first_id
                .iter()
                .zip(&entries.second_id)
                .map(|(a, b)| (row[a], row[b]))
                .collect();
            assert!(rows.windows(2).all(|w| w[0] <= w[1]), "{}", cell.category);
            if cell.symmetric {
                let mut flipped: Vec<(usize, usize)> = rows.iter().map(|&(a, b)| (b, a)).collect();
                flipped.sort_unstable();
                assert_eq!(flipped, rows, "{}", cell.category);
            }
            external |= entries
                .mrca_id
                .iter()
                .any(|m| *m >= 0 && !row.contains_key(m));
        }
        assert!(external);
    }

    /// Without the SE there are no entries, and a budget the entries do not
    /// fit is refused with their operation.
    #[test]
    fn entries_need_the_se_and_fit_the_budget() {
        let off = run(TraitKind::Binary, 1, false, Se::Off, true, 1 << 30).unwrap();
        assert!(off.entries.is_none());
        // Two binary traits with strata: the entries outgrow the SE.
        let call =
            |influence, budget| run(TraitKind::Binary, 2, true, Se::Estimates, influence, budget);
        let (mut lo, mut hi) = (1i64, 1i64 << 30);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if call(true, mid).is_ok() {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        assert!(call(false, lo).is_ok());
        match call(true, lo) {
            Err(Error::Pedigree(PgError::MemoryBudgetExceeded {
                operation,
                estimated_bytes,
                budget_bytes,
            })) => {
                assert_eq!(operation, OPERATION);
                assert_eq!((estimated_bytes, budget_bytes), (hi as u64, lo as u64));
            }
            other => panic!("{other:?}"),
        }
    }
}
