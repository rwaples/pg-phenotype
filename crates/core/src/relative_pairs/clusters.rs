//! The one engine pass of a Relative-Pair Correlation, with exact per-key
//! sufficient statistics for the cluster sandwich SE (ADR 0013).
//!
//! [`Pass::run`] is `relationship_moments` for `Receiver::Graph`, its
//! table byte for byte, with a key store riding the same lanes.  Each
//! entry (one oriented pair) the engine reduces adds two rows, one per
//! member: "member `i`, on this side of an entry of this engine cell, whose
//! MRCA is `m`".  The MRCA, person and (person, MRCA) clusters are
//! groupings of those rows, so one store holds all three.  A row keeps only
//! the partner half of CellReducer's stride (count, and per value column
//! the partners' sum and sum of squares); the member's own half follows
//! from its value ([`Row::stride`]).
//!
//! Lanes share nothing.  Each coalesces its rows in a hash index while the
//! pass runs, drops the index when it is merged, and keeps its rows as a
//! run; the runs are sorted after the pass and read back through one k-way
//! merge ([`merge`]) that adds a key several lanes hold by integer
//! addition, so what the store yields does not depend on lanes or threads.
//! [`Clusters::meats`] forms every float after that merge, in key order.
//!
//! With the covariance and two traits the lanes also sum the Within-Person
//! kernel ([`Kin`]), over categories the call may not have asked for.

use super::influence::Weights;
use super::mrca::{Ancestry, Cluster};
use super::within_person::{Kernel, KernelLane};
use super::{Members, Request};
use crate::error::Error;
use pedigree_graph_core::error::Error as PgError;
use pedigree_graph_core::relationships::{
    self, reduce_pairs, Category, CategorySet, CellReducer, MaxDegree, MomentsPlan, MomentsShape,
    MomentsTable, Product, Progress, Reducer, Side, Symmetric, N_CATEGORIES,
};
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::num::NonZeroUsize;
use std::ops::Range;

/// The operation the store's budget and allocation errors name.
pub(super) const OPERATION: &str = "relative_pairs_se";

/// Lane index bytes planned per row: a `(Key, u32)` bucket and its control
/// byte at the lowest load after a growth (7/16), 57 bytes, plus the half
/// as large table a growth frees only after rehashing into the new one.
const INDEX_BYTES_PER_ROW: u64 = 86;

/// Below this many rows per lane the plan trades lanes for room.
const MIN_LANE_ROWS: u64 = 1 << 16;

/// The smallest capacity a lane's rows grow to, and the fewest rows a
/// lane's share may hold: a lane that cannot take its first growth would
/// stop the pass at its first entries.
const MIN_GROWTH: usize = 1 << 10;

/// One row's place in the store.  Ordered (request, cluster, person, side,
/// cell), so every grouping the sweep needs is a prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Key {
    /// `request << 32 | cluster`.
    hi: u64,
    /// `person << 32 | side << 31 | cell`, `cell` being CellReducer's
    /// index: `slot * cells + first label * labels + second label`.
    lo: u64,
}

impl Key {
    /// `cell` is below `2^31` ([`Plan::new`] checks the cell count).
    #[inline]
    fn new(request: u32, cluster: Cluster, person: u32, side: Side, cell: u32) -> Key {
        Key {
            hi: u64::from(request) << 32 | u64::from(cluster.0),
            lo: u64::from(person) << 32 | u64::from(side.code()) << 31 | u64::from(cell),
        }
    }

    fn person(self) -> usize {
        (self.lo >> 32) as usize
    }

    fn side(self) -> Side {
        if self.lo >> 31 & 1 == 0 {
            Side::First
        } else {
            Side::Second
        }
    }

    fn cell(self) -> usize {
        (self.lo & 0x7fff_ffff) as usize
    }
}

/// An `i128` as little-endian `u64` limbs, so a row stays 8-aligned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Wide([u64; 2]);

impl Wide {
    #[inline]
    fn of(v: i128) -> Wide {
        Wide([v as u64, (v >> 64) as u64])
    }

    #[inline]
    fn get(self) -> i128 {
        (u128::from(self.0[1]) << 64 | u128::from(self.0[0])) as i128
    }
}

/// What one member's partners added in one engine cell: their count and,
/// per value column, their sum and sum of squares, exact.  A partner value
/// is at most `2^43` and a cell at most `2^40` pairs, so neither wraps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Partners<const K: usize> {
    count: u64,
    sum: [Wide; K],
    square: [Wide; K],
}

impl<const K: usize> Partners<K> {
    #[inline]
    fn of(values: [i64; K]) -> Partners<K> {
        Partners {
            count: 1,
            sum: values.map(|v| Wide::of(i128::from(v))),
            square: values.map(|v| Wide::of(i128::from(v) * i128::from(v))),
        }
    }

    #[inline]
    fn add(&mut self, other: &Partners<K>) {
        self.count += other.count;
        for c in 0..K {
            self.sum[c] = Wide::of(self.sum[c].get() + other.sum[c].get());
            self.square[c] = Wide::of(self.square[c].get() + other.square[c].get());
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row<const K: usize> {
    key: Key,
    partners: Partners<K>,
}

impl<const K: usize> Row<K> {
    /// The row as CellReducer's stride: the count, the first member's
    /// sums, the second's, their sums of squares, then each product's
    /// cross sum.  `own` is the row's member's values; every product is
    /// first x second, as [`Members`] requests them.
    #[inline(always)]
    fn stride(&self, own: &[i64], products: &[Product], out: &mut [i128]) {
        let p = &self.partners;
        let n = i128::from(p.count);
        let first = self.key.side() == Side::First;
        let (mine, theirs) = if first { (1, 1 + K) } else { (1 + K, 1) };
        out[0] = n;
        for (c, &v) in own.iter().enumerate() {
            let v = i128::from(v);
            let (mine, theirs) = (mine + c, theirs + c);
            out[mine] = n * v;
            out[theirs] = p.sum[c].get();
            out[mine + 2 * K] = n * v * v;
            out[theirs + 2 * K] = p.square[c].get();
        }
        for (i, product) in products.iter().enumerate() {
            debug_assert!(product.a.side == Side::First && product.b.side == Side::Second);
            let (mine, theirs) = if first {
                (product.a.column, product.b.column)
            } else {
                (product.b.column, product.a.column)
            };
            out[1 + 4 * K + i] = i128::from(own[mine]) * p.sum[theirs].get();
        }
    }
}

/// Why a lane stopped inserting rows; raised after the pass.
#[derive(Clone, Copy, Debug)]
enum Overflow {
    /// The lane reached its planned rows.
    Budget { rows: usize },
    /// The allocator refused the lane's rows or index.
    Allocation { rows: usize },
}

/// One engine lane, wrapped: CellReducer's accumulators untouched, the
/// lane's own rows and their index, the rows of the lanes merged into it,
/// each a run whose index is gone, and the kernel's sums.
struct KeyedLane<const K: usize, L> {
    cells: Vec<i128>,
    rows: Vec<Row<K>>,
    /// Each of `rows`' keys, to its position.  Never iterated, so its
    /// hasher reaches no result.
    index: HashMap<Key, u32, FxBuildHasher>,
    merged: Vec<Vec<Row<K>>>,
    /// The last symmetric entry and its cluster: `Symmetric::Both` sends
    /// `(b, a)` right after `(a, b)`.
    memo: Option<(Category, u32, u32, Cluster)>,
    /// Set instead of failing inside `reduce`.
    overflow: Option<Overflow>,
    kernel: L,
}

impl<const K: usize, L> KeyedLane<K, L> {
    /// Add `partners` to `key`'s row, or start one while the lane has room.
    #[inline]
    fn upsert(&mut self, key: Key, partners: Partners<K>, max_rows: usize) {
        if let Some(&at) = self.index.get(&key) {
            self.rows[at as usize].partners.add(&partners);
            return;
        }
        let rows = self.rows.len();
        if rows >= max_rows {
            self.overflow = Some(Overflow::Budget { rows });
            return;
        }
        if rows == self.rows.capacity() {
            let grown = (rows * 2).max(MIN_GROWTH).min(max_rows);
            if self.rows.try_reserve_exact(grown - rows).is_err() {
                self.overflow = Some(Overflow::Allocation { rows });
                return;
            }
        }
        if self.index.len() == self.index.capacity() && self.index.try_reserve(1).is_err() {
            self.overflow = Some(Overflow::Allocation { rows });
            return;
        }
        self.index.insert(key, rows as u32);
        self.rows.push(Row { key, partners });
    }
}

/// What the pass sums besides the table and the store: the Within-Person
/// kernel, or nothing ([`NoKernel`]).
pub(super) trait Kin: Sync {
    type Lane: Send;
    fn lane(&self) -> Self::Lane;
    /// Whether `cat` is traversed for the kernel alone: no cell, no row.
    fn only(&self, cat: Category) -> bool;
    /// One entry the engine sends, its members' engine labels `labels`.
    fn pair(&self, lane: &mut Self::Lane, cat: Category, first: u32, second: u32, labels: [u32; 2]);
    fn merge(&self, into: &mut Self::Lane, from: Self::Lane);
    /// The categories the pass traverses: `union` and the kernel's own.
    fn traversal(&self, union: CategorySet) -> CategorySet;
    fn finish(lane: Self::Lane) -> Option<KernelLane>;
}

/// The pass without the kernel: a separate monomorph whose `reduce` is the
/// one the SE had before the kernel.
pub(super) struct NoKernel;

impl Kin for NoKernel {
    type Lane = ();

    #[inline(always)]
    fn lane(&self) {}

    #[inline(always)]
    fn only(&self, _: Category) -> bool {
        false
    }

    #[inline(always)]
    fn pair(&self, _: &mut (), _: Category, _: u32, _: u32, _: [u32; 2]) {}

    #[inline(always)]
    fn merge(&self, _: &mut (), _: ()) {}

    fn traversal(&self, union: CategorySet) -> CategorySet {
        union
    }

    fn finish(_: ()) -> Option<KernelLane> {
        None
    }
}

/// pedigree-graph's CellReducer, extended: its reduce, then the entry's
/// two rows and its kernel term.
struct Keyed<'a, const K: usize, Q> {
    cells: CellReducer<'a>,
    labels: &'a [i32],
    values: &'a [i64],
    ancestry: &'a Ancestry<'a>,
    /// Per category: CellReducer's slot (its registry order among the
    /// requested, the rule `CellReducer::new` applies) and its request.
    slot: [u32; N_CATEGORIES],
    request: [u32; N_CATEGORIES],
    cells_per_category: u32,
    n_labels: u32,
    /// Per engine label, whether any trait is observed.  An entry whose
    /// members do not both show one feeds no estimate and adds no row.
    seen: Vec<bool>,
    /// The rows a lane may hold.
    max_rows: usize,
    kernel: Q,
}

impl<const K: usize, Q> Keyed<'_, K, Q> {
    #[inline]
    fn partners(&self, row: u32) -> Partners<K> {
        let at = row as usize * K;
        Partners::of(std::array::from_fn(|c| self.values[at + c]))
    }
}

impl<const K: usize, Q: Kin> Reducer for Keyed<'_, K, Q> {
    type Lane = KeyedLane<K, Q::Lane>;

    fn lane(&self) -> Result<Self::Lane, PgError> {
        Ok(KeyedLane {
            cells: self.cells.lane()?,
            rows: Vec::new(),
            index: HashMap::default(),
            merged: Vec::new(),
            memo: None,
            overflow: None,
            kernel: self.kernel.lane(),
        })
    }

    #[inline]
    fn reduce(&self, lane: &mut Self::Lane, cat: Category, first: u32, second: u32) {
        let labels = |a: u32, b: u32| {
            [
                self.labels[a as usize] as u32,
                self.labels[b as usize] as u32,
            ]
        };
        if self.kernel.only(cat) {
            let la_lb = labels(first, second);
            self.kernel
                .pair(&mut lane.kernel, cat, first, second, la_lb);
            return;
        }
        self.cells.reduce(&mut lane.cells, cat, first, second);
        if lane.overflow.is_some() {
            return;
        }
        let [la, lb] = labels(first, second);
        self.kernel
            .pair(&mut lane.kernel, cat, first, second, [la, lb]);
        if !(self.seen[la as usize] && self.seen[lb as usize]) {
            return;
        }
        let cluster = match lane.memo {
            Some((c, a, b, cluster)) if c == cat && a == second && b == first => cluster,
            _ => self.ancestry.mrca(cat, first, second),
        };
        lane.memo = cat.symmetric().then_some((cat, first, second, cluster));
        let cell = self.slot[cat.index()] * self.cells_per_category + la * self.n_labels + lb;
        let request = self.request[cat.index()];
        let max_rows = self.max_rows;
        let key = |person, side| Key::new(request, cluster, person, side, cell);
        lane.upsert(key(first, Side::First), self.partners(second), max_rows);
        lane.upsert(key(second, Side::Second), self.partners(first), max_rows);
    }

    fn merge(&self, into: &mut Self::Lane, from: Self::Lane) {
        self.cells.merge(&mut into.cells, from.cells);
        self.kernel.merge(&mut into.kernel, from.kernel);
        into.merged.push(from.rows);
        into.merged.extend(from.merged);
        into.overflow = into.overflow.or(from.overflow);
    }
}

/// The engine's lane plan and each lane's share of the rest of the budget.
///
/// ```text
/// E(W)     = MomentsPlan::new(shape, W, budget).estimated_peak_bytes
/// share(W) = (budget - E(W) - reserve) / W
/// max_rows = share(W) / (row bytes + INDEX_BYTES_PER_ROW)
/// ```
///
/// `W` starts at the engine's own lane count and drops while a lane would
/// hold fewer than [`MIN_LANE_ROWS`] rows; one lane that would hold fewer
/// than [`MIN_GROWTH`] is refused before the pass.  The peak during the
/// pass is `E(W) + W * share(W)`, at most the budget.  Which rows a lane draws
/// depends on the thread count, so near the budget edge whether a lane
/// overflows (an error, never a different result) can too.
struct Plan {
    engine: MomentsPlan,
    max_rows: usize,
}

impl Plan {
    /// # Errors
    ///
    /// The engine's `memory_budget_exceeded` (operation
    /// `relationship_moments`) when one engine lane does not fit; then
    /// `memory_budget_exceeded` (operation [`OPERATION`]) when one engine
    /// lane, `reserve` and [`MIN_GROWTH`] rows do not, or the cells do not
    /// fit a key.
    fn new(
        shape: MomentsShape,
        threads: NonZeroUsize,
        budget: u64,
        reserve: u64,
        row_bytes: u64,
    ) -> Result<Plan, Error> {
        let over = |estimated_bytes| PgError::MemoryBudgetExceeded {
            operation: OPERATION,
            estimated_bytes,
            budget_bytes: budget,
        };
        let mut lanes = MomentsPlan::new(shape, threads, budget)?.lanes;
        let one = MomentsPlan::new(shape, NonZeroUsize::MIN, budget)?;
        if (shape.n_categories as u64).saturating_mul(one.cells() as u64) > 1 << 31 {
            return Err(over(u64::MAX).into());
        }
        let fixed = one.estimated_peak_bytes.saturating_add(reserve);
        if fixed > budget {
            return Err(over(fixed).into());
        }
        loop {
            let engine = MomentsPlan::new(shape, lanes, budget)?;
            let rest = budget.saturating_sub(engine.estimated_peak_bytes.saturating_add(reserve));
            let rows = rest / engine.lanes.get() as u64 / (row_bytes + INDEX_BYTES_PER_ROW);
            match NonZeroUsize::new(engine.lanes.get() - 1) {
                Some(fewer) if rows < MIN_LANE_ROWS => lanes = fewer,
                _ if rows < MIN_GROWTH as u64 => {
                    let rows = MIN_GROWTH as u64 * (row_bytes + INDEX_BYTES_PER_ROW);
                    return Err(over(fixed.saturating_add(rows)).into());
                }
                _ => {
                    return Ok(Plan {
                        engine,
                        max_rows: rows.min(u64::from(u32::MAX)) as usize,
                    })
                }
            }
        }
    }
}

/// [`Pass::run`]'s inputs but the budget, the reserve and the kernel.
type Args<'a, 'p> = (
    &'p relationships::Pedigree<'p>,
    MaxDegree,
    CategorySet,
    &'p [Request],
    &'a Members,
    &'a Ancestry<'a>,
    NonZeroUsize,
);

/// The pass's outputs: the engine table, unchanged, the finished store and
/// the kernel's sums when it rode the pass.
pub(super) struct Pass<'a> {
    pub(super) table: MomentsTable<'static>,
    pub(super) clusters: Clusters<'a>,
    pub(super) kernel: Option<KernelLane>,
}

/// The store, one variant per value-column count.
pub(super) enum Clusters<'a> {
    Binary(Store<'a, 0>),
    One(Store<'a, 1>),
    Two(Store<'a, 2>),
}

/// Every lane's rows, each run sorted by key, its keys unique.
pub(super) struct Store<'a, const K: usize> {
    runs: Vec<Vec<Row<K>>>,
    n_rows: usize,
    /// The members' values, row-major `[rows, K]`.
    values: &'a [i64],
    products: &'static [Product],
}

impl<'a> Pass<'a> {
    /// `relationship_moments` for `Receiver::Graph` (`input.check`, the
    /// engine plan, `CellReducer::new`, `reduce_pairs` with `view = None`
    /// and `Symmetric::Both`, `CellReducer::finish`), its table identical,
    /// with the store riding the same lanes.  `reserve` is the bytes kept
    /// back for after the pass ([`Plan`]).  With `kernel` the pass also
    /// traverses the kernel's categories `union` lacks, for its sums alone,
    /// and `max_degree` covers them.
    ///
    /// # Errors
    ///
    /// The engine's errors, unchanged and in its order; then
    /// `memory_budget_exceeded` (operation [`OPERATION`]) before the pass
    /// when one engine lane and `reserve` do not fit, and after it when a
    /// lane outgrew its share; `allocation_failed` when the allocator
    /// refused a lane's rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(
        ped: &relationships::Pedigree<'_>,
        max_degree: MaxDegree,
        union: CategorySet,
        requests: &[Request],
        members: &'a Members,
        ancestry: &'a Ancestry<'a>,
        threads: NonZeroUsize,
        budget: u64,
        reserve: u64,
        kernel: Option<&Kernel>,
    ) -> Result<Pass<'a>, Error> {
        let args = (ped, max_degree, union, requests, members, ancestry, threads);
        match kernel {
            None => Pass::with(args, budget, reserve, NoKernel),
            Some(kernel) => Pass::with(args, budget, reserve, kernel),
        }
    }

    fn with<Q: Kin>(
        args: Args<'a, '_>,
        budget: u64,
        reserve: u64,
        kernel: Q,
    ) -> Result<Pass<'a>, Error> {
        match args.4.exponents.len() {
            0 => Pass::keyed(args, budget, reserve, kernel, Clusters::Binary),
            1 => Pass::keyed(args, budget, reserve, kernel, Clusters::One),
            _ => Pass::keyed(args, budget, reserve, kernel, Clusters::Two),
        }
    }

    /// [`Pass::run`] with `K` value columns, its store wrapped by `wrap`.
    fn keyed<const K: usize, Q: Kin>(
        (ped, max_degree, union, requests, members, ancestry, threads): Args<'a, '_>,
        budget: u64,
        reserve: u64,
        kernel: Q,
        wrap: fn(Store<'a, K>) -> Clusters<'a>,
    ) -> Result<Pass<'a>, Error> {
        let input = members.input();
        input.check(ped.len())?;
        let shape = input.shape(union);
        let row_bytes = std::mem::size_of::<Row<K>>() as u64;
        let plan = Plan::new(shape, threads, budget, reserve, row_bytes)?;
        let mut slot = [0u32; N_CATEGORIES];
        for (i, cat) in union.iter().enumerate() {
            slot[cat.index()] = i as u32;
        }
        let mut request = [0u32; N_CATEGORIES];
        for (r, req) in requests.iter().enumerate() {
            for cat in &req.members {
                request[cat.index()] = r as u32;
            }
        }
        let layout = members.layout;
        let n_joint = layout.n_joint();
        let seen = (0..input.n_labels_first)
            .map(|label| (0..layout.n_traits).any(|t| layout.shows(label % n_joint, t)))
            .collect();
        let traversal = kernel.traversal(union);
        let keyed = Keyed::<K, Q> {
            cells: CellReducer::new(input, union, plan.engine),
            labels: &members.labels,
            values: &members.values,
            ancestry,
            slot,
            request,
            cells_per_category: plan.engine.cells() as u32,
            n_labels: input.n_labels_second as u32,
            seen,
            max_rows: plan.max_rows,
            kernel,
        };
        let reduced = reduce_pairs(
            ped,
            max_degree,
            traversal,
            None,
            Symmetric::Both,
            plan.engine.lanes,
            &keyed,
            &Progress::default(),
        )?;
        let KeyedLane {
            cells,
            rows,
            index,
            mut merged,
            overflow,
            kernel,
            ..
        } = reduced.lane;
        let (width, bytes) = keyed.cells.finish(&cells)?;
        drop((cells, index));
        if let Some(overflow) = overflow {
            let lanes = plan.engine.lanes.get() as u64;
            let rows = |rows: usize| {
                lanes
                    .saturating_mul(rows as u64 + 1)
                    .saturating_mul(row_bytes + INDEX_BYTES_PER_ROW)
            };
            return Err(match overflow {
                Overflow::Budget { rows: at } => PgError::MemoryBudgetExceeded {
                    operation: OPERATION,
                    estimated_bytes: plan
                        .engine
                        .estimated_peak_bytes
                        .saturating_add(reserve)
                        .saturating_add(rows(at)),
                    budget_bytes: budget,
                },
                Overflow::Allocation { rows: at } => PgError::AllocationFailed {
                    operation: OPERATION,
                    requested_elements: (at + 1) * std::mem::size_of::<Row<K>>() / 8,
                    dtype: "uint64",
                },
            }
            .into());
        }
        merged.push(rows);
        merged
            .par_iter_mut()
            .for_each(|run| run.sort_unstable_by_key(|row| row.key));
        let table = MomentsTable::from_bytes(
            members.shape(shape.n_categories),
            input.n_columns,
            input.products.to_vec(),
            members.exponents.clone(),
            width,
            Cow::Owned(bytes),
        )?;
        Ok(Pass {
            table,
            clusters: wrap(Store {
                runs: merged,
                n_rows: members.labels.len(),
                values: &members.values,
                products: members.products(),
            }),
            kernel: Q::finish(kernel),
        })
    }
}

/// One estimate's meat `Σ_m Φ_m² + Σ_i S_i² − Σ_(i,m) S_(i,m)²`, its
/// `scale`: the same three sums of squares, added, had each of the `n`
/// entries the influence `1/n`, and `n`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Meat {
    pub(super) value: f64,
    pub(super) scale: f64,
    pub(super) entries: u64,
}

impl Meat {
    /// `sign · s²`.
    #[inline]
    fn add(&mut self, sign: f64, s: Sum) {
        self.value += sign * s.value * s.value;
        let entries = s.entries as f64;
        self.scale += entries * entries;
    }
}

/// A sum of rows' influences, and how many entries they count.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Sum {
    value: f64,
    entries: u64,
}

impl Sum {
    /// The influences summed.
    #[inline]
    pub(super) fn value(self) -> f64 {
        self.value
    }

    /// The entries summed.
    #[inline]
    pub(super) fn entries(self) -> u64 {
        self.entries
    }

    #[inline]
    fn add(&mut self, value: f64, entries: u64) {
        self.value += value;
        self.entries += entries;
    }

    /// A cluster's sum, whose entries each added two rows.
    fn half(self) -> Sum {
        Sum {
            value: self.value / 2.0,
            entries: self.entries / 2,
        }
    }
}

impl Clusters<'_> {
    /// Each of request `request`'s estimates' [`Meat`] over MRCA clusters
    /// `m` and members `i`, `Φ` and `S` the sums of the entries' influences
    /// ([`Weights`]) over each; 0 for an estimate with no influence.
    ///
    /// One sweep per `batch` estimates over the request's rows, merged in
    /// key order (cluster, person, side, cell): a cluster's rows and each
    /// (cluster, person) group's are consecutive.  An entry adds one row
    /// per member, so a cluster's rows sum to `2Φ_m`.  The per-person
    /// totals are dense and squared in row order at the end.  Batching
    /// repeats the sweep and never changes a float's order.  Each sweep
    /// hands its sums to the [`Sink`] `sink` makes for its range of
    /// estimates; the sinks come back in range order.
    // One function per sink, out of line, so `Discard`'s compiles as the
    // sweep did before it had a sink.
    #[inline(never)]
    pub(super) fn meats<S: Sink>(
        &self,
        request: u32,
        weights: &Weights,
        batch: usize,
        sink: impl FnMut(Range<usize>) -> S,
    ) -> (Vec<Meat>, Vec<S>) {
        match self {
            Clusters::Binary(store) => store.meats(request, weights, batch, sink),
            Clusters::One(store) => store.meats(request, weights, batch, sink),
            Clusters::Two(store) => store.meats(request, weights, batch, sink),
        }
    }

    /// The bytes the store holds.
    pub(super) fn bytes(&self) -> u64 {
        match self {
            Clusters::Binary(store) => store.bytes(),
            Clusters::One(store) => store.bytes(),
            Clusters::Two(store) => store.bytes(),
        }
    }

    /// How many rows request `request` holds over the lanes' runs, a key
    /// several runs hold counted in each.
    pub(super) fn n_rows(&self, request: u32) -> usize {
        match self {
            Clusters::Binary(store) => store.spans(request).map(<[_]>::len).sum(),
            Clusters::One(store) => store.spans(request).map(<[_]>::len).sum(),
            Clusters::Two(store) => store.spans(request).map(<[_]>::len).sum(),
        }
    }
}

impl<const K: usize> Store<'_, K> {
    /// Every distinct key's row in key order, a key several runs hold
    /// summed.
    #[cfg(test)]
    fn merged(&self) -> impl Iterator<Item = Row<K>> + '_ {
        merge(self.runs.iter().map(Vec::as_slice).collect())
    }

    /// Each run's rows of one request.
    fn spans(&self, request: u32) -> impl Iterator<Item = &[Row<K>]> {
        let of = |row: &Row<K>| (row.key.hi >> 32) as u32;
        self.runs.iter().map(move |run| {
            let start = run.partition_point(|row| of(row) < request);
            let end = start + run[start..].partition_point(|row| of(row) == request);
            &run[start..end]
        })
    }

    /// [`Store::merged`] over one request's rows.
    #[inline(always)]
    fn rows(&self, request: u32) -> impl Iterator<Item = Row<K>> + '_ {
        merge(self.spans(request).collect())
    }

    fn bytes(&self) -> u64 {
        let rows: usize = self.runs.iter().map(Vec::capacity).sum();
        (rows * size_of::<Row<K>>()) as u64
    }

    /// The values of `row`'s member.
    fn own(&self, row: &Row<K>) -> &[i64] {
        let at = row.key.person() * K;
        &self.values[at..at + K]
    }

    fn meats<S: Sink>(
        &self,
        request: u32,
        weights: &Weights,
        batch: usize,
        mut sink: impl FnMut(Range<usize>) -> S,
    ) -> (Vec<Meat>, Vec<S>) {
        let (n, batch) = (weights.n_estimates, batch.max(1));
        let mut meat = vec![Meat::default(); n];
        let mut sinks = Vec::new();
        for start in (0..n).step_by(batch) {
            let end = (start + batch).min(n);
            let mut s = sink(start..end);
            self.sweep(request, weights, start..end, &mut meat[start..end], &mut s);
            sinks.push(s);
        }
        (meat, sinks)
    }

    /// [`Clusters::meats`] of the estimates in `range`, into `meat`.
    fn sweep<S: Sink>(
        &self,
        request: u32,
        weights: &Weights,
        range: Range<usize>,
        meat: &mut [Meat],
        sink: &mut S,
    ) {
        let width = range.len();
        let mut cluster = vec![Sum::default(); width];
        let mut group = vec![Sum::default(); width];
        let mut person = vec![Sum::default(); self.n_rows * width];
        let mut entries = vec![0u64; width];
        let stride = 1 + 4 * K + self.products.len();
        let (mut sums, mut m) = (vec![0i128; stride], vec![0.0; stride]);
        let cells = weights.cells();
        let mut at: Option<(u64, usize)> = None;
        for row in self.rows(request) {
            let here = (row.key.hi, row.key.person());
            if let Some((hi, person)) = at.filter(|&at| at != here) {
                sink.group(Cluster(hi as u32), person as u32, &group);
                for (meat, g) in meat.iter_mut().zip(&mut group) {
                    meat.add(-1.0, std::mem::take(g));
                }
                if hi != here.0 {
                    sink.cluster(Cluster(hi as u32), &cluster);
                    for ((meat, c), n) in meat.iter_mut().zip(&mut cluster).zip(&mut entries) {
                        let c = std::mem::take(c).half();
                        *n += c.entries;
                        meat.add(1.0, c);
                    }
                }
            }
            at = Some(here);
            let items = weights.of(row.key.cell() % cells);
            if !items.iter().any(|w| range.contains(&(w.estimate as usize))) {
                continue;
            }
            row.stride(self.own(&row), self.products, &mut sums);
            for (m, &s) in m.iter_mut().zip(&sums) {
                *m = s as f64;
            }
            let totals = &mut person[here.1 * width..(here.1 + 1) * width];
            for w in items {
                let Some(e) = (w.estimate as usize).checked_sub(range.start) else {
                    continue;
                };
                if e >= width {
                    break;
                }
                let (v, n) = (w.apply(&m), row.partners.count);
                group[e].add(v, n);
                cluster[e].add(v, n);
                totals[e].add(v, n);
            }
        }
        if let Some((hi, person)) = at {
            sink.group(Cluster(hi as u32), person as u32, &group);
            sink.cluster(Cluster(hi as u32), &cluster);
        }
        for e in 0..width {
            let c = cluster[e].half();
            entries[e] += c.entries;
            meat[e].add(-1.0, group[e]);
            meat[e].add(1.0, c);
        }
        sink.persons(&person, width);
        for totals in person.chunks_exact(width) {
            for (meat, &t) in meat.iter_mut().zip(totals) {
                meat.add(1.0, t);
            }
        }
        for (meat, &n) in meat.iter_mut().zip(&entries) {
            meat.entries = n;
            if n > 0 {
                meat.scale /= (n as f64) * (n as f64);
            }
        }
    }
}

/// What a sweep hands out besides its meats: each sum of its range's
/// estimates before it is squared, in key order.
pub(super) trait Sink {
    /// A (cluster, person) group's sums.
    fn group(&mut self, cluster: Cluster, person: u32, sums: &[Sum]);
    /// A cluster's sums as summed: each entry's two rows, not yet halved.
    fn cluster(&mut self, cluster: Cluster, sums: &[Sum]);
    /// The per-person totals, row-major `[rows, width]`, once at the end.
    fn persons(&mut self, totals: &[Sum], width: usize);
}

/// The SE's own sweep: the sums go nowhere.  The sweep's helpers
/// ([`Row::stride`], [`Store::rows`], [`merge`]) are inlined always: with
/// two sinks they have two callers, and left to the inliner they come out
/// of the loop and slow it.
pub(super) struct Discard;

impl Sink for Discard {
    #[inline(always)]
    fn group(&mut self, _: Cluster, _: u32, _: &[Sum]) {}

    #[inline(always)]
    fn cluster(&mut self, _: Cluster, _: &[Sum]) {}

    #[inline(always)]
    fn persons(&mut self, _: &[Sum], _: usize) {}
}

/// Every distinct key's row of `runs` (each sorted, its keys unique) in
/// key order, a key several runs hold summed by integer addition.
#[inline(always)]
fn merge<const K: usize>(runs: Vec<&[Row<K>]>) -> Merge<'_, K> {
    let heap = runs
        .iter()
        .enumerate()
        .filter_map(|(r, run)| run.first().map(|row| Reverse((row.key, r))))
        .collect();
    Merge {
        at: vec![0usize; runs.len()],
        heap,
        runs,
    }
}

/// [`merge`]'s rows: each run, its next row's position, and the runs'
/// next keys.
struct Merge<'a, const K: usize> {
    runs: Vec<&'a [Row<K>]>,
    at: Vec<usize>,
    heap: BinaryHeap<Reverse<(Key, usize)>>,
}

impl<const K: usize> Iterator for Merge<'_, K> {
    type Item = Row<K>;

    #[inline(always)]
    fn next(&mut self) -> Option<Row<K>> {
        let Merge { runs, at, heap } = self;
        let Reverse((key, _)) = *heap.peek()?;
        let mut row: Option<Row<K>> = None;
        while let Some(&Reverse((next, r))) = heap.peek() {
            if next != key {
                break;
            }
            heap.pop();
            let from = &runs[r][at[r]];
            match &mut row {
                Some(row) => row.partners.add(&from.partners),
                None => row = Some(*from),
            }
            at[r] += 1;
            if let Some(following) = runs[r].get(at[r]) {
                heap.push(Reverse((following.key, r)));
            }
        }
        row
    }
}

#[cfg(test)]
mod tests {
    use super::super::mrca::tests::{engine_pairs, generated, Columns};
    use super::super::{default_requests, Layout};
    use super::*;
    use crate::input::{Trait, TraitKind};
    use crate::pedigree::Pedigree;
    use pedigree_graph_core::relationships::{relationship_moments, Receiver};
    use std::collections::BTreeMap;

    const BUDGET: u64 = 1 << 30;

    fn noise(i: usize) -> f64 {
        ((i * 7919 + 13) % 101) as f64 / 101.0
    }

    /// One or two traits of `kind` on `cols`, a few values (and every 23rd
    /// row's every value) missing, with
    /// generation strata (every 13th row unknown) when `strata`.
    fn members(cols: &Columns, kind: TraitKind, n_traits: usize, strata: bool) -> Members {
        let values: Vec<Vec<f64>> = (0..n_traits)
            .map(|t| {
                (0..cols.ids.len())
                    .map(|i| match ((i + 4 * t) % 9, kind) {
                        _ if i % 23 == 0 => f64::NAN,
                        (0, _) => f64::NAN,
                        (_, TraitKind::Binary) => f64::from(u8::from(noise(i + 31 * t) < 0.4)),
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
        let members = Members::of(&traits).unwrap();
        if !strata {
            return members;
        }
        let k = cols.generation.iter().max().unwrap() + 1;
        let codes: Vec<Option<usize>> = (0..cols.ids.len())
            .map(|i| (i % 13 != 0).then_some(cols.generation[i] as usize))
            .collect();
        members.stratified(&codes, k as usize + 1)
    }

    /// The store's rows in key order, each expanded to the engine stride,
    /// after checking that, summed by engine cell, they are twice the
    /// engine's table on every cell both of whose labels show a trait, and
    /// absent elsewhere.
    fn marginal<const K: usize>(
        store: &Store<'_, K>,
        table: &MomentsTable<'_>,
        layout: Layout,
        lanes: usize,
    ) -> Vec<(Key, Vec<i128>)> {
        assert_eq!(store.runs.len(), lanes);
        if lanes > 1 {
            assert!(store.runs.iter().filter(|run| !run.is_empty()).count() >= 2);
        }
        let stride = table.stride();
        let mut sums = vec![0i128; table.cells() * stride];
        let mut rows: Vec<(Key, Vec<i128>)> = Vec::new();
        for row in store.merged() {
            assert!(rows.last().is_none_or(|(last, _)| *last < row.key));
            let mut moments = vec![0i128; stride];
            row.stride(store.own(&row), store.products, &mut moments);
            let at = row.key.cell() * stride;
            for (sum, m) in sums[at..at + stride].iter_mut().zip(&moments) {
                *sum += m;
            }
            rows.push((row.key, moments));
        }
        let (n_joint, n_labels) = (layout.n_joint(), layout.n_codes * layout.n_joint());
        let seen = |label: usize| (0..layout.n_traits).any(|t| layout.shows(label % n_joint, t));
        let (mut stored, mut unstored) = (0, 0);
        for cell in 0..table.cells() {
            let (first, second) = (cell / n_labels % n_labels, cell % n_labels);
            let keyed = seen(first) && seen(second);
            for at in 0..stride {
                let engine = i128::try_from(table.read(cell, at)).unwrap();
                let expected = if keyed { 2 * engine } else { 0 };
                assert_eq!(sums[cell * stride + at], expected, "cell {cell} slot {at}");
            }
            let count = i128::try_from(table.read(cell, 0)).unwrap();
            if count > 0 {
                if keyed {
                    stored += 1;
                } else {
                    unstored += 1;
                }
            }
        }
        assert!(
            stored > 0 && unstored > 0,
            "{stored} stored, {unstored} not"
        );
        rows
    }

    /// The store, entry by entry: each entry the engine emits whose
    /// members both show a trait adds its CellReducer stride to its two
    /// members' keys.
    fn oracle(
        pedigree: &Pedigree,
        ancestry: &Ancestry<'_>,
        members: &Members,
        union: CategorySet,
    ) -> Vec<(Key, Vec<i128>)> {
        let layout = members.layout;
        let (n_joint, n_labels) = (layout.n_joint(), layout.n_codes * layout.n_joint());
        let seen = |label: usize| (0..layout.n_traits).any(|t| layout.shows(label % n_joint, t));
        let (k, products) = (members.exponents.len(), members.products());
        let slots: Vec<Category> = union.iter().collect();
        let requests = default_requests();
        let mut rows: BTreeMap<Key, Vec<i128>> = BTreeMap::new();
        for (cat, a, b) in engine_pairs(pedigree, Symmetric::Both) {
            let (la, lb) = (
                members.labels[a as usize] as usize,
                members.labels[b as usize] as usize,
            );
            if !(seen(la) && seen(lb)) {
                continue;
            }
            let slot = slots.iter().position(|&c| c == cat).unwrap();
            let request = requests
                .iter()
                .position(|r| r.members.contains(&cat))
                .unwrap();
            let cell = (slot * n_labels + la) * n_labels + lb;
            let cluster = ancestry.mrca(cat, a, b);
            let (va, vb) = (
                &members.values[a as usize * k..][..k],
                &members.values[b as usize * k..][..k],
            );
            let wide = |v: &[i64], c: usize| i128::from(v[c]);
            let mut entry = vec![1i128];
            entry.extend((0..k).map(|c| wide(va, c)));
            entry.extend((0..k).map(|c| wide(vb, c)));
            entry.extend((0..k).map(|c| wide(va, c) * wide(va, c)));
            entry.extend((0..k).map(|c| wide(vb, c) * wide(vb, c)));
            entry.extend(
                products
                    .iter()
                    .map(|p| wide(va, p.a.column) * wide(vb, p.b.column)),
            );
            for (person, side) in [(a, Side::First), (b, Side::Second)] {
                let key = Key::new(request as u32, cluster, person, side, cell as u32);
                let sum = rows.entry(key).or_insert_with(|| vec![0; entry.len()]);
                for (s, e) in sum.iter_mut().zip(&entry) {
                    *s += e;
                }
            }
        }
        rows.into_iter().collect()
    }

    #[test]
    fn the_store_is_every_entry_by_key_and_twice_the_engine_lane_and_the_table_is_relationship_moments(
    ) {
        let cols = generated(2400, 4, 7);
        assert!(cols.ids.len() > 4 * 2048, "{} rows", cols.ids.len());
        let pedigree = cols.pedigree();
        let ped = pedigree.relationships().unwrap();
        let ancestry = Ancestry::of(&pedigree, BUDGET).unwrap();
        let requests = default_requests();
        let union: CategorySet = requests.iter().flat_map(|r| r.members.clone()).collect();
        let max_degree = MaxDegree::try_new(union.top_degree().unwrap()).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (kind, n_traits, strata) in [
            (TraitKind::Binary, 1, false),
            (TraitKind::Continuous, 1, false),
            (TraitKind::Binary, 2, true),
            (TraitKind::Continuous, 2, true),
        ] {
            let members = members(&cols, kind, n_traits, strata);
            let moments = relationship_moments(
                &ped,
                max_degree,
                union,
                Receiver::Graph,
                &members.input(),
                Symmetric::Both,
                NonZeroUsize::MIN,
                BUDGET,
                &Progress::default(),
            )
            .unwrap();
            let expected = oracle(&pedigree, &ancestry, &members, union);
            for lanes in [1, 3, 8] {
                let what = format!("{kind:?} x{n_traits}, lanes {lanes}");
                let pass = pool
                    .install(|| {
                        Pass::run(
                            &ped,
                            max_degree,
                            union,
                            &requests,
                            &members,
                            &ancestry,
                            NonZeroUsize::new(lanes).unwrap(),
                            BUDGET,
                            0,
                            None,
                        )
                    })
                    .unwrap();
                assert_eq!(pass.table.width(), moments.width, "{what}");
                assert!(pass.table.bytes() == moments.table, "{what}");
                let layout = members.layout;
                let rows = match &pass.clusters {
                    Clusters::Binary(store) => marginal(store, &pass.table, layout, lanes),
                    Clusters::One(store) => marginal(store, &pass.table, layout, lanes),
                    Clusters::Two(store) => marginal(store, &pass.table, layout, lanes),
                };
                assert_eq!(rows.len(), expected.len(), "{what}");
                assert!(rows == expected, "{what}");
            }
        }
    }

    #[test]
    fn rows_are_as_small_as_planned() {
        assert_eq!(std::mem::size_of::<Row<0>>(), 24);
        assert_eq!(std::mem::size_of::<Row<1>>(), 56);
        assert_eq!(std::mem::size_of::<Row<2>>(), 88);
    }

    /// A budget whose one lane holds fewer rows than one growth is refused
    /// by the plan, before any pair is traversed, with the bytes one
    /// growth needs; one that holds a growth is planned at one lane.
    #[test]
    fn a_share_below_one_growth_is_refused_before_the_pass() {
        let cols = generated(200, 3, 1);
        let requests = default_requests();
        let union: CategorySet = requests.iter().flat_map(|r| r.members.clone()).collect();
        let members = members(&cols, TraitKind::Continuous, 1, false);
        let shape = members.input().shape(union);
        let one_lane = MomentsPlan::new(shape, NonZeroUsize::MIN, BUDGET)
            .unwrap()
            .estimated_peak_bytes;
        let row_bytes = std::mem::size_of::<Row<1>>() as u64;
        let (reserve, lanes) = (1000, NonZeroUsize::new(4).unwrap());
        let edge = one_lane + reserve + MIN_GROWTH as u64 * (row_bytes + INDEX_BYTES_PER_ROW);
        match Plan::new(shape, lanes, edge - 1, reserve, row_bytes) {
            Err(Error::Pedigree(PgError::MemoryBudgetExceeded {
                operation,
                estimated_bytes,
                budget_bytes,
            })) => assert_eq!(
                (operation, estimated_bytes, budget_bytes),
                (OPERATION, edge, edge - 1)
            ),
            Err(other) => panic!("{other:?}"),
            Ok(plan) => panic!("planned {} rows", plan.max_rows),
        }
        let plan = Plan::new(shape, lanes, edge, reserve, row_bytes).unwrap();
        assert_eq!((plan.engine.lanes.get(), plan.max_rows), (1, MIN_GROWTH));
    }

    #[test]
    fn the_store_is_held_to_the_budget() {
        let cols = generated(200, 3, 1);
        let pedigree = cols.pedigree();
        let ped = pedigree.relationships().unwrap();
        let ancestry = Ancestry::of(&pedigree, BUDGET).unwrap();
        let requests = default_requests();
        let union: CategorySet = requests.iter().flat_map(|r| r.members.clone()).collect();
        let max_degree = MaxDegree::try_new(3).unwrap();
        let members = members(&cols, TraitKind::Continuous, 1, false);
        let one_lane = MomentsPlan::new(members.input().shape(union), NonZeroUsize::MIN, BUDGET)
            .unwrap()
            .estimated_peak_bytes;
        let run = |budget, reserve| {
            Pass::run(
                &ped,
                max_degree,
                union,
                &requests,
                &members,
                &ancestry,
                NonZeroUsize::MIN,
                budget,
                reserve,
                None,
            )
            .err()
        };
        let refused = |err: Option<Error>, operation: &str, budget: u64| match err {
            Some(Error::Pedigree(PgError::MemoryBudgetExceeded {
                operation: o,
                estimated_bytes,
                budget_bytes,
            })) => o == operation && budget_bytes == budget && estimated_bytes > budget,
            _ => false,
        };
        assert!(refused(run(16, 0), "relationship_moments", 16));
        assert!(refused(run(one_lane, 1), OPERATION, one_lane));
        let ten_rows = one_lane + 10 * (56 + INDEX_BYTES_PER_ROW);
        assert!(refused(run(ten_rows, 0), OPERATION, ten_rows));
        assert!(run(BUDGET, 0).is_none());
    }
}
