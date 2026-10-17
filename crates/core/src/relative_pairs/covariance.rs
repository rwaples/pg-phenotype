//! The covariance of a Relative-Pair Correlation's estimates (ADR 0014),
//! over the forms the ACE Fit reads.  For estimates `j` and `k` whose
//! entries have influences `ψ`,
//!
//! ```text
//! Σ_jk = Σ_m Φ_jm Φ_km + Σ_i S_ji S_ki − Σ_(i,m) S_j(i,m) S_k(i,m)
//! ```
//!
//! the SE's three sums ([`super::clusters`]) joined across estimates: two
//! entries count once when they share an MRCA or a member, whichever
//! requests they are in, and `Σ_jj` is `j`'s meat.  Each sweep keeps its
//! covered estimates' sums in an [`Emission`], one key-sorted [`Tally`] per
//! grouping in the order the sweep forms them.  [`combine`] merge-joins
//! every two sources' tallies in key order, one task per block, and adds
//! the blocks in a fixed order, so no value depends on the threads or on
//! how the sweeps were batched.  The Within-Person cell's source holds its
//! rows' influences by person alone (`within_person`): a row meets a pair
//! of another cell when it is one of its members.

use super::clusters::{Sink, Sum};
use super::influence::Influence;
use super::mrca::Cluster;
use super::{Cell, Estimator, Fit, SeReason};
use crate::error::Error;
use crate::value::Value;
use pedigree_graph_core::error::Error as PgError;
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::ops::Range;

/// The operation the covariance's budget and allocation errors name.
pub(super) const OPERATION: &str = "relative_pairs_covariance";

/// Which estimate of a cell the ACE Fit reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Form {
    /// The pooled primary estimate.
    Pooled,
    /// The pooled phi, binary traits only.
    Phi,
    /// The stratified fit, with strata only.
    Stratified,
}

impl Form {
    pub const ALL: [Form; 3] = [Form::Pooled, Form::Phi, Form::Stratified];

    pub fn name(self) -> &'static str {
        match self {
            Form::Pooled => "pooled",
            Form::Phi => "phi",
            Form::Stratified => "stratified",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// The covariance of a result's estimates (ADR 0014): per form, a
/// symmetric `n x n` matrix, row-major, over the result's cells in order
/// and then, with two traits, the within-person cell.  Its diagonal is
/// each estimate's meat, so its square root is the SE bit for bit; the row
/// and column of an estimate with no SE are NaN.  `phi` is present for
/// binary traits and `stratified` with strata.  Two are equal when every
/// entry has the same bits, NaN included.
#[derive(Clone, Debug)]
pub struct Covariance {
    n: usize,
    /// By [`Form::index`].
    matrices: [Option<Box<[f64]>>; 3],
}

impl PartialEq for Covariance {
    fn eq(&self, other: &Covariance) -> bool {
        let bits = |m: &[f64]| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        self.n == other.n
            && Form::ALL
                .iter()
                .all(|&form| match (self.of(form), other.of(form)) {
                    (Some(a), Some(b)) => bits(a) == bits(b),
                    (a, b) => a.is_none() && b.is_none(),
                })
    }
}

impl Covariance {
    /// A host's matrices over `n` cells, each `n²` long and row-major.
    ///
    /// # Errors
    ///
    /// `covariance_shape_mismatch` for a matrix of another length.
    pub fn new(
        n: usize,
        pooled: Vec<f64>,
        phi: Option<Vec<f64>>,
        stratified: Option<Vec<f64>>,
    ) -> Result<Covariance, Error> {
        let len = n.saturating_mul(n);
        let matrices = [Some(pooled), phi, stratified];
        if let Some(m) = matrices.iter().flatten().find(|m| m.len() != len) {
            return Err(Error::CovarianceShapeMismatch {
                cells: n,
                expected_length: len,
                actual_length: m.len(),
            });
        }
        Ok(Covariance {
            n,
            matrices: matrices.map(|m| m.map(Vec::into_boxed_slice)),
        })
    }

    /// The cells it covers: the result's, and the within-person cell's.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The matrix of `form`, row-major; `None` when the result has none.
    pub fn of(&self, form: Form) -> Option<&[f64]> {
        self.matrices[form.index()].as_deref()
    }

    /// The covariance as a host-neutral tree: `n`, then each form's matrix
    /// row-major, `None` when absent.
    pub(super) fn to_value(&self) -> Value {
        let mut entries = vec![("n", (self.n as u64).into())];
        entries.extend(Form::ALL.map(|form| {
            let matrix = self.of(form).map(|m| Value::Floats(m.to_vec()));
            (form.name(), matrix.into())
        }));
        Value::Map(entries)
    }
}

/// One estimate the covariance covers: its matrix and its slot there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Column {
    pub(super) form: Form,
    pub(super) cell: u32,
}

/// A covered estimate of a request: its influence's index in the
/// request's list, and its column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Slot {
    pub(super) influence: u32,
    pub(super) column: Column,
}

/// The covered estimates of the cell at slot `cell` whose influences are
/// defined in its request's `influences`: the pooled primary, the pooled
/// phi and the stratified fit.
pub(super) fn slots<'a>(
    cell: u32,
    fit: &'a Cell<Fit>,
    influences: &'a [Result<Influence, SeReason>],
) -> impl Iterator<Item = Slot> + 'a {
    let phi = fit
        .estimates
        .iter()
        .find(|f| f.estimate.estimator == Estimator::Phi);
    let stratified = fit.stratified.as_ref().map(|s| &s.estimate);
    [
        (Form::Pooled, fit.estimates.first()),
        (Form::Phi, phi),
        (Form::Stratified, stratified),
    ]
    .into_iter()
    .filter_map(move |(form, fit)| {
        let influence = fit?.influence?;
        influences.get(influence)?.as_ref().ok()?;
        Some(Slot {
            influence: influence as u32,
            column: Column { form, cell },
        })
    })
}

/// The groupings, by index into a source's tallies: its key and its sign
/// in `Σ`.
const CLUSTER: usize = 0;
const PERSON: usize = 1;
const GROUP: usize = 2;

/// Keys for each grouping a request's tallies may hold: its clusters and
/// its (person, cluster) groups each at most its rows, its persons at most
/// its rows and the members.
pub(super) fn records(rows: usize, n_rows: usize) -> [usize; 3] {
    let mut out = [rows; 3];
    out[PERSON] = rows.min(n_rows);
    out
}

/// The bytes a request's tallies hold at most: `records` keys a grouping
/// in each of `sources` sweeps, and with them its `covered` estimates'
/// sums.
pub(super) fn tally_bytes(records: [usize; 3], covered: usize, sources: usize) -> u64 {
    let per_key = 8u64.saturating_mul(covered.saturating_add(sources) as u64);
    records
        .iter()
        .map(|&r| (r as u64).saturating_mul(per_key))
        .fold(0, u64::saturating_add)
}

/// The bytes the matrices and the combine's blocks hold: each grouping's
/// blocks fill at most one matrix per form.
pub(super) fn reserve(n: usize, forms: [bool; 3]) -> u64 {
    let matrix = (n as u64).saturating_mul(n as u64).saturating_mul(8);
    let present = forms.iter().filter(|&&f| f).count() as u64;
    matrix.saturating_mul(present).saturating_mul(4)
}

/// One grouping's sums by key, keys ascending and unique; `sums` is
/// row-major `[keys, width]`.
#[derive(Debug, Default)]
struct Tally {
    keys: Vec<u64>,
    sums: Vec<f64>,
}

/// One sweep's covered sums, its columns in order.
#[derive(Debug)]
pub(super) struct Source {
    columns: Box<[Column]>,
    tallies: [Tally; 3],
}

impl Source {
    /// A source that meets others at its members alone: `columns`' sums by
    /// person, `keys` ascending and `sums` row-major `[keys, columns]`.
    pub(super) fn persons(columns: Box<[Column]>, keys: Vec<u64>, sums: Vec<f64>) -> Source {
        debug_assert!(columns.is_sorted() && keys.is_sorted());
        debug_assert_eq!(keys.len() * columns.len(), sums.len());
        let mut tallies: [Tally; 3] = Default::default();
        tallies[PERSON] = Tally { keys, sums };
        Source { columns, tallies }
    }

    /// Its per-person keys and row-major sums.
    #[cfg(test)]
    pub(super) fn persons_tally(&self) -> (&[u64], &[f64]) {
        let tally = &self.tallies[PERSON];
        (&tally.keys, &tally.sums)
    }

    /// The bytes its tallies hold.
    pub(super) fn bytes(&self) -> u64 {
        let held = |t: &Tally| t.keys.capacity() * 8 + t.sums.capacity() * 8;
        self.tallies.iter().map(held).sum::<usize>() as u64
    }

    /// Each form's columns, a range since they are sorted.
    fn spans(&self) -> [Range<usize>; 3] {
        Form::ALL.map(|form| {
            let start = self.columns.partition_point(|c| c.form < form);
            start..self.columns.partition_point(|c| c.form <= form)
        })
    }
}

/// The sink a covariance sweep gets: it keeps its covered estimates' sums.
pub(super) struct Emission {
    /// Each covered estimate's offset in the sweep's range, in column
    /// order.
    picks: Box<[usize]>,
    columns: Box<[Column]>,
    tallies: [Tally; 3],
    /// The elements the allocator refused, set instead of failing inside
    /// the sweep.
    refused: Option<usize>,
}

impl Emission {
    /// The sink for a sweep of the estimates in `range` of a request whose
    /// covered estimates are `slots`, each tally holding up to `records`
    /// keys ([`records`]).  It allocates nothing and keeps nothing when the
    /// range covers none.
    pub(super) fn new(slots: &[Slot], range: Range<usize>, records: [usize; 3]) -> Emission {
        let mut picked: Vec<(Column, usize)> = slots
            .iter()
            .filter(|s| range.contains(&(s.influence as usize)))
            .map(|s| (s.column, s.influence as usize - range.start))
            .collect();
        picked.sort_unstable();
        let width = picked.len();
        let mut tallies: [Tally; 3] = Default::default();
        let mut refused = None;
        if width > 0 {
            for (tally, &records) in tallies.iter_mut().zip(&records) {
                let sums = records.saturating_mul(width);
                if tally.keys.try_reserve_exact(records).is_err()
                    || tally.sums.try_reserve_exact(sums).is_err()
                {
                    refused = Some(sums);
                    break;
                }
            }
        }
        Emission {
            picks: picked.iter().map(|&(_, p)| p).collect(),
            columns: picked.iter().map(|&(c, _)| c).collect(),
            tallies,
            refused,
        }
    }

    /// The sweep's source; `None` when it covers no estimate.
    ///
    /// # Errors
    ///
    /// `allocation_failed` (operation [`OPERATION`]) when the allocator
    /// refused the tallies.
    pub(super) fn finish(self) -> Result<Option<Source>, Error> {
        if let Some(requested_elements) = self.refused {
            return Err(PgError::AllocationFailed {
                operation: OPERATION,
                requested_elements,
                dtype: "float64",
            }
            .into());
        }
        if self.picks.is_empty() {
            return Ok(None);
        }
        Ok(Some(Source {
            columns: self.columns,
            tallies: self.tallies,
        }))
    }

    /// `key`'s sums of the picked estimates, each times `scale`, unless no
    /// picked estimate's entries reached it.
    #[inline]
    fn push(&mut self, grouping: usize, key: u64, sums: &[Sum], scale: f64) {
        if self.refused.is_some() || self.picks.iter().all(|&p| sums[p].entries() == 0) {
            return;
        }
        let tally = &mut self.tallies[grouping];
        debug_assert!(tally.keys.last().is_none_or(|&last| last < key));
        tally.keys.push(key);
        tally
            .sums
            .extend(self.picks.iter().map(|&p| sums[p].value() * scale));
    }
}

impl Sink for Emission {
    fn group(&mut self, cluster: Cluster, person: u32, sums: &[Sum]) {
        self.push(
            GROUP,
            u64::from(cluster.0) << 32 | u64::from(person),
            sums,
            1.0,
        );
    }

    /// Halved, as the meat's `Φ_m`: an entry adds a row per member.
    fn cluster(&mut self, cluster: Cluster, sums: &[Sum]) {
        self.push(CLUSTER, u64::from(cluster.0), sums, 0.5);
    }

    fn persons(&mut self, totals: &[Sum], width: usize) {
        for (person, sums) in totals.chunks_exact(width).enumerate() {
            self.push(PERSON, person as u64, sums, 1.0);
        }
    }
}

/// `Σ_jk` off the diagonal for every two covered estimates of one form:
/// per two sources `s ≤ t` and grouping, `Σ_key a_j b_k` over the keys
/// both tallies hold, in key order, one task each; then
/// `(clusters + persons) − groups` per entry.  A form absent from `forms`
/// gets no matrix; entries no task writes are 0.
pub(super) fn combine(sources: &[Source], n: usize, forms: [bool; 3]) -> [Option<Vec<f64>>; 3] {
    let spans: Vec<[Range<usize>; 3]> = sources.iter().map(Source::spans).collect();
    let pairs: Vec<(usize, usize)> = (0..sources.len())
        .flat_map(|s| (s..sources.len()).map(move |t| (s, t)))
        .collect();
    let tasks: Vec<(usize, usize)> = (0..3)
        .flat_map(|g| (0..pairs.len()).map(move |p| (g, p)))
        .collect();
    let mut order: Vec<usize> = (0..tasks.len()).collect();
    order.sort_by_key(|&i| {
        let (g, p) = tasks[i];
        let (s, t) = pairs[p];
        Reverse(sources[s].tallies[g].keys.len() + sources[t].tallies[g].keys.len())
    });
    let done: Vec<(usize, Vec<f64>)> = order
        .par_iter()
        .map(|&i| {
            let (g, p) = tasks[i];
            let (s, t) = pairs[p];
            (i, join(&sources[s], &sources[t], g, &spans[s], &spans[t]))
        })
        .collect();
    let mut blocks = vec![Vec::new(); tasks.len()];
    for (i, block) in done {
        blocks[i] = block;
    }
    let mut sigma = forms.map(|on| on.then(|| vec![0.0; n * n]));
    for (p, &(s, t)) in pairs.iter().enumerate() {
        let at = |g: usize| &blocks[g * pairs.len() + p];
        let (c, q, x) = (at(CLUSTER), at(PERSON), at(GROUP));
        let mut i = 0;
        for form in Form::ALL {
            let f = form.index();
            for a in spans[s][f].clone() {
                for b in spans[t][f].clone() {
                    let v = (c[i] + q[i]) - x[i];
                    i += 1;
                    let Some(m) = &mut sigma[f] else { continue };
                    let (j, k) = (
                        sources[s].columns[a].cell as usize,
                        sources[t].columns[b].cell as usize,
                    );
                    m[j * n + k] = v;
                    m[k * n + j] = v;
                }
            }
        }
    }
    sigma
}

/// Grouping `g`'s block of sources `s` and `t`: per form, `s`'s columns of
/// it by `t`'s, row-major, each `Σ_key a_j b_k` over the keys both hold.
fn join(
    s: &Source,
    t: &Source,
    g: usize,
    ss: &[Range<usize>; 3],
    ts: &[Range<usize>; 3],
) -> Vec<f64> {
    let (a, b) = (&s.tallies[g], &t.tallies[g]);
    let (wa, wb) = (s.columns.len(), t.columns.len());
    let size = (0..3).map(|f| ss[f].len() * ts[f].len()).sum();
    let mut out = vec![0.0; size];
    let (mut i, mut j) = (0, 0);
    while i < a.keys.len() && j < b.keys.len() {
        match a.keys[i].cmp(&b.keys[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                let (x, y) = (&a.sums[i * wa..][..wa], &b.sums[j * wb..][..wb]);
                let mut at = 0;
                for f in 0..3 {
                    for &u in &x[ss[f].clone()] {
                        for &v in &y[ts[f].clone()] {
                            out[at] += u * v;
                            at += 1;
                        }
                    }
                }
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// The result's covariance: `sigma` from [`combine`], each column's meat
/// in `meats` on the diagonal, and NaN in the row and column of every slot
/// `meats` does not give.
pub(super) fn finish(
    sigma: [Option<Vec<f64>>; 3],
    n: usize,
    meats: impl IntoIterator<Item = (Column, f64)>,
) -> Covariance {
    let mut diagonal = [(); 3].map(|_| vec![None; n]);
    for (column, meat) in meats {
        diagonal[column.form.index()][column.cell as usize] = Some(meat);
    }
    let mut forms = diagonal.iter();
    let matrices = sigma.map(|matrix| {
        let diagonal = forms.next()?;
        let mut m = matrix?;
        for (i, meat) in diagonal.iter().enumerate() {
            match meat {
                Some(meat) => m[i * n + i] = *meat,
                None => {
                    for j in 0..n {
                        m[i * n + j] = f64::NAN;
                        m[j * n + i] = f64::NAN;
                    }
                }
            }
        }
        Some(m.into_boxed_slice())
    });
    Covariance { n, matrices }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source over `columns` with each grouping's `(key, sums)` records.
    fn source(columns: &[(Form, u32)], tallies: [Vec<(u64, Vec<f64>)>; 3]) -> Source {
        let columns: Box<[Column]> = columns
            .iter()
            .map(|&(form, cell)| Column { form, cell })
            .collect();
        assert!(columns.is_sorted());
        let tallies = tallies.map(|records| {
            assert!(records.is_sorted_by_key(|r| r.0));
            Tally {
                keys: records.iter().map(|r| r.0).collect(),
                sums: records.iter().flat_map(|r| r.1.clone()).collect(),
            }
        });
        Source { columns, tallies }
    }

    /// `Σ` by definition: every two columns of one form, each grouping's
    /// `Σ_key a b` over a dense map of every key, then `(C + P) − G`.
    fn dense(sources: &[Source], n: usize, forms: [bool; 3]) -> [Option<Vec<f64>>; 3] {
        let value = |s: &Source, g: usize, col: usize, key: u64| {
            let t = &s.tallies[g];
            t.keys
                .iter()
                .position(|&k| k == key)
                .map_or(0.0, |at| t.sums[at * s.columns.len() + col])
        };
        let mut sigma = forms.map(|on| on.then(|| vec![0.0; n * n]));
        let keys: Vec<u64> = (0..64).collect();
        for s in sources {
            for t in sources {
                for (a, ca) in s.columns.iter().enumerate() {
                    for (b, cb) in t.columns.iter().enumerate() {
                        if ca.form != cb.form {
                            continue;
                        }
                        let term = |g: usize| -> f64 {
                            keys.iter()
                                .map(|&k| value(s, g, a, k) * value(t, g, b, k))
                                .sum()
                        };
                        let v = term(CLUSTER) + term(PERSON) - term(GROUP);
                        if let Some(m) = &mut sigma[ca.form.index()] {
                            m[ca.cell as usize * n + cb.cell as usize] = v;
                        }
                    }
                }
            }
        }
        sigma
    }

    /// Deterministic noise in `[-1, 1)`.
    fn noise(i: u64) -> f64 {
        ((i.wrapping_mul(2_654_435_761) % 1000) as f64) / 500.0 - 1.0
    }

    /// Records at every key below 64 that `keep` admits, `width` sums each.
    fn filled(width: usize, seed: u64, keep: impl Fn(u64) -> bool) -> Vec<(u64, Vec<f64>)> {
        (0..64)
            .filter(|&k| keep(k))
            .map(|k| {
                let sums = (0..width as u64)
                    .map(|c| noise(seed * 1000 + k * 10 + c))
                    .collect();
                (k, sums)
            })
            .collect()
    }

    #[test]
    fn combine_is_the_dense_sum_over_every_key_two_sources_share() {
        let n = 7;
        let forms = [true, false, true];
        let a = [(Form::Pooled, 0), (Form::Pooled, 2), (Form::Stratified, 0)];
        let b = [
            (Form::Pooled, 1),
            (Form::Stratified, 1),
            (Form::Stratified, 5),
        ];
        let c = [(Form::Pooled, 6)];
        let sources = vec![
            source(
                &a,
                [
                    filled(3, 1, |k| k % 3 == 0),
                    filled(3, 2, |k| k % 2 == 0),
                    filled(3, 3, |k| k % 5 != 1),
                ],
            ),
            source(
                &b,
                [
                    filled(3, 4, |k| k % 4 == 0),
                    filled(3, 5, |k| k % 3 != 2),
                    filled(3, 6, |k| k > 20),
                ],
            ),
            source(
                &c,
                [filled(1, 7, |k| k < 30), filled(1, 8, |_| true), Vec::new()],
            ),
        ];
        let got = combine(&sources, n, forms);
        let want = dense(&sources, n, forms);
        for f in 0..3 {
            let (Some(got), Some(want)) = (&got[f], &want[f]) else {
                assert!(got[f].is_none() && want[f].is_none());
                continue;
            };
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert!((g - w).abs() <= 1e-12 * (1.0 + w.abs()), "{f} {i}: {g} {w}");
                let (j, k) = (i / n, i % n);
                assert_eq!(g.to_bits(), got[k * n + j].to_bits(), "{f} ({j}, {k})");
            }
        }
        let pooled = got[0].as_ref().unwrap();
        assert!(pooled[n + 6] != 0.0, "sources 1 and 2 share keys");
        assert_eq!(pooled[3 * n + 4], 0.0, "slots no source covers");
    }

    /// Batching splits a request's columns over sources, and a sweep keeps
    /// only the keys its own columns reached: a key one batch holds and
    /// the other does not adds an exact zero, so `Σ` keeps its bits.
    #[test]
    fn combine_does_not_depend_on_how_columns_are_split() {
        let n = 4;
        let forms = [true, true, false];
        let whole = [
            (Form::Pooled, 0),
            (Form::Pooled, 1),
            (Form::Phi, 2),
            (Form::Phi, 3),
        ];
        let reached = |k: u64, c: usize| {
            (k + c as u64) % 4 != 0 && !(k % 5 == 0 && c % 2 == 0) && !(k % 7 == 0 && c % 2 == 1)
        };
        let full = [1, 2, 3].map(|seed| filled(4, seed, |_| true));
        let kept = |cols: &[usize]| {
            full.clone().map(|records| {
                records
                    .into_iter()
                    .filter(|&(k, _)| cols.iter().any(|&c| reached(k, c)))
                    .map(|(k, sums)| {
                        let at = |c: usize| if reached(k, c) { sums[c] } else { 0.0 };
                        (k, cols.iter().map(|&c| at(c)).collect())
                    })
                    .collect::<Vec<_>>()
            })
        };
        let one = [source(&whole, kept(&[0, 1, 2, 3]))];
        let two = [
            source(&[whole[0], whole[2]], kept(&[0, 2])),
            source(&[whole[1], whole[3]], kept(&[1, 3])),
        ];
        for g in 0..3 {
            assert!(two[0].tallies[g].keys.len() < one[0].tallies[g].keys.len());
        }
        let (one, two) = (combine(&one, n, forms), combine(&two, n, forms));
        for f in 0..2 {
            let (one, two) = (one[f].as_ref().unwrap(), two[f].as_ref().unwrap());
            for (i, (a, b)) in one.iter().zip(two).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{f} {i}");
            }
            let cross = [1, 2 * n + 3][f];
            assert!(one[cross] != 0.0, "the batches' columns share keys");
        }
    }

    #[test]
    fn finish_puts_the_meats_on_the_diagonal_and_nan_where_there_is_none() {
        let n = 3;
        let sigma = [Some(vec![0.5; 9]), None, Some(vec![0.25; 9])];
        let meats = [
            (
                Column {
                    form: Form::Pooled,
                    cell: 0,
                },
                4.0,
            ),
            (
                Column {
                    form: Form::Pooled,
                    cell: 2,
                },
                9.0,
            ),
            (
                Column {
                    form: Form::Stratified,
                    cell: 1,
                },
                1.0,
            ),
        ];
        let cov = finish(sigma, n, meats);
        let pooled = cov.of(Form::Pooled).unwrap();
        let nan = f64::NAN;
        let expect = [4.0, nan, 0.5, nan, nan, nan, 0.5, nan, 9.0];
        assert!(pooled
            .iter()
            .zip(expect)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(cov.of(Form::Phi).is_none());
        let stratified = cov.of(Form::Stratified).unwrap();
        assert_eq!(stratified.iter().filter(|v| !v.is_nan()).count(), 1);
        assert_eq!(stratified[n + 1], 1.0);
        assert_eq!(cov, cov.clone());
        let rebuilt = Covariance::new(n, pooled.to_vec(), None, Some(stratified.to_vec()));
        assert_eq!(rebuilt.as_ref(), Ok(&cov));
        assert!(Covariance::new(n, vec![0.0; 8], None, None).is_err());
        assert!(Covariance::new(n, vec![0.0; 9], Some(vec![0.0; 4]), None).is_err());
    }

    #[test]
    fn the_tally_bound_covers_every_grouping() {
        assert_eq!(records(10, 4), [10, 4, 10]);
        assert_eq!(tally_bytes(records(10, 4), 3, 2), 24 * 8 * 5);
        assert_eq!(reserve(5, [true, false, true]), 5 * 5 * 8 * 2 * 4);
    }
}
