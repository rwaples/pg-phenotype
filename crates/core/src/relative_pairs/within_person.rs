//! The Within-Person Cross-Trait Correlation's SE (ADR 0014), which comes
//! with the covariance.  A row is one entry, its own pair; its influence
//! `ψ_i` is its stride on its own engine cell under [`Weights`].  Rows
//! that are relatives are not independent, so the meat is the kinship
//! kernel
//!
//! ```text
//! meat = Σ_i ψ_i² + 2 Σ_(a<b) ψ_a ψ_b
//! ```
//!
//! over every relative pair of degree at most [`DEGREE`] whose members are
//! both entries, whichever categories the call asked for.  Each estimate's
//! `ψ` is quantized to integers of at most `2^40` before the pass, so the
//! pairs' products are summed exactly in `i128` lanes, merged by addition,
//! and become one float each after the pass.  A binary estimate's `ψ` is a
//! constant per engine label, so the table's counts give the requested
//! categories' terms and only the others ride the pass.
//!
//! An estimate reads the rows of the strata its blocks hold: the pooled and
//! stratified fits several, a stratum cell's one.  A row keeps a slot per
//! estimate that reads its stratum, those reading several first, so a pair
//! of two strata adds only the leading slots.

use super::clusters::{Kin, Meat};
use super::covariance::{self, Column, Slot, Source};
use super::influence::{Influence, Weight, Weights};
use super::{Members, SeReason};
use crate::error::Error;
use crate::input::TraitKind;
use pedigree_graph_core::error::Error as PgError;
use pedigree_graph_core::relationships::{Category, CategorySet, MomentsTable, N_CATEGORIES};

/// The kernel's deepest degree: MZ, MO, FO, FS, MHS, PHS, GP and Av, the
/// prototype's `kin2`, which calibrated (SD/SE 1.00-1.07).
pub(super) const DEGREE: u8 = 2;

/// The bits a quantized `ψ` may use: `|ψ̃| <= 2^40`, so a product is at
/// most `2^80`.
const BITS: i32 = 40;

/// The most entry terms the sums may take, each pair's counted as it is
/// added (twice, symmetric or not): each adds at most `2^80`, so a lane's
/// sum and the merged one stay within `2^126`, with `Σ ψ̃²` under `2^112`.
const MAX_TERMS: u64 = 1 << 46;

/// No stratum code: a label not both traits show, or an empty slot.
const NONE: u32 = u32::MAX;

/// The kernel's categories.
pub(super) fn categories() -> CategorySet {
    CategorySet::up_to_degree(DEGREE)
}

/// The within-person entries' kinship kernel, quantized before the pass.
pub(super) struct Kernel {
    /// Per engine label, its stratum code when both traits are observed,
    /// else [`NONE`].
    codes: Box<[u32]>,
    n_codes: usize,
    /// Each entry's quantized `ψ`, row-major `[index, width]`: indexed by
    /// row, or for binary traits by engine label.
    psi: Box<[i64]>,
    by_label: bool,
    width: usize,
    /// The leading slots: estimates that read several strata.
    shared: usize,
    /// Per lane sum, its estimate: the shared slots', then per code its
    /// own; [`NONE`] where a code has fewer own.
    estimates: Box<[u32]>,
    /// Per category, how many times an entry the engine sends counts in
    /// the pass: a symmetric category's pair comes in both orders, a
    /// directed one's once; 0 when the pass does not sum it.
    per_pair: [u8; N_CATEGORIES],
    /// Per category, traversed for the kernel alone.
    only: [bool; N_CATEGORIES],
    /// The categories the engine table's counts give.
    from_table: CategorySet,
    /// Per estimate: its quantization exponent, `Σ ψ̃²`, its rows, and
    /// whether every `ψ` was finite.
    exponents: Box<[i32]>,
    squares: Box<[i128]>,
    entries: Box<[u64]>,
    finite: Box<[bool]>,
    /// `[codes, estimates]`: whether the estimate reads the code's rows.
    reads: Box<[bool]>,
    /// Per influence, why it has none.
    reasons: Box<[Option<SeReason>]>,
}

/// One lane's kernel sums: per slot of [`Kernel::estimates`], `Σ ψ̃_a ψ̃_b`
/// over its terms, and per (code, code) its terms.
pub(super) struct KernelLane {
    sums: Vec<i128>,
    terms: Vec<u64>,
}

impl Kernel {
    /// The kernel of the within-person `influences` (in the order its fits
    /// made them), the call's requested categories `union`, and the source
    /// of its estimates the covariance covers, `covered`, keyed by row.
    ///
    /// # Errors
    ///
    /// `memory_budget_exceeded` (operation `relative_pairs_covariance`)
    /// before allocating when the weights or the quantized `ψ` and the
    /// source exceed `budget`.
    pub(super) fn new(
        members: &Members,
        influences: &[Result<Influence, SeReason>],
        union: CategorySet,
        covered: &[Slot],
        budget: u64,
    ) -> Result<(Kernel, Option<Source>), Error> {
        let layout = members.layout;
        let (n_codes, n_joint) = (layout.n_codes, layout.n_joint());
        let n_labels = n_codes * n_joint;
        let n_rows = members.labels.len();
        let n_estimates = influences.len();
        let over = |estimated_bytes: u64| -> Error {
            PgError::MemoryBudgetExceeded {
                operation: covariance::OPERATION,
                estimated_bytes,
                budget_bytes: budget,
            }
            .into()
        };
        let weights_bytes = Weights::bytes(layout, influences);
        if weights_bytes > budget {
            return Err(over(weights_bytes));
        }
        let weights = Weights::build(layout, &members.exponents, influences);
        let codes: Box<[u32]> = (0..n_labels)
            .map(|label| {
                let joint = label % n_joint;
                let both = layout.shows(joint, 0) && layout.shows(joint, 1);
                if both {
                    (label / n_joint) as u32
                } else {
                    NONE
                }
            })
            .collect();
        let own_cell = |label: usize| weights.of(label * n_labels + label);
        let mut reads = vec![false; n_codes * n_estimates];
        for (label, &code) in codes.iter().enumerate().filter(|(_, &c)| c != NONE) {
            for w in own_cell(label) {
                reads[code as usize * n_estimates + w.estimate as usize] = true;
            }
        }
        let n_read: Vec<usize> = (0..n_estimates)
            .map(|e| (0..n_codes).filter(|&c| reads[c * n_estimates + e]).count())
            .collect();
        let shared: Vec<u32> = (0..n_estimates)
            .filter(|&e| n_read[e] > 1)
            .map(|e| e as u32)
            .collect();
        let own: Vec<Vec<u32>> = (0..n_codes)
            .map(|c| {
                (0..n_estimates)
                    .filter(|&e| reads[c * n_estimates + e] && n_read[e] == 1)
                    .map(|e| e as u32)
                    .collect()
            })
            .collect();
        let n_own = own.iter().map(Vec::len).max().unwrap_or(0);
        let width = shared.len() + n_own;
        let mut estimates = shared.clone();
        for list in &own {
            estimates.extend(
                list.iter()
                    .copied()
                    .chain(std::iter::repeat(NONE))
                    .take(n_own),
            );
        }
        // A row's slot of each estimate reading its code.
        let mut slot_of = vec![NONE; n_codes * n_estimates];
        for c in 0..n_codes {
            let mine = shared.iter().chain(&own[c]).enumerate();
            for (slot, &e) in mine.filter(|&(_, &e)| reads[c * n_estimates + e as usize]) {
                slot_of[c * n_estimates + e as usize] = slot as u32;
            }
        }
        let by_label = layout.kind == TraitKind::Binary;
        let index = if by_label { n_labels } else { n_rows };
        let psi_bytes = (index as u64)
            .saturating_mul(width as u64)
            .saturating_mul(8);
        let source_bytes = (n_rows as u64)
            .saturating_mul(8)
            .saturating_mul(1 + covered.len() as u64);
        let held = weights_bytes
            .saturating_add(psi_bytes)
            .saturating_add(source_bytes);
        if held > budget {
            return Err(over(held));
        }
        let rows = Rows {
            members,
            codes: &codes,
            weights: &weights,
            n_labels,
        };
        let mut largest = vec![0.0f64; n_estimates];
        let mut finite = vec![true; n_estimates];
        rows.each(|_, _, psi| {
            for (w, v) in psi {
                let e = w.estimate as usize;
                finite[e] &= v.is_finite();
                largest[e] = largest[e].max(v.abs());
            }
        });
        let exponents: Box<[i32]> = largest
            .iter()
            .map(|&m| if m > 0.0 { BITS - libm::frexp(m).1 } else { 0 })
            .collect();
        let mut psi = vec![0i64; index * width];
        let mut squares = vec![0i128; n_estimates];
        let mut entries = vec![0u64; n_estimates];
        let mut columns: Vec<(Column, usize)> = covered
            .iter()
            .map(|s| (s.column, s.influence as usize))
            .collect();
        columns.sort_unstable();
        let mut picks = vec![Vec::new(); n_estimates];
        for (c, &(_, e)) in columns.iter().enumerate() {
            picks[e].push(c);
        }
        let (mut keys, mut sums) = (Vec::new(), Vec::new());
        let mut picked = vec![None; columns.len()];
        rows.each(|row, label, values| {
            let code = codes[label] as usize;
            let at = if by_label { label } else { row } * width;
            picked.fill(None);
            for (w, v) in values {
                let e = w.estimate as usize;
                let q = if finite[e] {
                    libm::ldexp(v, exponents[e]).round() as i64
                } else {
                    0
                };
                psi[at + slot_of[code * n_estimates + e] as usize] = q;
                squares[e] += i128::from(q) * i128::from(q);
                entries[e] += 1;
                for &c in &picks[e] {
                    picked[c] = Some(v);
                }
            }
            if picked.iter().any(Option::is_some) {
                keys.push(row as u64);
                sums.extend(picked.iter().map(|v| v.unwrap_or(0.0)));
            }
        });
        let source = (!columns.is_empty()).then(|| {
            let columns = columns.iter().map(|&(c, _)| c).collect();
            Source::persons(columns, keys, sums)
        });
        let mut per_pair = [0u8; N_CATEGORIES];
        let mut only = [false; N_CATEGORIES];
        let mut from_table = CategorySet::EMPTY;
        for cat in categories().iter() {
            only[cat.index()] = !union.contains(cat);
            if by_label && union.contains(cat) {
                from_table.insert(cat);
            } else {
                per_pair[cat.index()] = if cat.symmetric() { 1 } else { 2 };
            }
        }
        let kernel = Kernel {
            codes,
            n_codes,
            psi: psi.into_boxed_slice(),
            by_label,
            width,
            shared: shared.len(),
            estimates: estimates.into_boxed_slice(),
            per_pair,
            only,
            from_table,
            exponents,
            squares: squares.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
            finite: finite.into_boxed_slice(),
            reads: reads.into_boxed_slice(),
            reasons: influences
                .iter()
                .map(|i| i.as_ref().err().copied())
                .collect(),
        };
        Ok((kernel, source))
    }

    /// Whether any category's terms ride the pass.
    pub(super) fn traverses(&self) -> bool {
        self.per_pair.iter().any(|&t| t > 0)
    }

    /// The bytes it holds through the pass, `lanes` lanes' sums among them,
    /// and after it the table counts of one category.
    pub(super) fn bytes(&self, lanes: usize) -> u64 {
        let lane = self.estimates.len() * 16 + self.n_codes * self.n_codes * 8;
        let per_estimate = 2 * size_of::<i128>()
            + 2 * size_of::<u64>()
            + size_of::<i32>()
            + size_of::<bool>()
            + size_of::<Option<SeReason>>();
        let held = self.psi.len() * 8
            + self.codes.len() * 4
            + self.estimates.len() * 4
            + self.reads.len()
            + self.reasons.len() * per_estimate;
        let table = if self.from_table.iter().next().is_some() {
            self.codes.len() * self.codes.len() * 24
        } else {
            0
        };
        (held + table) as u64 + (lanes as u64).saturating_mul(lane as u64)
    }

    fn empty_lane(&self) -> KernelLane {
        KernelLane {
            sums: vec![0; self.estimates.len()],
            terms: vec![0; self.n_codes * self.n_codes],
        }
    }

    /// `times` terms `ψ̃_a ψ̃_b` of entries `a` and `b` of codes `ca` and
    /// `cb`: the leading slots always, a code's own when they share it.
    ///
    /// The sums wrap rather than panic: past [`MAX_TERMS`] they may wrap,
    /// and [`Kernel::meats`] reads them only after checking the term count.
    /// In a release build the wrapping operations are the plain ones.
    #[inline(always)]
    fn add(&self, lane: &mut KernelLane, [a, b]: [usize; 2], [ca, cb]: [u32; 2], times: u64) {
        let w = self.width;
        let (pa, pb) = (&self.psi[a * w..][..w], &self.psi[b * w..][..w]);
        let terms = &mut lane.terms[ca as usize * self.n_codes + cb as usize];
        *terms = terms.wrapping_add(times);
        let t = i128::from(times);
        let (shared, own) = lane.sums.split_at_mut(self.shared);
        for ((s, &x), &y) in shared.iter_mut().zip(pa).zip(pb) {
            *s = s.wrapping_add(t.wrapping_mul(i128::from(x) * i128::from(y)));
        }
        if ca == cb {
            let n_own = w - self.shared;
            let own = &mut own[ca as usize * n_own..][..n_own];
            let (pa, pb) = (&pa[self.shared..], &pb[self.shared..]);
            for ((s, &x), &y) in own.iter_mut().zip(pa).zip(pb) {
                *s = s.wrapping_add(t.wrapping_mul(i128::from(x) * i128::from(y)));
            }
        }
    }

    /// Each influence's meat, or why it has none: `Σ ψ̃² +` the pass's
    /// `lane` (none when the kernel did not ride it) `+` the counts of the
    /// categories `table` holds (its categories `order`), as one float.
    /// Its scale is `(n + 2P) / n²` over the estimate's `n` rows and `P`
    /// pairs, the meat had each entry the influence `1/n`.
    ///
    /// # Errors
    ///
    /// `arithmetic_overflow` (operation `relative_pairs_covariance`) past
    /// [`MAX_TERMS`] terms, the pass's and the table's, where a sum may have
    /// wrapped; checked before the table's terms are added.
    pub(super) fn meats(
        &self,
        lane: Option<KernelLane>,
        table: &MomentsTable<'_>,
        order: &[Category],
    ) -> Result<Vec<Result<Meat, SeReason>>, Error> {
        let mut lane = lane.unwrap_or_else(|| self.empty_lane());
        let n_labels = self.codes.len();
        let mut counted = Vec::new();
        for (p, &cat) in order.iter().enumerate() {
            if !self.from_table.contains(cat) {
                continue;
            }
            let counts = table.select(0, &[p])?.counts();
            let times: u64 = if cat.symmetric() { 1 } else { 2 };
            for (la, &ca) in self.codes.iter().enumerate().filter(|(_, &c)| c != NONE) {
                for (lb, &cb) in self.codes.iter().enumerate().filter(|(_, &c)| c != NONE) {
                    let n = counts[la * n_labels + lb] as u64;
                    if n > 0 {
                        counted.push(([la, lb], [ca, cb], n.saturating_mul(times)));
                    }
                }
            }
        }
        let terms = lane
            .terms
            .iter()
            .chain(counted.iter().map(|(_, _, n)| n))
            .fold(0u64, |total, &t| total.saturating_add(t));
        if terms > MAX_TERMS {
            return Err(PgError::ArithmeticOverflow {
                operation: covariance::OPERATION,
                dtype: "int128",
            }
            .into());
        }
        for (at, codes, n) in counted {
            self.add(&mut lane, at, codes, n);
        }
        let n_estimates = self.reasons.len();
        let mut value = self.squares.to_vec();
        for (&e, &s) in self.estimates.iter().zip(&lane.sums) {
            if e != NONE {
                value[e as usize] += s;
            }
        }
        let mut pairs = vec![0u64; n_estimates];
        for (at, &n) in lane.terms.iter().enumerate().filter(|(_, &n)| n > 0) {
            let (s, u) = (at / self.n_codes, at % self.n_codes);
            let (rs, ru) = (
                &self.reads[s * n_estimates..][..n_estimates],
                &self.reads[u * n_estimates..][..n_estimates],
            );
            for ((p, &a), &b) in pairs.iter_mut().zip(rs).zip(ru) {
                if a && b {
                    *p += n;
                }
            }
        }
        Ok((0..n_estimates)
            .map(|e| {
                if let Some(reason) = self.reasons[e] {
                    return Err(reason);
                }
                let n = self.entries[e];
                let value = if self.finite[e] {
                    libm::ldexp(value[e] as f64, -2 * self.exponents[e])
                } else {
                    f64::NAN
                };
                let scale = if n > 0 {
                    (n as f64 + pairs[e] as f64) / (n as f64 * n as f64)
                } else {
                    0.0
                };
                Ok(Meat {
                    value,
                    scale,
                    entries: n,
                })
            })
            .collect())
    }

    /// The same kernel with every category's terms in the pass, none from
    /// the table: the per-pair route the table's counts must match.
    #[cfg(test)]
    pub(super) fn per_pair_everywhere(mut self) -> Kernel {
        for cat in categories().iter() {
            self.per_pair[cat.index()] = if cat.symmetric() { 1 } else { 2 };
        }
        self.from_table = CategorySet::EMPTY;
        self
    }
}

impl Kin for &Kernel {
    type Lane = KernelLane;

    fn lane(&self) -> KernelLane {
        self.empty_lane()
    }

    #[inline(always)]
    fn only(&self, cat: Category) -> bool {
        self.only[cat.index()]
    }

    #[inline(always)]
    fn pair(
        &self,
        lane: &mut KernelLane,
        cat: Category,
        first: u32,
        second: u32,
        labels: [u32; 2],
    ) {
        let times = self.per_pair[cat.index()];
        if times == 0 {
            return;
        }
        let codes = labels.map(|l| self.codes[l as usize]);
        if codes.contains(&NONE) {
            return;
        }
        let at = if self.by_label {
            labels
        } else {
            [first, second]
        };
        self.add(lane, at.map(|i| i as usize), codes, u64::from(times));
    }

    fn merge(&self, into: &mut KernelLane, from: KernelLane) {
        for (a, b) in into.sums.iter_mut().zip(from.sums) {
            *a = a.wrapping_add(b);
        }
        for (a, b) in into.terms.iter_mut().zip(from.terms) {
            *a = a.wrapping_add(b);
        }
    }

    fn traversal(&self, union: CategorySet) -> CategorySet {
        let mut out = union;
        for cat in categories().iter().filter(|&c| self.only[c.index()]) {
            out.insert(cat);
        }
        out
    }

    fn finish(lane: KernelLane) -> Option<KernelLane> {
        Some(lane)
    }
}

/// Each row both traits show, with its weights on its own engine cell.
struct Rows<'a> {
    members: &'a Members,
    codes: &'a [u32],
    weights: &'a Weights,
    n_labels: usize,
}

impl Rows<'_> {
    /// `f(row, label, ψ)` per row both traits show, `ψ` each of its own
    /// cell's weights with the row's influence on that estimate: the
    /// weight applied to the row's stride (count 1, its values on both
    /// sides, their squares, its products).
    fn each(&self, mut f: impl FnMut(usize, usize, &mut dyn Iterator<Item = (Weight, f64)>)) {
        let members = self.members;
        let k = members.exponents.len();
        let products = members.products();
        let mut stride = vec![0.0f64; 1 + 4 * k + products.len()];
        stride[0] = 1.0;
        for (row, &label) in members.labels.iter().enumerate() {
            let label = label as usize;
            if self.codes[label] == NONE {
                continue;
            }
            let x = &members.values[row * k..(row + 1) * k];
            for (c, &v) in x.iter().enumerate() {
                let (v, square) = (i128::from(v), i128::from(v) * i128::from(v));
                stride[1 + c] = v as f64;
                stride[1 + k + c] = v as f64;
                stride[1 + 2 * k + c] = square as f64;
                stride[1 + 3 * k + c] = square as f64;
            }
            for (p, product) in products.iter().enumerate() {
                let xy = i128::from(x[product.a.column]) * i128::from(x[product.b.column]);
                stride[1 + 4 * k + p] = xy as f64;
            }
            let own = self.weights.of(label * self.n_labels + label);
            f(row, label, &mut own.iter().map(|w| (*w, w.apply(&stride))));
        }
    }
}

const _: () = assert!(
    (MAX_TERMS as i128) << (2 * BITS) < i128::MAX - ((u32::MAX as i128) << (2 * BITS)),
    "a lane's sum and the squares cannot wrap"
);

#[cfg(test)]
mod tests {
    use super::super::clusters::Pass;
    use super::super::mrca::tests::{engine_pairs, generated, Columns};
    use super::super::mrca::Ancestry;
    use super::super::{covariance, default_requests, single, Blocks, Influences, Request};
    use super::*;
    use crate::input::Trait;
    use pedigree_graph_core::relationships::{MaxDegree, Symmetric};
    use std::num::NonZeroUsize;

    const BUDGET: u64 = 1 << 30;

    fn noise(i: usize) -> f64 {
        ((i * 7919 + 13) % 101) as f64 / 101.0
    }

    /// Two traits of `kind` on `cols`, about one row in nine missing each,
    /// with generation strata (every 13th unknown).
    fn members(cols: &Columns, kind: TraitKind) -> Members {
        let values: Vec<Vec<f64>> = (0..2)
            .map(|t| {
                (0..cols.ids.len())
                    .map(|i| match ((i + 4 * t) % 9, kind) {
                        (0, _) => f64::NAN,
                        (_, TraitKind::Binary) => f64::from(u8::from(noise(i + 31 * t) < 0.4)),
                        _ => noise(i + 31 * t) + 0.7 * noise(cols.mother[i].max(0) as usize),
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
        let k = cols.generation.iter().max().unwrap() + 1;
        let codes: Vec<Option<usize>> = (0..cols.ids.len())
            .map(|i| (i % 13 != 0).then_some(cols.generation[i] as usize))
            .collect();
        Members::of(&traits)
            .unwrap()
            .stratified(&codes, k as usize + 1)
    }

    /// The within-person influences and the kernel over `union`.
    fn kernel(members: &Members, union: CategorySet) -> (Vec<Result<Influence, SeReason>>, Kernel) {
        let labels: Vec<i64> = (0..members.layout.n_codes as i64 - 1).collect();
        let mut influences = Influences(Some(Vec::new()));
        let request = Request {
            name: String::new(),
            members: Vec::new(),
            symmetric: false,
            roles: None,
        };
        Blocks::within_person(members, BUDGET)
            .unwrap()
            .cell(request, Some(&labels), &mut influences)
            .unwrap();
        let influences = influences.0.unwrap();
        let (kernel, _) = Kernel::new(members, &influences, union, &[], BUDGET).unwrap();
        (influences, kernel)
    }

    /// The kernel's meats from a pass over `union` at `lanes` lanes, the
    /// kernel riding it when it sums any category there, and the table; with
    /// `ride` false the pass is the SE's alone, its meats short of the
    /// pass's terms.
    fn meats(
        cols: &Columns,
        members: &Members,
        kernel: &Kernel,
        union: CategorySet,
        lanes: usize,
        ride: bool,
    ) -> (Vec<Result<Meat, SeReason>>, Vec<u8>) {
        let pedigree = cols.pedigree();
        let ped = pedigree.relationships().unwrap();
        let ancestry = Ancestry::of(&pedigree, BUDGET).unwrap();
        let requests: Vec<Request> = union.iter().map(single).collect();
        let traversal = if ride {
            (&kernel).traversal(union)
        } else {
            union
        };
        let max_degree = MaxDegree::try_new(traversal.top_degree().unwrap()).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(lanes)
            .build()
            .unwrap();
        let pass = pool
            .install(|| {
                Pass::run(
                    &ped,
                    max_degree,
                    union,
                    &requests,
                    members,
                    &ancestry,
                    NonZeroUsize::new(lanes).unwrap(),
                    BUDGET,
                    0,
                    Some(kernel).filter(|k| ride && k.traverses()),
                )
            })
            .unwrap();
        let order: Vec<Category> = union.iter().collect();
        let meats = kernel.meats(pass.kernel, &pass.table, &order).unwrap();
        (meats, pass.table.bytes().to_vec())
    }

    #[test]
    fn a_binary_kernel_from_the_table_counts_is_the_per_pair_sums() {
        let cols = generated(300, 4, 11);
        let members = members(&cols, TraitKind::Binary);
        let union = default_requests()
            .iter()
            .flat_map(|r| r.members.clone())
            .collect();
        let (_, from_table) = kernel(&members, union);
        assert!(!from_table.traverses());
        let (expected, _) = meats(&cols, &members, &from_table, union, 3, true);
        let per_pair = kernel(&members, union).1.per_pair_everywhere();
        let (got, _) = meats(&cols, &members, &per_pair, union, 3, true);
        assert_eq!(got, expected);
        let finite = expected.iter().flatten().filter(|m| m.value > 0.0).count();
        assert!(finite > 5, "{finite} meats");
    }

    /// The exact sums against a float sum over the engine's pairs with the
    /// unquantized `ψ`, at 1, 3 and 8 lanes, bit for bit across lanes; a
    /// narrow request traverses the rest for the kernel alone and leaves
    /// its own table as it was.
    #[test]
    fn a_continuous_kernel_is_its_float_sum_whatever_the_lanes_and_the_request() {
        let cols = generated(300, 4, 11);
        let pedigree = cols.pedigree();
        let members = members(&cols, TraitKind::Continuous);
        let union: CategorySet = default_requests()
            .iter()
            .flat_map(|r| r.members.clone())
            .collect();
        let (influences, kernel) = kernel(&members, union);
        assert!(kernel.traverses() && (&kernel).traversal(union) == union);
        let peak = kernel.psi.iter().map(|q| q.unsigned_abs()).max().unwrap();
        assert!(peak <= 1 << BITS && peak > 1 << (BITS - 1), "{peak}");
        let layout = members.layout;
        let weights = Weights::build(layout, &members.exponents, &influences);
        let n_labels = layout.n_codes * layout.n_joint();
        let rows = Rows {
            members: &members,
            codes: &kernel.codes,
            weights: &weights,
            n_labels,
        };
        let n = influences.len();
        let mut psi = vec![vec![0.0; n]; members.labels.len()];
        rows.each(|row, _, values| {
            for (w, v) in values {
                psi[row][w.estimate as usize] = v;
            }
        });
        let mut reference: Vec<f64> = (0..n)
            .map(|e| psi.iter().map(|p| p[e] * p[e]).sum())
            .collect();
        for (cat, a, b) in engine_pairs(&pedigree, Symmetric::Both) {
            if cat.degree() > DEGREE {
                continue;
            }
            let times = if cat.symmetric() { 1.0 } else { 2.0 };
            for (e, r) in reference.iter_mut().enumerate() {
                *r += times * psi[a as usize][e] * psi[b as usize][e];
            }
        }
        let (whole, table) = meats(&cols, &members, &kernel, union, 1, true);
        let mut checked = 0;
        for (meat, r) in whole.iter().zip(&reference) {
            let Ok(meat) = meat else { continue };
            assert!(
                (meat.value - r).abs() <= 1e-11 * r.abs(),
                "{} {r}",
                meat.value
            );
            checked += 1;
        }
        assert!(checked >= 5, "{checked}");
        for lanes in [3, 8] {
            assert_eq!(
                meats(&cols, &members, &kernel, union, lanes, true).0,
                whole,
                "{lanes} lanes"
            );
        }
        let mz: CategorySet = [Category::MZ].into_iter().collect();
        let (_, narrow) = self::kernel(&members, mz);
        assert!((&narrow).traversal(mz) == categories());
        let (got, mz_table) = meats(&cols, &members, &narrow, mz, 3, true);
        assert_eq!(got, whole);
        let (short, mz_alone) = meats(&cols, &members, &narrow, mz, 3, false);
        assert_eq!(mz_table, mz_alone);
        assert_ne!(short, whole);
        assert_ne!(mz_table, table);
    }

    #[test]
    fn every_column_of_an_influence_takes_its_sums() {
        let cols = generated(60, 3, 5);
        let members = members(&cols, TraitKind::Continuous);
        let (influences, _) = kernel(&members, categories());
        let e = influences.iter().position(Result::is_ok).unwrap() as u32;
        let slot = |cell| Slot {
            influence: e,
            column: Column {
                form: covariance::Form::Pooled,
                cell,
            },
        };
        let source = |covered: &[Slot]| {
            Kernel::new(&members, &influences, categories(), covered, BUDGET)
                .unwrap()
                .1
                .unwrap()
        };
        let (one, two) = (source(&[slot(0)]), source(&[slot(0), slot(1)]));
        let ((keys, sums), (both_keys, both_sums)) = (one.persons_tally(), two.persons_tally());
        assert_eq!(keys, both_keys);
        let twice: Vec<f64> = sums.iter().flat_map(|&v| [v, v]).collect();
        assert_eq!(both_sums, twice);
        assert!(sums.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn sums_wrap_without_a_panic_and_the_term_count_refuses_them() {
        let cols = generated(40, 2, 3);
        let members = members(&cols, TraitKind::Continuous);
        let (_, kernel) = kernel(&members, categories());
        assert!(kernel.shared > 0);
        let w = kernel.width;
        let row = (0..kernel.psi.len() / w)
            .find(|&r| kernel.psi[r * w] != 0)
            .unwrap();
        let code = kernel.codes.iter().copied().find(|&c| c != NONE).unwrap();
        let (mut lane, mut other) = (kernel.empty_lane(), kernel.empty_lane());
        lane.sums[0] = i128::MAX;
        lane.terms[0] = MAX_TERMS;
        kernel.add(&mut lane, [row, row], [code, code], 1);
        other.sums[0] = i128::MAX;
        (&kernel).merge(&mut lane, other);
        let table = MomentsTable::from_i128(vec![1], 0, Vec::new(), Vec::new(), &[0]).unwrap();
        assert!(matches!(
            kernel.meats(Some(lane), &table, &[]),
            Err(Error::Pedigree(PgError::ArithmeticOverflow { .. }))
        ));
    }

    #[test]
    fn past_the_most_terms_the_sums_may_have_wrapped() {
        let cols = generated(40, 2, 3);
        let members = members(&cols, TraitKind::Continuous);
        let (_, kernel) = kernel(&members, categories());
        let mut lane = kernel.empty_lane();
        lane.terms[0] = MAX_TERMS;
        let table = MomentsTable::from_i128(vec![1], 0, Vec::new(), Vec::new(), &[0]).unwrap();
        assert!(kernel.meats(Some(lane), &table, &[]).is_ok());
        let mut lane = kernel.empty_lane();
        lane.terms[0] = MAX_TERMS + 1;
        match kernel.meats(Some(lane), &table, &[]) {
            Err(Error::Pedigree(PgError::ArithmeticOverflow { operation, dtype })) => {
                assert_eq!((operation, dtype), (covariance::OPERATION, "int128"));
            }
            other => panic!("{other:?}"),
        }
    }
}
