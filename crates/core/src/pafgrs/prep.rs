//! The trait-independent relative structure of one pedigree at one degree.
//!
//! A proband's relatives are the rows whose closest relationship category is
//! at most `ndegree` and whose exact kinship to it is at least
//! `0.5^(ndegree+1) - 1e-6` (ADR 0003).  The categories come from
//! pedigree-graph-core's relationship engine and every kinship from its
//! pairwise recurrence, on a pedigree its validator accepted (ADR 0001).
//!
//! A prep holds, per proband, its relatives in ascending row order with
//! their kinship to the proband, and the upper triangle of the kinship among
//! them.  Nothing here depends on a trait: unphenotyped relatives stay in and
//! take `w = 0` when a score is computed.
//!
//! Full siblings share nearly all their relatives, so probands are walked in
//! sibship groups: one triangle over the union of the group's candidates and
//! its members gives every member's kinship to its candidates and every
//! member's relative triangle.  Groups are packed into chunks of at least
//! 256 probands, which are the unit of parallel work and of
//! storage, so neither depends on the thread count.

use super::triangle::Triangle;
use crate::error::Error;
use crate::input::PedigreeInput;
use crate::lineage::Terminals;
use pedigree_graph_core::graph::{self, IdIndex};
use pedigree_graph_core::kinship::ancestry::AncestorSignatures;
use pedigree_graph_core::kinship::pairwise::Walker;
use pedigree_graph_core::kinship::KinshipPedigree;
use pedigree_graph_core::relationships::{
    pair_blocks, CategorySet, Execution, MaxDegree, Pedigree, Progress,
};
use pedigree_graph_core::topology;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// The deepest `ndegree` pedigree-graph-core classifies.
pub const MAX_NDEGREE: u8 = 5;

/// Probands per chunk, at least; a chunk ends on a group boundary.
const PROBAND_CHUNK: usize = 256;

/// Memo entries after which a worker starts a fresh walker.  Values do not
/// depend on the walker, so this bounds memory without changing a bit.
const MEMO_CAP: usize = 1 << 22;

/// The relative structure every score call of one pedigree reads.
#[derive(Debug)]
pub struct Prep {
    pub(crate) ids: Vec<i64>,
    pub(crate) ndegree: u8,
    pub(crate) chunks: Vec<Chunk>,
    /// `(chunk, member)` of each proband, in ascending row order.
    locate: Vec<(u32, u32)>,
}

/// The probands of consecutive sibship groups and their relatives.
#[derive(Debug)]
pub(crate) struct Chunk {
    /// Proband rows, group by group.
    pub(crate) rows: Vec<u32>,
    /// Each proband's position among all probands in ascending row order.
    pub(crate) out: Vec<u32>,
    /// `rel_start[m]..rel_start[m + 1]` are member `m`'s relatives.
    rel_start: Vec<u32>,
    /// Relative rows, ascending within each member.
    rel_row: Vec<u32>,
    /// Kinship of each relative to its proband.
    rel_kin: Vec<f32>,
    /// `tri_start[m]` is where member `m`'s triangle begins.
    tri_start: Vec<usize>,
    /// Per member, kinship of relatives `(j, k)`, `j < k`, row-major.
    tri: Triangle,
}

/// One proband's relatives, borrowed from its chunk.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Relatives<'a> {
    pub(crate) rows: &'a [u32],
    pub(crate) kin: &'a [f32],
    tri: &'a Triangle,
    tri_base: usize,
}

impl Relatives<'_> {
    /// Kinship between relatives `j < k`.
    #[inline]
    pub(crate) fn pair(&self, j: usize, k: usize) -> f32 {
        self.tri
            .get(self.tri_base + tri_index(self.rows.len(), j, k))
    }
}

impl Chunk {
    pub(crate) fn relatives(&self, m: usize) -> Relatives<'_> {
        let range = self.rel_start[m] as usize..self.rel_start[m + 1] as usize;
        Relatives {
            rows: &self.rel_row[range.clone()],
            kin: &self.rel_kin[range],
            tri: &self.tri,
            tri_base: self.tri_start[m],
        }
    }

    fn bytes(&self) -> usize {
        (self.rows.len() + self.out.len() + self.rel_start.len()) * 4
            + (self.rel_row.len() + self.rel_kin.len()) * 4
            + self.tri_start.len() * 8
            + self.tri.bytes()
    }
}

impl Prep {
    /// The number of pedigree rows.
    pub fn n_rows(&self) -> usize {
        self.ids.len()
    }

    /// Pedigree ids, one per input row.
    pub fn ids(&self) -> &[i64] {
        &self.ids
    }

    pub fn ndegree(&self) -> u8 {
        self.ndegree
    }

    pub fn n_probands(&self) -> usize {
        self.locate.len()
    }

    /// Proband rows in ascending (input) order.
    pub fn probands(&self) -> Vec<u32> {
        self.locate
            .iter()
            .map(|&(c, m)| self.chunks[c as usize].rows[m as usize])
            .collect()
    }

    pub(crate) fn proband(&self, i: usize) -> Relatives<'_> {
        let (c, m) = self.locate[i];
        self.chunks[c as usize].relatives(m as usize)
    }

    /// Proband `i`'s (ascending order) relative rows and their kinship to it.
    pub fn relatives(&self, i: usize) -> (&[u32], &[f32]) {
        let r = self.proband(i);
        (r.rows, r.kin)
    }

    /// Kinship between proband `i`'s relatives `j` and `k` (positions in
    /// [`Prep::relatives`], `j != k`).
    pub fn pair_kinship(&self, i: usize, j: usize, k: usize) -> f32 {
        let (j, k) = if j < k { (j, k) } else { (k, j) };
        self.proband(i).pair(j, k)
    }

    /// Bytes held by the relative lists and the triangles.
    pub fn bytes(&self) -> usize {
        self.ids.len() * 8
            + self.locate.len() * 8
            + self.chunks.iter().map(Chunk::bytes).sum::<usize>()
    }
}

/// Position of `(j, k)`, `j < k < d`, in a row-major upper triangle.
#[inline]
pub(crate) fn tri_index(d: usize, j: usize, k: usize) -> usize {
    j * d - j * (j + 1) / 2 + (k - j - 1)
}

/// The smallest kinship a relative at `ndegree` may have.
pub fn kinship_threshold(ndegree: u8) -> f64 {
    0.5f64.powi(i32::from(ndegree) + 1) - 1e-6
}

/// Validate the pedigree and build its relative structure.
///
/// `probands` lists the ids to score, or `None` for every row.
///
/// # Errors
///
/// [`Error::DegreeOutOfRange`] for `ndegree` outside `1..=5`; any
/// pedigree-graph-core validation error; [`Error::UnknownProband`] and
/// [`Error::DuplicateProband`] for the proband list.
pub fn prepare(
    input: PedigreeInput<'_>,
    ndegree: u8,
    probands: Option<&[i64]>,
) -> Result<Prep, Error> {
    if !(1..=MAX_NDEGREE).contains(&ndegree) {
        return Err(Error::DegreeOutOfRange {
            value: i64::from(ndegree),
            minimum: 1,
            maximum: i64::from(MAX_NDEGREE),
        });
    }
    let built = input.validate()?;
    let probands = proband_rows(&built.ids, probands)?;
    let candidates = candidates(&built, &probands, ndegree)?;
    let groups = sibship_groups(&built, &probands);

    let depth = topology::structural_depth(&built.mother_rows, &built.father_rows);
    let kped = KinshipPedigree::try_new(
        &built.mother_rows,
        &built.father_rows,
        &built.twin_rows,
        &depth,
    )?;
    let chunks = walk_chunks(
        kped,
        &Walk {
            probands: &probands,
            candidates: &candidates,
            threshold: kinship_threshold(ndegree),
        },
        &groups,
    )?;

    let mut locate = vec![(0u32, 0u32); probands.len()];
    for (c, chunk) in chunks.iter().enumerate() {
        for (m, &out) in chunk.out.iter().enumerate() {
            locate[out as usize] = (c as u32, m as u32);
        }
    }
    Ok(Prep {
        ids: built.ids,
        ndegree,
        chunks,
        locate,
    })
}

/// The proband rows, ascending: every row, or the rows of `ids`.
fn proband_rows(pedigree_ids: &[i64], ids: Option<&[i64]>) -> Result<Vec<u32>, Error> {
    let Some(ids) = ids else {
        return Ok((0..pedigree_ids.len() as u32).collect());
    };
    let index = IdIndex::build(pedigree_ids);
    let mut seen = vec![usize::MAX; pedigree_ids.len()];
    let mut rows = Vec::with_capacity(ids.len());
    for (position, &id) in ids.iter().enumerate() {
        let row = index.row(id);
        if row < 0 {
            return Err(Error::UnknownProband { id, position });
        }
        let row = row as usize;
        if seen[row] != usize::MAX {
            return Err(Error::DuplicateProband {
                id,
                positions: [seen[row], position],
            });
        }
        seen[row] = position;
        rows.push(row as u32);
    }
    rows.sort_unstable();
    Ok(rows)
}

/// Each proband's candidate relatives: the rows its closest category puts
/// at most `ndegree` away, ascending.  Indexed like the proband rows.
struct Candidates {
    start: Vec<usize>,
    rows: Vec<u32>,
}

impl Candidates {
    fn of(&self, i: usize) -> &[u32] {
        &self.rows[self.start[i]..self.start[i + 1]]
    }
}

fn candidates(
    built: &graph::PedigreeGraph,
    probands: &[u32],
    ndegree: u8,
) -> Result<Candidates, Error> {
    let ped = Pedigree::try_new(
        &built.mother_rows,
        &built.father_rows,
        &built.twin_rows,
        &built.mother_ids,
        &built.father_ids,
    )?;
    let blocks = pair_blocks(
        &ped,
        MaxDegree::try_new(ndegree)?,
        CategorySet::up_to_degree(ndegree),
        None,
        Execution::Speed,
        &Progress::default(),
    )?;
    let mut slot = vec![u32::MAX; built.len()];
    for (i, &row) in probands.iter().enumerate() {
        slot[row as usize] = i as u32;
    }
    let pairs = || {
        blocks
            .0
            .iter()
            .flat_map(|b| b.first.iter().zip(&b.second))
            .flat_map(|(&a, &b)| [(a as usize, b as u32), (b as usize, a as u32)])
            .filter(|&(p, _)| slot[p] != u32::MAX)
    };
    let mut start = vec![0usize; probands.len() + 1];
    for (p, _) in pairs() {
        start[slot[p] as usize + 1] += 1;
    }
    for i in 0..probands.len() {
        start[i + 1] += start[i];
    }
    let mut rows = vec![0u32; start[probands.len()]];
    let mut fill = start[..probands.len()].to_vec();
    for (p, r) in pairs() {
        let i = slot[p] as usize;
        rows[fill[i]] = r;
        fill[i] += 1;
    }
    drop(blocks);
    for w in start.windows(2) {
        rows[w[0]..w[1]].sort_unstable();
    }
    Ok(Candidates { start, rows })
}

/// Probands grouped by their two represented parents (full sibs and MZ
/// twins); a proband missing a parent is a group of one.  Groups are in
/// order of their first member, members ascending, as proband indices.
fn sibship_groups(built: &graph::PedigreeGraph, probands: &[u32]) -> Vec<Vec<u32>> {
    let mut groups: Vec<Vec<u32>> = Vec::new();
    let mut by_parents: HashMap<(i32, i32), usize> = HashMap::new();
    for (i, &row) in probands.iter().enumerate() {
        let (m, f) = (
            built.mother_rows[row as usize],
            built.father_rows[row as usize],
        );
        if m >= 0 && f >= 0 {
            let g = *by_parents.entry((m, f)).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(i as u32);
        } else {
            groups.push(vec![i as u32]);
        }
    }
    groups
}

/// What every chunk of one prep reads.
struct Walk<'a> {
    probands: &'a [u32],
    candidates: &'a Candidates,
    threshold: f64,
}

/// The recurrence's columns and the terminal sets, shared by every
/// worker's local DP.
#[derive(Clone, Copy)]
struct Recurrence<'a> {
    mother: &'a [i32],
    father: &'a [i32],
    twin: &'a [i32],
    depth: &'a [i32],
    terminals: &'a Terminals,
}

/// Chunks of consecutive groups, each built by one worker with its walker
/// and a scratch row-to-position map.
fn walk_chunks(
    kped: KinshipPedigree<'_>,
    walk: &Walk<'_>,
    groups: &[Vec<u32>],
) -> Result<Vec<Chunk>, Error> {
    let mut bounds = vec![0usize];
    let mut members = 0;
    for (g, group) in groups.iter().enumerate() {
        members += group.len();
        if members >= PROBAND_CHUNK || g + 1 == groups.len() {
            bounds.push(g + 1);
            members = 0;
        }
    }
    let n_chunks = bounds.len() - 1;
    let signatures = AncestorSignatures::build(&kped)?;
    let terminals = Terminals::build(kped.mother(), kped.father(), kped.twin(), kped.depth());
    let rec = Recurrence {
        mother: kped.mother(),
        father: kped.father(),
        twin: kped.twin(),
        depth: kped.depth(),
        terminals: &terminals,
    };
    let slots: Vec<Mutex<Option<Chunk>>> = (0..n_chunks).map(|_| Mutex::new(None)).collect();

    let next = AtomicUsize::new(0);
    let workers = rayon::current_num_threads().min(n_chunks);
    (0..workers).into_par_iter().try_for_each(|_| {
        let mut walker: Option<Walker<'_>> = None;
        let mut pos = vec![u32::MAX; kped.len()];
        loop {
            let c = next.fetch_add(1, Ordering::Relaxed);
            if c >= n_chunks {
                return Ok(());
            }
            if walker
                .as_ref()
                .is_none_or(|w| w.memo().entries() > MEMO_CAP)
            {
                walker = Some(Walker::new(kped, &signatures)?);
            }
            let Some(w) = walker.as_mut() else {
                return Ok(());
            };
            let mut builder = ChunkBuilder::new(w, rec, &mut pos, walk);
            let result = groups[bounds[c]..bounds[c + 1]]
                .iter()
                .try_for_each(|group| builder.group(group));
            match result {
                Ok(()) => {
                    *slots[c].lock().unwrap_or_else(|e| e.into_inner()) = Some(builder.finish());
                }
                Err(err) => {
                    next.fetch_max(n_chunks, Ordering::Relaxed);
                    return Err(err);
                }
            }
        }
    })?;
    Ok(slots
        .into_iter()
        .filter_map(|slot| slot.into_inner().unwrap_or_else(|e| e.into_inner()))
        .collect())
}

/// One chunk as it grows, with the worker state it borrows.
struct ChunkBuilder<'w, 'a, 'p> {
    walker: &'w mut Walker<'a>,
    rec: Recurrence<'a>,
    /// Union position of each row of the current group, else `u32::MAX`.
    pos: &'w mut [u32],
    walk: &'w Walk<'p>,
    rows: Vec<u32>,
    out: Vec<u32>,
    rel_start: Vec<u32>,
    rel_row: Vec<u32>,
    rel_kin: Vec<f32>,
    tri_start: Vec<usize>,
    tri: Vec<f32>,
    union: Vec<u32>,
    /// Union positions in depth-major order (depth, then row).
    order: Vec<u32>,
    /// Dense kinship among the union, `d * d`, diagonal included.
    local: Vec<f32>,
    kept: Vec<u32>,
}

impl<'w, 'a, 'p> ChunkBuilder<'w, 'a, 'p> {
    fn new(
        walker: &'w mut Walker<'a>,
        rec: Recurrence<'a>,
        pos: &'w mut [u32],
        walk: &'w Walk<'p>,
    ) -> Self {
        ChunkBuilder {
            walker,
            rec,
            pos,
            walk,
            rows: Vec::new(),
            out: Vec::new(),
            rel_start: vec![0],
            rel_row: Vec::new(),
            rel_kin: Vec::new(),
            tri_start: vec![0],
            tri: Vec::new(),
            union: Vec::new(),
            order: Vec::new(),
            local: Vec::new(),
            kept: Vec::new(),
        }
    }

    fn finish(self) -> Chunk {
        Chunk {
            rows: self.rows,
            out: self.out,
            rel_start: self.rel_start,
            rel_row: self.rel_row,
            rel_kin: self.rel_kin,
            tri_start: self.tri_start,
            tri: Triangle::encode(&self.tri),
        }
    }

    /// Kinship among the group's union by the recurrence itself, in
    /// depth-major order so every operand is finished before it is read.
    ///
    /// Each value is the walker's: the later endpoint in (depth, row) order
    /// is the one the walk peels, and every step is the same float32
    /// half-sum.  An operand outside the union is zero when the two rows
    /// share no terminal ([`Terminals`]), and walked otherwise.
    fn union_kinship(&mut self) -> Result<(), Error> {
        let rec = self.rec;
        let d = self.union.len();
        let mut order = std::mem::take(&mut self.order);
        order.clear();
        order.extend(0..d as u32);
        let union = &self.union;
        order.sort_unstable_by_key(|&j| (rec.depth[union[j as usize] as usize], union[j as usize]));
        self.local.clear();
        self.local.resize(d * d, 0.0);
        let result = self.fill_local(rec, &order);
        self.order = order;
        result
    }

    fn fill_local(&mut self, rec: Recurrence<'a>, order: &[u32]) -> Result<(), Error> {
        let d = self.union.len();
        for (rank, &jb) in order.iter().enumerate() {
            let jb = jb as usize;
            let b = self.union[jb] as usize;
            let (m, f) = (rec.mother[b], rec.father[b]);
            let diag = if m < 0 || f < 0 {
                0.5
            } else {
                0.5f32 * (1.0f32 + self.operand(m, f as u32)?)
            };
            self.local[jb * d + jb] = diag;
            // A parent outside the union would send each operand to the
            // fallback; one exact test on `b`, whose terminals are its
            // parents' together, settles most such pairs first.
            let local = |parent: i32| parent < 0 || self.pos[parent as usize] != u32::MAX;
            let outside = !(local(m) && local(f));
            for &ja in &order[..rank] {
                let ja = ja as usize;
                let a = self.union[ja];
                let value = if rec.twin[b] == a as i32 || rec.twin[a as usize] == b as i32 {
                    diag
                } else if (m < 0 && f < 0) || (outside && rec.terminals.disjoint(a, b as u32)) {
                    0.0
                } else {
                    let vm = if m < 0 { 0.0 } else { self.operand(m, a)? };
                    let vf = if f < 0 { 0.0 } else { self.operand(f, a)? };
                    0.5f32 * (vm + vf)
                };
                self.local[ja * d + jb] = value;
                self.local[jb * d + ja] = value;
            }
        }
        Ok(())
    }

    /// Kinship of rows `x` and `y`: from the local matrix when both are in
    /// the union (and so already finished), else zero or walked.
    #[inline]
    fn operand(&mut self, x: i32, y: u32) -> Result<f32, Error> {
        let (jx, jy) = (self.pos[x as usize], self.pos[y as usize]);
        if jx != u32::MAX && jy != u32::MAX {
            return Ok(self.local[jx as usize * self.union.len() + jy as usize]);
        }
        if x as u32 != y && self.rec.terminals.disjoint(x as u32, y) {
            return Ok(0.0);
        }
        Ok(self.walker.resolve(x, y as i32)?)
    }

    /// Walk the triangle over the group's union, then cut each member's
    /// relatives and triangle out of it.
    fn group(&mut self, group: &[u32]) -> Result<(), Error> {
        let Walk {
            probands,
            candidates,
            threshold,
        } = *self.walk;
        self.union.clear();
        for &i in group {
            self.union.extend_from_slice(candidates.of(i as usize));
            self.union.push(probands[i as usize]);
        }
        // The parents of every member too: an in-law (a proband's spouse,
        // an aunt's husband) is no relative but is the other parent of one,
        // and holding it locally keeps the recurrence off the walker.
        let rec = self.rec;
        let n = self.union.len();
        for j in 0..n {
            let row = self.union[j] as usize;
            for parent in [rec.mother[row], rec.father[row]] {
                if parent >= 0 {
                    self.union.push(parent as u32);
                }
            }
        }
        self.union.sort_unstable();
        self.union.dedup();
        let d = self.union.len();
        for (j, &row) in self.union.iter().enumerate() {
            self.pos[row as usize] = j as u32;
        }
        self.union_kinship()?;
        for &i in group {
            let row = probands[i as usize];
            let p = self.pos[row as usize] as usize;
            self.kept.clear();
            for &r in candidates.of(i as usize) {
                let q = self.pos[r as usize] as usize;
                let k = self.local[p * d + q];
                if f64::from(k) >= threshold {
                    self.kept.push(q as u32);
                    self.rel_row.push(r);
                    self.rel_kin.push(k);
                }
            }
            for (j, &a) in self.kept.iter().enumerate() {
                for &b in &self.kept[j + 1..] {
                    self.tri.push(self.local[a as usize * d + b as usize]);
                }
            }
            self.rows.push(row);
            self.out.push(i);
            self.rel_start.push(self.rel_row.len() as u32);
            self.tri_start.push(self.tri.len());
        }
        for &row in &self.union {
            self.pos[row as usize] = u32::MAX;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Founders 10, 11; full sibs 12, 13; 14 is 12's child by external 99;
    /// 15 is unrelated.
    fn family() -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        (
            vec![10, 11, 12, 13, 14, 15],
            vec![-1, -1, 10, 10, 12, -1],
            vec![-1, -1, 11, 11, 99, -1],
        )
    }

    fn input<'a>(ids: &'a [i64], mother: &'a [i64], father: &'a [i64]) -> PedigreeInput<'a> {
        PedigreeInput {
            ids,
            mother,
            father,
            twin: None,
            sex: None,
        }
    }

    #[test]
    fn relatives_follow_degree_and_kinship() {
        let (ids, mother, father) = family();
        let prep = prepare(input(&ids, &mother, &father), 2, None).unwrap();
        assert_eq!(prep.probands(), &[0, 1, 2, 3, 4, 5]);
        let (rows, kin) = prep.relatives(2);
        assert_eq!(rows, &[0, 1, 3, 4]);
        assert_eq!(kin, &[0.25, 0.25, 0.25, 0.25]);
        let (rows, kin) = prep.relatives(4);
        assert_eq!(rows, &[0, 1, 2, 3]);
        assert_eq!(kin, &[0.125, 0.125, 0.25, 0.125]);
        assert!(prep.relatives(5).0.is_empty());
        // Proband 12's relatives 10, 11, 13, 14: founders unrelated; 13 and
        // 14 are aunt and niece.
        assert_eq!(prep.pair_kinship(2, 0, 1), 0.0);
        assert_eq!(prep.pair_kinship(2, 2, 3), 0.125);
        assert_eq!(prep.pair_kinship(2, 3, 0), 0.125);
    }

    #[test]
    fn degree_one_drops_grandparents_and_aunts() {
        let (ids, mother, father) = family();
        let prep = prepare(input(&ids, &mother, &father), 1, None).unwrap();
        assert_eq!(prep.relatives(4).0, &[2]);
        assert_eq!(prep.relatives(3).0, &[0, 1, 2]);
    }

    #[test]
    fn probands_restrict_the_scored_rows_not_the_relatives() {
        let (ids, mother, father) = family();
        let prep = prepare(input(&ids, &mother, &father), 2, Some(&[14, 12])).unwrap();
        assert_eq!(prep.probands(), &[2, 4]);
        assert_eq!(prep.relatives(1).0, &[0, 1, 2, 3]);
    }

    #[test]
    fn proband_list_errors() {
        let (ids, mother, father) = family();
        let err = prepare(input(&ids, &mother, &father), 2, Some(&[12, 77])).unwrap_err();
        assert_eq!(
            err,
            Error::UnknownProband {
                id: 77,
                position: 1
            }
        );
        let err = prepare(input(&ids, &mother, &father), 2, Some(&[12, 13, 12])).unwrap_err();
        assert_eq!(
            err,
            Error::DuplicateProband {
                id: 12,
                positions: [0, 2]
            }
        );
    }

    #[test]
    fn ndegree_is_checked_before_the_pedigree() {
        let (ids, mother, father) = family();
        for nd in [0u8, 6] {
            let err = prepare(input(&ids, &mother, &father), nd, None).unwrap_err();
            assert_eq!(err.code(), "degree_out_of_range");
        }
    }

    #[test]
    fn pedigree_errors_pass_through_with_their_code() {
        let ids = [1i64, 1];
        let err = prepare(input(&ids, &[-1, -1], &[-1, -1]), 2, None).unwrap_err();
        assert_eq!(err.code(), "duplicate_id");
    }

    #[test]
    fn mz_twins_are_relatives_at_half() {
        let ids = [1i64, 2, 3, 4];
        let mother = [-1i64, -1, 1, 1];
        let father = [-1i64, -1, 2, 2];
        let twin = [-1i64, -1, 4, 3];
        let sex = [0i64, 1, 0, 0];
        let prep = prepare(
            PedigreeInput {
                ids: &ids,
                mother: &mother,
                father: &father,
                twin: Some(&twin),
                sex: Some(&sex),
            },
            1,
            None,
        )
        .unwrap();
        let (rows, kin) = prep.relatives(2);
        assert_eq!(rows, &[0, 1, 3]);
        assert_eq!(kin, &[0.25, 0.25, 0.5]);
    }

    #[test]
    fn tri_index_is_row_major_upper() {
        let d = 4;
        let mut expected = 0;
        for j in 0..d {
            for k in j + 1..d {
                assert_eq!(tri_index(d, j, k), expected);
                expected += 1;
            }
        }
    }
}
