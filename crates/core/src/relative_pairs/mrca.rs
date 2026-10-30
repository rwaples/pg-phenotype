//! The MRCA cluster of a Relative Pair (ADR 0013): the shared ancestor its
//! relationship runs through, found by walking the category's meioses up
//! from each member.

use crate::error::Error;
use crate::pedigree::Pedigree;
use pedigree_graph_core::error::Error as PgError;
use pedigree_graph_core::relationships::{Category, N_CATEGORIES};

/// Per category in registry order, the meioses from a pair's first member
/// (in its roles' order) up to the MRCA, and from the MRCA down to the
/// second: pedigree_graph's `RELATIONSHIPS[code].up` and `.down`, which
/// Rust's `Category` lacks.  A Python test checks the two agree.
pub(super) const MEIOSES: [(u8, u8); N_CATEGORIES] = [
    (0, 0), // MZ
    (1, 0), // MO
    (1, 0), // FO
    (1, 1), // FS
    (1, 1), // MHS
    (1, 1), // PHS
    (2, 0), // GP
    (2, 1), // Av
    (3, 0), // GGP
    (2, 1), // HAv
    (3, 1), // GAv
    (2, 2), // 1C
    (4, 0), // GGGP
    (3, 1), // HGAv
    (4, 1), // GGAv
    (2, 2), // H1C
    (3, 2), // 1C1R
    (5, 0), // G3GP
    (4, 1), // HGGAv
    (5, 1), // G3Av
    (3, 2), // H1C1R
    (4, 2), // 1C2R
    (3, 3), // 2C
];

/// No parent.
const NONE: u32 = u32::MAX;

/// A cluster of the MRCA design: an ancestor by its original id, or a pair
/// with no common ancestor.  Dense, and the same partition as the original
/// id (ids are unique and nonnegative): a row's id is its row `0..n`, an
/// external parent's id `n + its rank` among the external ids ascending, and
/// a pair with no common ancestor `lone + its first member's row`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Cluster(pub(super) u32);

/// The external parents' ids, ascending: the ids of a row's parents that
/// have no row.  Cluster `n + i` is the `i`th.
pub(super) fn external(pedigree: &Pedigree) -> Vec<i64> {
    let mut external: Vec<i64> = [
        (pedigree.mother_rows(), pedigree.mother_ids()),
        (pedigree.father_rows(), pedigree.father_ids()),
    ]
    .iter()
    .flat_map(|&(rows, ids)| rows.iter().zip(ids))
    .filter(|&(&row, &id)| row < 0 && id >= 0)
    .map(|(_, &id)| id)
    .collect();
    external.sort_unstable();
    external.dedup();
    external
}

/// `cluster`'s ancestor's original id, `ids` being the rows' and
/// `external` [`external`]'s; `None` for a pair with no common ancestor.
pub(super) fn id(ids: &[i64], external: &[i64], cluster: Cluster) -> Option<i64> {
    let c = cluster.0 as usize;
    match c.checked_sub(ids.len()) {
        None => Some(ids[c]),
        Some(rank) => external.get(rank).copied(),
    }
}

/// Who is whose parent, as clusters, built once before the pass.
pub(super) struct Ancestry<'p> {
    mother_rows: &'p [i32],
    father_rows: &'p [i32],
    /// Per row, the mother's cluster (her row, or her external id's), or
    /// [`NONE`] when her id is missing.
    mother: Vec<u32>,
    father: Vec<u32>,
    /// The first lone cluster: rows' and external parents' come before it.
    lone: u32,
}

impl<'p> Ancestry<'p> {
    /// # Errors
    ///
    /// `memory_budget_exceeded` (operation `relative_pairs_se`, estimated
    /// `u64::MAX`) when the clusters, two per row plus the external ids, do
    /// not fit a `u32`.
    pub(super) fn of(pedigree: &'p Pedigree, budget_bytes: u64) -> Result<Ancestry<'p>, Error> {
        let (mother_rows, father_rows) = (pedigree.mother_rows(), pedigree.father_rows());
        let parents = [
            (mother_rows, pedigree.mother_ids()),
            (father_rows, pedigree.father_ids()),
        ];
        let external = external(pedigree);
        let n = mother_rows.len();
        let lone = u32::try_from(n + external.len())
            .ok()
            .filter(|&lone| u64::from(lone) + (n as u64) < u64::from(NONE))
            .ok_or(PgError::MemoryBudgetExceeded {
                operation: super::clusters::OPERATION,
                estimated_bytes: u64::MAX,
                budget_bytes,
            })?;
        let [mother, father] = parents.map(|(rows, ids)| {
            rows.iter()
                .zip(ids)
                .map(|(&row, &id)| match (u32::try_from(row), id) {
                    (Ok(row), _) => row,
                    (Err(_), id) if id >= 0 => (n + external.partition_point(|&e| e < id)) as u32,
                    _ => NONE,
                })
                .collect()
        });
        Ok(Ancestry {
            mother_rows,
            father_rows,
            mother,
            father,
            lone,
        })
    }

    /// The bytes it holds.
    pub(super) fn bytes(&self) -> u64 {
        ((self.mother.capacity() + self.father.capacity()) * size_of::<u32>()) as u64
    }

    /// The pair's MRCA cluster.  A symmetric category reads `(min, max)`, so
    /// both orientations agree.
    ///
    /// MZ: the mother's cluster, else the father's, else the pair's lone
    /// cluster.  Otherwise the first hit of [`Ancestry::walk`].  The engine
    /// emits no pair the walk misses; one would be its own lone cluster.
    #[inline]
    pub(super) fn mrca(&self, cat: Category, first: u32, second: u32) -> Cluster {
        let (a, b) = if cat.symmetric() {
            (first.min(second), first.max(second))
        } else {
            (first, second)
        };
        let found = self.walk(cat, a, b);
        debug_assert!(
            found.is_some(),
            "{} pair ({a}, {b}) has no MRCA",
            cat.code()
        );
        found.unwrap_or(Cluster(self.lone + a))
    }

    /// Each ancestor `x` of `a` `up - 1` meioses up, in path order, and for
    /// each the ancestors `y` of `b` `down - 1` meioses up (or `b` itself
    /// when `down` is 0): the first `x` whose parent is `b` (lineal), or
    /// whose mother (else father) is also `y`'s, gives the MRCA.  Sibs share
    /// a parent by id, so an external parent is an MRCA too.  This is the
    /// SE calibration tool's `top_keys`.
    fn walk(&self, cat: Category, a: u32, b: u32) -> Option<Cluster> {
        let (up, down) = MEIOSES[cat.index()];
        if up == 0 {
            let a = a as usize;
            let parent = if self.mother[a] != NONE {
                self.mother[a]
            } else {
                self.father[a]
            };
            return Some(if parent == NONE {
                Cluster(self.lone + a as u32)
            } else {
                Cluster(parent)
            });
        }
        for x in self.ancestors(a, up - 1) {
            if down == 0 {
                if self.mother_rows[x] == b as i32 || self.father_rows[x] == b as i32 {
                    return Some(Cluster(b));
                }
                continue;
            }
            let (m, f) = (self.mother[x], self.father[x]);
            for y in self.ancestors(b, down - 1).filter(|&y| y != x) {
                if m != NONE && m == self.mother[y] {
                    return Some(Cluster(m));
                }
                if f != NONE && f == self.father[y] {
                    return Some(Cluster(f));
                }
            }
        }
        None
    }

    /// The rows `depth` meioses up from `start`, one per path that stays in
    /// the pedigree, paths in order with the first hop most significant and
    /// the mother before the father.
    fn ancestors(&self, start: u32, depth: u8) -> impl Iterator<Item = usize> + '_ {
        (0u32..1 << depth).filter_map(move |path| {
            let mut x = start as usize;
            for hop in (0..depth).rev() {
                let parents = if path >> hop & 1 == 0 {
                    self.mother_rows
                } else {
                    self.father_rows
                };
                x = usize::try_from(parents[x]).ok()?;
            }
            Some(x)
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::input::PedigreeInput;
    use pedigree_graph_core::relationships::{
        reduce_pairs, CategorySet, MaxDegree, Progress, Reducer, Symmetric,
    };
    use std::collections::BTreeSet;
    use std::num::NonZeroUsize;

    /// Pedigree columns by id; ids are rows, a parent id past the rows is
    /// external.
    pub(in super::super) struct Columns {
        pub(in super::super) ids: Vec<i64>,
        pub(in super::super) mother: Vec<i64>,
        pub(in super::super) father: Vec<i64>,
        pub(in super::super) twin: Vec<i64>,
        /// Each row's generation, founders 0.
        pub(in super::super) generation: Vec<i64>,
    }

    impl Columns {
        pub(in super::super) fn pedigree(&self) -> Pedigree {
            Pedigree::new(PedigreeInput {
                ids: &self.ids,
                mother: &self.mother,
                father: &self.father,
                twin: Some(&self.twin),
                sex: None,
            })
            .unwrap()
        }

        fn push(&mut self, mother: i64, father: i64, generation: i64) -> usize {
            let row = self.ids.len();
            self.ids.push(row as i64);
            self.mother.push(mother);
            self.father.push(father);
            self.twin.push(-1);
            self.generation.push(generation);
            row
        }

        fn twins(&mut self, a: usize, b: usize) {
            self.twin[a] = b as i64;
            self.twin[b] = a as i64;
        }
    }

    /// External parent ids, past any row.
    const EXTERNAL: i64 = 1 << 40;

    /// A random pedigree of `founders` founders and `generations`
    /// generations: each woman of a generation mates (most of them), some
    /// twice, with random men of hers, a few with an external father from a
    /// small pool; one to three children each, the first two sometimes MZ
    /// twins.  Then fixed blocks: parentless MZ twins with children, double
    /// first cousins, and full sibs of two external parents with a child.
    pub(in super::super) fn generated(founders: usize, generations: usize, seed: u64) -> Columns {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % bound as u64) as usize
        };
        let mut cols = Columns {
            ids: Vec::new(),
            mother: Vec::new(),
            father: Vec::new(),
            twin: Vec::new(),
            generation: Vec::new(),
        };
        let mut male = Vec::new();
        let mut current: Vec<usize> = (0..founders)
            .map(|i| {
                male.push(i % 2 == 1);
                cols.push(-1, -1, 0)
            })
            .collect();
        for g in 1..=generations as i64 {
            let women: Vec<usize> = current.iter().copied().filter(|&r| !male[r]).collect();
            let men: Vec<usize> = current.iter().copied().filter(|&r| male[r]).collect();
            let mut children = Vec::new();
            for &woman in &women {
                let matings = match next(20) {
                    0..=2 => 0,
                    3..=4 => 2,
                    _ => 1,
                };
                for _ in 0..matings {
                    let father = match next(20) {
                        0 => EXTERNAL + next(4) as i64,
                        _ if men.is_empty() => continue,
                        _ => men[next(men.len())] as i64,
                    };
                    let n = 1 + next(3);
                    let twinned = n >= 2 && next(8) == 0;
                    let sex = next(2) == 1;
                    for k in 0..n {
                        let row = cols.push(woman as i64, father, g);
                        male.push(if twinned && k < 2 { sex } else { next(2) == 1 });
                        children.push(row);
                    }
                    if twinned {
                        let last = cols.ids.len();
                        cols.twins(last - n, last - n + 1);
                    }
                }
            }
            current = children;
        }
        // Parentless MZ twins (0, 1), and a child of each.
        let base = cols.ids.len();
        let (t0, t1) = (cols.push(-1, -1, 0), cols.push(-1, -1, 0));
        cols.twins(t0, t1);
        let mate = cols.push(-1, -1, 0);
        cols.push(t0 as i64, mate as i64, 1);
        cols.push(t1 as i64, mate as i64, 1);
        // Double first cousins: sibs (s1, b1) of (g1, g2) and (s2, b2) of
        // (g3, g4); s1 x b2 and s2 x b1.
        let g: Vec<i64> = (0..4).map(|_| cols.push(-1, -1, 0) as i64).collect();
        let s1 = cols.push(g[0], g[1], 1) as i64;
        let b1 = cols.push(g[0], g[1], 1) as i64;
        let s2 = cols.push(g[2], g[3], 1) as i64;
        let b2 = cols.push(g[2], g[3], 1) as i64;
        cols.push(s1, b2, 2);
        cols.push(s2, b1, 2);
        // Full sibs of two external parents, and a child of one of them.
        let d1 = cols.push(EXTERNAL + 10, EXTERNAL + 11, 1);
        cols.push(EXTERNAL + 10, EXTERNAL + 11, 1);
        cols.push(d1 as i64, base as i64 + 2, 2);
        cols
    }

    struct Collect;

    impl Reducer for Collect {
        type Lane = Vec<(Category, u32, u32)>;
        fn lane(&self) -> Result<Self::Lane, PgError> {
            Ok(Vec::new())
        }
        fn reduce(&self, lane: &mut Self::Lane, cat: Category, first: u32, second: u32) {
            lane.push((cat, first, second));
        }
        fn merge(&self, into: &mut Self::Lane, from: Self::Lane) {
            into.extend(from);
        }
    }

    /// Every entry the engine emits up to degree 3.
    pub(in super::super) fn engine_pairs(
        pedigree: &Pedigree,
        symmetric: Symmetric,
    ) -> Vec<(Category, u32, u32)> {
        reduce_pairs(
            &pedigree.relationships().unwrap(),
            MaxDegree::try_new(3).unwrap(),
            CategorySet::up_to_degree(3),
            None,
            symmetric,
            NonZeroUsize::MIN,
            &Collect,
            &Progress::default(),
        )
        .unwrap()
        .lane
    }

    /// The ids `depth` meioses up from `row` by parent id, through every
    /// path; an external id has no parents.
    fn ancestor_ids(cols: &Columns, row: usize, depth: u8) -> BTreeSet<i64> {
        let mut level = BTreeSet::from([cols.ids[row]]);
        for _ in 0..depth {
            level = level
                .iter()
                .filter(|&&id| id < EXTERNAL)
                .flat_map(|&id| [cols.mother[id as usize], cols.father[id as usize]])
                .filter(|&id| id >= 0)
                .collect();
        }
        level
    }

    /// The id a cluster stands for, `None` for a lone cluster, decoded
    /// from the external ids ranked here.
    fn decode(cols: &Columns, cluster: Cluster) -> Option<i64> {
        let n = cols.ids.len();
        let mut external: Vec<i64> = cols
            .mother
            .iter()
            .chain(&cols.father)
            .copied()
            .filter(|&id| id >= EXTERNAL)
            .collect();
        external.sort_unstable();
        external.dedup();
        let c = cluster.0 as usize;
        if c < n {
            Some(cols.ids[c])
        } else {
            external.get(c - n).copied()
        }
    }

    #[test]
    fn meioses_match_the_degree_rule() {
        for cat in Category::ALL {
            let (up, down) = MEIOSES[cat.index()];
            // Full: a collateral relationship through a mated pair.
            let full = matches!(
                cat,
                Category::FS
                    | Category::Av
                    | Category::GAv
                    | Category::C1
                    | Category::GGAv
                    | Category::C1R1
                    | Category::G3Av
                    | Category::C1R2
                    | Category::C2
            );
            let expected = (up + down).saturating_sub(u8::from(full));
            assert_eq!(cat.degree(), expected, "{}", cat.code());
        }
    }

    #[test]
    fn every_engine_pair_up_to_degree_3_has_an_mrca_shared_at_its_meioses() {
        for seed in 0..3 {
            let cols = generated(80, 4, seed);
            let pedigree = cols.pedigree();
            let ancestry = Ancestry::of(&pedigree, u64::MAX).unwrap();
            let pairs = engine_pairs(&pedigree, Symmetric::Canonical);
            let mut seen = BTreeSet::new();
            let (mut lone, mut external) = (0, 0);
            for &(cat, a, b) in &pairs {
                seen.insert(cat);
                let (up, down) = MEIOSES[cat.index()];
                let what = format!("seed {seed} {} ({a}, {b})", cat.code());
                let cluster = ancestry.walk(cat, a, b).expect(&what);
                if cat.symmetric() {
                    assert_eq!(ancestry.mrca(cat, b, a), cluster, "{what}");
                }
                let (a, b) = (a as usize, b as usize);
                let Some(id) = decode(&cols, cluster) else {
                    assert_eq!(cat, Category::MZ, "{what}");
                    assert_eq!((cols.mother[a], cols.father[a]), (-1, -1), "{what}");
                    assert_eq!(cluster.0, ancestry.lone + a as u32, "{what}");
                    lone += 1;
                    continue;
                };
                external += usize::from(id >= EXTERNAL);
                if up == 0 {
                    let parent = if cols.mother[a] >= 0 {
                        cols.mother[a]
                    } else {
                        cols.father[a]
                    };
                    assert_eq!(id, parent, "{what}");
                    continue;
                }
                assert!(ancestor_ids(&cols, a, up).contains(&id), "{what}: {id}");
                assert!(ancestor_ids(&cols, b, down).contains(&id), "{what}: {id}");
            }
            let wanted: BTreeSet<Category> = Category::ALL
                .into_iter()
                .filter(|c| c.degree() <= 3)
                .collect();
            assert_eq!(seen, wanted, "seed {seed}");
            assert!(
                lone > 0 && external > 0,
                "seed {seed}: {lone} lone, {external} external"
            );
        }
    }

    #[test]
    fn fixed_cases_find_their_mrca() {
        let cols = generated(0, 0, 0);
        let pedigree = cols.pedigree();
        let ancestry = Ancestry::of(&pedigree, u64::MAX).unwrap();
        let n = cols.ids.len() as u32;
        // Rows: twins 0, 1; mate 2; their children 3, 4; grandparents 5-8;
        // s1 9, b1 10, s2 11, b2 12; cousins 13, 14; external sibs 15, 16;
        // 17, the child of 15.
        assert_eq!(ancestry.mrca(Category::MZ, 1, 0), Cluster(ancestry.lone));
        // s1's mother 5 is b1's: found from 13, whose mother s1 comes first.
        assert_eq!(ancestry.mrca(Category::C1, 14, 13), Cluster(5));
        assert_eq!(ancestry.mrca(Category::C1, 13, 14), Cluster(5));
        // From 14 the first hit would be 7, s2's and b2's mother.
        assert_eq!(ancestry.walk(Category::C1, 14, 13), Some(Cluster(7)));
        // External ids rank ascending after the rows.
        assert_eq!(ancestry.mrca(Category::FS, 15, 16), Cluster(n));
        assert_eq!(ancestry.mrca(Category::Av, 17, 16), Cluster(n));
        assert_eq!(ancestry.mrca(Category::GP, 13, 5), Cluster(5));
    }
}
