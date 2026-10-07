//! An exact test that two rows share no ancestor, so their kinship is zero.
//!
//! Every upward path from a row ends at a *terminal*: a row with a missing
//! parent.  A common ancestor-or-self of two rows therefore has a terminal
//! ancestor-or-self that both rows reach, and two rows with a common
//! terminal share an ancestor.  Kinship is positive exactly when two rows
//! share an ancestor-or-self (or are MZ co-twins, which share a label), so
//! disjoint terminal sets prove a zero.
//!
//! pedigree-graph-core's 256-bit ancestor signatures make the same test
//! approximately; in a pedigree six generations deep each row sets about 60
//! bits, two unrelated rows almost always collide, and the walk then
//! reaches the founders to learn the zero.  The sets here are exact and
//! small: at most `2^depth` labels.  A row whose set would exceed
//! [`MAX_TERMINALS`] keeps none and is never declared disjoint.

/// The largest terminal set kept per row.
const MAX_TERMINALS: usize = 64;

/// The sorted terminal labels of every row, in CSR form.
pub(crate) struct Terminals {
    start: Vec<usize>,
    labels: Vec<u32>,
    /// Rows whose set overflowed.
    unknown: Vec<bool>,
}

impl Terminals {
    /// Build every row's set in depth-major order, parents first.
    pub(crate) fn build(mother: &[i32], father: &[i32], twin: &[i32], depth: &[i32]) -> Terminals {
        let n = mother.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_unstable_by_key(|&r| (depth[r as usize], r));
        let mut sets: Vec<Option<Vec<u32>>> = vec![None; n];
        let mut merged: Vec<u32> = Vec::new();
        for &r in &order {
            let r = r as usize;
            let (m, f) = (mother[r], father[r]);
            merged.clear();
            if m < 0 || f < 0 {
                let label = if twin[r] >= 0 {
                    (r as i32).min(twin[r])
                } else {
                    r as i32
                };
                merged.push(label as u32);
            }
            let mut overflow = false;
            for parent in [m, f] {
                if parent >= 0 {
                    match &sets[parent as usize] {
                        Some(set) => merged.extend_from_slice(set),
                        None => overflow = true,
                    }
                }
            }
            merged.sort_unstable();
            merged.dedup();
            sets[r] = (!overflow && merged.len() <= MAX_TERMINALS).then(|| merged.clone());
        }
        let mut start = Vec::with_capacity(n + 1);
        start.push(0);
        let mut labels = Vec::new();
        let mut unknown = vec![false; n];
        for (r, set) in sets.into_iter().enumerate() {
            match set {
                Some(set) => labels.extend_from_slice(&set),
                None => unknown[r] = true,
            }
            start.push(labels.len());
        }
        Terminals {
            start,
            labels,
            unknown,
        }
    }

    /// Whether rows `a` and `b` provably share no ancestor-or-self.
    #[inline]
    pub(crate) fn disjoint(&self, a: u32, b: u32) -> bool {
        let (a, b) = (a as usize, b as usize);
        if self.unknown[a] || self.unknown[b] {
            return false;
        }
        let (x, y) = (
            &self.labels[self.start[a]..self.start[a + 1]],
            &self.labels[self.start[b]..self.start[b + 1]],
        );
        let (mut i, mut j) = (0, 0);
        while i < x.len() && j < y.len() {
            match x[i].cmp(&y[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => return false,
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terminals(rows: &[(i32, i32)], twins: &[(i32, i32)]) -> Terminals {
        let mother: Vec<i32> = rows.iter().map(|r| r.0).collect();
        let father: Vec<i32> = rows.iter().map(|r| r.1).collect();
        let mut twin = vec![-1; rows.len()];
        for &(a, b) in twins {
            twin[a as usize] = b;
            twin[b as usize] = a;
        }
        let depth = pedigree_graph_core::topology::structural_depth(&mother, &father);
        Terminals::build(&mother, &father, &twin, &depth)
    }

    #[test]
    fn relatives_are_never_disjoint_and_strangers_are() {
        // 0, 1 founders; 2, 3 their children; 4 child of 2 and founder 5;
        // 6 a stranger; 7 child of 6 and one missing parent.
        let t = terminals(
            &[
                (-1, -1),
                (-1, -1),
                (0, 1),
                (0, 1),
                (2, 5),
                (-1, -1),
                (-1, -1),
                (6, -1),
            ],
            &[],
        );
        for (a, b) in [(2, 3), (3, 4), (0, 4), (4, 4), (6, 7)] {
            assert!(!t.disjoint(a, b), "{a} {b}");
        }
        for (a, b) in [(0, 1), (5, 3), (6, 4), (7, 2)] {
            assert!(t.disjoint(a, b), "{a} {b}");
        }
    }

    #[test]
    fn founder_mz_twins_share_a_label() {
        let t = terminals(&[(-1, -1), (-1, -1), (0, -1)], &[(0, 1)]);
        assert!(!t.disjoint(0, 1));
        assert!(!t.disjoint(1, 2));
    }
}
