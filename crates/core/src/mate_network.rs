//! Mating Pairs and the Mate Networks a sandwich SE clusters by.
//!
//! A method takes its networks under one of two policies: the networks of
//! exactly the pairs it analyses ([`Networks::of_pairs`], ADR 0006), or the
//! whole pedigree's networks read at each parent ([`ParentNetworks`], ADR
//! 0007).  Either way the labels come out dense, so a [`Sandwich`] counts G
//! networks.

/// A pedigree's Mating Pairs, as rows: pair `p` is `(mothers[p], fathers[p])`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MatingPairs {
    pub mothers: Vec<usize>,
    pub fathers: Vec<usize>,
}

/// Mating Pairs: distinct (mother, father) of the children whose parents are
/// both pedigree rows, in ascending (mother id, father id) order.
pub(crate) fn mating_pairs(ids: &[i64], mother_rows: &[i32], father_rows: &[i32]) -> MatingPairs {
    // Rows are unique, non-negative i32: (mother, father) packs into one u64
    // whose order is row order, which is id order when ids ascend by row.
    let mut keys: Vec<u64> = mother_rows
        .iter()
        .zip(father_rows)
        .filter(|(&m, &f)| m >= 0 && f >= 0)
        .map(|(&m, &f)| (m as u64) << 32 | f as u64)
        .collect();
    keys.sort_unstable();
    keys.dedup();
    let rows = |k: u64| ((k >> 32) as usize, k as u32 as usize);
    if !ids.is_sorted() {
        keys.sort_unstable_by_key(|&k| {
            let (m, f) = rows(k);
            (ids[m], ids[f])
        });
    }
    let (mothers, fathers) = keys.into_iter().map(rows).unzip();
    MatingPairs { mothers, fathers }
}

/// The Mate Network of each analysed unit, numbered `0..G` in order of each
/// network's first unit.  The numbering is dense by construction, so G is
/// the number of networks, and the bootstrap's draws depend on its order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Networks {
    labels: Vec<usize>,
    count: usize,
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

impl Networks {
    /// The Mate Networks of exactly these pairs: pair `p` is
    /// `(mothers[p], fathers[p])`, and mates are graph nodes by row.
    pub fn of_pairs(mothers: &[usize], fathers: &[usize]) -> Networks {
        let n_nodes = mothers.iter().chain(fathers).max().map_or(0, |&m| m + 1);
        let mut parent: Vec<usize> = (0..n_nodes).collect();
        for (&m, &f) in mothers.iter().zip(fathers) {
            let (a, b) = (find(&mut parent, m), find(&mut parent, f));
            if a != b {
                parent[a.max(b)] = a.min(b);
            }
        }
        let mut label = vec![usize::MAX; n_nodes];
        let mut count = 0;
        let labels = mothers
            .iter()
            .map(|&m| {
                let root = find(&mut parent, m);
                if label[root] == usize::MAX {
                    label[root] = count;
                    count += 1;
                }
                label[root]
            })
            .collect();
        Networks { labels, count }
    }

    /// Each unit's network.
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// G, the number of networks.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The largest network's share of the units, `None` without units.
    pub fn largest_share(&self) -> Option<f64> {
        let mut sizes = vec![0u64; self.count];
        for &l in &self.labels {
            sizes[l] += 1;
        }
        sizes
            .iter()
            .max()
            .map(|&largest| largest as f64 / self.labels.len() as f64)
    }

    /// The cluster-robust sandwich over these networks, `None` with fewer
    /// than two.
    pub fn sandwich(&self) -> Option<Sandwich<'_>> {
        (self.count >= 2).then_some(Sandwich { networks: self })
    }
}

/// The sandwich over G >= 2 Mate Networks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sandwich<'a> {
    networks: &'a Networks,
}

impl Sandwich<'_> {
    /// Each network's sum of `values`, one value per unit in unit order,
    /// in label order.
    pub fn sums(&self, values: impl IntoIterator<Item = f64>) -> Vec<f64> {
        let mut sums = vec![0.0; self.networks.count];
        let mut values = values.into_iter();
        for &l in &self.networks.labels {
            match values.next() {
                Some(v) => sums[l] += v,
                None => debug_assert!(false, "fewer values than units"),
            }
        }
        debug_assert!(values.next().is_none(), "more values than units");
        sums
    }

    /// `√(G/(G−1) · Σ_g S_g²)` of the networks' influence sums `S_g`
    /// ([`Sandwich::sums`]).
    pub fn se(&self, sums: &[f64]) -> f64 {
        debug_assert_eq!(sums.len(), self.networks.count, "one sum per network");
        let g = self.networks.count as f64;
        let ss = sums.iter().fold(0.0, |a, &s| a + s * s);
        (g / (g - 1.0) * ss).sqrt()
    }
}

/// The whole pedigree's Mate Networks read at each row as a parent: the
/// network of its Mating Pairs, or, when it is in no Mating Pair, a key of
/// its own past the networks.  Dropping a parent's children never splits its
/// network.
#[derive(Debug)]
pub(crate) struct ParentNetworks {
    network: Vec<u32>,
    n_networks: usize,
}

impl ParentNetworks {
    /// From the pedigree's Mating Pairs and their Mate Networks, over
    /// `n_rows` rows.
    pub fn of(pairs: &MatingPairs, networks: &Networks, n_rows: usize) -> ParentNetworks {
        debug_assert_eq!(
            networks.labels.len(),
            pairs.mothers.len(),
            "one network per pair"
        );
        let mut network = vec![u32::MAX; n_rows];
        for ((&m, &f), &label) in pairs
            .mothers
            .iter()
            .zip(&pairs.fathers)
            .zip(&networks.labels)
        {
            // Both mates of a pair, and every pair of one parent, share a network.
            network[m] = label as u32;
            network[f] = label as u32;
        }
        ParentNetworks {
            network,
            n_networks: networks.count,
        }
    }

    /// Row `parent`'s key.
    pub fn key(&self, parent: usize) -> usize {
        match self.network[parent] {
            u32::MAX => self.n_networks + parent,
            network => network as usize,
        }
    }

    /// Scratch for numbering the keys of one fit after another.
    pub fn relabel(&self) -> Relabel {
        Relabel::new(self.n_networks + self.network.len())
    }
}

/// Numbers keys below `n_keys` by first appearance, reusing one scratch.
pub(crate) struct Relabel {
    slot: Vec<usize>,
}

impl Relabel {
    pub fn new(n_keys: usize) -> Relabel {
        Relabel {
            slot: vec![usize::MAX; n_keys],
        }
    }

    /// The networks of units with these keys: G counts only the keys present.
    pub fn networks(&mut self, keys: &[usize]) -> Networks {
        let mut count = 0;
        let labels = keys
            .iter()
            .map(|&k| {
                if self.slot[k] == usize::MAX {
                    self.slot[k] = count;
                    count += 1;
                }
                self.slot[k]
            })
            .collect();
        for &k in keys {
            self.slot[k] = usize::MAX;
        }
        Networks { labels, count }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_sort_by_parent_id_and_networks_number_by_first_pair() {
        let ids = [50, 40, 30, 20, 10, 1, 2, 3, 4];
        let mother = [-1, -1, -1, -1, -1, 0, 1, 0, 0];
        let father = [-1, -1, -1, -1, -1, 3, 4, 2, 3];
        let MatingPairs {
            mothers: m,
            fathers: f,
        } = mating_pairs(&ids, &mother, &father);
        assert_eq!((m.clone(), f.clone()), (vec![1, 0, 0], vec![4, 3, 2]));
        let networks = Networks::of_pairs(&m, &f);
        assert_eq!((networks.labels(), networks.count()), (&[0, 1, 1][..], 2));
        assert_eq!(networks.largest_share(), Some(2.0 / 3.0));
        let none = Networks::of_pairs(&[], &[]);
        assert_eq!((none.count(), none.largest_share()), (0, None));
    }

    #[test]
    fn parents_in_no_pair_get_their_own_key_and_fits_are_renumbered() {
        // Pairs (0, 1) and (2, 3) over six rows; rows 4 and 5 are in none.
        let pairs = MatingPairs {
            mothers: vec![0, 2],
            fathers: vec![1, 3],
        };
        let parents = ParentNetworks::of(&pairs, &Networks::of_pairs(&[0, 2], &[1, 3]), 6);
        let keys: Vec<usize> = (0..6).map(|r| parents.key(r)).collect();
        assert_eq!(keys, vec![0, 0, 1, 1, 6, 7]);
        let mut relabel = parents.relabel();
        let fit = relabel.networks(&[1, 6, 6, 1, 7]);
        assert_eq!((fit.labels(), fit.count()), (&[0, 1, 1, 0, 2][..], 3));
        // The scratch is clean for the next fit.
        assert_eq!(relabel.networks(&[6]).labels(), &[0]);
    }

    #[test]
    fn the_sandwich_needs_two_networks_and_sums_by_label() {
        assert!(Relabel::new(4).networks(&[3, 3]).sandwich().is_none());
        assert!(Relabel::new(4).networks(&[]).sandwich().is_none());
        let networks = Relabel::new(8).networks(&[5, 2, 5, 7]);
        let sandwich = networks.sandwich().unwrap();
        let sums = sandwich.sums([1.0, -2.0, 0.5, 0.25]);
        assert_eq!(sums, vec![1.5, -2.0, 0.25]);
        let want = (3.0f64 / 2.0 * (1.5f64.powi(2) + 4.0 + 0.0625)).sqrt();
        assert_eq!(sandwich.se(&sums), want);
    }
}
