//! The analysed sample: Mating Pairs, Mate Networks, stratum codes, and the
//! thin- and degenerate-stratum fixed point (pedsum
//! `assortative_mating.py:883-900`, `:1185-1265`, `:1880-1897`,
//! `pedigree_ops.py:112-131`).

use super::input::Strata;
use super::kernels::{stratum_moments, Grid};

/// Mating Pairs: distinct (mother, father) of the children whose parents are
/// both pedigree rows, in ascending (mother id, father id) order.
pub(crate) fn mating_pairs(
    ids: &[i64],
    mother_rows: &[i32],
    father_rows: &[i32],
) -> (Vec<usize>, Vec<usize>) {
    let mut keys: Vec<(i64, i64, usize, usize)> = mother_rows
        .iter()
        .zip(father_rows)
        .filter(|(&m, &f)| m >= 0 && f >= 0)
        .map(|(&m, &f)| (ids[m as usize], ids[f as usize], m as usize, f as usize))
        .collect();
    keys.sort_unstable();
    keys.dedup_by_key(|k| (k.0, k.1));
    keys.into_iter().map(|k| (k.2, k.3)).unzip()
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Mate Network label of each pair, numbered in order of each network's
/// first pair; mates are graph nodes by row.
pub(crate) fn mate_networks(mothers: &[usize], fathers: &[usize]) -> Vec<usize> {
    let n_nodes = mothers.iter().chain(fathers).max().map_or(0, |&m| m + 1);
    let mut parent: Vec<usize> = (0..n_nodes).collect();
    for (&m, &f) in mothers.iter().zip(fathers) {
        let (a, b) = (find(&mut parent, m), find(&mut parent, f));
        if a != b {
            parent[a.max(b)] = a.min(b);
        }
    }
    let mut label = vec![usize::MAX; n_nodes];
    let mut next = 0;
    mothers
        .iter()
        .map(|&m| {
            let root = find(&mut parent, m);
            if label[root] == usize::MAX {
                label[root] = next;
                next += 1;
            }
            label[root]
        })
        .collect()
}

/// Networks and the largest one's share of the pairs.
pub(crate) fn network_summary(labels: &[usize]) -> (u64, Option<f64>) {
    let g = labels.iter().max().map_or(0, |&l| l + 1);
    let mut sizes = vec![0u64; g];
    for &l in labels {
        sizes[l] += 1;
    }
    let share = sizes
        .iter()
        .max()
        .map(|&largest| largest as f64 / labels.len() as f64);
    (g as u64, share)
}

/// Dense stratum code per row: the rank of its label among every row's
/// distinct label, unknown ranked first (pedsum's `-1`).  `None` is the
/// unknown code itself; without strata every row is code 0.
pub(crate) fn stratum_codes(strata: Option<Strata<'_>>, n_rows: usize) -> Vec<Option<usize>> {
    let Some(strata) = strata else {
        return vec![Some(0); n_rows];
    };
    let mut known: Vec<i64> = strata
        .labels
        .iter()
        .zip(strata.known)
        .filter(|(_, &k)| k)
        .map(|(&l, _)| l)
        .collect();
    known.sort_unstable();
    known.dedup();
    let offset = usize::from(strata.known.iter().any(|&k| !k));
    strata
        .labels
        .iter()
        .zip(strata.known)
        .map(|(l, &k)| k.then(|| offset + known.binary_search(l).unwrap_or(0)))
        .collect()
}

/// One cell's analysed pairs: values and each mate's stratum code, with the
/// levels each stratum shows for a discrete side.
#[derive(Clone, Debug)]
pub(crate) struct CellPairs {
    pub m: Vec<f64>,
    pub f: Vec<f64>,
    pub m_stratum: Vec<usize>,
    pub f_stratum: Vec<usize>,
    pub m_levels: Option<Grid<bool>>,
    pub f_levels: Option<Grid<bool>>,
}

/// Per stratum, which of the `k` levels `codes` shows there.
pub(crate) fn shown_levels(codes: &[f64], stratum: &[usize], k: usize) -> Grid<bool> {
    let n_strata = stratum.iter().max().map_or(0, |&s| s + 1);
    let mut out = Grid::filled(n_strata, k, false);
    for (&c, &s) in codes.iter().zip(stratum) {
        *out.at_mut(s, c as usize) = true;
    }
    out
}

/// The levels shown in any stratum, as one stratum.
pub(crate) fn pooled_levels(levels: &Grid<bool>) -> Grid<bool> {
    let mut out = Grid::filled(1, levels.cols, false);
    for c in 0..levels.cols {
        *out.at_mut(0, c) = (0..levels.rows).any(|s| levels.at(s, c));
    }
    out
}

pub(crate) fn n_codes(code: &[usize]) -> usize {
    code.iter().max().map_or(0, |&c| c + 1)
}

impl CellPairs {
    pub fn len(&self) -> usize {
        self.m.len()
    }

    pub fn take(&self, keep: &[usize]) -> CellPairs {
        let pick = |v: &[f64]| keep.iter().map(|&i| v[i]).collect();
        let pick_code = |v: &[usize]| keep.iter().map(|&i| v[i]).collect();
        CellPairs {
            m: pick(&self.m),
            f: pick(&self.f),
            m_stratum: pick_code(&self.m_stratum),
            f_stratum: pick_code(&self.f_stratum),
            m_levels: self.m_levels.clone(),
            f_levels: self.f_levels.clone(),
        }
    }

    /// Record the levels each stratum shows for each discrete side (`k` levels).
    pub fn with_levels(mut self, k_m: Option<usize>, k_f: Option<usize>) -> CellPairs {
        self.m_levels = k_m.map(|k| shown_levels(&self.m, &self.m_stratum, k));
        self.f_levels = k_f.map(|k| shown_levels(&self.f, &self.f_stratum, k));
        self
    }

    /// Dense stratum-code counts of the mother and father sides.
    pub fn n_strata(&self) -> (usize, usize) {
        (n_codes(&self.m_stratum), n_codes(&self.f_stratum))
    }

    /// The same pairs as one stratum, for a crude estimator.
    pub fn pooled(&self) -> CellPairs {
        let zeros = vec![0; self.len()];
        CellPairs {
            m: self.m.clone(),
            f: self.f.clone(),
            m_stratum: zeros.clone(),
            f_stratum: zeros,
            m_levels: self.m_levels.as_ref().map(pooled_levels),
            f_levels: self.f_levels.as_ref().map(pooled_levels),
        }
    }
}

/// Per stratum code, the distinct Mate Networks its pairs come from.
fn stratum_networks(code: &[usize], labels: &[usize]) -> Vec<usize> {
    let n_labels = labels.iter().max().map_or(0, |&l| l + 1);
    let mut keys: Vec<usize> = code
        .iter()
        .zip(labels)
        .map(|(&c, &l)| c * n_labels + l)
        .collect();
    keys.sort_unstable();
    keys.dedup();
    let mut out = vec![0; n_codes(code)];
    for k in keys {
        out[k / n_labels.max(1)] += 1;
    }
    out
}

/// Per stratum code, whether it is present and constant.
fn degenerate_strata(x: &[f64], code: &[usize]) -> Vec<bool> {
    let w = vec![1.0; x.len()];
    let m = stratum_moments(x, code, &w, n_codes(code), false);
    m.lo.iter().zip(&m.hi).map(|(l, h)| l == h).collect()
}

/// The pairs kept once every small or degenerate sex x stratum is dropped,
/// rules repeated until neither drops a pair: `(keep, n_small, labels)`.
pub(crate) fn drop_thin_strata(
    pairs: &CellPairs,
    mothers: &[usize],
    fathers: &[usize],
    min_networks: u64,
) -> (Vec<usize>, u64, Vec<usize>) {
    let mut keep: Vec<usize> = (0..pairs.len()).collect();
    let mut n_small = 0;
    loop {
        let m: Vec<usize> = keep.iter().map(|&i| mothers[i]).collect();
        let f: Vec<usize> = keep.iter().map(|&i| fathers[i]).collect();
        let labels = mate_networks(&m, &f);
        if keep.is_empty() {
            return (keep, n_small, labels);
        }
        let kept = pairs.take(&keep);
        let nets_m = stratum_networks(&kept.m_stratum, &labels);
        let nets_f = stratum_networks(&kept.f_stratum, &labels);
        let deg_m = degenerate_strata(&kept.m, &kept.m_stratum);
        let deg_f = degenerate_strata(&kept.f, &kept.f_stratum);
        let mut any_bad = false;
        let mut next = Vec::with_capacity(keep.len());
        for (j, &i) in keep.iter().enumerate() {
            let (s, t) = (kept.m_stratum[j], kept.f_stratum[j]);
            let small = (nets_m[s] as u64) < min_networks || (nets_f[t] as u64) < min_networks;
            if small {
                n_small += 1;
            }
            if small || deg_m[s] || deg_f[t] {
                any_bad = true;
            } else {
                next.push(i);
            }
        }
        if !any_bad {
            return (keep, n_small, labels);
        }
        keep = next;
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
        let (m, f) = mating_pairs(&ids, &mother, &father);
        assert_eq!((m.clone(), f.clone()), (vec![1, 0, 0], vec![4, 3, 2]));
        assert_eq!(mate_networks(&m, &f), vec![0, 1, 1]);
        assert_eq!(network_summary(&[0, 1, 1]), (2, Some(2.0 / 3.0)));
        assert_eq!(network_summary(&[]), (0, None));
    }

    #[test]
    fn codes_rank_labels_with_unknown_first() {
        let labels = [1990, 1950, -5, 1950];
        let strata = Strata {
            labels: &labels,
            known: &[true, true, false, true],
        };
        assert_eq!(
            stratum_codes(Some(strata), 4),
            vec![Some(2), Some(1), None, Some(1)]
        );
        let all = Strata {
            labels: &labels,
            known: &[true; 4],
        };
        assert_eq!(
            stratum_codes(Some(all), 4),
            vec![Some(2), Some(1), Some(0), Some(1)]
        );
    }
}
