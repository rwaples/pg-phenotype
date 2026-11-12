//! Properties of the Relative-Pair Correlation of one trait.
//!
//! Exact comparisons are between two calls in one process, never against a
//! stored value, so they hold on every platform.

#![expect(
    clippy::expect_used,
    reason = "test helpers outside #[test], which clippy.toml does not exempt"
)]

mod common;

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use common::{check, codes, kind, outbred_pedigree, pedigree, trait_of, values, Ped};
use pg_phenotype_core::relative_pairs::{
    correlation, correlation_with, Cell, Counts, Entry, Estimate, RelativePairCorrelation, Se,
    DEFAULT_MEMORY_BUDGET_BYTES,
};
use pg_phenotype_core::{PedigreeArg, TraitKind};
use proptest::prelude::*;

fn run(ped: &Ped, values: &[f64], kind: TraitKind, se: Se) -> RelativePairCorrelation {
    correlation_with(
        PedigreeArg::Columns(ped.input()),
        &[trait_of(values, kind)],
        None,
        None,
        DEFAULT_MEMORY_BUDGET_BYTES,
        se,
        se != Se::Off,
    )
    .expect("a valid pedigree and trait")
}

fn estimates(res: &RelativePairCorrelation) -> impl Iterator<Item = (&str, &Estimate)> {
    res.cells
        .iter()
        .flat_map(|c| c.estimates.iter().map(move |e| (c.category.as_str(), e)))
}

/// `a` and `b` have the same cells and estimates, each with one undefined
/// reason or values within `tol` of each other.  Returns how many defined
/// values it compared.
fn assert_same_values(
    a: &RelativePairCorrelation,
    b: &RelativePairCorrelation,
    tol: f64,
) -> Result<usize, TestCaseError> {
    let shape = |r: &RelativePairCorrelation| {
        r.cells
            .iter()
            .map(|c| {
                (
                    c.category.clone(),
                    c.estimates.iter().map(|e| e.estimator).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    prop_assert_eq!(shape(a), shape(b));
    let mut compared = 0;
    for ((cat, x), (_, y)) in estimates(a).zip(estimates(b)) {
        match (x.value, y.value) {
            (Ok(p), Ok(q)) => {
                prop_assert!(
                    (p - q).abs() <= tol,
                    "{} {:?}: {} vs {}",
                    cat,
                    x.estimator,
                    p,
                    q
                );
                compared += 1;
            }
            (p, q) => prop_assert_eq!(p.err(), q.err(), "{}", cat),
        }
        prop_assert_eq!(x.boundary, y.boundary, "{}", cat);
    }
    Ok(compared)
}

/// Defined estimates in `cells`.
fn defined(cells: &[Cell]) -> usize {
    cells
        .iter()
        .flat_map(|c| &c.estimates)
        .filter(|e| e.value.is_ok())
        .count()
}

/// A pedigree, values on its rows, and a shuffled row order.
fn shuffled(
    ped: impl Strategy<Value = Ped>,
) -> impl Strategy<Value = (Ped, Vec<u8>, TraitKind, Vec<usize>)> {
    (ped, codes(), kind()).prop_flat_map(|(ped, codes, kind)| {
        let order = Just((0..ped.len()).collect::<Vec<_>>()).prop_shuffle();
        (Just(ped), Just(codes), Just(kind), order)
    })
}

/// Cases per property in an ordinary run: each fits every cell of a
/// pedigree, so fewer than a cheap property draws.
const CASES: u32 = 64;

/// Every defined correlation is in [-1, 1].
#[test]
fn estimates_are_correlations() {
    let strategy = (pedigree(), codes(), kind());
    check(
        "in range",
        CASES,
        2.0,
        strategy,
        |(ped, codes, kind), checked| {
            let v = values(&codes, ped.len(), kind);
            let res = run(&ped, &v, kind, Se::Off);
            for (cat, e) in estimates(&res) {
                if let Ok(r) = e.value {
                    prop_assert!(
                        (-1.0..=1.0).contains(&r),
                        "{} {:?}: {}",
                        cat,
                        e.estimator,
                        r
                    );
                    checked.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        },
    );
}

/// A symmetric cell does not depend on the row order: its pairs are
/// unordered.  A directed cell can, where a loop puts one pair in its
/// category both ways round; see the outbred property.
#[test]
fn symmetric_cells_do_not_depend_on_row_order() {
    check(
        "symmetric cells",
        CASES,
        1.3,
        shuffled(pedigree()),
        |(ped, codes, kind, order), checked| {
            let v = values(&codes, ped.len(), kind);
            let shuffled_v: Vec<f64> = order.iter().map(|&r| v[r]).collect();
            let a = run(&ped, &v, kind, Se::Off);
            let b = run(&ped.permuted(&order), &shuffled_v, kind, Se::Off);
            for (x, y) in a.cells.iter().zip(&b.cells) {
                if x.symmetric {
                    prop_assert_eq!(x, y);
                    checked.fetch_add(defined(std::slice::from_ref(x)), Ordering::Relaxed);
                }
            }
            Ok(())
        },
    );
}

/// Without loops, every cell is row-order free.
#[test]
fn every_cell_is_row_order_free_without_loops() {
    let strategy = shuffled(outbred_pedigree());
    check(
        "outbred cells",
        CASES,
        3.0,
        strategy,
        |(ped, codes, kind, order), checked| {
            let v = values(&codes, ped.len(), kind);
            let shuffled_v: Vec<f64> = order.iter().map(|&r| v[r]).collect();
            let a = run(&ped, &v, kind, Se::Off);
            let b = run(&ped.permuted(&order), &shuffled_v, kind, Se::Off);
            prop_assert_eq!(&a.cells, &b.cells);
            checked.fetch_add(defined(&a.cells), Ordering::Relaxed);
            Ok(())
        },
    );
}

/// `a x + b` on the trait transforms both members, so Pearson keeps its
/// sign as well as its value.  `a` a power of two and `b` an integer keep
/// the quarter-integer values exact.
#[test]
fn an_affine_map_keeps_pearson() {
    let strategy = (
        pedigree(),
        codes(),
        prop_oneof![Just(-4.0), Just(-0.5), Just(0.5), Just(2.0), Just(8.0)],
        -3i32..=3,
    );
    check(
        "affine",
        CASES,
        1.8,
        strategy,
        |(ped, codes, a, b), checked| {
            let v = values(&codes, ped.len(), TraitKind::Continuous);
            let mapped: Vec<f64> = v.iter().map(|x| a * x + f64::from(b)).collect();
            let x = run(&ped, &v, TraitKind::Continuous, Se::Off);
            let y = run(&ped, &mapped, TraitKind::Continuous, Se::Off);
            checked.fetch_add(assert_same_values(&x, &y, 1e-12)?, Ordering::Relaxed);
            Ok(())
        },
    );
}

/// `1 - x` flips both members' levels, which rotates every 2 x 2 table
/// by 180 degrees: the tetrachoric and phi correlations stay.
#[test]
fn flipping_the_levels_keeps_the_binary_correlations() {
    check(
        "flipped levels",
        CASES,
        2.8,
        (pedigree(), codes()),
        |(ped, codes), checked| {
            let v = values(&codes, ped.len(), TraitKind::Binary);
            let flipped: Vec<f64> = v.iter().map(|x| 1.0 - x).collect();
            let x = run(&ped, &v, TraitKind::Binary, Se::Off);
            let y = run(&ped, &flipped, TraitKind::Binary, Se::Off);
            checked.fetch_add(assert_same_values(&x, &y, 1e-12)?, Ordering::Relaxed);
            Ok(())
        },
    );
}

/// One group's influences: their sum, the sum of their absolute values,
/// and how many there are.
#[derive(Clone, Copy, Default)]
struct Group {
    sum: f64,
    absolute: f64,
    entries: usize,
}

impl Group {
    fn add(&mut self, phi: f64) {
        self.sum += phi;
        self.absolute += phi.abs();
        self.entries += 1;
    }
}

/// The inclusion-exclusion meat of ADR 0013 over one cell's entries, and
/// an allowance for its rounding: `Σ_m Φ_m² + Σ_i S_i² − Σ_(i,m) S_(i,m)²`,
/// a pair with no MRCA its own cluster, each entry counted for both of its
/// members.  The groups are summed in key order, so a seed replays.
///
/// The allowance, in units of `ε`, is `Σ_g (8 n_g + 2 G) A_g²` over all
/// `G` groups, `n_g` a group's entries and `A_g = Σ (|φ| + 1/n)` over
/// them, `n` the cell's entries.  A sum of terms of total size `A_g` in
/// at most `2 n_g` additions (the engine adds a cluster's rows, up to two
/// per entry) rounds by at most `2 n_g ε A_g`, so its square by
/// `4 n_g ε A_g²`, and adding the `G` squares by at most `G ε Σ A_g²`;
/// the engine and this oracle each round so, in different orders.  The
/// sizes are absolute, never a signed sum: a group whose influences cancel
/// rounds by their sizes.  The `1/n` is an estimate, not a bound: an
/// influence is itself a sum of terms that cancel (to almost nothing when
/// `r` is near ±1), each of order `1/n` times the `(mx/sx)²` of
/// `Quadratic::standardised`, which is above 1 on a nearly constant side.
/// The worst-case counts leave room for that; commit 9ab7eb2 records the
/// headroom measured when the allowance was set.
fn meat(first: &[i64], second: &[i64], mrca: &[i64], phi: &[f64]) -> (f64, f64) {
    let mut cluster: BTreeMap<(i64, i64), Group> = BTreeMap::new();
    let mut member: BTreeMap<i64, Group> = BTreeMap::new();
    let mut member_cluster: BTreeMap<(i64, (i64, i64)), Group> = BTreeMap::new();
    for e in 0..phi.len() {
        let key = if mrca[e] == -1 {
            (-1 - first[e].min(second[e]), first[e].max(second[e]))
        } else {
            (mrca[e], -1)
        };
        cluster.entry(key).or_default().add(phi[e]);
        for i in [first[e], second[e]] {
            member.entry(i).or_default().add(phi[e]);
            member_cluster.entry((i, key)).or_default().add(phi[e]);
        }
    }
    let n_groups = (cluster.len() + member.len() + member_cluster.len()) as f64;
    let n = phi.len() as f64;
    let squares = |g: &Group| g.sum * g.sum;
    let allowance = |g: &Group| {
        let (entries, size) = (g.entries as f64, g.absolute + g.entries as f64 / n);
        (8.0 * entries + 2.0 * n_groups) * size * size
    };
    let (c, m, mc) = (cluster.values(), member.values(), member_cluster.values());
    let meat = c.clone().map(squares).sum::<f64>() + m.clone().map(squares).sum::<f64>()
        - mc.clone().map(squares).sum::<f64>();
    (meat, c.chain(m).chain(mc).map(allowance).sum())
}

/// Each SE is the root of the meat its exported influences give, and
/// asking for the SE and influences changes no estimate.
#[test]
fn the_se_is_the_meat_of_its_influences() {
    // Half the cases of the other properties: each fits with the SE.
    let strategy = (pedigree(), codes(), kind());
    check(
        "se meat",
        CASES / 2,
        1.2,
        strategy,
        |(ped, codes, kind), checked| {
            let v = values(&codes, ped.len(), kind);
            let res = run(&ped, &v, kind, Se::Estimates);
            let entries = res.entries.as_ref().expect("influences were asked for");
            for (cell, ent) in res.cells.iter().zip(entries) {
                for (e, phi) in cell.estimates.iter().zip(&ent.influence) {
                    let (Some(Ok(se)), Some(phi)) = (e.se, phi) else {
                        continue;
                    };
                    let (meat, allowance) = meat(&ent.first_id, &ent.second_id, &ent.mrca_id, phi);
                    let (gap, tol) = ((se * se - meat).abs(), f64::EPSILON * allowance);
                    prop_assert!(
                        gap <= tol,
                        "{} {:?}: se² {} vs meat {}, {} entries, gap {:e} of tolerance {:e}",
                        cell.category,
                        e.estimator,
                        se * se,
                        meat,
                        phi.len(),
                        gap,
                        tol
                    );
                    checked.fetch_add(1, Ordering::Relaxed);
                }
            }
            let plain = correlation(
                PedigreeArg::Columns(ped.input()),
                &[trait_of(&v, kind)],
                None,
                None,
                DEFAULT_MEMORY_BUDGET_BYTES,
                Se::Off,
            )
            .expect("a valid pedigree and trait");
            assert_same_values(&plain, &res, 0.0).map(|_| ())
        },
    );
}

/// The groups the additivity property requests: each a run of categories
/// of one kind (symmetric, or directed with one pair of roles).
const GROUPS: [(&str, &[&str]); 5] = [
    ("parents", &["MO", "FO"]),
    ("sibs", &["FS", "MHS", "PHS"]),
    ("ancestors", &["GP", "GGP", "GGGP"]),
    ("avuncular", &["Av", "HAv", "GAv"]),
    ("cousins", &["1C", "H1C", "2C"]),
];

/// The sum of `cells`' counts; `None` for no cells or cells whose
/// counts differ in kind.
fn summed(cells: &[&Cell]) -> Option<Counts> {
    let mut counts = cells.iter().map(|c| c.counts);
    let first = counts.next()?;
    counts.try_fold(first, |a, b| match (a, b) {
        (Counts::Table(s), Counts::Table(t)) => Some(Counts::Table(
            [0, 1].map(|i| [0, 1].map(|j| s[i][j] + t[i][j])),
        )),
        (Counts::Concordance(s), Counts::Concordance(t)) => {
            Some(Counts::Concordance([0, 1, 2].map(|i| s[i] + t[i])))
        }
        (Counts::None, Counts::None) => Some(Counts::None),
        _ => None,
    })
}

/// A Category Group's cell counts what its member categories' cells count:
/// `n`, `n_dropped`, `n_entries` and the binary counts are their sums.
#[test]
fn a_group_cell_sums_its_members() {
    let entry = |name: &str, members: &[&str], position| Entry::Group {
        name: name.to_string(),
        members: members.iter().map(|m| m.to_string()).collect(),
        position,
    };
    let groups: Vec<Entry> = GROUPS
        .iter()
        .enumerate()
        .map(|(i, (name, members))| entry(name, members, i))
        .collect();
    let singles: Vec<Entry> = GROUPS
        .iter()
        .flat_map(|(_, members)| members.iter())
        .enumerate()
        .map(|(i, code)| Entry::Category {
            code: code.to_string(),
            position: i,
        })
        .collect();
    check(
        "groups",
        CASES,
        0.9,
        (pedigree(), codes(), kind()),
        |(ped, codes, kind), checked| {
            let v = values(&codes, ped.len(), kind);
            let call = |entries: &[Entry]| {
                correlation(
                    PedigreeArg::Columns(ped.input()),
                    &[trait_of(&v, kind)],
                    None,
                    Some(entries),
                    DEFAULT_MEMORY_BUDGET_BYTES,
                    Se::Off,
                )
                .expect("a valid pedigree, trait and request")
            };
            let (grouped, single) = (call(&groups), call(&singles));
            prop_assert_eq!(grouped.cells.len(), GROUPS.len());
            prop_assert_eq!(
                single.cells.len(),
                GROUPS.iter().map(|(_, m)| m.len()).sum::<usize>()
            );
            for (cell, (name, members)) in grouped.cells.iter().zip(GROUPS) {
                prop_assert_eq!(&cell.category, name);
                let parts: Vec<&Cell> = single
                    .cells
                    .iter()
                    .filter(|c| members.contains(&c.category.as_str()))
                    .collect();
                prop_assert_eq!(parts.len(), members.len());
                let sum = |f: fn(&Cell) -> u64| parts.iter().map(|c| f(c)).sum::<u64>();
                prop_assert_eq!(cell.n, sum(|c| c.n), "{} n", name);
                prop_assert_eq!(cell.n_dropped, sum(|c| c.n_dropped), "{} n_dropped", name);
                let entries: Option<u64> = parts.iter().map(|c| c.n_entries).sum();
                prop_assert_eq!(cell.n_entries, entries, "{} n_entries", name);
                prop_assert_eq!(Some(cell.counts), summed(&parts), "{} counts", name);
                checked.fetch_add(usize::from(cell.n > 0), Ordering::Relaxed);
            }
            Ok(())
        },
    );
}
