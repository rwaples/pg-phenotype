//! Properties of the Relative-Pair Correlation of one trait.
//!
//! Exact comparisons are between two calls in one process, never against a
//! stored value, so they hold on every platform.

#![expect(
    clippy::expect_used,
    reason = "test helpers outside #[test], which clippy.toml does not exempt"
)]

mod common;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{codes, kind, outbred_pedigree, pedigree, trait_of, values, Ped};
use pg_phenotype_core::relative_pairs::{
    correlation, correlation_with, Estimate, RelativePairCorrelation, Se,
    DEFAULT_MEMORY_BUDGET_BYTES,
};
use pg_phenotype_core::{PedigreeArg, TraitKind};
use proptest::prelude::*;
use proptest::test_runner::TestRunner;

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
/// reason or values within `tol` of each other.
fn assert_same_values(
    a: &RelativePairCorrelation,
    b: &RelativePairCorrelation,
    tol: f64,
) -> Result<(), TestCaseError> {
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
    for ((cat, x), (_, y)) in estimates(a).zip(estimates(b)) {
        match (x.value, y.value) {
            (Ok(p), Ok(q)) => prop_assert!(
                (p - q).abs() <= tol,
                "{} {:?}: {} vs {}",
                cat,
                x.estimator,
                p,
                q
            ),
            (p, q) => prop_assert_eq!(p.err(), q.err(), "{}", cat),
        }
        prop_assert_eq!(x.boundary, y.boundary, "{}", cat);
    }
    Ok(())
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Every defined correlation is in [-1, 1].
    #[test]
    fn estimates_are_correlations(ped in pedigree(), codes in codes(), kind in kind()) {
        let v = values(&codes, ped.len(), kind);
        let res = run(&ped, &v, kind, Se::Off);
        for (cat, e) in estimates(&res) {
            if let Ok(r) = e.value {
                prop_assert!((-1.0..=1.0).contains(&r), "{} {:?}: {}", cat, e.estimator, r);
            }
        }
    }

    /// A symmetric cell does not depend on the row order: its pairs are
    /// unordered.  A directed cell can, where a loop puts one pair in its
    /// category both ways round; see the outbred property.
    #[test]
    fn symmetric_cells_do_not_depend_on_row_order((ped, codes, kind, order) in shuffled(pedigree())) {
        let v = values(&codes, ped.len(), kind);
        let shuffled_v: Vec<f64> = order.iter().map(|&r| v[r]).collect();
        let a = run(&ped, &v, kind, Se::Off);
        let b = run(&ped.permuted(&order), &shuffled_v, kind, Se::Off);
        for (x, y) in a.cells.iter().zip(&b.cells) {
            if x.symmetric {
                prop_assert_eq!(x, y);
            }
        }
    }

    /// Without loops, every cell is row-order free.
    #[test]
    fn every_cell_is_row_order_free_without_loops((ped, codes, kind, order) in shuffled(outbred_pedigree())) {
        let v = values(&codes, ped.len(), kind);
        let shuffled_v: Vec<f64> = order.iter().map(|&r| v[r]).collect();
        let a = run(&ped, &v, kind, Se::Off);
        let b = run(&ped.permuted(&order), &shuffled_v, kind, Se::Off);
        prop_assert_eq!(a.cells, b.cells);
    }

    /// `a x + b` on the trait transforms both members, so Pearson keeps its
    /// sign as well as its value.  `a` a power of two and `b` an integer keep
    /// the quarter-integer values exact.
    #[test]
    fn an_affine_map_keeps_pearson(
        ped in pedigree(),
        codes in codes(),
        a in prop_oneof![Just(-4.0), Just(-0.5), Just(0.5), Just(2.0), Just(8.0)],
        b in -3i32..=3,
    ) {
        let v = values(&codes, ped.len(), TraitKind::Continuous);
        let mapped: Vec<f64> = v.iter().map(|x| a * x + f64::from(b)).collect();
        let x = run(&ped, &v, TraitKind::Continuous, Se::Off);
        let y = run(&ped, &mapped, TraitKind::Continuous, Se::Off);
        assert_same_values(&x, &y, 1e-12)?;
    }

    /// `1 - x` flips both members' levels, which rotates every 2 x 2 table
    /// by 180 degrees: the tetrachoric and phi correlations stay.
    #[test]
    fn flipping_the_levels_keeps_the_binary_correlations(ped in pedigree(), codes in codes()) {
        let v = values(&codes, ped.len(), TraitKind::Binary);
        let flipped: Vec<f64> = v.iter().map(|x| 1.0 - x).collect();
        let x = run(&ped, &v, TraitKind::Binary, Se::Off);
        let y = run(&ped, &flipped, TraitKind::Binary, Se::Off);
        assert_same_values(&x, &y, 1e-12)?;
    }
}

/// The inclusion-exclusion meat of ADR 0013 over one cell's entries, and
/// its scale (the three sums of squares, added): `Σ_m Φ_m² + Σ_i S_i² −
/// Σ_(i,m) S_(i,m)²`, a pair with no MRCA its own cluster, each entry
/// counted for both of its members.
fn meat(first: &[i64], second: &[i64], mrca: &[i64], phi: &[f64]) -> (f64, f64) {
    let mut cluster: HashMap<(i64, i64), f64> = HashMap::new();
    let mut member: HashMap<i64, f64> = HashMap::new();
    let mut member_cluster: HashMap<(i64, (i64, i64)), f64> = HashMap::new();
    for e in 0..phi.len() {
        let key = if mrca[e] == -1 {
            (-1 - first[e].min(second[e]), first[e].max(second[e]))
        } else {
            (mrca[e], -1)
        };
        *cluster.entry(key).or_default() += phi[e];
        for i in [first[e], second[e]] {
            *member.entry(i).or_default() += phi[e];
            *member_cluster.entry((i, key)).or_default() += phi[e];
        }
    }
    let squares = |v: Vec<f64>| v.iter().map(|x| x * x).sum::<f64>();
    let (c, m, mc) = (
        squares(cluster.into_values().collect()),
        squares(member.into_values().collect()),
        squares(member_cluster.into_values().collect()),
    );
    (c + m - mc, c + m + mc)
}

/// Each SE is the root of the meat its exported influences give, and
/// asking for the SE and influences changes no estimate.
#[test]
fn the_se_is_the_meat_of_its_influences() {
    let checked = AtomicUsize::new(0);
    // A hand-built runner knows no source file unless told, and then
    // neither saves nor replays a failure.
    let mut runner = TestRunner::new(ProptestConfig {
        source_file: Some(file!()),
        ..ProptestConfig::with_cases(32)
    });
    let cases = (pedigree(), codes(), kind());
    let result = runner.run(&cases, |(ped, codes, kind)| {
        let v = values(&codes, ped.len(), kind);
        let res = run(&ped, &v, kind, Se::Estimates);
        let entries = res.entries.as_ref().expect("influences were asked for");
        for (cell, ent) in res.cells.iter().zip(entries) {
            for (e, phi) in cell.estimates.iter().zip(&ent.influence) {
                let (Some(Ok(se)), Some(phi)) = (e.se, phi) else {
                    continue;
                };
                let (meat, scale) = meat(&ent.first_id, &ent.second_id, &ent.mrca_id, phi);
                let tol = 4096.0 * f64::EPSILON * scale;
                prop_assert!(
                    (se * se - meat).abs() <= tol,
                    "{} {:?}: se² {} vs meat {}",
                    cell.category,
                    e.estimator,
                    se * se,
                    meat
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
        assert_same_values(&plain, &res, 0.0)
    });
    result.unwrap();
    // Not vacuous: most cases have SEs to check.
    let checked = checked.into_inner();
    assert!(checked >= 32, "{checked} SEs checked");
}
