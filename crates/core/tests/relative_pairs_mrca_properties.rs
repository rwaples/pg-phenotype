//! The MRCA of each exported entry checked against its definition (ADR
//! 0013, issue #31 item 4): an ancestor of both members at the category's
//! meioses, through either parent, rather than the walk's first hit.

#![expect(
    clippy::expect_used,
    reason = "test helpers outside #[test], which clippy.toml does not exempt"
)]

mod common;

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::Ordering;

use common::{check, codes, kind, pedigree, trait_of, values, Ped};
use pg_phenotype_core::relative_pairs::{
    correlation_with, meioses, RelativePairCorrelation, Se, DEFAULT_MEMORY_BUDGET_BYTES,
};
use pg_phenotype_core::{PedigreeArg, PedigreeInput, TraitKind};
use proptest::prelude::*;

fn run(ped: &Ped, values: &[f64], kind: TraitKind) -> RelativePairCorrelation {
    correlation_with(
        PedigreeArg::Columns(ped.input()),
        &[trait_of(values, kind)],
        None,
        None,
        DEFAULT_MEMORY_BUDGET_BYTES,
        Se::Estimates,
        true,
    )
    .expect("a valid pedigree and trait")
}

/// Each id's known parents, mother first: a row's, by id; an external
/// parent has none.
fn parents(ped: &Ped) -> HashMap<i64, Vec<i64>> {
    (0..ped.len())
        .map(|r| {
            let known = [ped.mother[r], ped.father[r]]
                .into_iter()
                .filter(|&p| p >= 0);
            (ped.ids[r], known.collect())
        })
        .collect()
}

/// The ids exactly `k` meioses up from `id`, through either parent.
fn up(parents: &HashMap<i64, Vec<i64>>, id: i64, k: u8) -> BTreeSet<i64> {
    let mut level = BTreeSet::from([id]);
    for _ in 0..k {
        level = level
            .iter()
            .flat_map(|i| parents.get(i).into_iter().flatten().copied())
            .collect();
    }
    level
}

/// Cases in an ordinary run: each fits every cell with its SE and
/// influences.
const CASES: u32 = 32;

/// Every entry's MRCA is `up` meioses above its first member and `down`
/// above its second (the second itself for a lineal category); an MZ
/// pair's is a parent of both, or `-1` when it has none.  A symmetric
/// pair's two orientations share one MRCA.
#[test]
fn every_mrca_is_a_common_ancestor_at_the_categorys_meioses() {
    let meioses: HashMap<&str, (u8, u8)> = meioses().map(|(c, u, d)| (c, (u, d))).collect();
    check(
        "mrca",
        CASES,
        25.0,
        (pedigree(), codes(), kind()),
        |(ped, codes, kind), checked| {
            let v = values(&codes, ped.len(), kind);
            let res = run(&ped, &v, kind);
            let parents = parents(&ped);
            let entries = res.entries.as_ref().expect("influences were asked for");
            prop_assert_eq!(res.cells.len(), entries.len());
            for (cell, ent) in res.cells.iter().zip(entries) {
                let code = cell.members[0].code();
                let (u, d) = meioses[code];
                let mut by_pair: HashMap<(i64, i64), i64> = HashMap::new();
                for e in 0..ent.mrca_id.len() {
                    let (a, b, m) = (ent.first_id[e], ent.second_id[e], ent.mrca_id[e]);
                    let where_ = format!("{code} ({a}, {b}) mrca {m}");
                    if m == -1 {
                        prop_assert!(
                            code == "MZ" && parents[&a].is_empty() && parents[&b].is_empty(),
                            "{}: -1 for a pair with an ancestor",
                            where_
                        );
                    } else if u == 0 {
                        prop_assert_eq!(code, "MZ");
                        prop_assert!(
                            parents[&a].contains(&m) && parents[&b].contains(&m),
                            "{}: not a parent of both",
                            where_
                        );
                    } else {
                        prop_assert!(
                            up(&parents, a, u).contains(&m),
                            "{}: not {} up from the first",
                            where_,
                            u
                        );
                        prop_assert!(
                            up(&parents, b, d).contains(&m),
                            "{}: not {} up from the second",
                            where_,
                            d
                        );
                    }
                    if cell.symmetric {
                        let key = (a.min(b), a.max(b));
                        let seen = *by_pair.entry(key).or_insert(m);
                        prop_assert_eq!(seen, m, "{}: the other orientation's MRCA", where_);
                    }
                    checked.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        },
    );
}

/// The walk matches a shared parent role for role because pedigree-graph
/// refuses an id that is one row's mother and another's father (its ADR
/// 0020), external ids included: the dual-role pedigree of 63af680 is an
/// error, not a pair with no MRCA.
#[test]
fn a_parent_in_both_roles_is_refused() {
    let nan = f64::NAN;
    // G1 0, G2 1, P 2; a 5 has mother P, b 6 has father P.
    let ids = [0i64, 1, 2, 3, 4, 5, 6];
    let mother = [-1i64, -1, 0, -1, -1, 2, 4];
    let father = [-1i64, -1, 1, -1, -1, 3, 2];
    // An external id in both roles: 90 is a's mother and b's father.
    let external = ([0i64, 1, 2], [90i64, -1, 91], [-1i64, 90, -1]);
    let v = [nan, nan, 0.0, nan, nan, 1.0, 2.0];
    for (ids, mother, father) in [
        (&ids[..], &mother[..], &father[..]),
        (&external.0[..], &external.1[..], &external.2[..]),
    ] {
        let pedigree = PedigreeInput {
            ids,
            mother,
            father,
            twin: None,
            sex: None,
        };
        let err = correlation_with(
            PedigreeArg::Columns(pedigree),
            &[trait_of(&v[..ids.len()], TraitKind::Continuous)],
            None,
            None,
            DEFAULT_MEMORY_BUDGET_BYTES,
            Se::Estimates,
            true,
        )
        .expect_err("a parent in both roles");
        assert_eq!(err.code(), "parent_in_both_roles");
    }
}
