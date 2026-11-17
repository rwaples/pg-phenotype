//! PA-FGRS relatives through a shared external parent id (pg-phenotype #40,
//! pedigree-graph #67).
//!
//! An external parent id (one with no row) that two or more rows name is a
//! founder of the kinship recurrence, so the rows that share it are related.
//! Every relative's kinship to its proband, and every kinship among the
//! relatives, is the pairwise walker's on that extended pedigree, bit for
//! bit.  Renaming each external id to one no other row names gives the
//! values of a missing parent.

#![expect(
    clippy::expect_used,
    reason = "test helpers outside #[test], which clippy.toml does not exempt"
)]

mod common;

use std::sync::atomic::Ordering;

use common::{check, pedigree, Ped};
use pedigree_graph_core::columns::GraphColumns;
use pedigree_graph_core::graph::{self, Columns, Limits, SexEncoding};
use pedigree_graph_core::kinship::{ExternalFounders, KinshipIndex};
use pedigree_graph_core::topology;
use pg_phenotype_core::pafgrs::{kinship_threshold, prepare, Prep};
use pg_phenotype_core::{PedigreeArg, PedigreeInput};
use proptest::prelude::*;

const CASES: u32 = 64;

fn prep(ped: &Ped, ndegree: i64) -> Prep {
    prepare(PedigreeArg::Columns(ped.input()), ndegree, None).expect("a valid pedigree")
}

/// Kinship of every pair of rows by pedigree-graph's walker on the pedigree
/// with its shared external ids as founders, `n * n`.
fn walked(ped: &Ped) -> Vec<f32> {
    let built = graph::build(
        Columns {
            ids: &ped.ids,
            mother: &ped.mother,
            father: &ped.father,
            twin: Some(&ped.twin),
            sex: None,
            generation: None,
            birth_year: None,
        },
        SexEncoding::Simace,
        Limits::default(),
    )
    .expect("a valid pedigree");
    let columns = GraphColumns {
        mother: &built.mother_rows,
        father: &built.father_rows,
        twin: &built.twin_rows,
    };
    let depth = topology::structural_depth(&built.mother_rows, &built.father_rows);
    let kped = columns
        .kinship(&depth, &built.mother_ids, &built.father_ids)
        .expect("kinship input");
    let founders = ExternalFounders::new(&kped).expect("founders");
    let index = KinshipIndex::new(&founders.pedigree()).expect("index");
    let mut walker = index.walker().expect("walker");
    let n = ped.len();
    let mut phi = vec![0.0f32; n * n];
    for a in 0..n {
        for b in 0..n {
            phi[a * n + b] = walker.resolve(a as i32, b as i32).expect("walk");
        }
    }
    phi
}

/// `ped` with each external father and some missing mothers renamed to one
/// of a few shared external ids, chosen by `dice`.  Co-twins keep their
/// parents, so they stay one genome.
fn shared(ped: &Ped, dice: &[u8]) -> Ped {
    let mut out = ped.clone();
    for r in 0..ped.len() {
        if ped.twin[r] >= 0 {
            continue;
        }
        let die = dice[r % dice.len()];
        if ped.father[r] >= 99_000 {
            out.father[r] = 99_000 + i64::from(die % 3);
        }
        if ped.mother[r] < 0 && die % 4 == 0 {
            out.mother[r] = 98_000 + i64::from(die / 4 % 2);
        }
    }
    out
}

/// Every relative's kinship, and every kinship among a proband's
/// relatives, as the walker's on `ped`; each relative clears the
/// threshold.  Returns how many pairs had a positive kinship only through
/// a shared external id, where `plain` is that pedigree's values without
/// founders (`None` to skip).
fn matches_walker(ped: &Ped, ndegree: i64, plain: Option<&[f32]>) -> Result<usize, TestCaseError> {
    let prep = prep(ped, ndegree);
    let phi = walked(ped);
    let n = ped.len();
    let threshold = kinship_threshold(ndegree as u8);
    let mut through_founder = 0;
    for (i, &p) in prep.probands().iter().enumerate() {
        let p = p as usize;
        let (rows, kin) = prep.relatives(i);
        for (&r, &k) in rows.iter().zip(kin) {
            let want = phi[p * n + r as usize];
            prop_assert_eq!(k.to_bits(), want.to_bits(), "proband {} relative {}", p, r);
            prop_assert!(f64::from(k) >= threshold);
            if plain.is_some_and(|plain| plain[p * n + r as usize] == 0.0) {
                through_founder += 1;
            }
        }
        for j in 0..rows.len() {
            for k in j + 1..rows.len() {
                let want = phi[rows[j] as usize * n + rows[k] as usize];
                let got = prep.pair_kinship(i, j, k);
                prop_assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "proband {} pair {} {}",
                    p,
                    j,
                    k
                );
            }
        }
    }
    Ok(through_founder)
}

/// Relatives and their kinship through shared external ids are the
/// walker's on the extended pedigree, and some of them are related only
/// through such an id.
#[test]
fn relatives_through_a_shared_external_id_are_the_walkers() {
    check(
        "external founders",
        CASES,
        0.2,
        (pedigree(), 1i64..=3, prop::collection::vec(any::<u8>(), 64)),
        |(ped, ndegree, dice), checked| {
            let merged = shared(&ped, &dice);
            let plain = walked(&ped);
            let gained = matches_walker(&merged, ndegree, Some(&plain))?;
            // Adding a shared founder only adds paths, so every relative of
            // the pedigree without one stays a relative.
            let (before, after) = (prep(&ped, ndegree), prep(&merged, ndegree));
            for i in 0..before.n_probands() {
                let after_rows = after.relatives(i).0;
                for r in before.relatives(i).0 {
                    prop_assert!(after_rows.contains(r), "proband {} lost {}", i, r);
                }
            }
            checked.fetch_add(usize::from(gained > 0), Ordering::Relaxed);
            Ok(())
        },
    );
}

/// External ids that no two rows share give every value a missing parent
/// gives, bit for bit.
#[test]
fn unshared_external_ids_are_missing_parents() {
    check(
        "unshared external ids",
        CASES,
        1.0,
        (pedigree(), 1i64..=3),
        |(ped, ndegree), checked| {
            let mut missing = ped.clone();
            for f in &mut missing.father {
                if *f >= 99_000 {
                    *f = -1;
                }
            }
            let (a, b) = (prep(&ped, ndegree), prep(&missing, ndegree));
            prop_assert_eq!(a.probands(), b.probands());
            for i in 0..a.n_probands() {
                let ((ra, ka), (rb, kb)) = (a.relatives(i), b.relatives(i));
                prop_assert_eq!(ra, rb);
                let bits = |k: &[f32]| k.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                prop_assert_eq!(bits(ka), bits(kb));
                for j in 0..ra.len() {
                    for k in j + 1..ra.len() {
                        prop_assert_eq!(
                            a.pair_kinship(i, j, k).to_bits(),
                            b.pair_kinship(i, j, k).to_bits()
                        );
                    }
                }
            }
            matches_walker(&ped, ndegree, None)?;
            checked.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    );
}

/// Paternal half sibs 3 and 4 by external father 99, and 5, a child of 4
/// by external father 98 that 6 also names: 5 and 6 are paternal half sibs.
#[test]
fn paternal_half_sibs_by_an_external_father_are_relatives() {
    let ids = [1i64, 2, 3, 4, 5, 6, 7];
    let mother = [-1i64, -1, 1, 2, 4, 7, -1];
    let father = [-1i64, -1, 99, 99, 98, 98, -1];
    let prep = prepare(
        PedigreeArg::Columns(PedigreeInput {
            ids: &ids,
            mother: &mother,
            father: &father,
            twin: None,
            sex: None,
        }),
        2,
        None,
    )
    .expect("a valid pedigree");
    // Proband 3 (row 2): mother 1 and half sib 4; half nephew 5 is third
    // degree.
    let (rows, kin) = prep.relatives(2);
    assert_eq!(rows, &[0, 3]);
    assert_eq!(kin, &[0.25, 0.125]);
    // Proband 5 (row 4): grandmother 2, mother 4, half sib 6.
    let (rows, kin) = prep.relatives(4);
    assert_eq!(rows, &[1, 3, 5]);
    assert_eq!(kin, &[0.125, 0.25, 0.125]);
    // Grandmother 2 and mother 4 are parent and child; half sib 6 is
    // related to neither.
    assert_eq!(prep.pair_kinship(4, 0, 1), 0.25);
    assert_eq!(prep.pair_kinship(4, 0, 2), 0.0);
    assert_eq!(prep.pair_kinship(4, 1, 2), 0.0);
}
