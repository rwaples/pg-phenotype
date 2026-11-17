//! PA-FGRS relatives through a shared external parent id (pg-phenotype #40,
//! pedigree-graph #67).
//!
//! An external parent id (one with no row) that two or more rows name is a
//! founder of the kinship recurrence, so the rows that share it are related.
//! Every relative's kinship to its proband, and every kinship among the
//! relatives, is the pairwise walker's on that extended pedigree, bit for
//! bit, and equals the values on the pedigree with every external id given
//! a parentless row of its own, an oracle that never runs
//! `ExternalFounders`.  Renaming each external id to one no other row names
//! gives the values of a missing parent.

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

/// The lower row of `r`'s MZ pair, or `None` when `r` is no twin.
fn pair_row(ped: &Ped, r: usize) -> Option<usize> {
    let co = ped.ids.iter().position(|&id| id == ped.twin[r])?;
    Some(r.min(co))
}

/// `ped` with some MZ pairs' father an external id that only the pair
/// names (the common generator gives a twin none).
fn twin_external(ped: &Ped, dice: &[u8]) -> Ped {
    let mut out = ped.clone();
    for r in 0..ped.len() {
        if let Some(low) = pair_row(ped, r) {
            if dice[low % dice.len()] % 3 == 0 {
                out.father[r] = 97_000 + low as i64;
            }
        }
    }
    out
}

/// `ped` with each external father and some missing mothers renamed to one
/// of a few shared external ids, chosen by `dice`.  Co-twins take the same
/// one, so an id can be named by a pair of co-twins alone or by co-twins
/// and other rows.
fn shared(ped: &Ped, dice: &[u8]) -> Ped {
    let mut out = ped.clone();
    for r in 0..ped.len() {
        let die = dice[pair_row(ped, r).unwrap_or(r) % dice.len()];
        if ped.twin[r] >= 0 {
            if ped.father[r] >= 97_000 {
                out.father[r] = 99_000 + i64::from(die % 3);
            }
            continue;
        }
        if ped.father[r] >= 99_000 {
            out.father[r] = 99_000 + i64::from(die % 3);
        }
        if ped.mother[r] < 0 && die % 4 == 0 {
            out.mother[r] = 98_000 + i64::from(die / 4 % 2);
        }
    }
    out
}

/// `ped` with a parentless row appended for every external parent id it
/// names, once or more: the pedigree in which no parent is external.
fn materialized(ped: &Ped) -> Ped {
    let mut out = ped.clone();
    let mut external: Vec<i64> = ped
        .mother
        .iter()
        .chain(&ped.father)
        .copied()
        .filter(|&id| id >= 0 && !ped.ids.contains(&id))
        .collect();
    external.sort_unstable();
    external.dedup();
    for id in external {
        out.ids.push(id);
        out.mother.push(-1);
        out.father.push(-1);
        out.twin.push(-1);
    }
    out
}

/// The relatives of every proband of `ped` and their kinship equal those of
/// [`materialized`] `ped` with its added rows left out, bit for bit: an
/// oracle that never runs `ExternalFounders`.
fn matches_materialized(ped: &Ped, ndegree: i64) -> Result<(), TestCaseError> {
    let n = ped.len() as u32;
    let (got, want) = (prep(ped, ndegree), prep(&materialized(ped), ndegree));
    for i in 0..got.n_probands() {
        let (rows, kin) = got.relatives(i);
        let (all_rows, all_kin) = want.relatives(i);
        let kept: Vec<usize> = (0..all_rows.len()).filter(|&j| all_rows[j] < n).collect();
        let want_rows: Vec<u32> = kept.iter().map(|&j| all_rows[j]).collect();
        let want_kin: Vec<u32> = kept.iter().map(|&j| all_kin[j].to_bits()).collect();
        let got_kin: Vec<u32> = kin.iter().map(|k| k.to_bits()).collect();
        prop_assert_eq!(rows, &want_rows[..], "proband {}", i);
        prop_assert_eq!(got_kin, want_kin, "proband {}", i);
        for (a, &ja) in kept.iter().enumerate() {
            for (b, &jb) in kept.iter().enumerate().skip(a + 1) {
                prop_assert_eq!(
                    got.pair_kinship(i, a, b).to_bits(),
                    want.pair_kinship(i, ja, jb).to_bits(),
                    "proband {} pair {} {}",
                    i,
                    a,
                    b
                );
            }
        }
    }
    Ok(())
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
        // About half of all cases gain one (549 of 1025 at PROPTEST_SCALE=16).
        0.2,
        (pedigree(), 1i64..=3, prop::collection::vec(any::<u8>(), 64)),
        |(ped, ndegree, dice), checked| {
            let ped = twin_external(&ped, &dice);
            let merged = shared(&ped, &dice);
            let plain = walked(&ped);
            let gained = matches_walker(&merged, ndegree, Some(&plain))?;
            matches_materialized(&merged, ndegree)?;
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
            matches_materialized(&ped, ndegree)?;
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
