//! Properties of the Relative-Pair Correlation of two traits: a sign flip
//! of one trait, a level flip of a binary one, and the traits' order
//! (issue #31 item 1).  They reach the cross-trait cells' product index,
//! the order of the trait digits in a cell's label, and the orientation of
//! a cross-trait table (issue #16).
//!
//! Values are exact comparisons between two calls in one process, on
//! quarter-integer values; SEs and covariances, whose cluster sums may add
//! in another order, compare within a tolerance.

#![expect(
    clippy::expect_used,
    reason = "test helpers outside #[test], which clippy.toml does not exempt"
)]

mod common;

use std::sync::atomic::Ordering;

use common::{check, codes, kind, pedigree, trait_of, values, Ped};
use pg_phenotype_core::ace::{
    self, AceCell, AceRequest, Components, CrudeEstimate, Estimator as AceEstimator, Pairs,
    Reason as AceReason, Solved, TraitFit, Weights,
};
use pg_phenotype_core::relative_pairs::{
    correlation, Cell, Counts, Estimate, Form, RelativePairCorrelation, Se,
    DEFAULT_MEMORY_BUDGET_BYTES,
};
use pg_phenotype_core::{PedigreeArg, TraitKind};
use proptest::prelude::*;

fn run(ped: &Ped, x: &[f64], y: &[f64], kind: TraitKind, se: Se) -> RelativePairCorrelation {
    correlation(
        PedigreeArg::Columns(ped.input()),
        &[trait_of(x, kind), trait_of(y, kind)],
        None,
        None,
        DEFAULT_MEMORY_BUDGET_BYTES,
        se,
    )
    .expect("a valid pedigree and two traits of one kind")
}

/// A pedigree and two traits of one kind on its rows.
fn two_traits(
    kind: impl Strategy<Value = TraitKind>,
) -> impl Strategy<Value = (Ped, Vec<f64>, Vec<f64>, TraitKind)> {
    (pedigree(), codes(), codes(), kind).prop_map(|(ped, cx, cy, kind)| {
        let n = ped.len();
        (ped, values(&cx, n, kind), values(&cy, n, kind), kind)
    })
}

/// The result's cells, then the within-person cell: the covariance's slots.
fn slots(res: &RelativePairCorrelation) -> Vec<&Cell> {
    res.cells.iter().chain(&res.within_person).collect()
}

fn cross(cell: &Cell) -> bool {
    cell.first_trait != cell.second_trait
}

fn transpose(counts: Counts) -> Counts {
    match counts {
        Counts::Table(t) => Counts::Table([[t[0][0], t[1][0]], [t[0][1], t[1][1]]]),
        other => other,
    }
}

/// `counts` with trait 0's levels flipped: a table's rows where the first
/// member reads trait 0, its columns where the second does; a
/// concordance, which only trait 0's own cells have, reversed.
fn flip_levels(cell: &Cell) -> Counts {
    match cell.counts {
        Counts::Table(mut t) => {
            if cell.first_trait == 0 {
                t.swap(0, 1);
            }
            if cell.second_trait == 0 {
                for row in &mut t {
                    row.swap(0, 1);
                }
            }
            Counts::Table(t)
        }
        Counts::Concordance([a, b, c]) if cell.first_trait == 0 => Counts::Concordance([c, b, a]),
        other => other,
    }
}

/// `cell` with each defined value negated and nothing else changed.
fn negated(cell: &Cell) -> Cell {
    let mut out = cell.clone();
    for e in &mut out.estimates {
        e.value = e.value.map(|v| -v);
    }
    out
}

/// Defined estimates in `cells`.
fn defined<'a>(cells: impl IntoIterator<Item = &'a Cell>) -> usize {
    cells
        .into_iter()
        .flat_map(|c| &c.estimates)
        .filter(|e| e.value.is_ok())
        .count()
}

/// A trait with its sign flipped.
fn neg(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| -x).collect()
}

/// A binary trait with its levels flipped.
fn flip(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| 1.0 - x).collect()
}

/// Cases per property in an ordinary run: each fits every cell of a
/// pedigree twice, for two traits.
const CASES: u32 = 64;

/// `x → −x` negates every cross-trait value, the within-person one
/// included, and leaves every same-trait cell bit for bit.
#[test]
fn negating_one_trait_negates_the_cross_trait_cells() {
    check(
        "negated trait",
        CASES,
        3.0,
        two_traits(Just(TraitKind::Continuous)),
        |(ped, x, y, kind), checked| {
            let a = run(&ped, &x, &y, kind, Se::Off);
            let b = run(&ped, &neg(&x), &y, kind, Se::Off);
            prop_assert_eq!(a.cells.len(), b.cells.len());
            prop_assert_eq!(&a.quantization_exponents, &b.quantization_exponents);
            for (p, q) in slots(&a).into_iter().zip(slots(&b)) {
                let expect = if cross(p) { negated(p) } else { p.clone() };
                prop_assert_eq!(&expect, q);
            }
            checked.fetch_add(
                defined(slots(&a).into_iter().filter(|c| cross(c))),
                Ordering::Relaxed,
            );
            Ok(())
        },
    );
}

/// `x → 1 − x` on a binary trait negates the cross-trait tetrachoric and
/// phi, the within-person ones included, and flips trait 0's levels in each
/// table; the boundary flags stay.  Trait 0's own cells keep their values
/// (the one-trait property) and trait 1's stay bit for bit.
#[test]
fn flipping_one_binary_trait_negates_the_cross_trait_cells() {
    check(
        "flipped trait",
        CASES,
        6.0,
        two_traits(Just(TraitKind::Binary)),
        |(ped, x, y, kind), checked| {
            let a = run(&ped, &x, &y, kind, Se::Off);
            let b = run(&ped, &flip(&x), &y, kind, Se::Off);
            prop_assert_eq!(a.cells.len(), b.cells.len());
            for (p, q) in slots(&a).into_iter().zip(slots(&b)) {
                let mut expect = if cross(p) { negated(p) } else { p.clone() };
                expect.counts = flip_levels(p);
                prop_assert_eq!(&expect, q);
            }
            checked.fetch_add(
                defined(slots(&a).into_iter().filter(|c| cross(c))),
                Ordering::Relaxed,
            );
            Ok(())
        },
    );
}

/// Where cell `p` of `[x, y]` is in the result for `[y, x]`: `(a, b)`
/// becomes `(1 − a, 1 − b)`, and a symmetric cross-trait cell, which has
/// only `(0, 1)`, stays.
fn swapped_index(p: &Cell, b: &RelativePairCorrelation) -> Option<usize> {
    let (f, s) = if p.symmetric && cross(p) {
        (0, 1)
    } else {
        (1 - p.first_trait, 1 - p.second_trait)
    };
    if p.category == "within_person" {
        return Some(b.cells.len());
    }
    b.cells
        .iter()
        .position(|q| q.category == p.category && q.first_trait == f && q.second_trait == s)
}

/// What cell `p` of `[x, y]` is in the result for `[y, x]`: the same
/// values and counts, its traits renamed, and a table that pairs a member
/// with itself or a symmetric pair's either member transposed.
fn swapped(p: &Cell) -> Cell {
    let mut out = p.clone();
    if p.category == "within_person" || (p.symmetric && cross(p)) {
        out.counts = transpose(p.counts);
    } else {
        out.first_trait = 1 - p.first_trait;
        out.second_trait = 1 - p.second_trait;
    }
    out
}

/// Reordering the traits to `[y, x]` maps each cell `(a, b)` to
/// `(1 − a, 1 − b)` with the same values, counts and pairs.
#[test]
fn reordering_the_traits_permutes_the_cells() {
    check(
        "reordered traits",
        CASES,
        10.0,
        two_traits(kind()),
        |(ped, x, y, kind), checked| {
            let a = run(&ped, &x, &y, kind, Se::Off);
            let b = run(&ped, &y, &x, kind, Se::Off);
            prop_assert_eq!(a.cells.len(), b.cells.len());
            let b_slots = slots(&b);
            for p in slots(&a) {
                let at = swapped_index(p, &b).expect("every cell has its image");
                prop_assert_eq!(&swapped(p), b_slots[at]);
            }
            checked.fetch_add(defined(slots(&a)), Ordering::Relaxed);
            Ok(())
        },
    );
}

/// Whether two floats from one quantity computed in two summation orders
/// agree: both NaN, or within `1e-10` (the covariances and SEs here are
/// at most about 1).
fn close(p: f64, q: f64) -> bool {
    (p.is_nan() && q.is_nan()) || (p - q).abs() <= 1e-10
}

/// Each estimate's SE matches: within [`close`] when both are defined,
/// with one reason otherwise.
fn assert_same_se(p: &Estimate, q: &Estimate, cell: &str) -> Result<(), TestCaseError> {
    match (p.se, q.se) {
        (Some(Ok(s)), Some(Ok(t))) => {
            prop_assert!(close(s, t), "{} {:?}: se {} vs {}", cell, p.estimator, s, t)
        }
        (s, t) => prop_assert_eq!(s, t, "{} {:?}", cell, p.estimator),
    }
    Ok(())
}

/// Under the slot map `to` and signs `sign`, covariance `b` is `a`:
/// `b[to j, to k] = sign_j sign_k a[j, k]`, and so are the SEs.  Returns
/// the defined entries compared.
fn assert_mapped_covariance(
    a: &RelativePairCorrelation,
    b: &RelativePairCorrelation,
    to: &[usize],
    sign: &[f64],
) -> Result<usize, TestCaseError> {
    let (ca, cb) = (
        a.covariance.as_ref().expect("asked for"),
        b.covariance.as_ref().expect("asked for"),
    );
    let n = ca.n();
    prop_assert_eq!(n, to.len());
    prop_assert_eq!(cb.n(), n);
    let (sa, sb) = (slots(a), slots(b));
    for (j, p) in sa.iter().enumerate() {
        for (e, f) in p.estimates.iter().zip(&sb[to[j]].estimates) {
            assert_same_se(e, f, &p.category)?;
        }
    }
    let mut compared = 0;
    for form in Form::ALL {
        let (ma, mb) = (ca.of(form), cb.of(form));
        prop_assert_eq!(ma.is_some(), mb.is_some(), "{:?}", form);
        let (Some(ma), Some(mb)) = (ma, mb) else {
            continue;
        };
        for j in 0..n {
            for k in 0..n {
                let (p, q) = (sign[j] * sign[k] * ma[j * n + k], mb[to[j] * n + to[k]]);
                prop_assert!(
                    close(p, q),
                    "{:?} [{} {}, {} {}]: {} vs {}",
                    form,
                    sa[j].category,
                    j,
                    sa[k].category,
                    k,
                    p,
                    q
                );
                compared += usize::from(!p.is_nan());
            }
        }
    }
    Ok(compared)
}

/// With the covariance, a sign flip of trait 0 (`−x`, or `1 − x` on a
/// binary trait) maps Σ to DΣD, D negating the cross-trait and
/// within-person slots; reordering the traits maps it to PΣPᵀ, P the
/// cells' permutation.  Each SE follows its estimate.
#[test]
fn the_covariance_follows_the_cells() {
    // A quarter of the other properties' cases: each fits three times with
    // the covariance.
    check(
        "covariance",
        CASES / 4,
        250.0,
        two_traits(kind()),
        |(ped, x, y, kind), checked| {
            let a = run(&ped, &x, &y, kind, Se::Covariance);
            let flipped = match kind {
                TraitKind::Binary => flip(&x),
                _ => neg(&x),
            };
            let b = run(&ped, &flipped, &y, kind, Se::Covariance);
            let identity: Vec<usize> = (0..slots(&a).len()).collect();
            let sign: Vec<f64> = slots(&a)
                .iter()
                .map(|c| if cross(c) { -1.0 } else { 1.0 })
                .collect();
            let mut compared = assert_mapped_covariance(&a, &b, &identity, &sign)?;
            let c = run(&ped, &y, &x, kind, Se::Covariance);
            let to: Vec<usize> = slots(&a)
                .iter()
                .map(|p| swapped_index(p, &c).expect("every cell has its image"))
                .collect();
            compared += assert_mapped_covariance(&a, &c, &to, &vec![1.0; to.len()])?;
            checked.fetch_add(compared, Ordering::Relaxed);
            Ok(())
        },
    );
}

/// `res`'s cells, then its within-person cell, as the ACE Fit reads them.
fn ace_cells(res: &RelativePairCorrelation) -> Vec<AceCell> {
    slots(res)
        .into_iter()
        .map(|c| AceCell {
            name: c.category.clone(),
            pairs: if c.category == "within_person" {
                Pairs::WithinPerson
            } else {
                Pairs::Relatives(c.members.clone())
            },
            first_trait: c.first_trait,
            second_trait: c.second_trait,
            crude: c
                .estimates
                .iter()
                .map(|e| CrudeEstimate {
                    estimator: e.estimator,
                    value: e.value.map_err(|r| r.name().to_string()),
                })
                .collect(),
            n: c.n,
            n_entries: c.n_entries,
            stratified: None,
        })
        .collect()
}

/// Two solved values agree: one reason, or values and SEs within a
/// relative `1e-8`, as two QR solves of permuted rows give.
fn assert_same_solved(
    p: &Result<Solved, AceReason>,
    q: &Result<Solved, AceReason>,
    what: &str,
) -> Result<usize, TestCaseError> {
    let near = |s: f64, t: f64| (s - t).abs() <= 1e-8 * s.abs().max(t.abs()).max(1.0);
    match (p, q) {
        (Ok(s), Ok(t)) => {
            prop_assert!(
                near(s.value, t.value),
                "{}: {} vs {}",
                what,
                s.value,
                t.value
            );
            match (s.se, t.se) {
                (Ok(u), Ok(v)) => prop_assert!(near(u, v), "{} se: {} vs {}", what, u, v),
                (u, v) => prop_assert_eq!(u, v, "{} se", what),
            }
            Ok(1)
        }
        (s, t) => {
            prop_assert_eq!(s, t, "{}", what);
            Ok(0)
        }
    }
}

/// The ACE Fit of the result for `[y, x]` swaps the traits' A, C and E
/// and keeps the cross solve, under pairs weights with the covariance's
/// SEs.
#[test]
fn reordering_the_traits_swaps_the_ace_fit() {
    ace_fit_swaps(Some(Weights::Pairs));
}

/// The same under the default weights, GLS where a fit's block of the
/// covariance has a Cholesky factor.  Fails: a singular block (two
/// cross-trait cells correlating 1) passes `linear::cholesky`'s pivot
/// test in one row order and fails it in the other, so the cross solve
/// is GLS one way and pairs the other.
#[test]
#[ignore = "finding: the default weights' Cholesky test depends on the row order; notes/handoff-pgp-issue-31.md"]
fn reordering_the_traits_swaps_the_default_weighted_ace_fit() {
    ace_fit_swaps(None);
}

fn ace_fit_swaps(weights: Option<Weights>) {
    check(
        "ace fit",
        CASES / 4,
        3.0,
        two_traits(kind()),
        |(ped, x, y, kind), checked| {
            let request = AceRequest {
                categories: None,
                components: Components::AC,
                shared_environment: Some(vec!["MZ".into(), "FS".into(), "MHS".into()]),
                estimator: AceEstimator::Primary,
                estimate: None,
                weights,
            };
            let fit = |res: &RelativePairCorrelation| {
                ace::fit(&ace_cells(res), &request, res.covariance.as_ref())
                    .expect("a fit of a result's own cells")
            };
            let (a, b) = (
                fit(&run(&ped, &x, &y, kind, Se::Covariance)),
                fit(&run(&ped, &y, &x, kind, Se::Covariance)),
            );
            let mut compared = 0;
            for (t, p) in a.traits.iter().enumerate() {
                let q = &b.traits[1 - t];
                prop_assert_eq!(p.weights, q.weights);
                for (name, s, u) in [("A", &p.a, &q.a), ("C", &p.c, &q.c), ("E", &p.e, &q.e)] {
                    compared += assert_same_solved(s, u, &format!("trait {t} {name}"))?;
                }
            }
            let (p, q) = (a.cross.expect("two traits"), b.cross.expect("two traits"));
            prop_assert_eq!(p.weights, q.weights);
            for (name, s, u) in [
                ("sA", &p.s_a, &q.s_a),
                ("sC", &p.s_c, &q.s_c),
                ("sE", &p.s_e, &q.s_e),
            ] {
                compared += assert_same_solved(s, u, name)?;
            }
            // `rA = sA / sqrt(A_1 A_2)` divides by the traits' components,
            // so where one is round-off from 0 (MO and FO cells that cancel
            // exactly solve A to 5.6e-17) its value and even its
            // nonpositive_component reason are round-off; compare the
            // others.
            let component = |t: &TraitFit, i: usize| match [&t.a, &t.c, &t.e][i] {
                Ok(s) => s.value.abs(),
                Err(_) => f64::INFINITY,
            };
            for (i, (name, s, u)) in [
                ("rA", &p.r_a, &q.r_a),
                ("rC", &p.r_c, &q.r_c),
                ("rE", &p.r_e, &q.r_e),
            ]
            .into_iter()
            .enumerate()
            {
                if a.traits.iter().all(|t| component(t, i) > 1e-9) {
                    compared += assert_same_solved(s, u, name)?;
                }
            }
            checked.fetch_add(compared, Ordering::Relaxed);
            Ok(())
        },
    );
}
