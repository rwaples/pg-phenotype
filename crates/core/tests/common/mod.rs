//! Pedigree and trait strategies for the property tests.
//!
//! A pedigree is built from its sizes and a list of dice, read in order and
//! 0 once used up, so every shrunk case is still a valid pedigree: shrinking
//! makes it smaller and its choices the first ones.

#![allow(dead_code, reason = "each test binary uses a subset")]

use std::sync::atomic::AtomicUsize;

use pg_phenotype_core::{PedigreeInput, Trait, TraitKind};
use proptest::prelude::*;
use proptest::test_runner::TestRunner;

#[derive(Clone, Debug)]
pub struct Ped {
    pub ids: Vec<i64>,
    pub mother: Vec<i64>,
    pub father: Vec<i64>,
    pub twin: Vec<i64>,
}

impl Ped {
    pub fn input(&self) -> PedigreeInput<'_> {
        PedigreeInput {
            ids: &self.ids,
            mother: &self.mother,
            father: &self.father,
            twin: Some(&self.twin),
            sex: None,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// The rows in `order`.
    pub fn permuted(&self, order: &[usize]) -> Ped {
        let pick = |v: &[i64]| order.iter().map(|&r| v[r]).collect();
        Ped {
            ids: pick(&self.ids),
            mother: pick(&self.mother),
            father: pick(&self.father),
            twin: pick(&self.twin),
        }
    }
}

struct Dice<'a> {
    values: &'a [u32],
    at: usize,
}

impl Dice<'_> {
    /// The next die in `0..n`.
    fn roll(&mut self, n: usize) -> usize {
        let v = self.values.get(self.at).copied().unwrap_or(0);
        self.at += 1;
        v as usize % n
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.roll(i + 1));
        }
    }
}

/// Rows as parent indices, then ids, missing and external parents, and a
/// shuffled row order: what [`Ped`] holds.
fn finish(mother: Vec<i64>, father: Vec<i64>, twin: Vec<i64>, dice: &mut Dice) -> Ped {
    let n = mother.len();
    let mut ids: Vec<i64> = (0..n as i64).map(|r| 1000 + 7 * r).collect();
    dice.shuffle(&mut ids);
    let id = |r: i64| if r < 0 { -1 } else { ids[r as usize] };
    let mut ped = Ped {
        ids: ids.clone(),
        mother: mother.iter().map(|&m| id(m)).collect(),
        father: father.iter().map(|&f| id(f)).collect(),
        twin: twin.iter().map(|&t| id(t)).collect(),
    };
    for r in 0..n {
        if ped.twin[r] >= 0 {
            continue;
        }
        if dice.roll(100) < 8 {
            ped.mother[r] = -1;
        }
        if dice.roll(100) < 5 {
            ped.father[r] = 99_000 + r as i64;
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    dice.shuffle(&mut order);
    ped.permuted(&order)
}

/// A small inbred pedigree with MZ twins, missing and external parents,
/// sparse unsorted ids: `tests/pedigrees.py::random_pedigree`.
pub fn pedigree() -> impl Strategy<Value = Ped> {
    (
        4usize..=10,
        1usize..=3,
        4usize..=10,
        prop::collection::vec(any::<u32>(), 0..300),
    )
        .prop_map(|(n_founders, generations, size, dice)| {
            let mut dice = Dice {
                values: &dice,
                at: 0,
            };
            let (mut mother, mut father, mut twin) = (
                vec![-1i64; n_founders],
                vec![-1i64; n_founders],
                vec![-1i64; n_founders],
            );
            let mut sex: Vec<usize> = (0..n_founders).map(|_| dice.roll(2)).collect();
            let mut previous: Vec<usize> = (0..n_founders).collect();
            for _ in 0..generations {
                let of = |s| {
                    previous
                        .iter()
                        .copied()
                        .filter(|&r| sex[r] == s)
                        .collect::<Vec<_>>()
                };
                let (mut females, mut males) = (of(0), of(1));
                // A generation of one sex mates with a new founder of the
                // other, so no id is a mother and a father: pedigree-graph
                // labels a cross-role half sib pair 1C (pg-phenotype
                // 63af680, pedigree-graph issue on dual-role parents).
                for (side, s) in [(&mut females, 0), (&mut males, 1)] {
                    if side.is_empty() {
                        side.push(mother.len());
                        mother.push(-1);
                        father.push(-1);
                        twin.push(-1);
                        sex.push(s);
                    }
                }
                let mut current = Vec::new();
                while current.len() < size {
                    let f = males[dice.roll(males.len())];
                    let m = females[dice.roll(females.len())];
                    for k in 0..1 + dice.roll(3) {
                        let r = mother.len();
                        mother.push(m as i64);
                        father.push(f as i64);
                        twin.push(-1);
                        sex.push(dice.roll(2));
                        current.push(r);
                        if k == 1 && dice.roll(10) < 3 {
                            twin[r] = r as i64 - 1;
                            twin[r - 1] = r as i64;
                            sex[r] = sex[r - 1];
                        }
                    }
                }
                previous = current;
            }
            finish(mother, father, twin, &mut dice)
        })
}

/// Founder couples whose descendants marry founders only: no loops, so no
/// pair fits a category both ways (`tests/test_relative_pairs.py::_outbred`).
pub fn outbred_pedigree() -> impl Strategy<Value = Ped> {
    (
        2usize..=4,
        1usize..=2,
        prop::collection::vec(any::<u32>(), 0..100),
    )
        .prop_map(|(couples, generations, dice)| {
            let mut dice = Dice {
                values: &dice,
                at: 0,
            };
            // Each child's sex travels with it in `current`; the pedigree
            // has no sex column.
            let (mut mother, mut father) = (Vec::new(), Vec::new());
            let mut add = |m: i64, f: i64| {
                mother.push(m);
                father.push(f);
                mother.len() as i64 - 1
            };
            let mut current = Vec::new();
            for _ in 0..couples {
                let (m, f) = (add(-1, -1), add(-1, -1));
                for _ in 0..1 + dice.roll(4) {
                    let s = dice.roll(2);
                    current.push((add(m, f), s));
                }
            }
            for _ in 0..generations {
                let mut next = Vec::new();
                for &(r, s) in &current {
                    if dice.roll(10) < 7 {
                        let spouse = add(-1, -1);
                        let (m, f) = if s == 0 { (r, spouse) } else { (spouse, r) };
                        for _ in 0..1 + dice.roll(3) {
                            let s = dice.roll(2);
                            next.push((add(m, f), s));
                        }
                    }
                }
                current = next;
            }
            let n = mother.len() as i64;
            let id = |r: i64| if r < 0 { -1 } else { r + 1 };
            Ped {
                ids: (1..=n).collect(),
                mother: mother.into_iter().map(id).collect(),
                father: father.into_iter().map(id).collect(),
                twin: vec![-1; n as usize],
            }
        })
}

/// Up to 64 values' codes; [`values`] reads the first `n`.
pub fn codes() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(0u8..=20, 64)
}

/// About 10% missing; a continuous value is a quarter-integer in
/// `[-2.25, 2.25]`, exact in binary.
pub fn values(codes: &[u8], n: usize, kind: TraitKind) -> Vec<f64> {
    (0..n)
        .map(|r| match codes[r % codes.len()] {
            19.. => f64::NAN,
            c if kind == TraitKind::Binary => f64::from(c % 2),
            c => (f64::from(c) - 9.0) / 4.0,
        })
        .collect()
}

pub fn kind() -> impl Strategy<Value = TraitKind> {
    prop_oneof![Just(TraitKind::Continuous), Just(TraitKind::Binary)]
}

pub fn trait_of(values: &[f64], kind: TraitKind) -> Trait<'_> {
    Trait {
        values,
        kind,
        n_levels: None,
    }
}

/// Cases for a property whose ordinary run draws `ordinary`:
/// `PROPTEST_CASES` when set (proptest's own variable, which its config
/// would ignore here because the runner sets `cases`), else `ordinary`.
fn cases(ordinary: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(ordinary)
}

/// Run `test` on `ordinary` draws of `strategy`, or `PROPTEST_CASES`, then
/// require that it found something to check at least `per_case` times per
/// case on average: `test` adds what it checked to the counter, so a
/// strategy that stops making such cases fails instead of passing
/// vacuously.  A failing case is saved beside the calling test file, as
/// `proptest!` does.
#[track_caller]
pub fn check<S: Strategy>(
    name: &str,
    ordinary: u32,
    per_case: f64,
    strategy: S,
    test: impl Fn(S::Value, &AtomicUsize) -> Result<(), TestCaseError>,
) {
    let cases = cases(ordinary);
    let floor = (per_case * f64::from(cases)).ceil() as usize;
    let checked = AtomicUsize::new(0);
    // A hand-built runner knows no source file unless told, and then
    // neither saves nor replays a failure.
    let mut runner = TestRunner::new(ProptestConfig {
        cases,
        source_file: Some(std::panic::Location::caller().file()),
        ..ProptestConfig::default()
    });
    let result = runner.run(&strategy, |value| test(value, &checked));
    assert!(result.is_ok(), "{name}: {result:?}");
    let checked = checked.into_inner();
    assert!(
        checked >= floor,
        "{name}: checked {checked} times in {cases} cases, expected at least {floor}"
    );
}
