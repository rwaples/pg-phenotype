//! Time `parent_offspring` on a synthetic pedigree: `gens` generations of
//! `per_gen` rows, random mating among the previous generation.  `mid` fits
//! the midparent line alone and `point` skips inference (issue #15).
//!
//!     cargo run --release --example po_profile -- <per_gen> <gens> <pooled|gen> <all|mid> <inference|point> <calls>

use pg_phenotype_core::assortative::Strata;
use pg_phenotype_core::regression::{parent_offspring, Lines, Parent, Settings};
use pg_phenotype_core::{Pedigree, PedigreeArg, PedigreeInput, Trait, TraitKind};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() as f64 + 0.5) / (1u64 << 53) as f64
    }
}

const USAGE: &str =
    "po_profile <per_gen >= 2> <gens >= 1> <pooled|gen> <all|mid> <inference|point> <calls >= 1>";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let count = |i: usize, least: usize| -> usize {
        match args.get(i).map(|a| a.parse::<usize>()) {
            Some(Ok(n)) if n >= least => n,
            _ => panic!("usage: {USAGE}"),
        }
    };
    assert_eq!(args.len(), 7, "usage: {USAGE}");
    let (per_gen, gens) = (count(1, 2), count(2, 1));
    let config = args[3].as_str();
    let settings = Settings {
        lines: if args[4] == "all" {
            Lines::ALL
        } else {
            Lines::from_iter([Parent::Midparent])
        },
        inference: args[5] == "inference",
    };
    let calls = count(6, 1);
    let n = per_gen * gens;
    let mut rng = Lcg(42);
    let ids: Vec<i64> = (0..n as i64).collect();
    let (mut mother, mut father) = (vec![-1i64; n], vec![-1i64; n]);
    let generation: Vec<i64> = (0..n).map(|i| (i / per_gen) as i64).collect();
    let mut liab: Vec<f64> = (0..n).map(|_| rng.unit() - 0.5).collect();
    for g in 1..gens {
        let prev = (g - 1) * per_gen;
        let half = per_gen / 2;
        // Even offsets are female, odd male; per_gen/2 pairs drawn at random.
        let pairs: Vec<(usize, usize)> = (0..half)
            .map(|_| (prev + 2 * rng.below(half), prev + 2 * rng.below(half) + 1))
            .collect();
        for i in g * per_gen..(g + 1) * per_gen {
            let (m, f) = pairs[rng.below(pairs.len())];
            mother[i] = m as i64;
            father[i] = f as i64;
            liab[i] += 0.25 * (liab[m] + liab[f]);
        }
    }
    let known = vec![true; n];
    let t = Instant::now();
    let ped = Pedigree::new(PedigreeInput {
        ids: &ids,
        mother: &mother,
        father: &father,
        twin: None,
        sex: None,
    })
    .expect("pedigree");
    println!("build {:.4} s", t.elapsed().as_secs_f64());
    let strata = (config == "gen").then_some(Strata {
        labels: &generation,
        known: &known,
    });
    let trait_ = Trait {
        values: &liab,
        kind: TraitKind::Continuous,
        n_levels: None,
    };
    let t = Instant::now();
    parent_offspring(PedigreeArg::Built(&ped), trait_, strata, settings).expect("warm");
    println!("first call {:.4} s", t.elapsed().as_secs_f64());
    let mut walls = Vec::new();
    for _ in 0..calls {
        let t = Instant::now();
        std::hint::black_box(
            parent_offspring(PedigreeArg::Built(&ped), trait_, strata, settings)
                .expect("fit")
                .to_value(),
        );
        walls.push(t.elapsed().as_secs_f64());
    }
    walls.sort_by(f64::total_cmp);
    println!(
        "{config} {} {}: median {:.4} s, min {:.4} s over {calls}",
        args[4],
        args[5],
        walls[calls / 2],
        walls[0]
    );
}
