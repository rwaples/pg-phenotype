//! Time `mate_correlation` on raw little-endian columns in a directory:
//! int64 `id.bin mother.bin father.bin stratum.bin`, float64 `liab.bin dx.bin`.
//! Config `a` is `dx` alone, unstratified; `b` is `liab` and `dx` by stratum.
//!
//!     cargo run --release --example am_bench -- <dir> <a|b> <threads> <permutations> <bootstrap>

use pg_phenotype_core::assortative::{mate_correlation, Settings, Strata};
use pg_phenotype_core::{configure_pool, PedigreeArg, PedigreeInput, Trait, TraitKind};
use std::num::NonZeroUsize;
use std::time::Instant;

fn bytes(dir: &str, name: &str) -> Vec<[u8; 8]> {
    let raw = std::fs::read(format!("{dir}/{name}.bin")).expect("column file");
    raw.chunks_exact(8)
        .map(|b| b.try_into().expect("8 bytes"))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dir, config) = (&args[1], args[2].as_str());
    let threads: usize = args[3].parse().expect("threads");
    let ints = |name| -> Vec<i64> {
        bytes(dir, name)
            .into_iter()
            .map(i64::from_le_bytes)
            .collect()
    };
    let floats = |name| -> Vec<f64> {
        bytes(dir, name)
            .into_iter()
            .map(f64::from_le_bytes)
            .collect()
    };
    let (ids, mother, father, stratum) =
        (ints("id"), ints("mother"), ints("father"), ints("stratum"));
    let (liab, dx) = (floats("liab"), floats("dx"));
    let known = vec![true; ids.len()];
    let trait_of = |values: &'static [f64], kind| Trait {
        values,
        kind,
        n_levels: None,
    };
    let liab: &'static [f64] = liab.leak();
    let dx: &'static [f64] = dx.leak();
    let traits = match config {
        "a" => vec![trait_of(dx, TraitKind::Binary)],
        _ => vec![
            trait_of(liab, TraitKind::Continuous),
            trait_of(dx, TraitKind::Binary),
        ],
    };
    let strata = (config == "b").then_some(Strata {
        labels: &stratum,
        known: &known,
    });
    let settings = Settings {
        permutations: args[4].parse().expect("permutations"),
        bootstrap: args[5].parse().expect("bootstrap"),
        ..Settings::default()
    };
    let pool = configure_pool(NonZeroUsize::new(threads).expect("threads")).expect("pool");
    let input = PedigreeArg::Columns(PedigreeInput {
        ids: &ids,
        mother: &mother,
        father: &father,
        twin: None,
        sex: None,
    });
    let start = Instant::now();
    let result = pool
        .install(|| mate_correlation(input, &traits, strata, settings))
        .expect("result");
    eprintln!(
        "{:.3} s, {} cells",
        start.elapsed().as_secs_f64(),
        result.cells.len()
    );
}
