//! Time `prepare` on raw little-endian int64 columns in a directory:
//! `id.bin mother.bin father.bin twin.bin sex.bin pheno_id.bin`.
//!
//!     cargo run --release --example prep_bench -- <dir> <ndegree> <threads> [all]

use pg_phenotype_core::pafgrs::prepare;
use pg_phenotype_core::{configure_pool, PedigreeInput};
use std::num::NonZeroUsize;
use std::time::Instant;

fn column(dir: &str, name: &str) -> Vec<i64> {
    let bytes = std::fs::read(format!("{dir}/{name}.bin")).expect("column file");
    bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let ndegree: i64 = args[2].parse().expect("ndegree");
    let threads: usize = args[3].parse().expect("threads");
    let all = args.get(4).is_some_and(|a| a == "all");
    let cols: Vec<Vec<i64>> = ["id", "mother", "father", "twin", "sex", "pheno_id"]
        .iter()
        .map(|c| column(dir, c))
        .collect();
    let pool = configure_pool(NonZeroUsize::new(threads).expect("threads")).expect("pool");
    let input = PedigreeInput {
        ids: &cols[0],
        mother: &cols[1],
        father: &cols[2],
        twin: Some(&cols[3]),
        sex: Some(&cols[4]),
    };
    let t = Instant::now();
    let prep = pool
        .install(|| prepare(input, ndegree, if all { None } else { Some(&cols[5]) }))
        .expect("prepare");
    println!(
        "prepare {:.3}s probands {} bytes {} MiB",
        t.elapsed().as_secs_f64(),
        prep.n_probands(),
        prep.bytes() >> 20
    );
}
