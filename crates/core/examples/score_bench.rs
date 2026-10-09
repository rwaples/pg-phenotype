//! Time the scores on raw columns in a directory (see prep_bench), plus
//! `affected{1,2}.bin age{1,2}.bin cip{1,2}.bin` as little-endian f64.
//!
//!     cargo run --release --example score_bench -- <dir> <ndegree> <threads> <uni|biv>

use pg_phenotype_core::pafgrs::{prepare, score_bivariate, score_univariate, BivParams, Cip};
use pg_phenotype_core::{configure_pool, PedigreeArg, PedigreeInput, Trait, TraitKind};
use std::num::NonZeroUsize;
use std::time::Instant;

fn read(dir: &str, name: &str) -> Vec<[u8; 8]> {
    let bytes = std::fs::read(format!("{dir}/{name}.bin")).expect("column file");
    bytes
        .chunks_exact(8)
        .map(|b| b.try_into().expect("8 bytes"))
        .collect()
}

fn ints(dir: &str, name: &str) -> Vec<i64> {
    read(dir, name)
        .into_iter()
        .map(i64::from_le_bytes)
        .collect()
}

fn floats(dir: &str, name: &str) -> Vec<f64> {
    read(dir, name)
        .into_iter()
        .map(f64::from_le_bytes)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dir, ndegree, threads, which) = (
        &args[1],
        args[2].parse::<i64>().expect("ndegree"),
        args[3].parse::<usize>().expect("threads"),
        args[4].as_str(),
    );
    let cols: Vec<Vec<i64>> = ["id", "mother", "father", "twin", "sex", "pheno_id"]
        .iter()
        .map(|c| ints(dir, c))
        .collect();
    let pool = configure_pool(NonZeroUsize::new(threads).expect("threads")).expect("pool");
    let input = PedigreeArg::Columns(PedigreeInput {
        ids: &cols[0],
        mother: &cols[1],
        father: &cols[2],
        twin: Some(&cols[3]),
        sex: Some(&cols[4]),
    });
    let prep = pool
        .install(|| prepare(input, ndegree, Some(&cols[5])))
        .expect("prepare");
    let status: Vec<Vec<f64>> = (1..=2)
        .map(|t| floats(dir, &format!("affected{t}")))
        .collect();
    let age: Vec<Vec<f64>> = (1..=2).map(|t| floats(dir, &format!("age{t}"))).collect();
    let cips: Vec<Cip> = (1..=2)
        .map(|t| {
            let v = floats(dir, &format!("cip{t}"));
            let half = v.len() / 2;
            Cip::new(v[..half].to_vec(), v[half..].to_vec()).expect("cip")
        })
        .collect();
    let traits = [0, 1].map(|t| Trait {
        values: &status[t],
        kind: TraitKind::Binary,
        n_levels: None,
    });
    let t = Instant::now();
    if which == "uni" {
        pool.install(|| score_univariate(&prep, traits[0], &age[0], &cips[0], 0.3))
            .expect("score");
    } else {
        let params = BivParams::new([0.3, 0.5], 0.5, None).expect("params");
        pool.install(|| {
            score_bivariate(
                &prep,
                traits,
                [&age[0], &age[1]],
                [&cips[0], &cips[1]],
                params,
            )
        })
        .expect("score");
    }
    println!("{which} {:.3}s", t.elapsed().as_secs_f64());
}
