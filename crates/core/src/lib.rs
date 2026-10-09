//! pg-phenotype-core: phenotypes in the context of a pedigree.
//!
//! A pedigree goes in through pedigree-graph-core's validator, and
//! relationships and kinship stay inside (ADR 0001).  Each method is a
//! module ([`pafgrs`]); the pedigree and [`Trait`] inputs, errors, and the
//! thread pool are shared.

#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented
    )
)]

/// The pedigree-graph revision this build links, as `Cargo.toml` pins it.
pub const PG_CORE_REV: &str = "727159175db19e0315e3743299c1e6538710c0ab";

pub mod assortative;
pub mod correlation;
pub mod error;
pub mod input;
mod lineage;
pub(crate) mod normal;
pub mod pafgrs;
pub mod pedigree;
pub mod threads;
pub mod value;

pub use error::Error;
pub use input::{PedigreeInput, Trait, TraitKind};
pub use pedigree::{Pedigree, PedigreeArg};
pub use rayon;

/// This build's one Rayon pool, built on first use with `threads` workers.
///
/// Every host configures it from its committed thread budget and runs each
/// call inside it.  The same size again is a no-op; a different size is
/// pedigree-graph-core's `thread_pool_conflict`.
///
/// # Errors
///
/// [`Error::Pedigree`] with `thread_pool_conflict` or
/// `thread_pool_unavailable`.
pub fn configure_pool(
    threads: std::num::NonZeroUsize,
) -> Result<&'static rayon::ThreadPool, Error> {
    Ok(pedigree_graph_core::pool::configure(threads)?)
}

#[cfg(test)]
mod tests {
    #[test]
    fn pg_core_rev_is_the_rev_cargo_toml_pins() {
        let manifest = include_str!("../../../Cargo.toml");
        let pin = format!("rev = \"{}\"", super::PG_CORE_REV);
        assert!(manifest.contains(&pin), "Cargo.toml does not pin {pin}");
    }
}
