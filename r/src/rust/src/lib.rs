//! extendr host binding for pg-phenotype-core.
//!
//! Every exported function returns its value or, on failure, the classed
//! list the R wrapper `.pgp_call()` signals (`errors.rs`).  Nothing here
//! raises into R.  A method crosses into the core through `crossing.rs`.  Each method is a module, as in the core
//! ([`pafgrs`]).

// No user-reachable panics (pedigree-graph ADR 0007): input errors cross as
// host error values, as in the core.  A surviving `expect` / `unreachable!`
// names the internal invariant it relies on in an `#[expect(..., reason)]`.
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

mod assortative;
mod correlation;
mod crossing;
mod errors;
mod input;
mod pafgrs;
mod pedigree;
mod regression;
mod relative_pairs;
mod test_hooks;
mod threads;
mod value;

use errors::{finish, HostError};
use extendr_api::prelude::*;

/// An R list with one name per value.
#[expect(
    clippy::expect_used,
    reason = "every caller passes one name per value, and a length mismatch is \
              the only way setting the names of a fresh list fails"
)]
pub(crate) fn named_list<N, V>(names: N, values: V) -> Robj
where
    N: IntoIterator,
    N::IntoIter: ExactSizeIterator,
    N::Item: extendr_api::ToVectorValue + AsRef<str>,
    V: IntoIterator,
    V::IntoIter: ExactSizeIterator,
    V::Item: Into<Robj>,
{
    List::from_names_and_values(names, values)
        .expect("one name per value")
        .into_robj()
}

/// `robj` with the S3 class `class`.
#[expect(
    clippy::expect_used,
    reason = "every caller classes a fresh vector or external pointer, never \
              NULL, and R sets a character class on any of those"
)]
pub(crate) fn with_class(mut robj: Robj, class: &str) -> Robj {
    robj.set_class([class])
        .expect("a character class attribute");
    robj
}

/// The binding crate's version, kept equal to the workspace and the R package.
#[extendr]
fn pgphenotype_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The pedigree-graph-core git revision this build links.
#[extendr]
fn pg_core_rev() -> &'static str {
    pg_phenotype_core::PG_CORE_REV
}

/// Record a thread budget; R passes `n` as a double (or `NaN` for a non-number).
#[extendr]
fn configure_threads(n: f64) -> Robj {
    if !(n.is_finite() && n == n.trunc() && n >= 0.0 && n <= usize::MAX as f64) {
        return HostError::usage(format!(
            "configure_threads(n) requires a whole number from 1 to {}, got {n}",
            pg_phenotype_core::threads::MAX_THREADS
        ))
        .into_robj();
    }
    finish(threads::configure(n as usize).map(|()| ().into()))
}

/// The committed thread budget, committing it on first call.
#[extendr]
fn thread_budget() -> Robj {
    finish(threads::budget().map(|n| (n as i32).into()))
}

extendr_module! {
    mod pgphenotype;
    use assortative;
    use correlation;
    use input;
    use pafgrs;
    use pedigree;
    use regression;
    use relative_pairs;
    use test_hooks;
    fn pgphenotype_version;
    fn pg_core_rev;
    fn configure_threads;
    fn thread_budget;
}
