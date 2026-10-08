//! extendr host binding for pg-phenotype-core.
//!
//! Every exported function returns its value or, on failure, the classed
//! list the R wrapper `.pgp_call()` signals (`errors.rs`).  Nothing here
//! raises into R.  Work runs in the core pool sized by the package thread
//! budget (`threads.rs`).  Each method is a module, as in the core
//! ([`pafgrs`]).

mod assortative;
mod errors;
mod input;
mod pafgrs;
mod test_hooks;
mod threads;

use errors::{finish, HostError};
use extendr_api::prelude::*;

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
            threads::MAX_THREADS
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
    use input;
    use pafgrs;
    use test_hooks;
    fn pgphenotype_version;
    fn pg_core_rev;
    fn configure_threads;
    fn thread_budget;
}
