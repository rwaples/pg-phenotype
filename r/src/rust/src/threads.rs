//! The package thread budget: the core's (`pg_phenotype_core::threads`),
//! which Python reads too, with its refusals as R conditions.

use crate::errors::{HostError, HostResult};
use pg_phenotype_core::threads::{self, BudgetError, ENV_VAR, MAX_THREADS};
use std::num::NonZeroUsize;

fn host_error(err: BudgetError) -> HostError {
    match err {
        BudgetError::OutOfRange { requested } => HostError::usage(format!(
            "configure_threads(n) requires a whole number from 1 to {MAX_THREADS}, got {requested}"
        )),
        BudgetError::Conflict {
            committed,
            requested,
        } => HostError::thread_conflict(committed, requested),
        BudgetError::InvalidEnv { raw } => HostError::usage(format!(
            "{ENV_VAR} must be a decimal integer from 1 to {MAX_THREADS}, got {raw:?}"
        )),
    }
}

pub fn configure(n: usize) -> HostResult<()> {
    threads::configure(n).map_err(host_error)
}

pub fn budget() -> HostResult<usize> {
    threads::budget().map_err(host_error)
}

/// The core pool, sized by the committed budget; every native call runs in it.
pub fn pool() -> HostResult<&'static pg_phenotype_core::rayon::ThreadPool> {
    let threads = NonZeroUsize::new(budget()?).expect("a committed budget is at least 1");
    Ok(pg_phenotype_core::configure_pool(threads)?)
}
