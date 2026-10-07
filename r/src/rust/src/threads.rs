//! The package thread budget, with Python's rules (`pg_phenotype/_threads.py`).
//!
//! `configure_threads(n)` records `n`; the first `thread_budget()` commits
//! it, else `PG_PHENOTYPE_THREADS`, else 1; after that only the committed value
//! may be configured again.  The state lives here, not in R, so there is one
//! source of truth per process.

use crate::errors::{HostError, HostResult};
use std::num::NonZeroUsize;
use std::sync::Mutex;

const ENV_VAR: &str = "PG_PHENOTYPE_THREADS";

/// The largest budget: `thread_budget()` returns it as an R integer.
pub const MAX_THREADS: usize = i32::MAX as usize;

struct Budget {
    configured: Option<usize>,
    committed: Option<usize>,
}

static BUDGET: Mutex<Budget> = Mutex::new(Budget {
    configured: None,
    committed: None,
});

pub fn configure(n: usize) -> HostResult<()> {
    if !(1..=MAX_THREADS).contains(&n) {
        return Err(HostError::usage(format!(
            "configure_threads(n) requires a whole number from 1 to {MAX_THREADS}, got {n}"
        )));
    }
    let mut budget = BUDGET.lock().unwrap_or_else(|e| e.into_inner());
    match budget.committed {
        Some(committed) if committed != n => Err(HostError::thread_conflict(committed, n)),
        Some(_) => Ok(()),
        None => {
            budget.configured = Some(n);
            Ok(())
        }
    }
}

fn from_env() -> HostResult<usize> {
    let raw = match std::env::var(ENV_VAR) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(1),
        Err(std::env::VarError::NotUnicode(raw)) => {
            return Err(HostError::usage(format!(
                "{ENV_VAR} must be a decimal integer >= 1, got {raw:?}"
            )))
        }
    };
    match raw.parse::<usize>() {
        Ok(n) if (1..=MAX_THREADS).contains(&n) && raw.bytes().all(|b| b.is_ascii_digit()) => Ok(n),
        _ => Err(HostError::usage(format!(
            "{ENV_VAR} must be a decimal integer from 1 to {MAX_THREADS}, got {raw:?}"
        ))),
    }
}

pub fn budget() -> HostResult<usize> {
    let mut budget = BUDGET.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(committed) = budget.committed {
        return Ok(committed);
    }
    let committed = match budget.configured {
        Some(n) => n,
        None => from_env()?,
    };
    budget.committed = Some(committed);
    Ok(committed)
}

/// The core pool, sized by the committed budget; every native call runs in it.
pub fn pool() -> HostResult<&'static pg_phenotype_core::rayon::ThreadPool> {
    let threads = NonZeroUsize::new(budget()?).expect("a committed budget is at least 1");
    Ok(pg_phenotype_core::configure_pool(threads)?)
}
