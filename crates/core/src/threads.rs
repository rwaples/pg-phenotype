//! The package thread budget: one per process, shared by every host.
//!
//! `configure(n)` records `n`; the first [`budget`] commits it, else
//! `PG_PHENOTYPE_THREADS`, else 1; after that only the committed value may
//! be configured again, so work already dispatched under the committed
//! budget is never invalidated behind its back (pedigree-graph ADR 0007).
//! The budget sizes this crate's pool ([`crate::configure_pool`]).

use std::sync::Mutex;

/// The environment variable a budget falls back to.
pub const ENV_VAR: &str = "PG_PHENOTYPE_THREADS";

/// The largest budget: R reports it as an integer.
pub const MAX_THREADS: usize = i32::MAX as usize;

/// Why a budget was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetError {
    /// `configure(n)` with `n` outside `1..=MAX_THREADS`.
    OutOfRange { requested: usize },
    /// `configure(n)` after the budget was committed to another value.
    Conflict { committed: usize, requested: usize },
    /// [`ENV_VAR`] is set to anything but a decimal integer in
    /// `1..=MAX_THREADS`.
    InvalidEnv { raw: String },
}

struct Budget {
    configured: Option<usize>,
    committed: Option<usize>,
}

static BUDGET: Mutex<Budget> = Mutex::new(Budget {
    configured: None,
    committed: None,
});

fn state() -> std::sync::MutexGuard<'static, Budget> {
    BUDGET.lock().unwrap_or_else(|e| e.into_inner())
}

/// Record `n` as the budget, or confirm it once committed.
///
/// # Errors
///
/// [`BudgetError::OutOfRange`], and [`BudgetError::Conflict`] once a
/// different budget is committed.
pub fn configure(n: usize) -> Result<(), BudgetError> {
    if !(1..=MAX_THREADS).contains(&n) {
        return Err(BudgetError::OutOfRange { requested: n });
    }
    let mut budget = state();
    match budget.committed {
        Some(committed) if committed != n => Err(BudgetError::Conflict {
            committed,
            requested: n,
        }),
        Some(_) => Ok(()),
        None => {
            budget.configured = Some(n);
            Ok(())
        }
    }
}

fn from_env() -> Result<usize, BudgetError> {
    let raw = match std::env::var(ENV_VAR) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(1),
        Err(std::env::VarError::NotUnicode(raw)) => {
            return Err(BudgetError::InvalidEnv {
                raw: raw.to_string_lossy().into_owned(),
            })
        }
    };
    match raw.parse::<usize>() {
        Ok(n) if (1..=MAX_THREADS).contains(&n) && raw.bytes().all(|b| b.is_ascii_digit()) => Ok(n),
        _ => Err(BudgetError::InvalidEnv { raw }),
    }
}

/// The committed budget, committing it on the first call.
///
/// # Errors
///
/// [`BudgetError::InvalidEnv`] when the budget would come from a bad
/// [`ENV_VAR`]; nothing is committed then.
pub fn budget() -> Result<usize, BudgetError> {
    let mut budget = state();
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

/// Clear the configured and committed budget.  For tests only: the pool,
/// once built, keeps its size.
#[doc(hidden)]
pub fn reset() {
    let mut budget = state();
    budget.configured = None;
    budget.committed = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configure_commit_and_conflict() {
        reset();
        assert_eq!(configure(0), Err(BudgetError::OutOfRange { requested: 0 }));
        configure(3).unwrap();
        configure(2).unwrap();
        assert_eq!(budget(), Ok(2));
        assert_eq!(configure(2), Ok(()));
        assert_eq!(
            configure(4),
            Err(BudgetError::Conflict {
                committed: 2,
                requested: 4
            })
        );
        reset();
    }
}
