//! How a method crosses from R into the core, held once for every method:
//! the Trait whole, the free-memory release, the pool the thread budget
//! sizes, and the result tree stamped with the versions.  A method's
//! binding states only its own parameters.
//!
//! Every call that runs in the pool first hands the process's free heap
//! pages back to the system (`malloc_trim`, glibc only).  The pool threads
//! allocate in arenas of their own, so memory freed on R's thread would
//! otherwise stay resident beside them.

use crate::errors::{HostError, HostResult};
use crate::input;
use crate::threads;
use crate::value::to_robj;
use extendr_api::prelude::*;
use pg_phenotype_core::value::Value;
use pg_phenotype_core::{Error, Trait, TraitKind};

/// An R `trait()`, read by its elements; the wrapper checked its class.
pub struct TraitArg {
    values: Robj,
    kind: TraitKind,
    n_levels: Option<usize>,
}

impl TraitArg {
    pub fn read(robj: &Robj) -> HostResult<TraitArg> {
        let [values, kind, levels] =
            input::elements(robj, "trait", "trait()", ["values", "kind", "levels"])?;
        Ok(TraitArg {
            values,
            kind: input::trait_kind(kind.as_str().unwrap_or_default())?,
            n_levels: (!levels.is_null()).then(|| levels.len()),
        })
    }

    pub fn get(&self) -> HostResult<Trait<'_>> {
        Ok(Trait {
            values: input::doubles("trait", &self.values)?,
            kind: self.kind,
            n_levels: self.n_levels,
        })
    }
}

/// Run `compute` in the pool the committed thread budget sizes, after
/// handing free memory back.
pub fn run<T, F>(compute: F) -> HostResult<T>
where
    F: FnOnce() -> Result<T, Error> + Send,
    T: Send,
{
    let pool = threads::pool()?;
    release_free_memory();
    Ok(pool.install(compute)?)
}

/// A method's result tree as nested R lists, with its `metadata`.
pub fn result(value: Value) -> HostResult<Robj> {
    let Value::Map(mut entries) = value else {
        return Err(HostError::usage(
            "a method's result tree is not a map".to_string(),
        ));
    };
    entries.push(("metadata", metadata(Vec::new())));
    Ok(to_robj(&Value::Map(entries)))
}

/// A result's metadata: the method's `entries`, then the versions.
pub fn metadata(mut entries: Vec<(&'static str, Value)>) -> Value {
    entries.push(("pg_phenotype_version", env!("CARGO_PKG_VERSION").into()));
    entries.push((
        "pedigree_graph_core_rev",
        pg_phenotype_core::PG_CORE_REV.into(),
    ));
    Value::Map(entries)
}

/// Releases so far, which the package tests read to see that every
/// crossing releases.
#[cfg(feature = "test-hooks")]
pub static RELEASES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Return the process's free heap pages to the system.
pub fn release_free_memory() {
    #[cfg(feature = "test-hooks")]
    RELEASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    malloc_trim();
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn malloc_trim() {
    // SAFETY: glibc declares `int malloc_trim(size_t pad)`.  It takes and
    // returns an integer, and no pointer crosses.
    unsafe extern "C" {
        #[link_name = "malloc_trim"]
        safe fn glibc_malloc_trim(pad: usize) -> std::ffi::c_int;
    }
    glibc_malloc_trim(0);
}

/// musl, macOS and Windows have no `malloc_trim`.
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn malloc_trim() {}
