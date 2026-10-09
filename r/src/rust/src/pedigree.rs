//! `pedigree()`: a validated pedigree that methods share, held behind an
//! external pointer with the storage type of its ids.

use crate::errors::{finish, HostResult};
use crate::input::{self, IdType};
use extendr_api::prelude::*;
use pg_phenotype_core::{Pedigree, PedigreeArg};

/// What a `pgphenotype_pedigree`'s pointer owns.
pub struct Handle {
    pedigree: Pedigree,
    id_type: IdType,
}

/// Handles freed so far, which the package tests read to see a drop.
#[cfg(feature = "test-hooks")]
pub static DROPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "test-hooks")]
impl Drop for Handle {
    fn drop(&mut self) {
        DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The pedigree a method was given: a `pedigree()` pointer, or columns.
pub enum Source<'a> {
    Built(&'a Handle),
    Columns(Box<input::Pedigree>),
}

impl<'a> Source<'a> {
    /// The handle behind `ptr`, or, when it is `NULL`, the coerced columns.
    pub fn read(ptr: &'a Robj, columns: [Robj; 5]) -> HostResult<Source<'a>> {
        Ok(if ptr.is_null() {
            Source::Columns(Box::new(input::Pedigree::coerce(columns)?))
        } else {
            Source::Built(input::handle(ptr, "pedigree", "pedigree()")?)
        })
    }

    pub fn arg(&self) -> PedigreeArg<'_> {
        match self {
            Source::Built(h) => PedigreeArg::Built(&h.pedigree),
            Source::Columns(c) => PedigreeArg::Columns(c.input()),
        }
    }

    /// The R storage of the ids, for ids a result returns.
    pub fn id_type(&self) -> IdType {
        match self {
            Source::Built(h) => h.id_type,
            Source::Columns(c) => c.ids.storage,
        }
    }
}

fn new_impl(columns: [Robj; 5]) -> HostResult<Robj> {
    let columns = input::Pedigree::coerce(columns)?;
    let pedigree = Pedigree::new(columns.input())?;
    let id_type = columns.ids.storage;
    let ids = id_type.to_robj(pedigree.ids());
    let ptr: Robj = ExternalPtr::new(Handle { pedigree, id_type }).into();
    Ok(list!(ptr = ptr, ids = ids).into_robj())
}

/// Validate a pedigree: `list(ptr, ids)`, the ids in the storage given.
#[extendr]
fn pedigree_new(id: Robj, mother: Robj, father: Robj, twin: Robj, sex: Robj) -> Robj {
    finish(new_impl([id, mother, father, twin, sex]))
}

extendr_module! {
    mod pedigree;
    fn pedigree_new;
}
