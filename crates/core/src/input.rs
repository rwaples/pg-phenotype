//! What every method takes: the pedigree and the traits on its rows.

use crate::error::Error;
use pedigree_graph_core::graph::{self, Columns, Limits, PedigreeGraph, SexEncoding};

/// The pedigree columns a caller passes, one entry per row.
#[derive(Clone, Copy, Debug)]
pub struct PedigreeInput<'a> {
    pub ids: &'a [i64],
    /// Mother id, `-1` when missing.
    pub mother: &'a [i64],
    /// Father id, `-1` when missing.
    pub father: &'a [i64],
    /// MZ co-twin id, `-1` when none.
    pub twin: Option<&'a [i64]>,
    /// `0` female, `1` male, `-1` unknown.
    pub sex: Option<&'a [i64]>,
}

impl PedigreeInput<'_> {
    /// The validated graph, by pedigree-graph-core's own rules (ADR 0001).
    ///
    /// # Errors
    ///
    /// Any pedigree-graph-core validation error, with its code.
    pub fn validate(&self) -> Result<PedigreeGraph, Error> {
        Ok(graph::build(
            Columns {
                ids: self.ids,
                mother: self.mother,
                father: self.father,
                twin: self.twin,
                sex: self.sex,
                generation: None,
                birth_year: None,
            },
            SexEncoding::Simace,
            Limits::default(),
        )?)
    }
}

/// How a trait's values are read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraitKind {
    Continuous,
    /// `0` or `1`.
    Binary,
    /// Ordered integer codes.
    Ordinal,
    /// Unordered integer codes.
    Categorical,
}

impl TraitKind {
    /// The name hosts use.
    pub fn name(self) -> &'static str {
        match self {
            TraitKind::Continuous => "continuous",
            TraitKind::Binary => "binary",
            TraitKind::Ordinal => "ordinal",
            TraitKind::Categorical => "categorical",
        }
    }
}

/// One phenotype column aligned to pedigree rows, NaN where missing.
#[derive(Clone, Copy, Debug)]
pub struct Trait<'a> {
    pub values: &'a [f64],
    pub kind: TraitKind,
    /// The declared number of levels of a binary, ordinal or categorical
    /// trait (codes `0..n_levels`), when the host knows it: Python
    /// `Trait.levels`, R factor levels.  `None` means max code + 1.
    pub n_levels: Option<usize>,
}
