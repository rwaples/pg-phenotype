//! PA-FGRS: Pearson-Aitken family genetic risk scores.
//!
//! [`prepare`] builds the trait-independent relative structure of a pedigree
//! once; [`score_univariate`] and [`score_bivariate`] condition each proband
//! on its relatives per trait and parameter variant.

mod biv;
mod cip;
mod order;
mod pa;
mod prep;
mod triangle;
mod uni;

pub use biv::{score_bivariate, BivParams, BivScores};
pub use cip::Cip;
pub use prep::{kinship_threshold, prepare, Prep, MAX_NDEGREE};
pub use uni::{score_univariate, UniScores};
