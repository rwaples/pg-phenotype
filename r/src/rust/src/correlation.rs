//! `correlation_tetrachoric()`: a 2 x 2 table's tetrachoric correlation as a
//! named list, the core's `to_value` tree that the Python binding converts too.

use crate::assortative::to_robj;
use crate::errors::{finish, HostError, HostResult};
use crate::input;
use extendr_api::prelude::*;
use pg_phenotype_core::correlation;

/// The core's largest count, which an R double holds exactly.
const MAX_COUNT: f64 = correlation::MAX_COUNT as f64;

fn table_impl(table: &Robj) -> HostResult<Robj> {
    let cells = input::doubles("table", table)?;
    let [n00, n01, n10, n11] = <[f64; 4]>::try_from(cells)
        .map_err(|_| HostError::usage("`table` must be a 2 x 2 matrix".to_string()))?;
    let mut counts = [[0u64; 2]; 2];
    for (k, v) in [n00, n01, n10, n11].into_iter().enumerate() {
        let (row, column) = (k / 2, k % 2);
        if !(v.is_finite() && v == v.trunc() && (0.0..=MAX_COUNT).contains(&v)) {
            return Err(HostError::validation(
                "invalid_table",
                format!(
                    "table[{}, {}] = {v} is not a count (a whole number from 0 to 2^53)",
                    row + 1,
                    column + 1
                ),
                vec![
                    ("field", "table".into()),
                    ("row", (row as f64 + 1.0).into()),
                    ("column", (column as f64 + 1.0).into()),
                    ("value", v.into()),
                ],
            ));
        }
        counts[row][column] = v as u64;
    }
    Ok(to_robj(&correlation::tetrachoric(counts)?.to_value()))
}

/// The tetrachoric correlation of a table, its cells row by row.
#[extendr]
fn correlation_tetrachoric_table(table: Robj) -> Robj {
    finish(table_impl(&table))
}

/// The tetrachoric correlation of paired binary values, `NA` missing.
#[extendr]
fn correlation_tetrachoric_pairs(x: Robj, y: Robj) -> Robj {
    finish((|| {
        let x = input::doubles("x", &x)?;
        let y = input::doubles("y", &y)?;
        Ok(to_robj(&correlation::tetrachoric_pairs(x, y)?.to_value()))
    })())
}

extendr_module! {
    mod correlation;
    fn correlation_tetrachoric_table;
    fn correlation_tetrachoric_pairs;
}
