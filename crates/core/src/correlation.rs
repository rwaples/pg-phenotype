//! Correlation estimators on their own, without a pedigree: the tetrachoric
//! correlation of a 2 x 2 table.
//!
//! The fit, its boundary flag and its SE are the Mate Correlation's
//! (ADR 0006): two-step maximum likelihood on the assortative-mating
//! numerics (ADR 0005), and the two-step sandwich with every pair its own
//! cluster.

use crate::assortative::estimators::Tables;
use crate::assortative::fit::newton_rho;
use crate::assortative::kernels::{Grid, Status, Table};
use crate::assortative::sandwich::{cell_influence, wald_ci};
use crate::assortative::{Ci, CiMethod, CiScale, Point, Reason};
use crate::error::Error;
use crate::input::TraitKind;
use crate::value::Value;

/// The largest count a table takes: 2^53, the last integer every count's
/// `f64` holds exactly.
pub const MAX_COUNT: u64 = 1 << 53;

/// The SE this module reports.
pub const SE_METHOD: &str =
    "sandwich of the two-step estimating equations (thresholds, then rho), \
    each pair its own cluster, n/(n-1) small-sample factor";

/// A defined estimate with its inference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub point: Point,
    /// The two-step sandwich SE of ρ̂.
    pub se: Result<f64, Reason>,
    /// The Wald interval on the Fisher-z scale, at the Mate Correlation's level.
    pub ci: Result<Ci, Reason>,
}

/// The tetrachoric correlation of one 2 x 2 table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tetrachoric {
    /// Pair counts, rows by the `x` level, columns by the `y` level.
    pub table: [[u64; 2]; 2],
    /// Pairs counted.
    pub n: u64,
    /// Pairs left out for a missing value (0 for a table).
    pub n_dropped: u64,
    pub outcome: Result<Estimate, Reason>,
}

/// The tetrachoric correlation of `table`, rows by the `x` level and columns
/// by the `y` level.
///
/// `no_complete_pairs` for an empty table and `constant_margin` when a side
/// shows one level.  The SE (and the CI) is withheld at a bound (`boundary`)
/// or when the sandwich is not finite (`sandwich_undefined`).
///
/// # Errors
///
/// `parameter_out_of_range` for a count above [`MAX_COUNT`].
pub fn tetrachoric(table: [[u64; 2]; 2]) -> Result<Tetrachoric, Error> {
    if let Some(&count) = table.iter().flatten().find(|&&c| c > MAX_COUNT) {
        return Err(Error::ParameterOutOfRange {
            name: "table",
            value: count as f64,
            domain: "[0, 2^53]",
        });
    }
    Ok(tetrachoric_of(table, 0))
}

/// The tetrachoric correlation of paired binary values, `NaN` missing.
/// Pairs with a missing value are dropped, as [`Tetrachoric::n_dropped`]
/// counts.
///
/// # Errors
///
/// `pair_length_mismatch` when `y` is not `x`'s length, then
/// `invalid_trait_value` for a value that is not 0, 1 or `NaN`.
pub fn tetrachoric_pairs(x: &[f64], y: &[f64]) -> Result<Tetrachoric, Error> {
    if x.len() != y.len() {
        return Err(Error::PairLength {
            expected_length: x.len(),
            actual_length: y.len(),
        });
    }
    let level = |field: &'static str, position: usize, value: f64| match value {
        0.0 => Ok(Some(0)),
        1.0 => Ok(Some(1)),
        v if v.is_nan() => Ok(None),
        _ => Err(Error::InvalidTraitCode {
            field,
            kind: TraitKind::Binary,
            position,
            value,
        }),
    };
    let mut table = [[0u64; 2]; 2];
    let mut dropped = 0;
    for (p, (&a, &b)) in x.iter().zip(y).enumerate() {
        match (level("x", p, a)?, level("y", p, b)?) {
            (Some(i), Some(j)) => table[i][j] += 1,
            _ => dropped += 1,
        }
    }
    Ok(tetrachoric_of(table, dropped))
}

fn tetrachoric_of(table: [[u64; 2]; 2], n_dropped: u64) -> Tetrachoric {
    Tetrachoric {
        table,
        n: table.iter().flatten().sum(),
        n_dropped,
        outcome: fit(table),
    }
}

fn fit(table: [[u64; 2]; 2]) -> Result<Estimate, Reason> {
    let mut counts = Table::zeros(1, 1, 2, 2);
    counts.data = table.iter().flatten().map(|&c| c as f64).collect();
    // A side's levels are the ones it shows, as a Mate Correlation cell's are.
    let shown = |margin: Grid<f64>| Grid {
        rows: 1,
        cols: 2,
        data: margin.data.iter().map(|&v| v > 0.0).collect(),
    };
    let (rows, cols) = (shown(counts.mother_margin()), shown(counts.father_margin()));
    let tables = Tables::from_counts(counts, &rows, &cols).map_err(Status::reason)?;
    let point = newton_rho(|r| tables.terms(r), |r| tables.nll(r), 0.0);
    let se = sandwich_se(&tables, point);
    let ci = se.map(|se| Ci {
        bounds: wald_ci(point.value, se, CiScale::FisherZ),
        method: CiMethod::Sandwich,
    });
    Ok(Estimate { point, se, ci })
}

/// `√(n/(n−1) · Σ_cells n_ij IF_ij²)`: the Mate Network sandwich with every
/// pair its own network.
fn sandwich_se(tables: &Tables, point: Point) -> Result<f64, Reason> {
    if point.boundary == Some(true) {
        return Err(Reason::Boundary);
    }
    let influence = cell_influence(tables, point.value)?;
    let (mut n, mut ss) = (0.0, 0.0);
    for i in 0..2 {
        for j in 0..2 {
            let count = tables.n.at(0, 0, i, j);
            let v = influence.at(0, 0, i, j);
            n += count;
            ss += count * v * v;
        }
    }
    let se = (n / (n - 1.0) * ss).sqrt();
    if se.is_finite() {
        Ok(se)
    } else {
        Err(Reason::SandwichUndefined)
    }
}

impl Tetrachoric {
    /// The result as a [`Value`] tree, in the key order hosts present.
    pub fn to_value(&self) -> Value {
        let mut out = vec![
            ("estimator", "tetrachoric".into()),
            (
                "table",
                Value::List(
                    self.table
                        .iter()
                        .map(|row| Value::Counts(row.to_vec()))
                        .collect(),
                ),
            ),
            ("n", self.n.into()),
            ("n_dropped", self.n_dropped.into()),
        ];
        let unavailable = |out: &mut Vec<_>, key, reason_key, reason: Reason| {
            out.extend([(key, Value::Null), (reason_key, reason.name().into())]);
        };
        match &self.outcome {
            Err(reason) => {
                out.extend([
                    ("value", Value::Null),
                    ("reason", reason.name().into()),
                    ("boundary", Value::Null),
                ]);
                unavailable(&mut out, "se", "se_unavailable_reason", *reason);
                unavailable(&mut out, "ci", "ci_unavailable_reason", *reason);
                out.push(("ci_method", Value::Null));
            }
            Ok(e) => {
                out.extend([
                    ("value", e.point.value.into()),
                    ("reason", Value::Null),
                    ("boundary", e.point.boundary.into()),
                ]);
                match e.se {
                    Ok(se) => {
                        out.extend([("se", se.into()), ("se_unavailable_reason", Value::Null)])
                    }
                    Err(r) => unavailable(&mut out, "se", "se_unavailable_reason", r),
                }
                match e.ci {
                    Ok(ci) => out.extend([
                        ("ci", Value::Floats(ci.bounds.to_vec())),
                        ("ci_unavailable_reason", Value::Null),
                        ("ci_method", ci.method.name().into()),
                    ]),
                    Err(r) => {
                        unavailable(&mut out, "ci", "ci_unavailable_reason", r);
                        out.push(("ci_method", Value::Null));
                    }
                }
            }
        }
        out.extend([
            ("ci_level", crate::assortative::CI_LEVEL.into()),
            ("se_method", SE_METHOD.into()),
        ]);
        Value::Map(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independence_gives_zero_and_the_delta_method_se() {
        let t = tetrachoric([[250, 250], [250, 250]]).unwrap();
        let e = t.outcome.unwrap();
        assert!(e.point.value.abs() < 1e-12);
        assert_eq!(e.point.boundary, Some(false));
        // Thresholds at 0, ρ = 0: Var = (π/2)² / n, then n/(n-1).
        let expected = std::f64::consts::FRAC_PI_2 / 1000f64.sqrt() * (1000.0f64 / 999.0).sqrt();
        assert!((e.se.unwrap() - expected).abs() < 1e-12, "{:?}", e.se);
    }

    #[test]
    fn degenerate_tables_carry_a_reason() {
        assert_eq!(
            tetrachoric([[0, 0], [0, 0]]).unwrap().outcome,
            Err(Reason::NoCompletePairs)
        );
        assert_eq!(
            tetrachoric([[3, 4], [0, 0]]).unwrap().outcome,
            Err(Reason::ConstantMargin)
        );
        let edge = tetrachoric([[10, 0], [0, 10]]).unwrap().outcome.unwrap();
        let big = tetrachoric([[MAX_COUNT + 1, 1], [1, 1]]).unwrap_err();
        assert_eq!(big.code(), "parameter_out_of_range");
        assert_eq!(edge.point.boundary, Some(true));
        assert_eq!(edge.se, Err(Reason::Boundary));
        assert_eq!(edge.ci, Err(Reason::Boundary));
    }

    #[test]
    fn pairs_count_into_the_table_and_drop_missing() {
        let x = [0.0, 1.0, 1.0, f64::NAN, 0.0, 1.0];
        let y = [0.0, 1.0, 0.0, 1.0, f64::NAN, 1.0];
        let t = tetrachoric_pairs(&x, &y).unwrap();
        assert_eq!(t.table, [[1, 0], [1, 2]]);
        assert_eq!((t.n, t.n_dropped), (4, 2));
        assert_eq!(
            tetrachoric_pairs(&x, &y[..5]).unwrap_err().code(),
            "pair_length_mismatch"
        );
        let err = tetrachoric_pairs(&[0.0, 2.0], &[0.0, 1.0]).unwrap_err();
        assert_eq!(err.code(), "invalid_trait_value");
    }
}
