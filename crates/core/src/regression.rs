//! Parent-offspring regression: the ordinary least-squares line of each
//! child's trait on its midparent's, its mother's or its father's, pooled
//! and within strata (ADR 0007).
//!
//! The slope's SE is the cluster-robust sandwich over Mate Networks, with a
//! Wald p-value and interval, beside the plain OLS SE.  Networks are the
//! whole pedigree's: dropping children never splits one.  A parent in no
//! Mating Pair is a network of its own.

use crate::assortative::sandwich::cluster_se;
use crate::assortative::{check_strata, Strata};
use crate::error::Error;
use crate::input::{Trait, ValueRule};
use crate::normal;
use crate::pedigree::{Pedigree, PedigreeArg};
use crate::value::Value;
use crate::wald::{CI_LEVEL, WALD_Z};

/// The trait's field name, as errors report it.
const FIELD: &str = "trait";

/// The parent value a child's trait is regressed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parent {
    /// `(mother + father) / 2`; both parents are pedigree rows.
    Midparent,
    /// The mother's value; the father may be missing.
    Mother,
    /// The father's value; the mother may be missing.
    Father,
}

impl Parent {
    /// Every regression, in result order.
    pub const ALL: [Parent; 3] = [Parent::Midparent, Parent::Mother, Parent::Father];

    pub fn name(self) -> &'static str {
        match self {
            Parent::Midparent => "midparent",
            Parent::Mother => "mother",
            Parent::Father => "father",
        }
    }
}

/// Why a value is missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// No child has its own and its parent value observed.
    NoCompleteChildren,
    /// Every analysed child has the same parent value, so the slope is undefined.
    ConstantParentValue,
    /// Every analysed child has the same value: the slope is 0 and r undefined.
    ConstantChildValue,
    /// The OLS SE needs three children.
    TooFewChildren,
    /// The sandwich SE needs two Mate Networks.
    SingleMateNetwork,
    /// The sandwich SE is not finite.
    SandwichUndefined,
    /// The sandwich SE is exactly 0 (a perfect fit), so there is no Wald test.
    ZeroSe,
}

impl Reason {
    pub fn name(self) -> &'static str {
        match self {
            Reason::NoCompleteChildren => "no_complete_children",
            Reason::ConstantParentValue => "constant_parent_value",
            Reason::ConstantChildValue => "constant_child_value",
            Reason::TooFewChildren => "too_few_children",
            Reason::SingleMateNetwork => "single_mate_network",
            Reason::SandwichUndefined => "sandwich_undefined",
            Reason::ZeroSe => "zero_se",
        }
    }
}

/// Rows a regression leaves out, each under the first reason that applies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dropped {
    /// A parent the regression needs is not a pedigree row (founders included).
    pub parent_not_in_pedigree: u64,
    pub child_value_missing: u64,
    pub parent_value_missing: u64,
}

impl Dropped {
    pub fn total(&self) -> u64 {
        self.parent_not_in_pedigree + self.child_value_missing + self.parent_value_missing
    }
}

/// The Wald test and interval of a slope from its sandwich SE.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wald {
    pub p: f64,
    pub ci: [f64; 2],
}

/// A defined regression line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    pub slope: f64,
    pub intercept: f64,
    pub r: Result<f64, Reason>,
    /// Mate Network sandwich SE of the slope.
    pub se: Result<f64, Reason>,
    /// OLS SE of the slope, `sqrt(SSres / (n - 2) / Sxx)`.
    pub se_ols: Result<f64, Reason>,
    pub wald: Result<Wald, Reason>,
}

/// One regression over one subset of children.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    pub n: u64,
    /// Mate Networks with at least one analysed child.
    pub n_mate_networks: u64,
    pub dropped: Dropped,
    pub line: Result<Line, Reason>,
}

/// The three regressions of one subset, in [`Parent::ALL`] order.
#[derive(Clone, Debug, PartialEq)]
pub struct Subset {
    pub fits: [Fit; 3],
}

impl Subset {
    pub fn fit(&self, parent: Parent) -> &Fit {
        &self.fits[parent as usize]
    }
}

/// One stratum's label and regressions.
#[derive(Clone, Debug, PartialEq)]
pub struct Stratum {
    pub label: i64,
    pub subset: Subset,
}

/// The result of [`parent_offspring`].
#[derive(Clone, Debug, PartialEq)]
pub struct ParentOffspring {
    /// Every child, whatever its stratum.
    pub pooled: Subset,
    /// Each known stratum label of any row, ascending; `None` without strata.
    pub strata: Option<Vec<Stratum>>,
    /// Rows whose stratum is unknown: in `pooled`, in no stratum.
    pub n_unknown_stratum: u64,
}

/// How the SE, p-value and interval are computed.
const METHOD: [(&str, &str); 4] = [
    (
        "se_method",
        "cluster-robust sandwich over Mate Networks of the OLS slope, G/(G-1) small-sample factor only; \
         a parent in no Mating Pair is a network of its own; networks are the whole pedigree's",
    ),
    ("se_ols_method", "sqrt(SSres / (n - 2) / Sxx)"),
    ("p_method", "two-sided Wald, normal reference, from the sandwich SE"),
    (
        "dependence",
        "Mate Networks are independent. Dependence between networks through ancestry \
         (a child is a parent elsewhere, cousins' parents are sibs) is not modelled.",
    ),
];

/// The parent-offspring regressions of `trait_` over `pedigree`, pooled and,
/// with `strata`, within each stratum of the child.
///
/// # Errors
///
/// Any pedigree-graph-core validation error of columns, then
/// `trait_length_mismatch`, `unsupported_trait_kind` (categorical),
/// `invalid_trait_value`, then `stratum_length_mismatch`.
pub fn parent_offspring(
    pedigree: PedigreeArg<'_>,
    trait_: Trait<'_>,
    strata: Option<Strata<'_>>,
) -> Result<ParentOffspring, Error> {
    let pedigree = pedigree.resolve()?;
    let n_rows = pedigree.len();
    let y = check_trait(trait_, n_rows)?;
    if let Some(s) = strata {
        check_strata(s, n_rows)?;
    }
    let networks = row_networks(&pedigree);
    let (codes, labels) = strata.map(Strata::codes).unzip();
    let n_strata = labels.as_ref().map_or(0, Vec::len);
    let mut scratch = Relabel::new(networks.n_keys);
    // Per parent, one fit per subset: pooled first, then each stratum.
    let fits = Parent::ALL.map(|parent| {
        let mut subsets = vec![Children::default(); 1 + n_strata];
        for child in 0..n_rows {
            let outcome = observe(&pedigree, &networks, y, parent, child);
            subsets[0].add(outcome);
            if let Some(code) = codes.as_ref().and_then(|c| c[child]) {
                subsets[1 + code].add(outcome);
            }
        }
        subsets
            .iter()
            .map(|s| s.fit(&mut scratch))
            .collect::<Vec<Fit>>()
    });
    let subset = |i: usize| Subset {
        fits: [0, 1, 2].map(|p| fits[p][i]),
    };
    let n_unknown_stratum = codes
        .as_ref()
        .map_or(0, |c| c.iter().filter(|c| c.is_none()).count() as u64);
    Ok(ParentOffspring {
        pooled: subset(0),
        strata: labels.map(|labels| {
            labels
                .into_iter()
                .enumerate()
                .map(|(i, label)| Stratum {
                    label,
                    subset: subset(1 + i),
                })
                .collect()
        }),
        n_unknown_stratum,
    })
}

/// The trait's values, after the checks this method needs.  An all-missing
/// or constant trait is not an error here: its fits carry a reason.
fn check_trait<'a>(t: Trait<'a>, n_rows: usize) -> Result<&'a [f64], Error> {
    let rule = ValueRule::of(&t, FIELD, n_rows)?;
    for (position, &value) in t.values.iter().enumerate() {
        if !value.is_nan() {
            rule.check(position, value)?;
        }
    }
    Ok(t.values)
}

/// Each row's cluster key as a parent: its Mate Network, or past the
/// networks a key of its own when it is in no Mating Pair.
struct RowNetworks {
    network: Vec<u32>,
    n_networks: usize,
    /// Networks plus one key per row.
    n_keys: usize,
}

impl RowNetworks {
    fn key(&self, row: usize) -> usize {
        match self.network[row] {
            u32::MAX => self.n_networks + row,
            network => network as usize,
        }
    }
}

fn row_networks(pedigree: &Pedigree) -> RowNetworks {
    let pairs = pedigree.mating_pairs();
    let labels = pedigree.networks();
    let mut network = vec![u32::MAX; pedigree.len()];
    for ((&m, &f), &label) in pairs.mothers.iter().zip(&pairs.fathers).zip(labels) {
        // Both mates of a pair, and every pair of one parent, share a network.
        network[m] = label as u32;
        network[f] = label as u32;
    }
    let n_networks = labels.iter().max().map_or(0, |&l| l + 1);
    RowNetworks {
        network,
        n_networks,
        n_keys: n_networks + pedigree.len(),
    }
}

/// One row as a child in one regression.
#[derive(Clone, Copy, Debug)]
enum Outcome {
    /// Parent value, child value, the parent's cluster key.
    Analysed {
        x: f64,
        y: f64,
        key: usize,
    },
    ParentNotInPedigree,
    ChildValueMissing,
    ParentValueMissing,
}

fn observe(
    pedigree: &Pedigree,
    networks: &RowNetworks,
    y: &[f64],
    parent: Parent,
    child: usize,
) -> Outcome {
    let row = |r: i32| usize::try_from(r).ok();
    let mother = row(pedigree.mother_rows()[child]);
    let father = row(pedigree.father_rows()[child]);
    let (x, key) = match (parent, mother, father) {
        (Parent::Midparent, Some(m), Some(f)) => ((y[m] + y[f]) / 2.0, networks.key(m)),
        (Parent::Mother, Some(m), _) => (y[m], networks.key(m)),
        (Parent::Father, _, Some(f)) => (y[f], networks.key(f)),
        _ => return Outcome::ParentNotInPedigree,
    };
    if y[child].is_nan() {
        Outcome::ChildValueMissing
    } else if x.is_nan() {
        Outcome::ParentValueMissing
    } else {
        Outcome::Analysed {
            x,
            y: y[child],
            key,
        }
    }
}

/// The analysed children of one regression x subset, and its drops.
#[derive(Clone, Default)]
struct Children {
    x: Vec<f64>,
    y: Vec<f64>,
    keys: Vec<usize>,
    dropped: Dropped,
}

impl Children {
    fn add(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Analysed { x, y, key } => {
                self.x.push(x);
                self.y.push(y);
                self.keys.push(key);
            }
            Outcome::ParentNotInPedigree => self.dropped.parent_not_in_pedigree += 1,
            Outcome::ChildValueMissing => self.dropped.child_value_missing += 1,
            Outcome::ParentValueMissing => self.dropped.parent_value_missing += 1,
        }
    }

    fn fit(&self, relabel: &mut Relabel) -> Fit {
        let labels = relabel.dense(&self.keys);
        let n_mate_networks = labels.iter().max().map_or(0, |&l| l + 1);
        Fit {
            n: self.x.len() as u64,
            n_mate_networks: n_mate_networks as u64,
            dropped: self.dropped,
            line: line(&self.x, &self.y, &labels, n_mate_networks),
        }
    }
}

/// Renumbers cluster keys densely `0..G` in first-seen order, so
/// [`cluster_se`] (which sizes by the largest label) counts G networks.
struct Relabel {
    slot: Vec<usize>,
}

impl Relabel {
    fn new(n_keys: usize) -> Relabel {
        Relabel {
            slot: vec![usize::MAX; n_keys],
        }
    }

    fn dense(&mut self, keys: &[usize]) -> Vec<usize> {
        let mut next = 0;
        let labels = keys
            .iter()
            .map(|&k| {
                if self.slot[k] == usize::MAX {
                    self.slot[k] = next;
                    next += 1;
                }
                self.slot[k]
            })
            .collect();
        for &k in keys {
            self.slot[k] = usize::MAX;
        }
        labels
    }
}

/// The OLS line of `y` on `x` with both SEs, or why there is none.
///
/// Sums are two-pass and centred, as simACE's `_linregress_core`, and
/// `se_ols` uses its residuals `y - (slope x + intercept)`.  A parent value
/// whose centred squares sum to 0 (equal, or differences that underflow) has
/// no line.  A constant child value fits exactly: slope 0, both SEs 0.  The
/// sandwich SE is also 0 when every network's score sum is within its
/// rounding bound (see [`sandwich_se`]).  A zero SE has no Wald test
/// ([`Reason::ZeroSe`]).
fn line(x: &[f64], y: &[f64], labels: &[usize], n_networks: usize) -> Result<Line, Reason> {
    let n = x.len();
    let Some(&x0) = x.first() else {
        return Err(Reason::NoCompleteChildren);
    };
    if x.iter().all(|&v| v == x0) {
        return Err(Reason::ConstantParentValue);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n as f64;
    let (mx, my) = (mean(x), mean(y));
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for (&xi, &yi) in x.iter().zip(y) {
        let (dx, dy) = (xi - mx, yi - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx == 0.0 {
        return Err(Reason::ConstantParentValue);
    }
    let constant_child = syy == 0.0 || y.iter().all(|&v| v == y[0]);
    let (slope, intercept, r) = if constant_child {
        (0.0, y[0], Err(Reason::ConstantChildValue))
    } else {
        let slope = sxy / sxx;
        // simACE's sqrt(Sxx Syy), split only when the product underflows.
        let product = (sxx * syy).sqrt();
        let denominator = if product > 0.0 {
            product
        } else {
            sxx.sqrt() * syy.sqrt()
        };
        let r = (sxy / denominator).clamp(-1.0, 1.0);
        (slope, my - slope * mx, Ok(r))
    };
    let se_ols = if n < 3 {
        Err(Reason::TooFewChildren)
    } else if constant_child {
        Ok(0.0)
    } else {
        let ss_res: f64 = x
            .iter()
            .zip(y)
            .map(|(&xi, &yi)| (yi - (slope * xi + intercept)).powi(2))
            .sum();
        Ok((ss_res / (n - 2) as f64 / sxx).sqrt())
    };
    let se = if n_networks < 2 {
        Err(Reason::SingleMateNetwork)
    } else if constant_child {
        Ok(0.0)
    } else {
        sandwich_se(x, y, labels, n_networks, [mx, my, sxx, slope])
    };
    let wald = match se {
        Err(reason) => Err(reason),
        Ok(0.0) => Err(Reason::ZeroSe),
        Ok(se) => Ok(Wald {
            p: 2.0 * normal::sf((slope / se).abs()),
            ci: [slope - WALD_Z * se, slope + WALD_Z * se],
        }),
    };
    Ok(Line {
        slope,
        intercept,
        r,
        se,
        se_ols,
        wald,
    })
}

/// The Mate Network sandwich SE of the slope, exactly 0 when no network's
/// score sum is distinguishable from rounding.
///
/// Residuals are centred, `e_i = (y_i - my) - slope (x_i - mx)`, so shifting
/// the trait by a constant changes nothing.  Network g's sum
/// `s_g = sum (x_i - mx) e_i / Sxx` is within rounding when
/// `|s_g| <= n eps sum |x_i - mx| (|y_i - my| + |slope (x_i - mx)|) / Sxx`,
/// the worst-case bound on summing its n terms and on each residual.  This
/// happens for a perfect fit, and when each network has one parent value and
/// there are two networks: the line then passes through both networks'
/// means, so each `s_g` is 0 in exact arithmetic.
fn sandwich_se(
    x: &[f64],
    y: &[f64],
    labels: &[usize],
    n_networks: usize,
    [mx, my, sxx, slope]: [f64; 4],
) -> Result<f64, Reason> {
    let mut influence = Vec::with_capacity(x.len());
    let mut sums = vec![0.0; n_networks];
    let mut bounds = vec![0.0; n_networks];
    for ((&xi, &yi), &l) in x.iter().zip(y).zip(labels) {
        let (dx, dy) = (xi - mx, yi - my);
        let term = dx * (dy - slope * dx) / sxx;
        influence.push(term);
        sums[l] += term;
        bounds[l] += dx.abs() * (dy.abs() + (slope * dx).abs()) / sxx;
    }
    let tolerance = x.len() as f64 * f64::EPSILON;
    if sums
        .iter()
        .zip(&bounds)
        .all(|(s, b)| s.abs() <= tolerance * b)
    {
        return Ok(0.0);
    }
    let se = cluster_se(&influence, labels);
    if se.is_finite() {
        Ok(se)
    } else {
        Err(Reason::SandwichUndefined)
    }
}

fn either<T: Into<Value>>(
    out: &mut Vec<(&'static str, Value)>,
    value: &'static str,
    reason: &'static str,
    result: Result<T, Reason>,
) {
    match result {
        Ok(v) => out.extend([(value, v.into()), (reason, Value::Null)]),
        Err(r) => out.extend([(value, Value::Null), (reason, r.name().into())]),
    }
}

impl Fit {
    pub fn to_value(&self) -> Value {
        let d = self.dropped;
        let mut out = vec![
            ("n", self.n.into()),
            ("n_mate_networks", self.n_mate_networks.into()),
            (
                "dropped",
                Value::Map(vec![
                    ("parent_not_in_pedigree", d.parent_not_in_pedigree.into()),
                    ("child_value_missing", d.child_value_missing.into()),
                    ("parent_value_missing", d.parent_value_missing.into()),
                ]),
            ),
        ];
        let line = self.line;
        either(&mut out, "slope", "reason", line.map(|l| l.slope));
        out.push(("intercept", line.ok().map(|l| l.intercept).into()));
        let r = line.and_then(|l| l.r);
        either(&mut out, "r", "r_unavailable_reason", r);
        out.push(("r2", r.ok().map(|r| r * r).into()));
        either(
            &mut out,
            "se",
            "se_unavailable_reason",
            line.and_then(|l| l.se),
        );
        either(
            &mut out,
            "se_ols",
            "se_ols_unavailable_reason",
            line.and_then(|l| l.se_ols),
        );
        let wald = line.and_then(|l| l.wald);
        either(&mut out, "p", "p_unavailable_reason", wald.map(|w| w.p));
        either(
            &mut out,
            "ci",
            "ci_unavailable_reason",
            wald.map(|w| Value::Floats(w.ci.to_vec())),
        );
        Value::Map(out)
    }
}

impl Subset {
    fn entries(&self) -> Vec<(&'static str, Value)> {
        Parent::ALL
            .iter()
            .map(|&p| (p.name(), self.fit(p).to_value()))
            .collect()
    }
}

impl ParentOffspring {
    /// The result as a host-neutral tree; its keys are the hosts' contract.
    pub fn to_value(&self) -> Value {
        let strata = self.strata.as_ref().map(|strata| {
            Value::List(
                strata
                    .iter()
                    .map(|s| {
                        let mut entries = vec![("stratum", Value::Int(s.label))];
                        entries.extend(s.subset.entries());
                        Value::Map(entries)
                    })
                    .collect(),
            )
        });
        Value::Map(vec![
            ("pooled", Value::Map(self.pooled.entries())),
            ("strata", strata.unwrap_or(Value::Null)),
            ("n_unknown_stratum", self.n_unknown_stratum.into()),
            ("settings", Value::Map(vec![("ci_level", CI_LEVEL.into())])),
            (
                "method",
                Value::Map(METHOD.iter().map(|&(k, v)| (k, v.into())).collect()),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{PedigreeInput, TraitKind};

    const NA: f64 = f64::NAN;

    /// Rows: 0-3 founders (mothers 0, 2; fathers 1, 3), children 4-7 of
    /// (0, 1) and 8-9 of (2, 3), 10 of mother 2 alone, 11 of father 3 alone.
    fn family() -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let ids: Vec<i64> = (0..12).collect();
        let mother = vec![-1, -1, -1, -1, 0, 0, 0, 0, 2, 2, 2, -1];
        let father = vec![-1, -1, -1, -1, 1, 1, 1, 1, 3, 3, -1, 3];
        (ids, mother, father)
    }

    fn run(
        ped: (&[i64], &[i64], &[i64]),
        values: &[f64],
        strata: Option<Strata<'_>>,
    ) -> Result<ParentOffspring, Error> {
        let input = PedigreeInput {
            ids: ped.0,
            mother: ped.1,
            father: ped.2,
            twin: None,
            sex: None,
        };
        parent_offspring(
            PedigreeArg::Columns(input),
            Trait {
                values,
                kind: TraitKind::Continuous,
                n_levels: None,
            },
            strata,
        )
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()), "{a} vs {b}");
    }

    /// The textbook line, residuals and cluster sums, computed by hand.
    fn reference(x: &[f64], y: &[f64], clusters: &[usize]) -> (f64, f64, f64, f64) {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
        let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let slope = sxy / sxx;
        let intercept = my - slope * mx;
        let e: Vec<f64> = x
            .iter()
            .zip(y)
            .map(|(a, b)| b - intercept - slope * a)
            .collect();
        let ss: f64 = e.iter().map(|v| v * v).sum();
        let se_ols = (ss / (n - 2.0) / sxx).sqrt();
        let g = clusters.iter().max().unwrap() + 1;
        let mut sums = vec![0.0; g];
        for i in 0..x.len() {
            sums[clusters[i]] += (x[i] - mx) * e[i] / sxx;
        }
        let se = (g as f64 / (g as f64 - 1.0) * sums.iter().map(|s| s * s).sum::<f64>()).sqrt();
        (slope, intercept, se_ols, se)
    }

    #[test]
    fn midparent_mother_and_father_match_the_hand_computation() {
        let (ids, mother, father) = family();
        let v = [1.0, 3.0, -1.0, 0.5, 2.5, 1.0, 2.2, 3.1, 0.4, -0.3, 0.9, 0.2];
        let res = run((&ids, &mother, &father), &v, None).unwrap();
        let mid = res.pooled.fit(Parent::Midparent);
        // Children 4-7 (midparent 2.0, network of pair (0,1)) and 8-9 (-0.25).
        let x = [2.0, 2.0, 2.0, 2.0, -0.25, -0.25];
        let y = [2.5, 1.0, 2.2, 3.1, 0.4, -0.3];
        let (slope, intercept, se_ols, _) = reference(&x, &y, &[0, 0, 0, 0, 1, 1]);
        let line = mid.line.unwrap();
        assert_eq!((mid.n, mid.n_mate_networks), (6, 2));
        close(line.slope, slope);
        close(line.intercept, intercept);
        close(line.se_ols.unwrap(), se_ols);
        // Mother: children 4-10 (10 has no father but mother 2 is in a pair).
        let mo = res.pooled.fit(Parent::Mother);
        assert_eq!((mo.n, mo.n_mate_networks), (7, 2));
        // Father: children 4-9 and 11.
        let fa = res.pooled.fit(Parent::Father);
        assert_eq!((fa.n, fa.n_mate_networks), (7, 2));
        let x = [3.0, 3.0, 3.0, 3.0, 0.5, 0.5, 0.5];
        let y = [2.5, 1.0, 2.2, 3.1, 0.4, -0.3, 0.2];
        let (slope, ..) = reference(&x, &y, &[0, 0, 0, 0, 1, 1, 1]);
        close(fa.line.unwrap().slope, slope);
    }

    #[test]
    fn drops_count_each_row_once_and_sum_to_the_rows() {
        let (ids, mother, father) = family();
        let mut v = [1.0, 3.0, -1.0, 0.5, 2.5, 1.0, 2.2, 3.1, 0.4, -0.3, 0.9, 0.2];
        v[5] = NA; // a child
        v[1] = NA; // father of 4-7
        let res = run((&ids, &mother, &father), &v, None).unwrap();
        for p in Parent::ALL {
            let fit = res.pooled.fit(p);
            assert_eq!(fit.n + fit.dropped.total(), 12, "{p:?}");
        }
        let mid = res.pooled.fit(Parent::Midparent);
        // Founders and 10, 11 have no pair of parents; child 5 missing;
        // 4, 6, 7 lose their father's value.
        assert_eq!(
            mid.dropped,
            Dropped {
                parent_not_in_pedigree: 6,
                child_value_missing: 1,
                parent_value_missing: 3,
            }
        );
        assert_eq!(mid.n, 2);
    }

    #[test]
    fn a_parent_missing_the_trait_still_joins_two_sibships() {
        // Father 2 has children with mothers 0 and 1; his value is missing,
        // so no midparent uses him, but both sibships stay one network.
        let ids: Vec<i64> = (0..8).collect();
        let mother = vec![-1, -1, -1, 0, 0, 1, 1, 1];
        let father = vec![-1, -1, -1, 2, 2, 2, 2, 2];
        let v = [1.0, 2.0, NA, 0.5, 1.5, 1.8, 2.4, 2.0];
        let res = run((&ids, &mother, &father), &v, None).unwrap();
        let mo = res.pooled.fit(Parent::Mother);
        assert_eq!((mo.n, mo.n_mate_networks), (5, 1));
        assert_eq!(mo.line.unwrap().se, Err(Reason::SingleMateNetwork));
        assert_eq!(mo.line.unwrap().wald, Err(Reason::SingleMateNetwork));
        assert!(mo.line.unwrap().se_ols.is_ok());
    }

    #[test]
    fn a_single_parent_is_its_own_network_shared_by_its_children() {
        // Mothers 0 and 1, fathers unknown: two networks of two children.
        let ids: Vec<i64> = (0..6).collect();
        let mother = vec![-1, -1, 0, 0, 1, 1];
        let father = vec![-1; 6];
        let v = [1.0, 2.0, 0.8, 1.4, 2.5, 1.7];
        let res = run((&ids, &mother, &father), &v, None).unwrap();
        let mo = res.pooled.fit(Parent::Mother);
        assert_eq!((mo.n, mo.n_mate_networks), (4, 2));
        // One parent value per network, two networks: an exact zero SE.
        assert_eq!(mo.line.unwrap().se, Ok(0.0));
        assert_eq!(
            res.pooled.fit(Parent::Midparent).line,
            Err(Reason::NoCompleteChildren)
        );
    }

    #[test]
    fn sparse_keys_are_renumbered_before_the_cluster_count() {
        let mut relabel = Relabel::new(10);
        assert_eq!(relabel.dense(&[2, 7, 7, 2, 9]), vec![0, 1, 1, 0, 2]);
        // The scratch is clean for the next subset.
        assert_eq!(relabel.dense(&[7]), vec![0]);
    }

    #[test]
    fn a_stratum_counts_only_its_own_networks() {
        // Pairs (0,1), (2,3), (4,5), (6,7) are networks 0-3.  The stratum
        // holds the children of networks 1-3 only: G is 3 (keys 1, 2, 3
        // renumbered), not 4, and the sandwich matches the hand sums.
        let ids: Vec<i64> = (0..15).collect();
        let mother = [vec![-1; 8], vec![0, 2, 2, 4, 4, 6, 6]].concat();
        let father = [vec![-1; 8], vec![1, 3, 3, 5, 5, 7, 7]].concat();
        let v = [
            0.0, 1.0, 0.2, 0.6, 1.4, 1.0, -0.5, 0.1, 0.7, 0.1, 0.9, 1.5, 1.0, -0.6, 0.3,
        ];
        let labels = [vec![0; 9], vec![1; 6]].concat();
        let known = [true; 15];
        let strata = Strata {
            labels: &labels,
            known: &known,
        };
        let res = run((&ids, &mother, &father), &v, Some(strata)).unwrap();
        let fit = *res.strata.unwrap()[1].subset.fit(Parent::Midparent);
        assert_eq!((fit.n, fit.n_mate_networks), (6, 3));
        let x = [0.4, 0.4, 1.2, 1.2, -0.2, -0.2];
        let y = [0.1, 0.9, 1.5, 1.0, -0.6, 0.3];
        let (slope, _, se_ols, se) = reference(&x, &y, &[0, 0, 1, 1, 2, 2]);
        let line = fit.line.unwrap();
        assert!(se > 0.01, "{se}");
        close(line.slope, slope);
        close(line.se_ols.unwrap(), se_ols);
        close(line.se.unwrap(), se);
        assert!(line.wald.is_ok());
    }

    #[test]
    fn undefined_lines_and_zero_se_carry_reasons() {
        let (ids, mother, father) = family();
        // Every parent equal: no slope.
        let mut v = [1.0; 12];
        v[4] = 2.0;
        let res = run((&ids, &mother, &father), &v, None).unwrap();
        assert_eq!(
            res.pooled.fit(Parent::Midparent).line,
            Err(Reason::ConstantParentValue)
        );
        // Every child equal, parents not: slope 0, r undefined, zero SEs.
        let v = [1.0, 3.0, -1.0, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let res = run((&ids, &mother, &father), &v, None).unwrap();
        let line = res.pooled.fit(Parent::Midparent).line.unwrap();
        assert_eq!((line.slope, line.intercept), (0.0, 2.0));
        assert_eq!(line.r, Err(Reason::ConstantChildValue));
        assert_eq!((line.se, line.se_ols), (Ok(0.0), Ok(0.0)));
        assert_eq!(line.wald, Err(Reason::ZeroSe));
        // All missing: nothing to fit.
        let res = run((&ids, &mother, &father), &[NA; 12], None).unwrap();
        assert_eq!(
            res.pooled.fit(Parent::Father).line,
            Err(Reason::NoCompleteChildren)
        );
    }

    #[test]
    fn two_networks_of_one_parent_value_each_have_no_wald_test() {
        // The line passes through both sibships' means: every score sum is 0
        // in exact arithmetic, so the SE is 0, not rounding noise.
        let (ids, mother, father) = family();
        let v = [1.0, 3.0, -1.0, 0.5, 2.5, 1.0, 2.2, 3.1, 0.4, -0.3, 0.9, 0.2];
        let res = run((&ids, &mother, &father), &v, None).unwrap();
        let line = res.pooled.fit(Parent::Midparent).line.unwrap();
        assert_eq!(line.se, Ok(0.0));
        assert_eq!(line.wald, Err(Reason::ZeroSe));
        assert!(line.se_ols.unwrap() > 0.1);
    }

    #[test]
    fn shifting_the_trait_leaves_the_sandwich_se_alone() {
        // 10^6 children in 4 networks on a 1/1024 grid, so a shift of 2^27
        // is exact.  It grows |y| but not the residuals: the SE must match
        // the unshifted one, not fall to zero_se.
        let n = 1_000_000;
        let mut state: u64 = 12345;
        let mut draw = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 54) as f64 / 1024.0
        };
        let x: Vec<f64> = (0..n).map(|_| draw()).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 0.5 * xi + draw()).collect();
        let labels: Vec<usize> = (0..n).map(|i| i % 4).collect();
        let shift = |v: &[f64]| v.iter().map(|a| a + 2f64.powi(27)).collect::<Vec<_>>();
        let base = line(&x, &y, &labels, 4).unwrap();
        let shifted = line(&shift(&x), &shift(&y), &labels, 4).unwrap();
        let (a, b) = (base.se.unwrap(), shifted.se.unwrap());
        // The shifted means carry about sqrt(n) eps 2^27 of rounding, which
        // moves the slope and SE by about 1e-5: precision, not a zero SE.
        assert!(a > 0.0 && (a - b).abs() <= 1e-4 * a, "{a} vs {b}");
        assert!(shifted.wald.is_ok(), "{:?}", shifted.wald);
        assert!((base.slope - shifted.slope).abs() <= 1e-4 * base.slope);
    }

    #[test]
    fn exact_lines_keep_r_within_one() {
        // y = 2x + 1 on non-dyadic x: rounding can put sxy / sqrt(sxx syy)
        // just past 1.
        let mut worst: f64 = 0.0;
        for k in 3..400 {
            let x: Vec<f64> = (0..k).map(|i| (i as f64 * 0.1).sin() / 3.0).collect();
            let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
            let r = line(&x, &y, &vec![0; k], 1).unwrap().r.unwrap();
            worst = worst.max(r);
            assert!(r <= 1.0, "k = {k}: r = {r}");
        }
        assert_eq!(worst, 1.0);
    }

    #[test]
    fn parent_differences_that_underflow_have_no_line() {
        let x = [0.0, 1e-170, 2e-170];
        let y = [0.1, 0.4, 0.2];
        assert_eq!(
            line(&x, &y, &[0, 1, 2], 3),
            Err(Reason::ConstantParentValue)
        );
        // Sxx Syy underflows though neither factor does: r still comes out
        // (NumPy's corrcoef of the unscaled values), to the few digits a
        // subnormal Syy (2e-320) keeps.
        let x = [0.0, 1e-150, 3e-150];
        let y = [0.0, 1e-160, 2e-160];
        let r = line(&x, &y, &[0, 1, 2], 3).unwrap().r.unwrap();
        assert!((r - 0.9819805060619656).abs() < 1e-4, "{r}");
    }

    #[test]
    fn two_children_in_two_networks_fit_exactly() {
        let ids: Vec<i64> = (0..6).collect();
        let mother = vec![-1, -1, -1, -1, 0, 2];
        let father = vec![-1, -1, -1, -1, 1, 3];
        let v = [0.3, 1.1, -0.7, 2.9, 0.37, 1.13];
        let res = run((&ids, &mother, &father), &v, None).unwrap();
        let fit = res.pooled.fit(Parent::Midparent);
        let line = fit.line.unwrap();
        assert_eq!((fit.n, fit.n_mate_networks), (2, 2));
        assert_eq!(line.se, Ok(0.0));
        assert_eq!(line.se_ols, Err(Reason::TooFewChildren));
        assert_eq!(line.wald, Err(Reason::ZeroSe));
    }

    #[test]
    fn strata_partition_children_and_unknown_stay_pooled() {
        let (ids, mother, father) = family();
        let v = [1.0, 3.0, -1.0, 0.5, 2.5, 1.0, 2.2, 3.1, 0.4, -0.3, 0.9, 0.2];
        let labels = [0, 0, 0, 0, 7, 7, 7, 7, 3, 3, 3, 3];
        let mut known = [true; 12];
        known[11] = false;
        let strata = Strata {
            labels: &labels,
            known: &known,
        };
        let res = run((&ids, &mother, &father), &v, Some(strata)).unwrap();
        let strata = res.strata.unwrap();
        assert_eq!(
            strata.iter().map(|s| s.label).collect::<Vec<_>>(),
            vec![0, 3, 7]
        );
        assert_eq!(res.n_unknown_stratum, 1);
        for p in Parent::ALL {
            let rows: u64 = strata
                .iter()
                .map(|s| s.subset.fit(p).n + s.subset.fit(p).dropped.total())
                .sum();
            assert_eq!(rows + res.n_unknown_stratum, 12);
        }
        // Stratum 0 holds only founders.
        let founders = strata[0].subset.fit(Parent::Mother);
        assert_eq!(founders.line, Err(Reason::NoCompleteChildren));
        assert_eq!(founders.dropped.parent_not_in_pedigree, 4);
        // Stratum 7: one sibship, so one network.
        let sibs = strata[2].subset.fit(Parent::Father);
        assert_eq!((sibs.n, sibs.n_mate_networks), (4, 1));
        // Child 11 (unknown stratum) is pooled.
        assert_eq!(res.pooled.fit(Parent::Father).n, 7);
    }

    #[test]
    fn trait_checks() {
        let (ids, mother, father) = family();
        let input = PedigreeInput {
            ids: &ids,
            mother: &mother,
            father: &father,
            twin: None,
            sex: None,
        };
        let code = |values: &[f64], kind| {
            parent_offspring(
                PedigreeArg::Columns(input),
                Trait {
                    values,
                    kind,
                    n_levels: None,
                },
                None,
            )
            .unwrap_err()
            .code()
        };
        assert_eq!(
            code(&[0.0; 3], TraitKind::Continuous),
            "trait_length_mismatch"
        );
        assert_eq!(
            code(&[0.0; 12], TraitKind::Categorical),
            "unsupported_trait_kind"
        );
        let mut v = [0.0; 12];
        v[3] = 2.0;
        assert_eq!(code(&v, TraitKind::Binary), "invalid_trait_value");
        v[3] = 0.5;
        assert_eq!(code(&v, TraitKind::Ordinal), "invalid_trait_value");
        v[3] = f64::INFINITY;
        assert_eq!(code(&v, TraitKind::Continuous), "invalid_trait_value");
        let labels = [0i64; 3];
        let known = [true; 3];
        let err = parent_offspring(
            PedigreeArg::Columns(input),
            Trait {
                values: &[0.0; 12],
                kind: TraitKind::Binary,
                n_levels: None,
            },
            Some(Strata {
                labels: &labels,
                known: &known,
            }),
        )
        .unwrap_err();
        assert_eq!(err.code(), "stratum_length_mismatch");
    }

    #[test]
    fn binary_traits_regress_on_codes() {
        let (ids, mother, father) = family();
        let v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, NA, 1.0];
        let input = PedigreeInput {
            ids: &ids,
            mother: &mother,
            father: &father,
            twin: None,
            sex: None,
        };
        let res = parent_offspring(
            PedigreeArg::Columns(input),
            Trait {
                values: &v,
                kind: TraitKind::Binary,
                n_levels: None,
            },
            None,
        )
        .unwrap();
        let x = [0.5, 0.5, 0.5, 0.5, 0.0, 0.0];
        let y = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let (slope, ..) = reference(&x, &y, &[0, 0, 0, 0, 1, 1]);
        close(res.pooled.fit(Parent::Midparent).line.unwrap().slope, slope);
    }
}
