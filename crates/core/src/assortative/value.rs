//! [`MateCorrelation`] as a [`Value`] tree: the one walk both bindings
//! convert (Python dicts the package wraps in dataclasses, R named lists).
//!
//! An undefined estimate carries `reason` and no estimate keys; an
//! unavailable `se`, `ci` or `p` is `null` with its `*_unavailable_reason`.

use super::result::{
    Cell, Draws, Estimate, EstimatorResult, MateCorrelation, Permutation, Point, Reason,
    WithinPerson,
};
use crate::value::Value;

type Entries = Vec<(&'static str, Value)>;

/// `key` and its `*_unavailable_reason`.
fn reason_or(
    out: &mut Entries,
    [key, reason_key]: [&'static str; 2],
    value: Result<Value, Reason>,
) {
    match value {
        Ok(v) => out.extend([(key, v), (reason_key, Value::Null)]),
        Err(r) => out.extend([(key, Value::Null), (reason_key, r.name().into())]),
    }
}

fn draws(d: &Draws) -> Value {
    Value::Map(vec![
        ("requested", d.requested.into()),
        ("valid", d.valid.into()),
        ("failed", d.failed.into()),
        (
            "failure_reasons",
            Value::Map(
                d.failure_reasons
                    .iter()
                    .map(|(r, n)| (r.name(), (*n).into()))
                    .collect(),
            ),
        ),
    ])
}

fn permutation(p: &Permutation) -> Value {
    let mut out = vec![("statistic", p.statistic.name().into())];
    reason_or(&mut out, ["p", "p_unavailable_reason"], p.p.map(Into::into));
    out.extend([
        ("draws", draws(&p.draws)),
        ("seed", Value::Int(p.seed)),
        ("n_fixed_fathers", p.n_fixed_fathers.into()),
        ("stopped_early", p.stopped_early.into()),
        ("draws_used", p.draws_used.into()),
        ("sequential_h", p.sequential_h.into()),
    ]);
    Value::Map(out)
}

fn point(out: &mut Entries, p: &Point) {
    out.extend([("value", p.value.into()), ("boundary", p.boundary.into())]);
}

fn estimate(out: &mut Entries, e: &Estimate) {
    point(out, &e.point);
    reason_or(out, ["se", "se_unavailable_reason"], e.se.map(Into::into));
    reason_or(
        out,
        ["ci", "ci_unavailable_reason"],
        e.ci.map(|c| Value::Floats(c.bounds.to_vec())),
    );
    out.extend([
        ("ci_method", e.ci.ok().map(|c| c.method.name()).into()),
        ("bootstrap", e.bootstrap.as_ref().map_or(Value::Null, draws)),
        (
            "permutation",
            e.permutation.as_ref().map_or(Value::Null, permutation),
        ),
    ]);
}

fn estimator_result(r: &EstimatorResult, extra: Entries) -> Value {
    let mut out = vec![
        ("estimator", r.estimator.name().into()),
        ("primary", r.primary.into()),
    ];
    match &r.outcome {
        Ok(e) => {
            out.push(("reason", Value::Null));
            estimate(&mut out, e);
        }
        Err(reason) => out.push(("reason", reason.name().into())),
    }
    out.extend(extra);
    Value::Map(out)
}

fn cell(c: &Cell) -> Value {
    let d = &c.n_dropped;
    let stratified = c.stratified.as_ref().map_or(Value::Null, |s| {
        estimator_result(
            &s.result,
            vec![
                ("n_strata_mothers", s.n_strata_mothers.into()),
                ("n_strata_fathers", s.n_strata_fathers.into()),
            ],
        )
    });
    let table = c.table.map_or(Value::Null, |t| {
        Value::List(t.iter().map(|row| Value::Counts(row.to_vec())).collect())
    });
    Value::Map(vec![
        ("mother_trait", (c.mother_trait as u64).into()),
        ("father_trait", (c.father_trait as u64).into()),
        ("n", c.n.into()),
        (
            "n_dropped",
            Value::Map(vec![
                ("mother_missing", d.mother_missing.into()),
                ("father_missing", d.father_missing.into()),
                ("both_missing", d.both_missing.into()),
                ("small_stratum", d.small_stratum.into()),
                ("degenerate_stratum", d.degenerate_stratum.into()),
            ]),
        ),
        ("n_mate_networks", c.n_mate_networks.into()),
        (
            "largest_mate_network_share",
            c.largest_mate_network_share.into(),
        ),
        ("table", table),
        (
            "crude",
            Value::List(
                c.crude
                    .iter()
                    .map(|r| estimator_result(r, Vec::new()))
                    .collect(),
            ),
        ),
        ("stratified", stratified),
    ])
}

fn within(w: &WithinPerson) -> Value {
    let mut out = vec![("estimator", w.estimator.name().into()), ("n", w.n.into())];
    match &w.outcome {
        Ok(p) => {
            point(&mut out, p);
            out.push(("reason", Value::Null));
        }
        Err(r) => out.extend([
            ("value", Value::Null),
            ("boundary", Value::Null),
            ("reason", r.name().into()),
        ]),
    }
    Value::Map(out)
}

impl MateCorrelation {
    /// The result as a [`Value`] tree, in the key order hosts present.
    pub fn to_value(&self) -> Value {
        let s = &self.sample;
        let e = &self.settings;
        let m = &self.method;
        Value::Map(vec![
            (
                "sample",
                Value::Map(vec![
                    ("n_total", s.n_total.into()),
                    (
                        "n_dropped_unknown_stratum",
                        s.n_dropped_unknown_stratum.into(),
                    ),
                    ("n_mate_networks", s.n_mate_networks.into()),
                    (
                        "largest_mate_network_share",
                        s.largest_mate_network_share.into(),
                    ),
                    (
                        "n_mothers_multiple_mates",
                        s.n_mothers_multiple_mates.into(),
                    ),
                    (
                        "n_fathers_multiple_mates",
                        s.n_fathers_multiple_mates.into(),
                    ),
                ]),
            ),
            ("cells", Value::List(self.cells.iter().map(cell).collect())),
            (
                "within_person",
                self.within_person.as_ref().map_or(Value::Null, |w| {
                    Value::Map(vec![
                        ("mothers", within(&w.mothers)),
                        ("fathers", within(&w.fathers)),
                    ])
                }),
            ),
            (
                "settings",
                Value::Map(vec![
                    ("permutations", e.permutations.into()),
                    ("bootstrap", e.bootstrap.into()),
                    ("seed", Value::Int(e.seed)),
                    ("threads", (e.threads as u64).into()),
                    ("ci_level", e.ci_level.into()),
                    ("min_stratum_networks", e.min_stratum_networks.into()),
                ]),
            ),
            (
                "method",
                Value::Map(vec![
                    ("se_method", m.se_method.into()),
                    ("ci_scale", m.ci_scale.into()),
                    ("bootstrap_unit", m.bootstrap_unit.into()),
                    ("bootstrap_assumption", m.bootstrap_assumption.into()),
                    ("bootstrap_method", m.bootstrap_method.into()),
                    ("permutation_null", m.permutation_null.into()),
                    ("permutation_blocks", m.permutation_blocks.into()),
                    ("permutation_statistic", m.permutation_statistic.into()),
                    ("permutation_stopping", m.permutation_stopping.into()),
                    ("permutation_stop_h", m.permutation_stop_h.into()),
                ]),
            ),
        ])
    }
}
