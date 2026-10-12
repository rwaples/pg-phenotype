//! Bivariate PA-FGRS: two genetically correlated traits scored jointly.

use super::cip::{Cip, Observed};
use super::order;
use super::pa::{self, Obs};
use super::prep::{Prep, Relatives};
use super::uni::check_h2;
use crate::error::{Error, Inconsistency};
use crate::input::Trait;
use crate::value::Value;

/// Slack on the positive-semidefinite check of [`BivParams::new`], so a
/// `rho_within` computed at the boundary in floating point is not rejected
/// for its last-bit rounding (ADR 0002).
const PSD_SLACK: f64 = 1e-12;

/// The genetic and within-person parameters of a bivariate score.
///
/// The fields are private so every value has passed [`BivParams::new`]'s
/// checks (ADR 0002).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BivParams {
    h2: [f64; 2],
    rg: f64,
    rho_within: f64,
}

impl BivParams {
    /// Check the parameters; `rho_within` defaults to `rg sqrt(h2_1 h2_2)`,
    /// the genetic covariance (no shared non-genetic correlation).
    ///
    /// # Errors
    ///
    /// [`Error::ParameterOutOfRange`] for an `h2` outside `(0, 1]`, `rg`
    /// outside `[-1, 1]` or `|rho_within| > 1`, and
    /// [`Error::InconsistentParameters`] when `rho_within` leaves a
    /// non-genetic covariance that is not positive semidefinite.
    pub fn new(h2: [f64; 2], rg: f64, rho_within: Option<f64>) -> Result<BivParams, Error> {
        check_h2("h2_1", h2[0])?;
        check_h2("h2_2", h2[1])?;
        if !(-1.0..=1.0).contains(&rg) {
            return Err(Error::ParameterOutOfRange {
                name: "rg",
                value: rg,
                domain: "[-1, 1]",
            });
        }
        let cov_g = rg * (h2[0] * h2[1]).sqrt();
        let rho_within = rho_within.unwrap_or(cov_g);
        if !(-1.0..=1.0).contains(&rho_within) {
            return Err(Error::ParameterOutOfRange {
                name: "rho_within",
                value: rho_within,
                domain: "[-1, 1]",
            });
        }
        let residual = rho_within - cov_g;
        if residual * residual > (1.0 - h2[0]) * (1.0 - h2[1]) + PSD_SLACK {
            return Err(Error::InconsistentParameters {
                reason: Inconsistency::NonGeneticCovarianceNotPsd,
            });
        }
        Ok(BivParams { h2, rg, rho_within })
    }

    /// The two heritabilities.
    pub fn h2(&self) -> [f64; 2] {
        self.h2
    }

    /// Genetic correlation.
    pub fn rg(&self) -> f64 {
        self.rg
    }

    /// A person's cross-trait liability correlation.
    pub fn rho_within(&self) -> f64 {
        self.rho_within
    }

    /// The genetic covariance `rg sqrt(h2_1 h2_2)`.
    pub fn cov_g(&self) -> f64 {
        self.rg * (self.h2[0] * self.h2[1]).sqrt()
    }
}

/// One joint score per proband, in ascending row order.
#[derive(Clone, Debug, PartialEq)]
pub struct BivScores {
    pub ids: Vec<i64>,
    pub est: [Vec<f64>; 2],
    pub var: [Vec<f64>; 2],
    pub cov12: Vec<f64>,
    /// Relatives observed (`w > 0`) on at least one trait.
    pub n_relatives: Vec<u32>,
    /// Observations per trait.
    pub n_obs: [Vec<u32>; 2],
    /// The parameters scored with, `rho_within` as used.
    pub params: BivParams,
    pub prevalence: [f64; 2],
    pub threshold: [f64; 2],
    pub controls_without_age: [usize; 2],
    pub ndegree: u8,
}

impl BivScores {
    /// The result's metadata, in the hosts' key order; each binding adds
    /// the versions.
    pub fn metadata(&self) -> Vec<(&'static str, Value)> {
        let p = &self.params;
        vec![
            ("h2", Value::Floats(p.h2().to_vec())),
            ("rg", p.rg().into()),
            ("rho_within", p.rho_within().into()),
            ("prevalence", Value::Floats(self.prevalence.to_vec())),
            ("threshold", Value::Floats(self.threshold.to_vec())),
            (
                "controls_without_age",
                Value::Counts(self.controls_without_age.map(|c| c as u64).to_vec()),
            ),
            ("n_probands", (self.ids.len() as u64).into()),
            ("ndegree", u64::from(self.ndegree).into()),
        ]
    }
}

/// The result of one proband.
#[derive(Clone, Copy, Default)]
struct One {
    est: [f64; 2],
    var: [f64; 2],
    cov12: f64,
    n_relatives: u32,
    n_obs: [u32; 2],
}

/// Score every proband of `prep` on two traits jointly.
///
/// # Errors
///
/// Each trait's length, kind and value errors, then its age's; the
/// parameters were checked when `params` was built.
pub fn score_bivariate(
    prep: &Prep,
    values: [Trait<'_>; 2],
    ages: [&[f64]; 2],
    cips: [&Cip; 2],
    params: BivParams,
) -> Result<BivScores, Error> {
    let obs = [
        Observed::new(
            values[0],
            ages[0],
            cips[0],
            prep.n_rows(),
            ["trait1", "age1"],
        )?,
        Observed::new(
            values[1],
            ages[1],
            cips[1],
            prep.n_rows(),
            ["trait2", "age2"],
        )?,
    ];
    let one = prep.score_each(|s: &mut Scratch, rel| s.score(rel, &obs, &params));
    let per_trait = |f: fn(&One) -> [f64; 2], t: usize| one.iter().map(|o| f(o)[t]).collect();
    Ok(BivScores {
        ids: prep.proband_ids(),
        est: [per_trait(|o| o.est, 0), per_trait(|o| o.est, 1)],
        var: [per_trait(|o| o.var, 0), per_trait(|o| o.var, 1)],
        cov12: one.iter().map(|o| o.cov12).collect(),
        n_relatives: one.iter().map(|o| o.n_relatives).collect(),
        n_obs: [0, 1].map(|t| one.iter().map(|o| o.n_obs[t]).collect()),
        params,
        prevalence: cips.map(Cip::prevalence),
        threshold: [obs[0].threshold, obs[1].threshold],
        controls_without_age: [obs[0].controls_without_age, obs[1].controls_without_age],
        ndegree: prep.ndegree(),
    })
}

/// Per-worker buffers, reused across probands.
#[derive(Default)]
struct Scratch {
    /// Relative positions observed on at least one trait.
    people: Vec<usize>,
    /// Kinship among `people`, dense.
    phi: Vec<f64>,
    /// Per person, whether each trait is observed.
    seen: Vec<[bool; 2]>,
    keys: Vec<order::Key>,
    obs: Vec<Obs>,
    cov: Vec<f64>,
    mu: Vec<f64>,
    col: Vec<f64>,
}

impl Scratch {
    fn score(&mut self, rel: Relatives<'_>, obs: &[Observed; 2], params: &BivParams) -> One {
        let [h1, h2] = params.h2;
        let cov_g = params.cov_g();
        let g = [[h1, cov_g], [cov_g, h2]];
        self.people.clear();
        self.seen.clear();
        let mut n_obs = [0u32; 2];
        for (j, &r) in rel.rows.iter().enumerate() {
            let seen = [obs[0].w[r as usize] > 0.0, obs[1].w[r as usize] > 0.0];
            if seen[0] || seen[1] {
                self.people.push(j);
                self.seen.push(seen);
                n_obs[0] += u32::from(seen[0]);
                n_obs[1] += u32::from(seen[1]);
            }
        }
        let n = self.people.len();
        if n == 0 {
            return One {
                est: [0.0, 0.0],
                var: [h1, h2],
                cov12: cov_g,
                n_relatives: 0,
                n_obs,
            };
        }
        rel.dense_kinship(&self.people, &mut self.phi);
        let abs_cov = cov_g.abs();
        let abs_rho = params.rho_within.abs();
        self.keys.clear();
        for (x, &j) in self.people.iter().enumerate() {
            let row = rel.rows[j];
            let phi_p = f64::from(rel.kin[j]);
            let mut sums = [0.0f64; 2];
            for (y, seen) in self.seen.iter().enumerate() {
                if y != x {
                    for t in 0..2 {
                        if seen[t] {
                            sums[t] += self.phi[x * n + y];
                        }
                    }
                }
            }
            for t in 0..2 {
                if !self.seen[x][t] {
                    continue;
                }
                let to_proband = 2.0 * phi_p * (params.h2[t] + abs_cov);
                let rho_term = if self.seen[x][1 - t] { abs_rho } else { 0.0 };
                let row_sum = to_proband
                    + 1.0
                    + rho_term
                    + 2.0 * (params.h2[t] * sums[t] + abs_cov * sums[1 - t]);
                self.keys.push(order::Key {
                    w: obs[t].w[row as usize],
                    to_proband,
                    row_sum,
                    row,
                    trait_index: t as u8,
                    index: x,
                });
            }
        }
        order::sort(&mut self.keys);

        let m = self.keys.len();
        let size = m + 2;
        self.cov.clear();
        self.cov.resize(size * size, 0.0);
        self.mu.resize(size, 0.0);
        self.obs.clear();
        self.cov[0] = h1;
        self.cov[1] = cov_g;
        self.cov[size + 1] = h2;
        for (i, key) in self.keys.iter().enumerate() {
            let t = key.trait_index as usize;
            let phi_p = f64::from(rel.kin[self.people[key.index]]);
            let d = 2 + i;
            self.cov[d] = 2.0 * phi_p * g[0][t];
            self.cov[size + d] = 2.0 * phi_p * g[1][t];
            self.cov[d * size + d] = 1.0;
            for (k, other) in self.keys.iter().enumerate().skip(i + 1) {
                let u = other.trait_index as usize;
                self.cov[d * size + 2 + k] = if other.index == key.index {
                    params.rho_within
                } else {
                    2.0 * self.phi[key.index * n + other.index] * g[t][u]
                };
            }
            self.obs.push(obs[t].obs(key.row as usize));
        }
        pa::condition(&mut self.cov, &mut self.mu, 2, &self.obs, &mut self.col);
        One {
            est: [self.mu[0], self.mu[1]],
            var: [self.cov[0].max(0.0), self.cov[size + 1].max(0.0)],
            cov12: self.cov[1],
            n_relatives: n as u32,
            n_obs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pafgrs::uni::tests::{case, keys, AGE};

    #[test]
    fn metadata_follows_the_hosts_key_order() {
        let (prep, cip, status) = case();
        let params = BivParams::new([0.5, 0.4], 0.3, None).unwrap();
        let biv = score_bivariate(&prep, [status; 2], [&AGE; 2], [&cip; 2], params).unwrap();
        let meta = biv.metadata();
        assert_eq!(
            keys(&meta),
            [
                "h2",
                "rg",
                "rho_within",
                "prevalence",
                "threshold",
                "controls_without_age",
                "n_probands",
                "ndegree"
            ]
        );
        assert_eq!(meta[0].1, Value::Floats(vec![0.5, 0.4]));
        assert_eq!(meta[2].1, Value::Float(params.rho_within()));
        assert_eq!(meta[5].1, Value::Counts(vec![1, 1]));
    }
}
