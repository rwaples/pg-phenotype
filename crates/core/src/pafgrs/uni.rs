//! Univariate PA-FGRS: one trait, one parameter variant, every proband.

use super::cip::{Cip, Observed};
use super::order;
use super::pa::{self, Obs};
use super::prep::Prep;
use crate::error::Error;
use crate::input::Trait;

/// One score per proband, in ascending row order.
#[derive(Clone, Debug, PartialEq)]
pub struct UniScores {
    /// Proband ids.
    pub ids: Vec<i64>,
    /// Posterior mean of the proband's genetic liability.
    pub est: Vec<f64>,
    /// Posterior variance.
    pub var: Vec<f64>,
    /// Relatives with `w > 0`.
    pub n_relatives: Vec<u32>,
    /// Controls with no age, scored as unobserved.
    pub controls_without_age: usize,
    /// The liability threshold the CIP table gives.
    pub threshold: f64,
}

/// Check that `h2` is in `(0, 1]`.
pub(crate) fn check_h2(name: &'static str, h2: f64) -> Result<(), Error> {
    if h2 > 0.0 && h2 <= 1.0 {
        return Ok(());
    }
    Err(Error::ParameterOutOfRange {
        name,
        value: h2,
        domain: "(0, 1]",
    })
}

/// Score every proband of `prep` on one trait.
///
/// # Errors
///
/// [`Error::ParameterOutOfRange`] for `h2` outside `(0, 1]`, checked first;
/// then the trait's kind, length, value, and age errors.
pub fn score_univariate(
    prep: &Prep,
    values: Trait<'_>,
    age: &[f64],
    cip: &Cip,
    h2: f64,
) -> Result<UniScores, Error> {
    check_h2("h2", h2)?;
    let obs = Observed::new(values, age, cip, prep.n_rows(), ["trait", "age"])?;
    let one = prep.score_each(|s: &mut Scratch, rel| s.score(rel, &obs, h2));
    Ok(UniScores {
        ids: prep.proband_ids(),
        est: one.iter().map(|o| o.est).collect(),
        var: one.iter().map(|o| o.var).collect(),
        n_relatives: one.iter().map(|o| o.n_relatives).collect(),
        controls_without_age: obs.controls_without_age,
        threshold: obs.threshold,
    })
}

/// The result of one proband.
#[derive(Clone, Copy, Default)]
struct One {
    est: f64,
    var: f64,
    n_relatives: u32,
}

/// Per-worker buffers, reused across probands.
#[derive(Default)]
struct Scratch {
    /// Positions (in the proband's relative list) with `w > 0`.
    valid: Vec<usize>,
    /// Kinship among the valid relatives, dense, in `valid` order.
    phi: Vec<f64>,
    keys: Vec<order::Key>,
    obs: Vec<Obs>,
    cov: Vec<f64>,
    mu: Vec<f64>,
    col: Vec<f64>,
}

impl Scratch {
    fn score(&mut self, rel: super::prep::Relatives<'_>, obs: &Observed, h2: f64) -> One {
        self.valid.clear();
        self.valid.extend(
            rel.rows
                .iter()
                .enumerate()
                .filter(|&(_, &r)| obs.w[r as usize] > 0.0)
                .map(|(j, _)| j),
        );
        let n = self.valid.len();
        if n == 0 {
            return One {
                est: 0.0,
                var: h2,
                n_relatives: 0,
            };
        }
        rel.dense_kinship(&self.valid, &mut self.phi);
        self.keys.clear();
        for (x, &j) in self.valid.iter().enumerate() {
            let row = rel.rows[j];
            let to_proband = f64::from(rel.kin[j]);
            self.keys.push(order::Key {
                w: obs.w[row as usize],
                to_proband,
                row_sum: to_proband + self.phi[x * n..(x + 1) * n].iter().sum::<f64>(),
                row,
                trait_index: 0,
                index: x,
            });
        }
        order::sort(&mut self.keys);

        let size = n + 1;
        self.cov.clear();
        self.cov.resize(size * size, 0.0);
        self.mu.resize(size, 0.0);
        self.obs.clear();
        self.cov[0] = h2;
        for (i, key) in self.keys.iter().enumerate() {
            self.cov[1 + i] = 2.0 * key.to_proband * h2;
            self.cov[(1 + i) * size + 1 + i] = 1.0;
            for (k, other) in self.keys.iter().enumerate().skip(i + 1) {
                self.cov[(1 + i) * size + 1 + k] = 2.0 * self.phi[key.index * n + other.index] * h2;
            }
            self.obs.push(obs.obs(key.row as usize));
        }
        pa::condition(&mut self.cov, &mut self.mu, 1, &self.obs, &mut self.col);
        One {
            est: self.mu[0],
            var: self.cov[0].max(0.0),
            n_relatives: n as u32,
        }
    }
}
