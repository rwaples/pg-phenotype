//! What [`mate_correlation`](super::mate_correlation) returns.
//!
//! A defined estimate and the reason an estimate, an SE, a CI or a p-value
//! is missing are alternatives, so each is a `Result`: never a value and a
//! reason together, never neither.

/// A cell estimator, as pedsum names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimator {
    Pearson,
    Spearman,
    OddsRatio,
    Phi,
    PointBiserial,
    Tetrachoric,
    Polychoric,
    Biserial,
    Polyserial,
}

/// The scale of an estimator's sandwich SE and Wald interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CiScale {
    /// Fisher z, `atanh` of a correlation.
    FisherZ,
    /// `log` of the odds ratio.
    Log,
}

impl Estimator {
    /// The name hosts use.
    pub fn name(self) -> &'static str {
        match self {
            Estimator::Pearson => "pearson",
            Estimator::Spearman => "spearman",
            Estimator::OddsRatio => "odds_ratio",
            Estimator::Phi => "phi",
            Estimator::PointBiserial => "point_biserial",
            Estimator::Tetrachoric => "tetrachoric",
            Estimator::Polychoric => "polychoric",
            Estimator::Biserial => "biserial",
            Estimator::Polyserial => "polyserial",
        }
    }

    /// The key pedsum writes the value under: `r`, `rho` or `value`.
    pub fn value_key(self) -> &'static str {
        match self {
            Estimator::Pearson
            | Estimator::Spearman
            | Estimator::Phi
            | Estimator::PointBiserial => "r",
            Estimator::Tetrachoric
            | Estimator::Polychoric
            | Estimator::Biserial
            | Estimator::Polyserial => "rho",
            Estimator::OddsRatio => "value",
        }
    }

    /// The scale of the SE and the Wald CI.
    pub fn ci_scale(self) -> CiScale {
        match self {
            Estimator::OddsRatio => CiScale::Log,
            _ => CiScale::FisherZ,
        }
    }

    /// A latent (liability) correlation, fitted by maximum likelihood.
    pub fn is_latent(self) -> bool {
        matches!(
            self,
            Estimator::Tetrachoric
                | Estimator::Polychoric
                | Estimator::Biserial
                | Estimator::Polyserial
        )
    }
}

/// Why an estimate, SE, CI, p-value or draw is missing.  One table for every
/// place a reason appears.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reason {
    BootstrapNotRequested,
    Boundary,
    ConstantMargin,
    DegenerateStratum,
    EmptyCategory,
    InfiniteOddsRatio,
    NoCompletePairs,
    NoInformativePermutations,
    NoValidPermutations,
    NotRequested,
    SandwichUndefined,
    SingleMateNetwork,
    TooManyFailedDraws,
}

impl Reason {
    /// Every reason, in name order.
    pub const ALL: [Reason; 13] = [
        Reason::BootstrapNotRequested,
        Reason::Boundary,
        Reason::ConstantMargin,
        Reason::DegenerateStratum,
        Reason::EmptyCategory,
        Reason::InfiniteOddsRatio,
        Reason::NoCompletePairs,
        Reason::NoInformativePermutations,
        Reason::NoValidPermutations,
        Reason::NotRequested,
        Reason::SandwichUndefined,
        Reason::SingleMateNetwork,
        Reason::TooManyFailedDraws,
    ];

    /// The name hosts use.
    pub fn name(self) -> &'static str {
        match self {
            Reason::BootstrapNotRequested => "bootstrap_not_requested",
            Reason::Boundary => "boundary",
            Reason::ConstantMargin => "constant_margin",
            Reason::DegenerateStratum => "degenerate_stratum",
            Reason::EmptyCategory => "empty_category",
            Reason::InfiniteOddsRatio => "infinite_odds_ratio",
            Reason::NoCompletePairs => "no_complete_pairs",
            Reason::NoInformativePermutations => "no_informative_permutations",
            Reason::NoValidPermutations => "no_valid_permutations",
            Reason::NotRequested => "not_requested",
            Reason::SandwichUndefined => "sandwich_undefined",
            Reason::SingleMateNetwork => "single_mate_network",
            Reason::TooManyFailedDraws => "too_many_failed_draws",
        }
    }
}

/// A defined point estimate.  `boundary` is set for a latent correlation
/// only: whether ρ̂ sits at, or on a likelihood plateau reaching, a bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub value: f64,
    pub boundary: Option<bool>,
}

/// How a CI was built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CiMethod {
    /// Percentile interval of the Mate Network bootstrap.
    Bootstrap,
    /// Wald interval from the sandwich SE, on the estimator's [`CiScale`].
    Sandwich,
}

impl CiMethod {
    /// The name hosts use.
    pub fn name(self) -> &'static str {
        match self {
            CiMethod::Bootstrap => "bootstrap",
            CiMethod::Sandwich => "sandwich",
        }
    }
}

/// An interval at the result's `ci_level` and how it was built.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ci {
    pub bounds: [f64; 2],
    pub method: CiMethod,
}

/// Draws requested, how many gave a value, and why the others did not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Draws {
    pub requested: u64,
    pub valid: u64,
    pub failed: u64,
    /// Failure counts by reason, in [`Reason`] name order, zero counts left out.
    pub failure_reasons: Vec<(Reason, u64)>,
}

/// What a permutation test's statistic is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermutationStatistic {
    /// Pearson r (continuous x continuous).
    Pearson,
    /// The latent-correlation score at ρ = 0, margins refit per draw.
    ScoreAtZero,
}

impl PermutationStatistic {
    /// The name hosts use.
    pub fn name(self) -> &'static str {
        match self {
            PermutationStatistic::Pearson => "pearson",
            PermutationStatistic::ScoreAtZero => "score_at_zero",
        }
    }
}

/// The permutation test of a primary estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct Permutation {
    pub statistic: PermutationStatistic,
    /// Two-sided sequential Monte Carlo p-value (Besag & Clifford, closed).
    pub p: Result<f64, Reason>,
    /// Counts over the draws read (`draws_used`), `requested` as asked.
    pub draws: Draws,
    pub seed: i64,
    /// The cell's distinct fathers alone in their permutation block.
    pub n_fixed_fathers: u64,
    pub stopped_early: bool,
    pub draws_used: u64,
    pub sequential_h: u64,
}

/// A defined estimate with its inference.
#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    pub point: Point,
    /// The Mate Network cluster-robust sandwich SE, on the scale of the
    /// estimate: a correlation's raw r (its Wald interval converts it to
    /// Fisher z), the odds ratio's log.
    pub se: Result<f64, Reason>,
    pub ci: Result<Ci, Reason>,
    /// Present when a bootstrap was requested.
    pub bootstrap: Option<Draws>,
    /// Present for a primary estimate.
    pub permutation: Option<Permutation>,
}

/// One estimator of one cell.
#[derive(Clone, Debug, PartialEq)]
pub struct EstimatorResult {
    pub estimator: Estimator,
    /// The estimator that gets the permutation test: the first crude one, and
    /// the stratified one.
    pub primary: bool,
    pub outcome: Result<Estimate, Reason>,
}

/// The stratified form of a cell's primary estimator.
#[derive(Clone, Debug, PartialEq)]
pub struct Stratified {
    pub result: EstimatorResult,
    /// Distinct strata among the cell's mothers and fathers.
    pub n_strata_mothers: u64,
    pub n_strata_fathers: u64,
}

/// Pairs a cell leaves out, by why.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dropped {
    pub mother_missing: u64,
    pub father_missing: u64,
    pub both_missing: u64,
    pub small_stratum: u64,
    pub degenerate_stratum: u64,
}

/// One Mate Correlation cell: the mother's trait against the father's.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    /// Index into the traits passed.
    pub mother_trait: usize,
    pub father_trait: usize,
    /// Analysed Mating Pairs.
    pub n: u64,
    pub n_dropped: Dropped,
    pub n_mate_networks: u64,
    pub largest_mate_network_share: Option<f64>,
    /// Pair counts of a binary x binary cell, rows by mother level, columns
    /// by father level.
    pub table: Option<[[u64; 2]; 2]>,
    /// Crude estimators, the primary first.
    pub crude: Vec<EstimatorResult>,
    /// Present with strata.
    pub stratified: Option<Stratified>,
}

/// The Mating Pairs before any cell's filtering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub n_total: u64,
    pub n_dropped_unknown_stratum: u64,
    pub n_mate_networks: u64,
    pub largest_mate_network_share: Option<f64>,
    pub n_mothers_multiple_mates: u64,
    pub n_fathers_multiple_mates: u64,
}

/// The Within-Person Cross-Trait Correlation of one sex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WithinPerson {
    pub estimator: Estimator,
    /// Distinct people with both traits and an analysed Mating Pair.
    pub n: u64,
    pub outcome: Result<Point, Reason>,
}

/// Both sexes' Within-Person Cross-Trait Correlation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WithinPersonPair {
    pub mothers: WithinPerson,
    pub fathers: WithinPerson,
}

/// The settings a result was computed with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SettingsEcho {
    pub permutations: u64,
    pub bootstrap: u64,
    pub seed: i64,
    pub threads: usize,
    pub ci_level: f64,
    /// `Some` with strata.
    pub min_stratum_networks: Option<u64>,
    pub spearman: bool,
}

/// The method description: what pedsum writes as its `inference` block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Method {
    pub se_method: &'static str,
    pub ci_scale: &'static str,
    pub bootstrap_unit: &'static str,
    pub bootstrap_assumption: &'static str,
    pub bootstrap_method: &'static str,
    pub permutation_null: &'static str,
    pub permutation_blocks: &'static str,
    pub permutation_statistic: &'static str,
    pub permutation_stopping: &'static str,
    pub permutation_stop_h: u64,
}

/// The method this build computes, pedsum `INFERENCE` (`assortative_mating.py:39-57`) verbatim.
pub const METHOD: Method = Method {
    se_method: "cluster-robust sandwich over Mate Networks of the stacked two-step estimating equations \
        (thresholds, stratum means and variances, then rho), G/(G-1) small-sample factor",
    ci_scale: "Fisher z for correlations, log for the odds ratio",
    bootstrap_unit: "mate_network",
    bootstrap_assumption: "Mate Networks are independent. Dependence from remating is kept; \
        dependence between networks through ancestry or siblings is not modelled.",
    bootstrap_method: "one_step: each draw refits the first step (thresholds, stratum moments) and every \
        closed-form estimator exactly on the resampled Mate Networks, and takes one Newton step for rho from \
        the observed estimate with the draw-weighted score and the full-sample Hessian",
    permutation_null: "father trait vectors exchangeable within blocks",
    permutation_blocks: "father_stratum x father_missingness_pattern",
    permutation_statistic: "score_at_zero: the score of the latent-correlation log-likelihood at rho = 0 with \
        every margin refit on the permuted pairs (polychoric, tetrachoric, polyserial, biserial); \
        pearson: Pearson r (continuous x continuous)",
    permutation_stopping: "besag_clifford_closed",
    permutation_stop_h: 20,
};

/// The Mate Correlation of one or two traits over a pedigree's Mating Pairs.
#[derive(Clone, Debug, PartialEq)]
pub struct MateCorrelation {
    pub sample: Sample,
    /// Mother trait 0 x father trait 0, then (0, 1), (1, 0), (1, 1).
    pub cells: Vec<Cell>,
    /// Present with two traits.
    pub within_person: Option<WithinPersonPair>,
    pub settings: SettingsEcho,
    pub method: Method,
}

#[cfg(test)]
mod tests {
    use super::Reason;

    #[test]
    fn reasons_are_declared_in_name_order() {
        // Failure histograms sort by `Reason`'s derived order, which pedsum sorts by name.
        let names: Vec<&str> = Reason::ALL.iter().map(|r| r.name()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
        assert!(Reason::ALL.windows(2).all(|w| w[0] < w[1]));
    }
}
