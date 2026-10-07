# Changelog

## Unreleased (0.1.0)

First release.  The shared `Trait` input and PA-FGRS (`pafgrs.prepare`,
`pafgrs.score_univariate`, `pafgrs.score_bivariate`) for Python and R, on
pg-phenotype-core linked to pedigree-graph-core v0.12.2 (`62c82fa`).

Assortative mating (`pg_phenotype.assortative.mate_correlation`, R
`assortative_mate_correlation()`): the Mate Correlation of one or two traits
with sandwich SEs, the one-step Mate Network bootstrap and the sequential
father-permutation test, ported from pedsum #13 (`142adf3`).  The Rust
`Trait` gains `n_levels`, the declared level count.
