# The native routines, registered by useDynLib(.registration = TRUE).

.native_pedigree_new <- function(id, mother, father, twin, sex) {
  .Call(wrap__pedigree_new, id, mother, father, twin, sex)
}
.native_pedigree_rows <- function(ptr, ids) {
  .Call(wrap__pedigree_rows, ptr, ids)
}
.native_pedigree_align <- function(ptr, ids, values) {
  .Call(wrap__pedigree_align, ptr, ids, values)
}
.native_pafgrs_prepare <- function(pedigree, id, mother, father, twin, sex, ndegree, probands) {
  .Call(wrap__pafgrs_prepare, pedigree, id, mother, father, twin, sex, ndegree, probands)
}
.native_pafgrs_prep_info <- function(prep) .Call(wrap__pafgrs_prep_info, prep)
.native_pafgrs_check_cip <- function(ages, cip, interpolation) {
  .Call(wrap__pafgrs_check_cip, ages, cip, interpolation)
}
.native_pafgrs_cip_aalen_johansen <- function(age, status, entry) {
  .Call(wrap__pafgrs_cip_aalen_johansen, age, status, entry)
}
.native_pafgrs_score_univariate <- function(prep, values, kind, age, cip_ages, cip_values, cip_interpolation,
                                            h2) {
  .Call(wrap__pafgrs_score_univariate, prep, values, kind, age, cip_ages, cip_values, cip_interpolation, h2)
}
.native_pafgrs_score_bivariate <- function(prep, values1, kind1, age1, values2, kind2, age2,
                                           cip1_ages, cip1_values, cip1_interpolation,
                                           cip2_ages, cip2_values, cip2_interpolation,
                                           h2, rg, rho_within) {
  .Call(wrap__pafgrs_score_bivariate, prep, values1, kind1, age1, values2, kind2, age2,
        cip1_ages, cip1_values, cip1_interpolation, cip2_ages, cip2_values, cip2_interpolation,
        h2, rg, rho_within)
}
.native_configure_threads <- function(n) .Call(wrap__configure_threads, n)
.native_thread_budget <- function() .Call(wrap__thread_budget)
.native_trait_kinds <- function() .Call(wrap__trait_kinds)
.native_assortative_mate_correlation <- function(pedigree, id, mother, father, twin, sex, values, kinds, n_levels,
                                                 stratum, permutations, bootstrap, seed, min_stratum_networks,
                                                 spearman, inference) {
  .Call(wrap__assortative_mate_correlation, pedigree, id, mother, father, twin, sex, values, kinds, n_levels,
        stratum, permutations, bootstrap, seed, min_stratum_networks, spearman, inference)
}
.native_correlation_tetrachoric_table <- function(table) .Call(wrap__correlation_tetrachoric_table, table)
.native_correlation_tetrachoric_pairs <- function(x, y) .Call(wrap__correlation_tetrachoric_pairs, x, y)
.native_regression_parent_offspring <- function(pedigree, id, mother, father, twin, sex, values, kind, n_levels,
                                                stratum) {
  .Call(wrap__regression_parent_offspring, pedigree, id, mother, father, twin, sex, values, kind, n_levels, stratum)
}
.native_relative_pairs_correlation <- function(pedigree, id, mother, father, twin, sex, values, kind, n_levels,
                                               names, members, memory_budget_bytes) {
  .Call(wrap__relative_pairs_correlation, pedigree, id, mother, father, twin, sex, values, kind, n_levels,
        names, members, memory_budget_bytes)
}
