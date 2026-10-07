# The native routines, registered by useDynLib(.registration = TRUE).

.native_pafgrs_prepare <- function(id, mother, father, twin, sex, ndegree, probands) {
  .Call(wrap__pafgrs_prepare, id, mother, father, twin, sex, ndegree, probands)
}
.native_pafgrs_prep_info <- function(prep) .Call(wrap__pafgrs_prep_info, prep)
.native_pafgrs_check_cip <- function(ages, cip) .Call(wrap__pafgrs_check_cip, ages, cip)
.native_pafgrs_score_univariate <- function(prep, values, kind, age, cip_ages, cip_values, h2) {
  .Call(wrap__pafgrs_score_univariate, prep, values, kind, age, cip_ages, cip_values, h2)
}
.native_pafgrs_score_bivariate <- function(prep, values1, kind1, age1, values2, kind2, age2,
                                           cip1_ages, cip1_values, cip2_ages, cip2_values,
                                           h2, rg, rho_within) {
  .Call(wrap__pafgrs_score_bivariate, prep, values1, kind1, age1, values2, kind2, age2,
        cip1_ages, cip1_values, cip2_ages, cip2_values, h2, rg, rho_within)
}
.native_configure_threads <- function(n) .Call(wrap__configure_threads, n)
.native_thread_budget <- function() .Call(wrap__thread_budget)
