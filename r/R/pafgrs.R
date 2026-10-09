#' Prepare a pedigree for PA-FGRS scoring
#'
#' Builds each proband's relatives up to `ndegree`.  Call it once per
#' pedigree and degree, then score each trait and parameter variant against
#' the result; scoring many traits needs only the result.  A relative of a
#' proband is a person whose closest relationship category is at most
#' `ndegree` and whose exact kinship to the proband is at least
#' `0.5^(ndegree + 1) - 1e-6`.
#'
#' @param pedigree A [pedigree()], or a data frame or a named list with
#'   columns `id`, `mother` and `father`, and optionally `twin` (the MZ
#'   co-twin's id) and `sex` (0 female, 1 male, -1 unknown).  Columns are
#'   read first and checked by pedigree-graph's rules after `ndegree`, for
#'   this call alone.  Columns may be integer, whole-number
#'   double, or `bit64::integer64`; `NA` or -1 marks a missing parent, and
#'   `id` may not be `NA`.  Other columns are ignored.  A [pedigree()] helps
#'   when preparing again (another `ndegree` or set of probands) or running
#'   other methods on the same pedigree.
#' @param ndegree The deepest relationship degree that counts, 1 to 5.
#' @param probands Ids to score, or `NULL` to score every row.
#' @return An opaque handle of class `pgphenotype_pafgrs_prep`, held in
#'   memory only: it does not survive [saveRDS()] or a new R session.  It
#'   does not hold `pedigree`.
#' @examples
#' ped <- data.frame(
#'   id = 1:6,
#'   mother = c(NA, NA, 1, 1, NA, 3),
#'   father = c(NA, NA, 2, 2, NA, 5)
#' )
#' prep <- pafgrs_prepare(ped, ndegree = 2)
#' prep
#' @export
pafgrs_prepare <- function(pedigree, ndegree = 2L, probands = NULL) {
  args <- .pgp_pedigree_args(pedigree, sys.call())
  ndegree <- .pgp_number(ndegree, "ndegree")
  .pgp_call(do.call(.native_pafgrs_prepare, c(args, list(ndegree, probands))))
}

#' @export
print.pgphenotype_pafgrs_prep <- function(x, ...) {
  info <- .pgp_call(.native_pafgrs_prep_info(x))
  cat(sprintf(
    "<pgphenotype_pafgrs_prep> %d rows, %d probands, ndegree %d\n",
    info$n_rows, info$n_probands, info$ndegree
  ))
  invisible(x)
}

#' PA-FGRS CIP table
#'
#' Cumulative incidence proportion by age.  CIP at an age is linear
#' interpolation: 0 below the first age, the last value `K` (the lifetime
#' prevalence) at and above the last age.  A control observed at age `a`
#' carries weight `CIP(a) / K`; the liability threshold is `qnorm(1 - K)`.
#'
#' @param ages Ages, finite and strictly increasing.
#' @param cip CIP at each age, non-decreasing in `[0, 1)` with a positive
#'   last value.
#' @return An object of class `pgphenotype_pafgrs_cip` holding `ages`, `cip`,
#'   `prevalence` and `threshold`.  An invalid table is a
#'   `pgphenotype_parameter_error` with code `invalid_cip`.
#' @examples
#' pafgrs_cip(c(0, 40, 80), c(0, 0.05, 0.1))
#' @export
pafgrs_cip <- function(ages, cip) {
  ages <- .pgp_doubles(ages, "ages")
  values <- .pgp_doubles(cip, "cip")
  checked <- .pgp_call(.native_pafgrs_check_cip(ages, values))
  structure(
    list(ages = ages, cip = values, prevalence = checked$prevalence, threshold = checked$threshold),
    class = "pgphenotype_pafgrs_cip"
  )
}

#' @export
print.pgphenotype_pafgrs_cip <- function(x, ...) {
  cat(sprintf(
    "<pgphenotype_pafgrs_cip> %d ages from %g to %g, prevalence %g, threshold %g\n",
    length(x$ages), x$ages[1L], x$ages[length(x$ages)], x$prevalence, x$threshold
  ))
  invisible(x)
}

.pgp_check_class <- function(x, class, name, from, call) {
  if (!inherits(x, class)) .pgp_usage(sprintf("`%s` must come from %s", name, from), call)
}

# Two of something, as a plain list (not one of the objects itself).
.pgp_check_pair <- function(x, class, name, from, call) {
  if (!is.list(x) || inherits(x, class) || length(x) != 2L) {
    .pgp_usage(sprintf("`%s` must be a list of two %ss", name, from), call)
  }
}

# A data frame built without method dispatch, so an integer64 id column
# stays one whether or not bit64 is loaded.
.pgp_frame <- function(columns, metadata) {
  structure(
    columns,
    class = "data.frame", row.names = c(NA_integer_, -length(columns[[1L]])), metadata = metadata
  )
}

.pgp_metadata <- function(prep, raw) {
  list(
    n_probands = length(raw$id),
    ndegree = .pgp_call(.native_pafgrs_prep_info(prep))$ndegree,
    pg_phenotype_version = pgphenotype_version(),
    pedigree_graph_core_rev = pg_core_rev()
  )
}

#' Univariate PA-FGRS scores
#'
#' Scores every proband of `prep` on one binary trait: the posterior mean and
#' variance of its genetic liability given its relatives' observations.
#'
#' @param prep From [pafgrs_prepare()].
#' @param trait A binary [trait()]: 1 affected, 0 unaffected, `NA` unknown
#'   (weight 0).
#' @param age Per pedigree row, the age at onset for a case and at last
#'   observation for a control, `NA` when unknown.  A control without an age
#'   is unobserved and counted in `controls_without_age`; a case needs none.
#' @param cip The trait's table, from [pafgrs_cip()].
#' @param h2 Liability-scale heritability, in `(0, 1]`.
#' @return A data frame with one row per proband in pedigree input-row order
#'   and columns `id`, `est`, `var` and `n_relatives` (relatives with
#'   weight > 0).  A proband with no informative relative gets `est = 0`,
#'   `var = h2`.  `attr(, "metadata")` is a list: `h2`, `prevalence`,
#'   `threshold`, `controls_without_age`, `n_probands`, `ndegree`,
#'   `pg_phenotype_version` and `pedigree_graph_core_rev`.  A trait that is
#'   not binary is a `pgphenotype_validation_error` with code
#'   `trait_kind_mismatch`; a value outside \{0, 1\} gives
#'   `invalid_trait_value`.
#' @examples
#' ped <- data.frame(id = 1:4, mother = c(NA, NA, 1, 1), father = c(NA, NA, 2, 2))
#' pafgrs_score_univariate(
#'   pafgrs_prepare(ped), trait(c(1, 0, 0, NA)), age = c(50, 60, 30, NA),
#'   cip = pafgrs_cip(c(0, 80), c(0, 0.1)), h2 = 0.5
#' )
#' @export
pafgrs_score_univariate <- function(prep, trait, age, cip, h2) {
  call <- sys.call()
  .pgp_check_class(trait, "pgphenotype_trait", "trait", "trait()", call)
  .pgp_check_class(cip, "pgphenotype_pafgrs_cip", "cip", "pafgrs_cip()", call)
  age <- .pgp_doubles(age, "age", call)
  h2 <- .pgp_number(h2, "h2", call)
  raw <- .pgp_call(.native_pafgrs_score_univariate(
    prep, trait$values, trait$kind, age, cip$ages, cip$cip, h2
  ), call)
  metadata <- c(
    list(h2 = h2, prevalence = raw$prevalence, threshold = raw$threshold,
         controls_without_age = raw$controls_without_age),
    .pgp_metadata(prep, raw)
  )
  .pgp_frame(raw[c("id", "est", "var", "n_relatives")], metadata)
}

#' Bivariate PA-FGRS scores
#'
#' Scores every proband of `prep` on two genetically correlated binary
#' traits jointly.
#'
#' @param prep From [pafgrs_prepare()].
#' @param traits A list of two binary [trait()]s.
#' @param ages A list of each trait's ages, as [pafgrs_score_univariate()]
#'   takes them.
#' @param cips A list of each trait's [pafgrs_cip()] table.
#' @param h2 The two liability-scale heritabilities, each in `(0, 1]`.
#' @param rg Genetic correlation, in `[-1, 1]`.
#' @param rho_within A person's cross-trait liability correlation, in
#'   `[-1, 1]`; `NULL` means `rg * sqrt(h2[1] * h2[2])`.
#' @return A data frame with one row per proband in pedigree input-row order
#'   and columns `id`, `est1`, `est2`, `var1`, `var2`, `cov12`,
#'   `n_relatives` (people with weight > 0 on either trait), `n_obs1` and
#'   `n_obs2`.  With no informative relative: `est = 0`, `var = h2`,
#'   `cov12 = rg * sqrt(h2[1] * h2[2])`.  `attr(, "metadata")` is a list:
#'   `h2`, `rg`, `rho_within` (as used), and per trait `prevalence`,
#'   `threshold` and `controls_without_age`, then `n_probands`, `ndegree`,
#'   `pg_phenotype_version` and `pedigree_graph_core_rev`.
#' @examples
#' ped <- data.frame(id = 1:4, mother = c(NA, NA, 1, 1), father = c(NA, NA, 2, 2))
#' table <- pafgrs_cip(c(0, 80), c(0, 0.1))
#' pafgrs_score_bivariate(
#'   pafgrs_prepare(ped),
#'   traits = list(trait(c(1, 0, 0, NA)), trait(c(0, 1, NA, 0))),
#'   ages = list(c(50, 60, 30, NA), c(70, 40, NA, 20)),
#'   cips = list(table, table),
#'   h2 = c(0.5, 0.4), rg = 0.3
#' )
#' @export
pafgrs_score_bivariate <- function(prep, traits, ages, cips, h2, rg, rho_within = NULL) {
  call <- sys.call()
  .pgp_check_pair(traits, "pgphenotype_trait", "traits", "trait()", call)
  .pgp_check_pair(cips, "pgphenotype_pafgrs_cip", "cips", "pafgrs_cip()", call)
  if (!is.list(ages) || length(ages) != 2L) .pgp_usage("`ages` must be a list of two vectors", call)
  for (k in 1:2) {
    .pgp_check_class(traits[[k]], "pgphenotype_trait", sprintf("traits[[%d]]", k), "trait()", call)
    .pgp_check_class(cips[[k]], "pgphenotype_pafgrs_cip", sprintf("cips[[%d]]", k), "pafgrs_cip()", call)
  }
  age1 <- .pgp_doubles(ages[[1L]], "ages[[1]]", call)
  age2 <- .pgp_doubles(ages[[2L]], "ages[[2]]", call)
  if (!is.numeric(h2) || length(h2) != 2L) .pgp_usage("`h2` must be two numbers", call)
  h2 <- as.double(h2)
  rg <- .pgp_number(rg, "rg", call)
  if (!is.null(rho_within)) rho_within <- .pgp_number(rho_within, "rho_within", call)
  t1 <- traits[[1L]]
  t2 <- traits[[2L]]
  raw <- .pgp_call(.native_pafgrs_score_bivariate(
    prep, t1$values, t1$kind, age1, t2$values, t2$kind, age2,
    cips[[1L]]$ages, cips[[1L]]$cip, cips[[2L]]$ages, cips[[2L]]$cip, h2, rg, rho_within
  ), call)
  metadata <- c(
    list(h2 = h2, rg = rg, rho_within = raw$rho_within, prevalence = raw$prevalence,
         threshold = raw$threshold, controls_without_age = raw$controls_without_age),
    .pgp_metadata(prep, raw)
  )
  columns <- c("id", "est1", "est2", "var1", "var2", "cov12", "n_relatives", "n_obs1", "n_obs2")
  .pgp_frame(raw[columns], metadata)
}
