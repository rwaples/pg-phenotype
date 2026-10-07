#' Mate Correlation: assortative mating over a pedigree's Mating Pairs
#'
#' A Mating Pair is a distinct mother and father, both pedigree rows, of at
#' least one child; a Mate Network is a connected set of pairs through shared
#' mates.  Each cell (mother trait x father trait) gets crude estimators on
#' its pairs and, with `stratum`, the primary estimator standardised within
#' each sex x stratum.  Inference is a Mate Network cluster-robust sandwich
#' SE, an optional one-step Mate Network bootstrap, and a father-permutation
#' test of each primary estimate.  The same inputs give the same numbers as
#' the Python package's `pg_phenotype.assortative.mate_correlation`.
#'
#' @param pedigree A data frame or a named list with columns `id`, `mother`
#'   and `father` (`NA` or -1 when missing), optionally `twin` and `sex`.
#' @param traits One [trait()] or a list of one or two: continuous, binary or
#'   ordinal.  Binary and ordinal values are level codes 0, 1, ...; every
#'   level (a factor's levels included) must be taken by some row.
#' @param stratum `NULL`, or one whole-number label per row (a Depth, a
#'   birth-year bin), `NA` where unknown.  Pairs with a mate of unknown
#'   stratum are dropped; each cell then drops its pairs in any sex x stratum
#'   spanning fewer than `min_stratum_networks` Mate Networks or constant
#'   there, and adds the stratified primary estimator.
#' @param permutations Father permutations per primary estimate; 0 turns them off.
#' @param bootstrap Mate Network bootstrap draws; 0 gives each estimate a
#'   Wald CI from its sandwich SE.
#' @param seed Keys every permutation and bootstrap draw (a whole number).
#' @param min_stratum_networks The thin-stratum rule, used only with `stratum`.
#' @return An object of class `pgphenotype_mate_correlation`: a list with
#'   `sample`, `cells` (mother trait 1 x father trait 1, then 1 x 2, 2 x 1,
#'   2 x 2), `within_person` (with two traits), `settings`, `method` (the
#'   method description) and `metadata`.  Each cell holds `crude` (a list of
#'   estimator records, the primary first) and `stratified`; a record has
#'   `value` or a `reason`, and `se`, `ci` and the permutation `p` each come
#'   with an `*_unavailable_reason`.  Trait indices (`mother_trait`,
#'   `father_trait`) count from 0, as in Python.
#' @examples
#' ped <- data.frame(
#'   id = 1:12,
#'   mother = c(rep(NA, 8), 1, 2, 3, 4),
#'   father = c(rep(NA, 8), 5, 6, 7, 8)
#' )
#' x <- trait(c(0.1, 1.2, -0.3, 0.8, 0.3, 1.1, -0.6, 0.2, rep(NA, 4)))
#' r <- assortative_mate_correlation(ped, x, permutations = 99)
#' r
#' @export
assortative_mate_correlation <- function(pedigree, traits, stratum = NULL, permutations = 999, bootstrap = 0,
                                         seed = 0, min_stratum_networks = 10) {
  call <- sys.call()
  if (!is.list(pedigree)) {
    .pgp_usage("`pedigree` must be a data frame or a named list of columns", call)
  }
  if (inherits(traits, "pgphenotype_trait")) traits <- list(traits)
  if (!is.list(traits) || !all(vapply(traits, inherits, logical(1), "pgphenotype_trait"))) {
    .pgp_usage("`traits` must be a trait() or a list of them", call)
  }
  column <- function(name) if (name %in% names(pedigree)) pedigree[[name]] else NULL
  n_levels <- vapply(traits, function(t) if (is.null(t$levels)) NA_real_ else as.double(length(t$levels)), 0)
  raw <- .pgp_call(.native_assortative_mate_correlation(
    column("id"), column("mother"), column("father"), column("twin"), column("sex"),
    lapply(traits, `[[`, "values"), vapply(traits, `[[`, "", "kind"), n_levels, stratum,
    .pgp_number(permutations, "permutations", call), .pgp_number(bootstrap, "bootstrap", call),
    .pgp_number(seed, "seed", call), .pgp_number(min_stratum_networks, "min_stratum_networks", call)
  ), call)
  raw$metadata <- list(pg_phenotype_version = pgphenotype_version(), pedigree_graph_core_rev = pg_core_rev())
  structure(raw, class = "pgphenotype_mate_correlation")
}

#' @export
print.pgphenotype_mate_correlation <- function(x, ...) {
  cat(sprintf("<pgphenotype_mate_correlation> %d Mating Pairs, %d Mate Networks\n",
              x$sample$n_total, x$sample$n_mate_networks))
  for (cell in x$cells) {
    for (r in c(list(cell$crude[[1L]]), if (!is.null(cell$stratified)) list(cell$stratified))) {
      form <- if (identical(r, cell$crude[[1L]])) "crude" else "stratified"
      value <- if (is.null(r$reason)) sprintf("%.4f", r$value) else paste("undefined:", r$reason)
      ci <- if (!is.null(r$ci)) sprintf(" [%.4f, %.4f]", r$ci[1L], r$ci[2L]) else ""
      p <- if (!is.null(r$permutation$p)) sprintf(" p = %.4g", r$permutation$p) else ""
      cat(sprintf("  mother %d x father %d, n %d, %s %s: %s%s%s\n", cell$mother_trait, cell$father_trait,
                  cell$n, form, r$estimator, value, ci, p))
    }
  }
  invisible(x)
}
