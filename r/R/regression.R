#' Parent-offspring regression
#'
#' The ordinary least-squares lines of each child's trait on its midparent's
#' (the mean of both parents), its mother's and its father's, pooled and
#' within strata of the child.  The slope's SE is the cluster-robust
#' sandwich over Mate Networks with a G/(G-1) factor (a parent in no Mating
#' Pair is a network of its own), with a two-sided Wald p-value and interval,
#' beside the plain OLS SE.  Slopes are raw: under the usual assumptions (no
#' environment shared by parent and child, random mating) the midparent
#' slope estimates h2 and a single-parent slope h2/2, on the observed scale
#' for a 0/1 trait.  The same inputs give the same numbers as the Python
#' package's `pg_phenotype.regression.parent_offspring`.
#'
#' @param pedigree A [pedigree()] or a data frame or a named list with
#'   columns `id`, `mother` and `father` (`NA` or -1 when missing),
#'   optionally `twin` and `sex`, checked for this call alone.
#' @param trait A continuous, binary or ordinal [trait()] on the pedigree's
#'   rows; codes are regressed as numbers.  An all-missing or constant trait
#'   gives fits with a reason.
#' @param stratum `NULL`, or one whole-number label per row (a generation, a
#'   sex x generation code), `NA` where unknown, read at the child.  Each
#'   known label gets its own regressions; children of unknown stratum are
#'   pooled only.
#' @return An object of class `pgphenotype_parent_offspring`: a list with
#'   `pooled`, `strata` (`NULL` without `stratum`, else one record per known
#'   label, ascending, with its `stratum`), `n_unknown_stratum`, `settings`,
#'   `method` and `metadata`.  Each subset holds `midparent`, `mother` and
#'   `father` fits: `n`, `n_mate_networks`, `dropped` (rows left out, by
#'   reason), `slope` and `intercept` or a `reason`, and `r`, `se`, `se_ols`,
#'   `p` and `ci`, each with an `*_unavailable_reason`.
#' @examples
#' ped <- data.frame(
#'   id = 1:12,
#'   mother = c(rep(NA, 6), 1, 1, 2, 2, 3, 3),
#'   father = c(rep(NA, 6), 4, 4, 5, 5, 6, 6)
#' )
#' x <- trait(c(0.1, 1.2, -0.3, 0.8, 0.3, 1.1, 0.4, 0.2, 0.9, 1.0, 0.6, -0.1))
#' regression_parent_offspring(ped, x)
#' @export
regression_parent_offspring <- function(pedigree, trait, stratum = NULL) {
  call <- sys.call()
  pedigree <- .pgp_pedigree_arg(pedigree, call)
  .pgp_check_class(trait, "pgphenotype_trait", "trait", "trait()", call)
  raw <- .pgp_call(.native_regression_parent_offspring(pedigree, trait, stratum), call)
  structure(raw, class = "pgphenotype_parent_offspring")
}

#' @export
print.pgphenotype_parent_offspring <- function(x, ...) {
  cat("<pgphenotype_parent_offspring>\n")
  line <- function(label, subset) {
    for (parent in c("midparent", "mother", "father")) {
      f <- subset[[parent]]
      value <- if (is.null(f$reason)) sprintf("slope %.4f", f$slope) else paste("undefined:", f$reason)
      se <- if (!is.null(f$se)) sprintf(" (SE %.4f)", f$se) else ""
      cat(sprintf("  %s %s, n %s: %s%s\n", label, parent, format(f$n, scientific = FALSE), value, se))
    }
  }
  line("pooled", x$pooled)
  for (s in x$strata) line(paste("stratum", s$stratum), s)
  invisible(x)
}
