#' Tetrachoric correlation of two binary variables
#'
#' The tetrachoric correlation from paired binary values or from a 2 x 2
#' table.  The fit, boundary flag and SE are the Mate Correlation's
#' ([assortative_mate_correlation()]): two-step maximum likelihood,
#' thresholds from the margins and then rho by Newton with a bounded-Brent
#' fallback on (-0.9999, 0.9999), and the two-step sandwich SE with every
#' pair its own cluster.  The same inputs give the same numbers as the
#' Python package's `pg_phenotype.correlation.tetrachoric`.
#'
#' @param x,y Binary values (0/1 or logical), `NA` for missing, of one
#'   length.  Pairs with a missing value are dropped and counted in
#'   `n_dropped`.
#' @param table Instead of `x` and `y`: a 2 x 2 matrix of pair counts, rows
#'   by the `x` level (0, 1) and columns by the `y` level.
#' @return An object of class `pgphenotype_tetrachoric`: a list with
#'   `estimator`, `table` (rows as integer vectors), `n`, `n_dropped`,
#'   `value` or a `reason` (`no_complete_pairs`, `constant_margin`),
#'   `boundary`, and `se` and `ci` (a Wald interval on the Fisher-z scale at
#'   `ci_level`), each with an `*_unavailable_reason`.
#' @examples
#' correlation_tetrachoric(table = matrix(c(40, 15, 10, 35), 2))
#' correlation_tetrachoric(c(1, 0, 1, NA, 1, 0, 0), c(1, 0, 0, 1, NA, 0, 1))
#' @export
correlation_tetrachoric <- function(x = NULL, y = NULL, table = NULL) {
  call <- sys.call()
  raw <- if (!is.null(table)) {
    if (!is.null(x) || !is.null(y)) .pgp_usage("pass `x` and `y`, or `table`, not both", call)
    if (!is.numeric(table) || !identical(as.integer(dim(table)), c(2L, 2L))) {
      .pgp_usage("`table` must be a 2 x 2 numeric matrix", call)
    }
    .pgp_call(.native_correlation_tetrachoric_table(as.double(t(table))), call)
  } else if (!is.null(x) && !is.null(y)) {
    .pgp_call(.native_correlation_tetrachoric_pairs(.pgp_doubles(x, "x", call), .pgp_doubles(y, "y", call)), call)
  } else {
    .pgp_usage("pass `x` and `y`, or `table`", call)
  }
  structure(raw, class = "pgphenotype_tetrachoric")
}

#' @export
print.pgphenotype_tetrachoric <- function(x, ...) {
  value <- if (is.null(x$reason)) sprintf("%.4f", x$value) else paste("undefined:", x$reason)
  se <- if (!is.null(x$se)) sprintf(" (SE %.4f)", x$se) else ""
  ci <- if (!is.null(x$ci)) sprintf(" [%.4f, %.4f]", x$ci[1L], x$ci[2L]) else ""
  cat(sprintf("<pgphenotype_tetrachoric> n %s: %s%s%s\n", format(x$n, scientific = FALSE), value, se, ci))
  invisible(x)
}
