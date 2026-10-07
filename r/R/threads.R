#' Package thread budget
#'
#' `configure_threads(n)` sets the number of threads every native call uses,
#' [pafgrs_prepare()] and the score functions alike.  It must run before the budget is first used; the
#' first use commits `n`, else the `PG_PHENOTYPE_THREADS` environment variable,
#' else 1, and afterwards only the committed value may be configured again
#' (`pgphenotype_thread_conflict_error`).  Scores are identical for every
#' budget.
#'
#' @param n A whole number >= 1.
#' @return `configure_threads()` returns `NULL` invisibly;
#'   `thread_budget()` returns the committed budget.
#' @examples
#' thread_budget()
#' @export
configure_threads <- function(n) {
  if (!is.numeric(n) || length(n) != 1L || is.na(n)) n <- NaN
  invisible(.pgp_call(.native_configure_threads(as.double(n))))
}

#' @rdname configure_threads
#' @export
thread_budget <- function() .pgp_call(.native_thread_budget())
