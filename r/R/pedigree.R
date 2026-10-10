#' A validated pedigree that methods share
#'
#' Validates a pedigree once, for several methods or calls.
#' [assortative_mate_correlation()] and [pafgrs_prepare()] take the result
#' wherever they take pedigree columns.  It keeps the Mating Pairs the first
#' `assortative_mate_correlation()` call finds and the Mate Networks the first
#' with inference builds, so a later call skips validation and finding them.
#'
#' Row *i* of the pedigree is row *i* of `x`.  Traits, ages, strata and
#' probands are matched to it by position: move values from another table
#' onto its rows with [pedigree_align()] (or [pedigree_rows()]) first.  A
#' misaligned trait is not detected.
#'
#' @param x A data frame or a named list with columns `id`, `mother` and
#'   `father`, and optionally `twin` (the MZ co-twin's id) and `sex` (0
#'   female, 1 male, -1 unknown), as [pafgrs_prepare()] takes them.
#' @return An object of class `pgphenotype_pedigree`, held in memory only: it
#'   does not survive [saveRDS()] or a new R session.  `length(ped)` is its
#'   number of rows and `ped$ids` its ids in row order, in the storage type
#'   `x$id` had (integer, double or `bit64::integer64`), which results that
#'   return ids keep.
#' @examples
#' ped <- pedigree(data.frame(
#'   id = 1:6,
#'   mother = c(NA, NA, 1, 1, NA, 3),
#'   father = c(NA, NA, 2, 2, NA, 5)
#' ))
#' ped
#' prep <- pafgrs_prepare(ped, ndegree = 2)
#' @export
pedigree <- function(x) {
  call <- sys.call()
  if (!is.list(x) || inherits(x, "pgphenotype_pedigree")) {
    .pgp_usage("`x` must be a data frame or a named list of columns", call)
  }
  column <- function(name) if (name %in% names(x)) x[[name]] else NULL
  out <- .pgp_call(.native_pedigree_new(
    column("id"), column("mother"), column("father"), column("twin"), column("sex")
  ), call)
  structure(out, class = "pgphenotype_pedigree")
}

#' Rows of a pedigree by id
#'
#' `pedigree_rows()` gives the row of each id.  `pedigree_align()` moves
#' values given per id onto the pedigree's rows, as traits, ages and strata
#' take them.  The first call builds an id index the pedigree keeps (12 bytes
#' per row).
#'
#' @param pedigree A [pedigree()].
#' @param ids Pedigree ids, each at most once: integer, whole double or
#'   `bit64::integer64`, with no `NA`.
#' @param values A numeric or logical vector, one value per id.
#' @return `pedigree_rows()`: an integer vector of 1-based rows, one per id.
#'   `pedigree_align()`: a double vector with one value per pedigree row, `NA`
#'   where no id names the row.
#' @section Errors: `unknown_id` for an id that is not a pedigree row's,
#'   `duplicate_lookup_id` for an id listed twice, `invalid_integer_value`
#'   for an `NA` or non-integer id, and, from `pedigree_align()`,
#'   `pair_length_mismatch` when `values` is not `ids`'s length.
#' @examples
#' ped <- pedigree(data.frame(
#'   id = 1:6,
#'   mother = c(NA, NA, 1, 1, NA, 3),
#'   father = c(NA, NA, 2, 2, NA, 5)
#' ))
#' pheno <- data.frame(id = c(6, 3, 4), dx = c(1, 0, 1))
#' pedigree_rows(ped, pheno$id)
#' pedigree_align(ped, pheno$id, pheno$dx)
#' @export
pedigree_rows <- function(pedigree, ids) {
  call <- sys.call()
  .pgp_check_class(pedigree, "pgphenotype_pedigree", "pedigree", "pedigree()", call)
  .pgp_call(.native_pedigree_rows(pedigree$ptr, ids), call)
}

#' @rdname pedigree_rows
#' @export
pedigree_align <- function(pedigree, ids, values) {
  call <- sys.call()
  .pgp_check_class(pedigree, "pgphenotype_pedigree", "pedigree", "pedigree()", call)
  values <- .pgp_doubles(values, "values", call)
  .pgp_call(.native_pedigree_align(pedigree$ptr, ids, values), call)
}

#' @export
length.pgphenotype_pedigree <- function(x) length(x$ids)

#' @export
print.pgphenotype_pedigree <- function(x, ...) {
  cat(sprintf("<pgphenotype_pedigree> %d rows\n", length(x)))
  invisible(x)
}

# The native pedigree arguments: a pedigree()'s pointer and NULL columns, or
# a NULL pointer and the columns of a data frame or list.
.pgp_pedigree_args <- function(pedigree, call) {
  if (inherits(pedigree, "pgphenotype_pedigree")) {
    return(list(pedigree$ptr, NULL, NULL, NULL, NULL, NULL))
  }
  if (!is.list(pedigree)) {
    .pgp_usage("`pedigree` must be a pedigree(), a data frame or a named list of columns", call)
  }
  column <- function(name) if (name %in% names(pedigree)) pedigree[[name]] else NULL
  list(NULL, column("id"), column("mother"), column("father"), column("twin"), column("sex"))
}
