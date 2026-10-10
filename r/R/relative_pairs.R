#' Relative-Pair Correlation
#'
#' One trait's correlation between the members of each Relationship
#' Category's Relative Pairs (pedigree-graph's MZ, MO, FO, FS, MHS, ..., 1C),
#' or of a Category Group's (such as PO for MO and FO).  A continuous trait
#' gets Pearson's r; a binary trait gets the tetrachoric correlation
#' (primary) and phi.  A symmetric category's pairs enter in both orders, so
#' its Pearson is intraclass and its tetrachoric fits one threshold; a
#' directed category's pairs enter once, in its roles' order (offspring then
#' parent).  There is no SE yet: relative pairs share people, and the
#' cluster design waits on a calibration study.  The same inputs give the
#' same numbers as the Python package's
#' `pg_phenotype.relative_pairs.correlation`.
#'
#' @param pedigree A [pedigree()] or a data frame or a named list with
#'   columns `id`, `mother` and `father` (`NA` or -1 when missing),
#'   optionally `twin` and `sex`, checked for this call alone.
#' @param trait A continuous or binary [trait()] on the pedigree's rows.  An
#'   all-missing or constant trait gives cells with a reason.
#' @param categories `NULL` for every category up to degree 3, one cell
#'   each; else the cells in order, as a character vector of codes or a list
#'   whose unnamed elements are codes and whose named elements are Category
#'   Groups: `list("MZ", "FS", PO = c("MO", "FO"))`.  A group's codes are all
#'   symmetric, or all directed with the same roles (mother and father read
#'   as parent); no code is in two entries, and no group is named like a
#'   code.  A category with no pairs still gets a cell, with `n = 0`.
#' @param memory_budget_bytes The bound on the engine's accumulators.
#' @return An object of class `pgphenotype_relative_pairs`: a list with
#'   `cells` (one per entry, in request order), `settings`, `method` and
#'   `metadata`.  Each cell holds `category` (the code or the group's name),
#'   `members`, `symmetric`, `roles` (`NULL` when symmetric), `n` (unordered
#'   pairs with both values observed), `n_dropped` (with either missing),
#'   `concordance` (a binary symmetric cell's both 0, discordant, both 1),
#'   `table` (a binary directed cell's 2 x 2 counts, rows by the first role's
#'   value) and `crude`: each estimator's `estimator`, `primary`, `value`,
#'   `reason` (why `value` is `NULL`) and `boundary`, the primary first.
#' @examples
#' ped <- data.frame(
#'   id = 1:9,
#'   mother = c(NA, NA, 1, NA, 3, 3, 3, 3, NA),
#'   father = c(NA, NA, 2, NA, 4, 4, 4, 9, NA)
#' )
#' x <- trait(c(0.3, -1.2, 0.8, 2.0, 1.5, -0.4, 0.9, 0.1, 1.1))
#' relative_pairs_correlation(ped, x, categories = list("FS", "MHS", PO = c("MO", "FO")))
#' @export
relative_pairs_correlation <- function(pedigree, trait, categories = NULL, memory_budget_bytes = 2^30) {
  call <- sys.call()
  args <- .pgp_pedigree_args(pedigree, call)
  .pgp_check_class(trait, "pgphenotype_trait", "trait", "trait()", call)
  entries <- .pgp_category_entries(categories, call)
  budget <- .pgp_number(memory_budget_bytes, "memory_budget_bytes", call)
  n_levels <- if (is.null(trait$levels)) NA_real_ else as.double(length(trait$levels))
  raw <- .pgp_call(do.call(.native_relative_pairs_correlation, c(args, list(
    trait$values, trait$kind, n_levels, entries$names, entries$members, budget
  ))), call)
  raw$metadata <- list(pg_phenotype_version = pgphenotype_version(), pedigree_graph_core_rev = pg_core_rev())
  structure(raw, class = "pgphenotype_relative_pairs")
}

# `categories` as parallel entry names and members (NULL for a code).
.pgp_category_entries <- function(categories, call) {
  if (is.null(categories)) {
    return(list(names = NULL, members = NULL))
  }
  bad <- function() {
    .pgp_usage("`categories` must be NULL, a character vector of codes, or a list of codes and named groups", call)
  }
  if (is.character(categories)) {
    if (!is.null(names(categories)) || anyNA(categories)) bad()
    return(list(names = categories, members = vector("list", length(categories))))
  }
  if (!is.list(categories)) bad()
  labels <- names(categories)
  if (is.null(labels)) labels <- rep("", length(categories))
  out <- list(names = character(length(categories)), members = vector("list", length(categories)))
  for (i in seq_along(categories)) {
    item <- categories[[i]]
    if (!is.character(item) || anyNA(item)) bad()
    if (is.na(labels[[i]]) || labels[[i]] == "") {
      if (length(item) != 1L) bad()
      out$names[[i]] <- item
    } else {
      out$names[[i]] <- labels[[i]]
      out$members[i] <- list(item)
    }
  }
  out
}

#' @export
print.pgphenotype_relative_pairs <- function(x, ...) {
  cat("<pgphenotype_relative_pairs>\n")
  for (cell in x$cells) {
    estimates <- vapply(cell$crude, function(e) {
      value <- if (is.null(e$value)) paste("undefined:", e$reason) else sprintf("%.4f", e$value)
      paste(e$estimator, value)
    }, "")
    cat(sprintf("  %s, n %s: %s\n", cell$category, format(cell$n, scientific = FALSE),
                paste(estimates, collapse = ", ")))
  }
  invisible(x)
}
