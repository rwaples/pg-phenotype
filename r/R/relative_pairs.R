#' Relative-Pair Correlation
#'
#' One or two traits' correlation between the members of each Relationship
#' Category's Relative Pairs (pedigree-graph's MZ, MO, FO, FS, MHS, ..., 1C),
#' or of a Category Group's (such as PO for MO and FO).  A continuous trait
#' gets Pearson's r; a binary trait gets the tetrachoric correlation
#' (primary) and phi.  A symmetric category's pairs enter in both orders, so
#' its same-trait Pearson is intraclass and its tetrachoric fits one
#' threshold; a directed category's pairs enter once, in its roles' order
#' (offspring then parent).  With two traits of one kind, each entry gets a
#' cell per trait combination, and `within_person` is the Within-Person
#' Cross-Trait Correlation: each row's trait 1 against its trait 2.  With
#' `se = TRUE`, each estimate also gets a cluster sandwich SE: relative
#' pairs share people, so two pairs are in one cluster when they share a
#' member or their most recent common ancestor (MRCA).  With `stratum`, each
#' cell also gets one cell per known stratum and a stratified fit: a directed pair
#' is in its first role's stratum (the offspring's, the descendant's),
#' whatever the second member's; a symmetric pair is in a stratum only when
#' both members are.  The same inputs give the same numbers as the Python
#' package's `pg_phenotype.relative_pairs.correlation`.
#'
#' @param pedigree A [pedigree()] or a data frame or a named list with
#'   columns `id`, `mother` and `father` (`NA` or -1 when missing),
#'   optionally `twin` and `sex`, checked for this call alone.
#' @param traits A continuous or binary [trait()] on the pedigree's rows, or
#'   a list of one or two of one kind.  An all-missing or constant trait
#'   gives cells with a reason.
#' @param categories `NULL` for every category up to degree 3; else the
#'   entries in order, as a character vector of codes or a list
#'   whose unnamed elements are codes and whose named elements are Category
#'   Groups: `list("MZ", "FS", PO = c("MO", "FO"))`.  A group's codes are all
#'   symmetric, or all directed with the same roles (mother and father read
#'   as parent); no code is in two entries, and no group is named like a
#'   code or `"within_person"`.  A category with no pairs still gets its cells, with `n = 0`.
#' @param stratum `NULL`, or one whole-number label per row (a generation, a
#'   birth cohort), `NA` where unknown.  `within_person` puts each row in its
#'   own stratum.  The pooled fields do not change.
#' @param memory_budget_bytes The bound on the engine's accumulators and,
#'   with two traits, the within-person accumulator, and with `se` the
#'   per-cluster sums.
#' @param se `TRUE` gives every estimate, in `crude`, each stratum's `crude`
#'   and `stratified`, its cluster sandwich SE (ADR 0013) as `se`, or `NULL`
#'   with `se_reason` saying why; `FALSE` (the default) leaves both fields
#'   out.  Each entry (a pair's complete orientation) adds its influence on
#'   the estimate, divided by the fit's entries, to its pair's MRCA, to each
#'   member, and to each (member, MRCA); the meat is the sum of squares over
#'   MRCAs plus over members minus over (member, MRCA), and the SE its square
#'   root, with no small-sample factor.  A pair's MRCA is the ancestor member
#'   of a lineal pair, the shared parent of a half relationship and the
#'   shared mother of a full one (MZ: the twins' mother), and can be an
#'   external parent.  `se_reason` is `"no_estimate"` (the estimate has no
#'   value), `"boundary"` (the tetrachoric fit stopped at a bound),
#'   `"sandwich_undefined"` (the two-step fit's Hessian is not negative
#'   definite, or the meat is not a number), `"degenerate_meat"` (the meat
#'   is zero to within rounding, at most max(1024, 4 sqrt(n)) machine
#'   epsilons of its scale over n entries, as when a cell's pairs share one
#'   MRCA or its estimate is +-1),
#'   `"negative_meat"` (the meat is negative) or `"within_person"` (the
#'   Within-Person cell has no cluster design).  The SE comes from the same
#'   pass over the pedigree as the cells, costs 1.5 to 1.8 times the call's
#'   time with one trait and 2.1 to 3.7 times with two traits and strata,
#'   and does not depend on the thread count.
#' @return An object of class `pgphenotype_relative_pairs`: a list with
#'   `cells` (each entry's, in request order: one with one trait; with two,
#'   a symmetric entry's trait 1 x 1, 2 x 2, 1 x 2 and a directed entry's
#'   1 x 1, 1 x 2, 2 x 1, 2 x 2), `within_person` (with two traits, a cell
#'   over the rows; `NULL` with one), `settings` (`se`, whether the SE was
#'   asked for, and with continuous traits' `quantization_exponents` and
#'   `quantization_steps`, one per trait),
#'   `method` and `metadata`.  Each cell holds `category` (the code, the
#'   group's name, or `"within_person"`), `members`, `symmetric`, `roles`
#'   (`NULL` when symmetric), `first_trait` and `second_trait` (the traits,
#'   counted from 1, read in the first member and the second), `n`
#'   (unordered pairs with a complete orientation: trait `first_trait` of
#'   one member and `second_trait` of the other observed), `n_dropped` (the
#'   other pairs), `n_entries` (a symmetric cross-trait cell's complete
#'   orientations, which its estimates read; else `NULL`), `concordance` (a
#'   binary symmetric same-trait cell's both 0, discordant, both 1), `table`
#'   (any other binary cell's 2 x 2 counts, rows by trait `first_trait` of
#'   one member, a directed cell's first role, and columns by `second_trait`
#'   of the other, a symmetric cell's orientations summed) and `crude`: each
#'   estimator's `estimator`, `primary`, `value`, `reason` (why `value` is
#'   `NULL`), `boundary` and, with `se = TRUE`, `se` and `se_reason`, the
#'   primary first.
#'   With `stratum`, a cell also holds `strata` (one record per known label,
#'   ascending, with its `stratum` and the pooled fields `n`, `n_dropped`,
#'   `n_entries`, `concordance`, `table` and `crude` over its pairs; `n = 0`
#'   when it has none), `n_cross_stratum` (symmetric pairs with a complete
#'   orientation whose members are in two known strata), `n_unknown_stratum`
#'   (pairs with a complete orientation in no stratum for an unknown
#'   stratum: a directed pair's first member's, either member's of a
#'   symmetric pair; `n` is the strata's `n` plus these two) and
#'   `stratified`: the primary estimator
#'   fitted over every pair of two known strata at once (rows by the first
#'   member's stratum, columns by the second's, a threshold or a mean and SD
#'   per stratum and side, one correlation), with `n` (pairs fitted),
#'   `n_entries` (a symmetric cross-trait cell's complete orientations
#'   fitted; else `NULL`), `n_degenerate_stratum` (pairs left out in the strata whose side is
#'   constant, dropped until none is) and `estimate`.  Without `stratum`
#'   these are `NULL`, 0, 0 and `NULL`.
#' @examples
#' ped <- data.frame(
#'   id = 1:9,
#'   mother = c(NA, NA, 1, NA, 3, 3, 3, 3, NA),
#'   father = c(NA, NA, 2, NA, 4, 4, 4, 9, NA)
#' )
#' x <- trait(c(0.3, -1.2, 0.8, 2.0, 1.5, -0.4, 0.9, 0.1, 1.1))
#' relative_pairs_correlation(ped, x, categories = list("FS", "MHS", PO = c("MO", "FO")))
#' relative_pairs_correlation(ped, x, categories = "MO", stratum = c(0, 0, 1, NA, 2, 2, 2, 2, 1))
#' y <- trait(c(1.0, 0.2, -0.5, 0.7, 0.4, 1.3, -0.8, 0.6, 0.4))
#' r <- relative_pairs_correlation(ped, list(x, y), categories = c("FS", "MO"))
#' r$within_person
#' relative_pairs_correlation(ped, x, categories = c("FO", "FS"), se = TRUE)
#' @export
relative_pairs_correlation <- function(pedigree, traits, categories = NULL, stratum = NULL,
                                       memory_budget_bytes = 2^30, se = FALSE) {
  call <- sys.call()
  pedigree <- .pgp_pedigree_arg(pedigree, call)
  traits <- .pgp_trait_list(traits, call)
  entries <- .pgp_category_entries(categories, call)
  budget <- .pgp_number(memory_budget_bytes, "memory_budget_bytes", call)
  se <- .pgp_flag(se, "se", call)
  raw <- .pgp_call(
    .native_relative_pairs_correlation(pedigree, traits, stratum, entries$names, entries$members, budget, se), call
  )
  raw$cells <- lapply(raw$cells, .pgp_traits_from_one)
  if (!is.null(raw$within_person)) raw$within_person <- .pgp_traits_from_one(raw$within_person)
  structure(raw, class = "pgphenotype_relative_pairs")
}

# A cell's trait indices counted from 1, as R indexes `traits`.
.pgp_traits_from_one <- function(cell) {
  cell$first_trait <- cell$first_trait + 1L
  cell$second_trait <- cell$second_trait + 1L
  cell
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
  two <- !is.null(x$within_person)
  cells <- if (two) c(x$cells, list(x$within_person)) else x$cells
  shown <- function(e) {
    out <- paste(e$estimator, if (is.null(e$value)) paste("undefined:", e$reason) else sprintf("%.4f", e$value))
    if (!is.null(e$se)) {
      out <- sprintf("%s (se %.4f)", out, e$se)
    } else if (!is.null(e$se_reason)) {
      out <- sprintf("%s (no se: %s)", out, e$se_reason)
    }
    out
  }
  for (cell in cells) {
    name <- if (two) sprintf("%s %d x %d", cell$category, cell$first_trait, cell$second_trait) else cell$category
    cat(sprintf("  %s, n %s: %s\n", name, format(cell$n, scientific = FALSE),
                paste(vapply(cell$crude, shown, ""), collapse = ", ")))
    s <- cell$stratified
    if (!is.null(s)) {
      cat(sprintf("    stratified, n %s: %s\n", format(s$n, scientific = FALSE), shown(s$estimate)))
    }
  }
  invisible(x)
}

#' ACE Fit of a Relative-Pair Correlation
#'
#' Solves a [relative_pairs_correlation()] result's cells for the additive
#' genetic (A), common environment (C) and unique environment (E) shares, and
#' with two traits their cross-trait correlations.  On the correlation scale
#' (each trait's variance 1) a cell of trait t expects
#' `r = 2 phi A_t + s C_t`, `phi = 2^-(degree+1)` the nominal kinship of its
#' categories and `s` 1 for a Shared-Environment Category, else 0; a Category
#' Group's members must share both.  Each trait's components are the weighted
#' least-squares solution over its same-trait cells, each weighted by the
#' unordered pairs that entered the estimate read (the pooled cell's `n`, or
#' its `stratified$n`), with `E = 1 - A - C`.  A component not requested is
#' fixed at 0.  With two traits the cross-trait cells give
#' `r12 = 2 phi sA + s sC` and the within-person cell `r_wp = sA + sC + sE`,
#' solved together, a cross-trait cell weighted by its complete orientations so
#' each counts once (a symmetric cell's `n_entries` or `stratified$n_entries`,
#' a directed cell's `n`) and the within-person cell by its `n`; then `rA = sA / sqrt(A_1 A_2)`, `rC` and `rE` alike.
#' Values are as solved, unconstrained: a correlation is `NULL` with
#' `nonpositive_component` when its component is <= 0 in a trait, and
#' `|r| > 1` is kept and flagged `out_of_range`.  A component no entered cell
#' informs is `not_identified` and left out, so the others are still solved:
#' without a shared cell, C is `not_identified` and A is solved alone; without
#' a shared cross-trait cell, sC and sE meet only in the within-person cell,
#' which is left out, and sA is solved alone.  Otherwise fewer independent
#' cells than unknowns gives `not_identified`.  There is no SE.
#'
#' The binary primary estimate, the tetrachoric, is on the liability scale, so
#' the shares are too, with no Dempster-Lerner step; `estimator = "phi",
#' estimate = "pooled"` reads phi, on the observed scale.  An exactly
#' identified call is a closed form (`r` a cell's primary estimate):
#' `categories = "FS", components = "A"` gives `A = 2 r_FS`;
#' `categories = c("MZ", "FS"), shared_environment = c("MZ", "FS")` gives
#' Falconer's `A = 2 (r_MZ - r_FS)`, `C = 2 r_FS - r_MZ`;
#' `categories = "PHS", components = "A"` gives `A = 4 r_PHS`;
#' `categories = "1C", components = "A"` gives `A = 8 r_1C`.  The same inputs
#' give the same numbers as the Python package's
#' `pg_phenotype.relative_pairs.ace`.
#'
#' @param result A [relative_pairs_correlation()] result.
#' @param categories `NULL` for every entry of `result`; else the entries to
#'   fit, by category code or group name.  The within-person cell enters the
#'   cross solve either way.
#' @param components `"A"`, `"C"`, or both; E is the remainder.
#' @param shared_environment The codes of the Shared-Environment Categories,
#'   whose pairs share the common environment (such as MZ, FS, MHS when the
#'   household follows the mother); required with C.
#' @param estimator `"primary"` (Pearson or tetrachoric) or `"phi"`.
#' @param estimate `"stratified"` reads each cell's stratified fit, `"pooled"`
#'   its pooled estimate; `NULL` is `"stratified"` when `result` has strata,
#'   else `"pooled"`.
#' @return An object of class `pgphenotype_ace`: a list with `traits` (per
#'   trait, its `trait` counted from 1, `A`, `C` and `E`, each with a
#'   `_reason`, `NULL` unless the value is, `not_requested` or
#'   `not_identified`, and `rows`, the entries that entered), `cross` (with two
#'   traits: `sA`, `sC`, `sE`, `rA`, `rC`, `rE` with their `_reason`, each
#'   correlation's `_out_of_range`, and `rows`; `NULL` with one trait), `rows`
#'   (every cell considered: `entry`, `first_trait` and `second_trait` counted
#'   from 1, `within_person`, `kinship`, `shared`, the `value` read, its
#'   `weight` (a symmetric cross-trait cell's `n_entries`, else its `n`),
#'   `used` and the `reason` it did not enter), `settings` (the request,
#'   `estimate` resolved), `method` and `metadata`.
#' @examples
#' ped <- data.frame(
#'   id = 1:9,
#'   mother = c(NA, NA, 1, NA, 3, 3, 3, 3, NA),
#'   father = c(NA, NA, 2, NA, 4, 4, 4, 9, NA)
#' )
#' x <- trait(c(0.3, -1.2, 0.8, 2.0, 1.5, -0.4, 0.9, 0.1, 1.1))
#' r <- relative_pairs_correlation(ped, x, categories = c("FS", "MHS"))
#' relative_pairs_ace(r, categories = "FS", components = "A")
#' @export
relative_pairs_ace <- function(result, categories = NULL, components = c("A", "C"),
                               shared_environment = NULL, estimator = "primary", estimate = NULL) {
  call <- sys.call()
  if (!inherits(result, "pgphenotype_relative_pairs")) {
    .pgp_usage("`result` must come from relative_pairs_correlation()", call)
  }
  codes <- function(x, name) {
    if (!is.null(x) && (!is.character(x) || anyNA(x))) {
      .pgp_usage(sprintf("`%s` must be NULL or a character vector", name), call)
    }
    x
  }
  one_of <- function(x, values, name) {
    if (!is.character(x) || length(x) != 1L || !(x %in% values)) {
      .pgp_usage(sprintf("`%s` must be one of %s", name, paste0("\"", values, "\"", collapse = ", ")), call)
    }
    x
  }
  categories <- codes(categories, "categories")
  shared_environment <- codes(shared_environment, "shared_environment")
  if (!is.character(components) || length(components) == 0L || anyDuplicated(components) > 0L ||
      !all(components %in% c("A", "C"))) {
    .pgp_usage("`components` must be \"A\", \"C\" or both, each once", call)
  }
  estimator <- one_of(estimator, c("primary", "phi"), "estimator")
  if (is.null(estimate)) {
    estimate <- if (any(vapply(result$cells, function(cell) !is.null(cell$stratified), TRUE))) "stratified" else "pooled"
  }
  estimate <- one_of(estimate, c("stratified", "pooled"), "estimate")
  cells <- lapply(result$cells, .pgp_ace_cell, within_person = FALSE)
  if (!is.null(result$within_person)) {
    cells <- c(cells, list(.pgp_ace_cell(result$within_person, within_person = TRUE)))
  }
  raw <- .pgp_call(
    .native_relative_pairs_ace(cells, categories, components, shared_environment, estimator, estimate), call
  )
  raw$traits <- lapply(raw$traits, function(t) {
    t$trait <- t$trait + 1L
    t
  })
  raw$rows <- lapply(raw$rows, .pgp_traits_from_one)
  structure(raw, class = "pgphenotype_ace")
}

# A result's cell as the binding reads it: traits from 0, the crude
# estimates as parallel vectors with NA where undefined, NULL members within
# person.
.pgp_ace_cell <- function(cell, within_person) {
  or_na <- function(x, na) if (is.null(x)) na else x
  s <- cell$stratified
  list(
    name = cell$category,
    members = if (within_person) NULL else as.character(unlist(cell$members)),
    first_trait = as.double(cell$first_trait - 1L),
    second_trait = as.double(cell$second_trait - 1L),
    estimators = vapply(cell$crude, `[[`, "", "estimator"),
    values = vapply(cell$crude, function(e) or_na(e$value, NA_real_), 0),
    reasons = vapply(cell$crude, function(e) or_na(e$reason, NA_character_), ""),
    n = as.double(cell$n),
    n_entries = if (!is.null(cell$n_entries)) as.double(cell$n_entries),
    stratified = if (!is.null(s)) {
      list(or_na(s$estimate$value, NA_real_), or_na(s$estimate$reason, NA_character_), as.double(s$n),
           if (!is.null(s$n_entries)) as.double(s$n_entries))
    }
  )
}

#' @export
print.pgphenotype_ace <- function(x, ...) {
  cat("<pgphenotype_ace>\n")
  show <- function(fit, key) {
    value <- fit[[key]]
    if (is.null(value)) paste(key, "undefined:", fit[[paste0(key, "_reason")]]) else sprintf("%s %.4f", key, value)
  }
  entries <- function(rows) if (length(rows) == 0L) "none" else paste(unlist(rows), collapse = ", ")
  for (t in x$traits) {
    cat(sprintf("  trait %d: %s; from %s\n", t$trait,
                paste(vapply(c("A", "C", "E"), function(k) show(t, k), ""), collapse = ", "), entries(t$rows)))
  }
  if (!is.null(x$cross)) {
    flagged <- function(k) {
      out <- show(x$cross, k)
      if (isTRUE(x$cross[[paste0(k, "_out_of_range")]])) paste(out, "(out of range)") else out
    }
    cat(sprintf("  cross: %s; from %s\n",
                paste(vapply(c("sA", "sC", "sE", "rA", "rC", "rE"), flagged, ""), collapse = ", "),
                entries(x$cross$rows)))
  }
  cat(sprintf("  %s estimator, %s estimate\n", x$settings$estimator, x$settings$estimate))
  invisible(x)
}
