# The kinds the core reads, in its order (`TraitKind::ALL`).
.pgp_trait_kinds <- function() .native_trait_kinds()

#' One phenotype column
#'
#' A trait's values aligned to the pedigree's input rows, with how they are
#' read.  The same input gives the same trait as the Python package's
#' `pg_phenotype.Trait`.
#'
#' @param values One value per pedigree row, `NA` where unknown: a logical,
#'   numeric, character or factor vector.  Character values are category
#'   labels coded 0, 1, ... in sorted (C-locale) order; a factor is coded
#'   0, 1, ... in its level order, unused levels kept.
#' @param kind `"continuous"`, `"binary"`, `"ordinal"` or `"categorical"`.
#'   `NULL` infers it: logical values and values all in \{0, 1\} are binary,
#'   an ordered factor ordinal, character values and other factors
#'   categorical, and any non-integer number continuous.  Other integer
#'   values need an explicit `kind`.
#' @return An object of class `pgphenotype_trait` holding `values` (doubles,
#'   `NA` where unknown, category codes for character or factor input),
#'   `kind`, and `levels` (the category labels in code order, or `NULL`).
#'   An unknown `kind` is a `pgphenotype_validation_error` with code
#'   `invalid_trait_kind`; integer values beyond \{0, 1\} with no `kind` give
#'   `ambiguous_trait_kind`; a non-integer ordinal or categorical number
#'   gives `invalid_trait_value`.
#' @examples
#' trait(c(1, 0, NA, 1))
#' trait(c(TRUE, FALSE, NA))
#' trait(c(1.7, 2.2, NA))
#' trait(c("b", "a", NA, "c"))
#' trait(c(0L, 2L, 1L), kind = "ordinal")
#' trait(factor(c("mild", "severe", NA), levels = c("mild", "moderate", "severe"), ordered = TRUE))
#' @export
trait <- function(values, kind = NULL) {
  call <- sys.call()
  if (!is.null(kind)) {
    if (!is.character(kind) || length(kind) != 1L || is.na(kind)) {
      .pgp_usage("`kind` must be NULL or a single string", call)
    }
    if (!kind %in% .pgp_trait_kinds()) {
      .pgp_signal(
        "validation", "invalid_trait_kind",
        sprintf("kind must be one of %s, got \"%s\"", paste(.pgp_trait_kinds(), collapse = ", "), kind),
        list(kind = kind), call
      )
    }
  }
  if (!(is.numeric(values) || is.logical(values) || is.character(values) || is.factor(values))) {
    .pgp_usage("`values` must be a logical, numeric, character or factor vector", call)
  }
  levels <- NULL
  if (is.factor(values)) {
    levels <- levels(values)
    codes <- as.double(unclass(values)) - 1
  } else if (is.character(values) && !all(is.na(values))) {
    levels <- sort(unique(values[!is.na(values)]), method = "radix")
    codes <- as.double(match(values, levels)) - 1
  } else {
    codes <- as.double(values)
  }
  if (is.null(kind)) {
    kind <- if (!is.null(levels)) {
      if (is.ordered(values)) "ordinal" else "categorical"
    } else {
      .pgp_infer_kind(codes, is.logical(values), call)
    }
  }
  if (kind %in% c("ordinal", "categorical") && is.null(levels)) {
    bad <- which(!is.na(codes) & codes != trunc(codes))
    if (length(bad)) {
      position <- bad[1L]
      .pgp_signal(
        "validation", "invalid_trait_value",
        sprintf("trait[%d] = %s is not an integer %s code", position, format(codes[position]), kind),
        list(field = "trait", position = as.double(position), value = codes[position]), call
      )
    }
  }
  structure(list(values = codes, kind = kind, levels = levels), class = "pgphenotype_trait")
}

.pgp_infer_kind <- function(codes, logical, call) {
  present <- codes[!is.na(codes)]
  if (logical || all(present == 0 | present == 1)) {
    return("binary")
  }
  if (any(present != trunc(present))) {
    return("continuous")
  }
  .pgp_signal(
    "validation", "ambiguous_trait_kind",
    "integer trait values beyond {0, 1}: pass kind = continuous, ordinal or categorical",
    list(field = "trait"), call
  )
}

#' @export
print.pgphenotype_trait <- function(x, ...) {
  levels <- if (is.null(x$levels)) "" else paste0("; levels ", paste(x$levels, collapse = ", "))
  cat(sprintf(
    "<pgphenotype_trait> %s, %d values, %d missing%s\n",
    x$kind, length(x$values), sum(is.na(x$values)), levels
  ))
  invisible(x)
}
